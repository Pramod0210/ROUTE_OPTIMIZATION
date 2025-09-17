def process_file(self,
                 csv_path: Path,
                 school_columns_map: Optional[Dict[str, str]] = None,
                 force_regeocode: bool = False,
                 original_raw_file: Optional[Path] = None) -> pd.DataFrame:
    """
    Process a single CSV (csv_path) and return the processed DataFrame.
    Responsibilities:
      - normalize column names (snake_case)
      - lowercase text columns
      - detect address-like columns and ensure <col>_lat / <col>_lon exist
      - geocode missing coordinates using self.geocode_with_cache (if present)
      - validate geocodes against school area using self._is_within_school_area()
      - set <col>_geocode_flag for problematic rows ("geocode_failed", "out_of_radius:NN.NNkm")
      - collect file-level issues to logs/<file>_geocode_issues.csv
    Notes:
      - relies on self.cfg for lat/lon suffixes and validation.max_city_radius_km
      - will call save_cache() from utils.cache_loader if available
    """
    import math
    from pathlib import Path
    import pandas as pd

    self.log.info(f"Processing CSV: {csv_path}")

    # read CSV
    df = pd.read_csv(csv_path, dtype=str).fillna("")

    # normalize column names
    df.rename(columns={c: c.strip().lower().replace(" ", "_") for c in df.columns}, inplace=True)

    # lowercase textual columns (safe)
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip().str.lower()

    # address keyword heuristics from config
    addr_keywords_cfg = self.cfg.get("address_keywords", ["address", "pickup", "drop", "location", "parking", "society", "stop", "depot"])
    # normalize keywords for matching (replace spaces with underscore)
    addr_keywords = [k.strip().lower().replace(" ", "_") for k in addr_keywords_cfg]

    lat_suffix = self.cfg.get("lat_suffix", "_lat")
    lon_suffix = self.cfg.get("lon_suffix", "_lon")

    # detect candidate address columns
    address_cols = [c for c in df.columns if any(k in c for k in addr_keywords)]
    self.log.info(f"Detected address-like columns: {address_cols}")

    # ensure school centers loaded for proximity checks
    if getattr(self, "_school_centers", None) is None:
        try:
            self._load_school_centers()
        except Exception as e:
            self.log.debug(f"_load_school_centers() failed: {e}")
            self._school_centers = {}

    # prepare cache utilities if available
    try:
        from utils.cache_loader import load_cache, save_cache
        cache = load_cache()
    except Exception:
        cache = {}
        save_cache = None

    # geocode availability
    do_geocode = hasattr(self, "geocode_with_cache") and callable(getattr(self, "geocode_with_cache"))

    # prepare per-file issues list
    file_issues = []


    # validation radius (km)
    max_radius_km = float(self.cfg.get("validation", {}).get("max_city_radius_km", 50.0))

    # For each address-like column: ensure lat/lon exist and geocode missing
    for col in address_cols:
        lat_col = f"{col}{lat_suffix}"
        lon_col = f"{col}{lon_suffix}"
        flag_col = f"{col}_geocode_flag"

        if lat_col not in df.columns:
            df[lat_col] = ""
        if lon_col not in df.columns:
            df[lon_col] = ""
        if flag_col not in df.columns:
            df[flag_col] = ""

        # convert lat/lon columns to numeric where present
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

        # rows that need geocoding (missing coords) or force_regeocode
        mask_need = (df[lat_col].isna() | df[lon_col].isna()) if not force_regeocode else pd.Series([True] * len(df), index=df.index)
        mask_need = mask_need & df[col].astype(bool)

        unique_addrs = df.loc[mask_need, col].dropna().unique().tolist()
        self.log.info(f"{len(unique_addrs)} unique addresses to geocode for column '{col}'")

        for addr in unique_addrs:
            # sample rows for this address
            idxs = df.index[df[col] == addr].tolist()
            sample_idx = idxs[0] if idxs else None

            # determine per-row school key & hint_city (if school_columns_map provided)
            per_row_school_key = None
            hint_city = None
            if sample_idx is not None and school_columns_map:
                try:
                    sname_col = school_columns_map.get("school_name")
                    sbranch_col = school_columns_map.get("school_branch")
                    if sname_col in df.columns and sbranch_col in df.columns:
                        sname = str(df.at[sample_idx, sname_col]).strip().lower()
                        sbranch = str(df.at[sample_idx, sbranch_col]).strip().lower()
                        if sname:
                            per_row_school_key = f"{sname}__{sbranch}"
                            center = getattr(self, "_school_centers", {}).get(per_row_school_key)
                            if center and center.get("location"):
                                hint_city = center["location"]
                except Exception:
                    per_row_school_key = None
                    hint_city = None

            # call geocode (try to pass hint_city if supported)
            res = None
            if do_geocode:
                try:
                    # prefer hint_city-aware call
                    try:
                        res = self.geocode_with_cache(addr, hint_city=hint_city)
                    except TypeError:
                        res = self.geocode_with_cache(addr)
                except Exception as e:
                    self.log.debug(f"geocode_with_cache exception for '{addr}': {e}")
                    res = None

            # apply or flag results
            if res and res.get("lat") is not None:
                lat_val = float(res["lat"])
                lon_val = float(res["lon"])

                # proximity check against school center if available
                too_far = False
                dist_km = None
                if per_row_school_key:
                    center = getattr(self, "_school_centers", {}).get(per_row_school_key)
                    if center and center.get("lat") is not None and center.get("lon") is not None:
                        dist_m = _haversine_m(center["lat"], center["lon"], lat_val, lon_val)
                        if dist_m is not None:
                            dist_km = dist_m / 1000.0
                            if dist_km > max_radius_km:
                                too_far = True

                # if too far and we have a hint_city, try retry with appended hint
                if too_far and hint_city:
                    appended = f"{addr}, {hint_city}"
                    try:
                        try:
                            retry_res = self.geocode_with_cache(appended, hint_city=hint_city)
                        except TypeError:
                            retry_res = self.geocode_with_cache(appended)
                        if retry_res and retry_res.get("lat") is not None:
                            lat_val = float(retry_res["lat"])
                            lon_val = float(retry_res["lon"])
                            # recompute distance
                            if per_row_school_key and center and center.get("lat") is not None:
                                dist_m2 = _haversine_m(center["lat"], center["lon"], lat_val, lon_val)
                                dist_km2 = (dist_m2 / 1000.0) if dist_m2 is not None else None
                                if dist_km2 is not None and dist_km2 <= max_radius_km:
                                    too_far = False
                                    dist_km = dist_km2
                                else:
                                    too_far = True
                                    dist_km = dist_km2
                            else:
                                # accept retry if no center to compare
                                too_far = False
                                dist_km = None
                            # update cache variable (if used)
                            res = retry_res
                    except Exception as e:
                        self.log.debug(f"Retry geocode exception for '{addr}': {e}")

                # Final apply or flag
                if not too_far:
                    for i in idxs:
                        # do not overwrite existing coords unless force_regeocode
                        if not force_regeocode and not pd.isna(df.at[i, lat_col]) and not pd.isna(df.at[i, lon_col]):
                            continue
                        df.at[i, lat_col] = lat_val
                        df.at[i, lon_col] = lon_val
                        df.at[i, flag_col] = ""  # clear flag
                else:
                    # mark as out_of_radius and record issue
                    for i in idxs:
                        # optionally still write the geocoded coords but flag them
                        df.at[i, lat_col] = lat_val
                        df.at[i, lon_col] = lon_val
                        flag_msg = f"out_of_radius:{dist_km:.2f}km" if dist_km is not None else "out_of_radius"
                        df.at[i, flag_col] = flag_msg
                        file_issues.append({
                            "file": Path(csv_path).name,
                            "row_index": int(i),
                            "address_col": col,
                            "address": addr,
                            "geocoded_lat": lat_val,
                            "geocoded_lon": lon_val,
                            "issue": "out_of_radius",
                            "distance_km": dist_km
                        })
                        self.log.warning(f"Geocode out_of_radius file={Path(csv_path).name} row={i} addr='{addr}' dist_km={dist_km}")
            else:
                # geocode failed - flag all matching rows
                for i in idxs:
                    df.at[i, flag_col] = "geocode_failed"
                    file_issues.append({
                        "file": Path(csv_path).name,
                        "row_index": int(i),
                        "address_col": col,
                        "address": addr,
                        "issue": "geocode_failed"
                    })
                    self.log.debug(f"Geocode failed for file={Path(csv_path).name} row={i} addr='{addr}'")

        # end for each unique addr

        # save cache if updated
        try:
            if save_cache is not None:
                save_cache(cache)
        except Exception as e:
            self.log.debug(f"save_cache failed: {e}")

    # end for each address column

    # write file_issues if any
    try:
        if file_issues:
            issues_df = pd.DataFrame(file_issues)
            logs_dir = Path(self.cfg.get("paths", {}).get("logs", "logs"))
            logs_dir.mkdir(parents=True, exist_ok=True)
            out_issues = logs_dir / f"geocode_issues__{Path(csv_path).stem}.csv"
            # append if exists to preserve previous runs
            if out_issues.exists():
                issues_df.to_csv(out_issues, mode="a", header=False, index=False)
            else:
                issues_df.to_csv(out_issues, index=False)
            self.log.info(f"Wrote {len(file_issues)} geocode issues to {out_issues}")
    except Exception as e:
        self.log.debug(f"Failed to write geocode issues: {e}")

    self.log.info(f"Completed processing for {csv_path.name}")
    return df
