from pathlib import Path
import os
import time
import math
import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import yaml
from dotenv import load_dotenv
from tqdm import tqdm
from utils.config_loader import load_config
from utils.cache_loader import safe_float, load_cache, save_cache
from utils.haversine_distance import haversine_km
from utils.helpers import is_text_dtype, empty_directory
from logger.custom_logger import CustomLogger
from exception.custom_exception import CustomException
import re
import tempfile
import time
import uuid

# Optional imports - googlemaps may be absent
try:
    import googlemaps
except Exception:
    googlemaps = None

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeopyError

# ---------- Main class ----------
class GeocodeCleaner:
    def __init__(self):

        self.log = CustomLogger().get_logger(__name__)
        # load config
        self.cfg = load_config()
        # setup paths
        self.base = Path.cwd()
        self.raw_dir = self.base / Path(self.cfg["paths"]["raw_data"])
        self.proc_dir = self.base / Path(self.cfg["paths"]["processed_data"])
        self.issues_dir = self.base / Path(self.cfg["paths"]["issues_data"])
        self.issues_dir.mkdir(parents=True, exist_ok=True)
        # self.cache_path = self.base / Path(self.cfg["paths"].get("geocode_cache", "data/geocode_cache.csv"))
        self.log_dir = self.base / Path(self.cfg["paths"].get("logs", "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.proc_dir.mkdir(parents=True, exist_ok=True)
        # load env
        load_dotenv()
        self.gmaps_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
        self.use_google = bool(self.gmaps_key) and (googlemaps is not None)
        self.gmaps_client = googlemaps.Client(key=self.gmaps_key) if self.use_google else None
        # set geocoder fallback
        self.nominatim = Nominatim(user_agent=self.cfg.get("app_user_agent", "school-transport-optimizer"))
        self.nom_rate_limiter = RateLimiter(self.nominatim.geocode, min_delay_seconds=self.cfg["geocode"].get("nominatim_delay", 1.1))
        # behavior params
        self.max_retries = self.cfg["geocode"].get("max_retries", 3)
        self.sleep_between_calls = self.cfg["geocode"].get("sleep_between_calls", 0.2)
        self.address_keywords = [k.lower() for k in self.cfg["address_keywords"]]
        self.lat_suffix = self.cfg.get("lat_suffix", "_lat")
        self.lon_suffix = self.cfg.get("lon_suffix", "_lon")
        # city validation params
        self.school_master_file = Path(self.cfg["paths"]["school_master"])
        self.max_city_radius_km = float(self.cfg["validation"].get("max_city_radius_km", 30.0))
        # load cache
        self.cache = load_cache()
        # configure logging
        # logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
        # logging.info(f"GeocodeCleaner initialized. use_google={self.use_google}")

    # ---------- Helpers ----------

    def _normalize_text(self, x: str) -> str:
        """Lowercase and remove non-alphanumeric characters for stable matching."""
        if not isinstance(x, str):
            return ""
        x = x.strip().lower()
        # replace runs of non-alphanumeric with single underscore
        x = re.sub(r"[^0-9a-z]+", "_", x)
        # strip leading/trailing underscores
        x = x.strip("_")
        return x
    
    
    def _find_address_cols(self, df: pd.DataFrame) -> List[str]:
        """
        Robustly find address-like columns by normalizing both column names and keywords.
        """
        cols = []
        # pre-normalize keywords once
        normalized_keywords = [ self._normalize_text(k) for k in self.address_keywords ]
        self.log.info(f"Normalized address keywords: {normalized_keywords}")

        for c in df.columns:
            low = str(c).strip().lower()
            norm_col = self._normalize_text(low)
            self.log.debug(f"Checking column '{c}' -> normalized: '{norm_col}'")
            # Match if any normalized keyword is substring of normalized column name
            if any(k in norm_col for k in normalized_keywords):
                cols.append(c)
        self.log.info(f"Found address columns: {cols}")
        return cols

    
    def geocode_with_cache(self, query: str, hint_city: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Geocode a query using cache + multiple fallbacks.
        If hint_city provided, try variants with the city appended.
        """
        if not isinstance(query, str) or not query.strip():
            return None
        key = query.strip().lower()
        if key in self.cache:
            return self.cache[key]

        tried = []
        variants = [query.strip()]
        # normalized variant (remove punctuation)
        import re
        norm = re.sub(r"[^\w\s]", " ", query).strip()
        if norm and norm not in variants:
            variants.append(norm)
        # append hint_city variants
        if hint_city:
            hc = hint_city.strip()
            v1 = f"{query.strip()}, {hc}"
            v2 = f"{norm}, {hc}"
            for v in (v1, v2):
                if v not in variants:
                    variants.append(v)
        # try Google Geocode (if available), then Google Places, then Nominatim
        result = None
        if self.use_google and self.gmaps_client:
            for q in variants:
                tried.append(("google_geocode", q))
                try:
                    res = self.gmaps_client.geocode(q)
                    if res:
                        loc = res[0]["geometry"]["location"]
                        result = {"lat": float(loc["lat"]), "lon": float(loc["lng"]), "provider": "google_geocode"}
                        break
                except Exception as e:
                    self.log.info(f"Google geocode error for '{q}': {e}")
                    # self.debug(f"Google geocode error for '{q}': {e}")
            # try Places if geocode failed
            if result is None:
                for q in variants:
                    tried.append(("google_places", q))
                    try:
                        places_res = self.gmaps_client.places(query=q)
                        if places_res and places_res.get("results"):
                            place = places_res["results"][0]
                            loc = place["geometry"]["location"]
                            result = {"lat": float(loc["lat"]), "lon": float(loc["lng"]), "provider": "google_places"}
                            break
                    except Exception as e:
                        self.log.error(f"Google places error for '{q}': {e}")
                        # logging.debug(f"Google places error for '{q}': {e}")

        # fallback to Nominatim
        if result is None:
            for q in variants:
                tried.append(("nominatim", q))
                try:
                    loc = self.nom_rate_limiter(q)
                    if loc:
                        result = {"lat": float(loc.latitude), "lon": float(loc.longitude), "provider": "nominatim"}
                        break
                except Exception as e:
                    self.log.error(f"Nominatim error for '{q}': {e}")
                    # logging.debug(f"Nominatim error for '{q}': {e}")

        # store result in cache (even negative)
        if result is None:
            self.cache[key] = {"lat": None, "lon": None, "provider": None, "tried": tried}
        else:
            self.cache[key] = {"lat": result["lat"], "lon": result["lon"], "provider": result["provider"], "tried": tried}

        try:
            save_cache(self.cache)
        except Exception as e:
            self.log.error(f"Failed to save cache: {e}")

        self.log.info(f"Geocode attempts for '{query}': {tried}; result: {self.cache[key]}")
        return self.cache[key]


    # ---------- City validation ----------
    def _load_school_centers(self) -> Dict[str, Dict[str, Any]]:
        # expects school_master path csv with columns school_name, school_branch, school_location (human) or lat/lon present
        """
        Load school centers from a CSV file.

        Expects a CSV file at `self.school_master_file` with columns:

        - school_name
        - school_branch
        - school_location (human-readable, optional)
        - school_lat (optional)
        - school_lon (optional)

        Returns a dictionary with school_name__school_branch as keys and
        another dictionary with lat, lon and location_text as values.

        If the file is not found, city validation is disabled and an empty
        dictionary is returned.
        """
        p = Path(self.school_master_file)
        if not p.exists():
            self.log.warning(f"School master {p} not found; city validation disabled")
            return {}
        
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p, dtype=str).fillna("")
        elif p.suffix.lower() == ".xlsx":
            df = pd.read_excel(p, dtype=str, engine="openpyxl").fillna("")
        else:
            self.log.warning(f"Skipping unsupported file type: {p.name}")


        # df = pd.read_excel(p, dtype=str).fillna("")
        # normalize columns
        df.rename(columns={c: c.strip().lower().replace(" ", "_") for c in df.columns}, inplace=True)
        centers = {}
        for _, r in df.iterrows():
            sname = str(r.get("school_name", "")).strip().lower()
            sbranch = str(r.get("school_branch", "")).strip().lower()
            if not sname:
                continue
            key = f"{sname}__{sbranch}"
            # pick location text from 'school_location' (or other fallback columns)
            location_text = r.get("school_location", "") or r.get("location", "") or ""
            lat = safe_float(r.get("school_lat") or r.get("schoollat") or r.get("school_latitude"))
            lon = safe_float(r.get("school_lon") or r.get("schoollon") or r.get("school_longitude"))
            centers[key] = {"lat": lat, "lon": lon, "location": location_text.strip()}
        self._school_centers = centers
        self.log.info(f"Loaded {len(centers)} school centers for city-hinting.")
        return centers

    def _is_within_school_area(self, school_key: str, lat: float, lon: float) -> bool:
        """
        Check if a given lat, lon is within the area of a school (with key school_key)
        in the school master list.
        """
        centers = getattr(self, "_school_centers", None)
        if centers is None:
            self._school_centers = self._load_school_centers()
            centers = self._school_centers
        center = centers.get(school_key)
        if not center:
            # no center available -> can't validate
            return True
        if center["lat"] is None or center["lon"] is None:
            return True
        d = haversine_km(center["lat"], center["lon"], lat, lon)
        if d is None:
            return False
        return d <= self.max_city_radius_km



    def process_file(self,
                        csv_path: Path,
                        school_columns_map: Optional[Dict[str, str]] = None,
                        force_regeocode: bool = False,
                        original_raw_file: Optional[Path] = None) -> pd.DataFrame:
            """
            Process a single CSV file path and return the processed dataframe.
            - Normalizes column names to snake_case lowercase.
            - Lowercases text columns.
            - Detects address-like columns and, if geocoding available, geocodes missing lat/lon.
            NOTE: This function DOES NOT write the final file; caller writes canonical file.
            """
            self.log.info(f"Processing CSV for normalization: {csv_path.name}")
            # Read CSV into df (ensure encoding safe)
            df = pd.read_csv(csv_path, dtype=str).fillna("")

            # Normalize column names to snake_case
            normalized_cols = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
            df.rename(columns=normalized_cols, inplace=True)
            self.log.debug(f"Normalized columns: {list(df.columns)}")

            # Lowercase textual columns
            for col in df.columns:
                if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                    df[col] = df[col].astype(str).str.strip().str.lower()

            # Detect address-like columns (basic heuristic from config)
            keywords = [k.strip().lower() for k in self.cfg.get("address_keywords", ["address", "pickup", "drop", "location", "parking", "society", "stop", "depot"])]
            address_cols = [c for c in df.columns if any(kw.replace(" ", "_") in c or kw in c for kw in keywords)]
            self.log.info(f"Detected address-like columns: {address_cols}")

            # For each address column, ensure lat/lon existence and geocode missing points if geocode available
            lat_suffix = self.cfg.get("lat_suffix", "_lat")
            lon_suffix = self.cfg.get("lon_suffix", "_lon")

            # Prepare school hint mapping if requested: mapping school_name__school_branch -> school_location
            school_hints = {}
            school_master_path = Path(self.cfg.get("paths", {}).get("school_master", ""))  # may be Excel or CSV
            if school_master_path and school_master_path.exists():
                try:
                    # support excel/csv for school master
                    if school_master_path.suffix.lower() in [".xlsx", ".xls"]:
                        sm_df = pd.read_excel(school_master_path, dtype=str, engine="openpyxl")
                    else:
                        sm_df = pd.read_csv(school_master_path, dtype=str)
                    sm_df.rename(columns={c: c.strip().lower().replace(" ", "_") for c in sm_df.columns}, inplace=True)
                    for _, r in sm_df.fillna("").iterrows():
                        key = f"{str(r.get('school_name','')).strip().lower()}__{str(r.get('school_branch','')).strip().lower()}"
                        # prefer school_location column if present
                        school_hints[key] = str(r.get('school_location','')).strip()
                except Exception as e:
                    self.log.debug(f"Could not load school master for hints: {e}")

            # Load cache if loader present
            try:
                cache = load_cache()
            except Exception:
                cache = {}

            # Simple geocode call if method exists on self
            do_geocode = hasattr(self, "geocode_with_cache") and callable(getattr(self, "geocode_with_cache"))

            for col in address_cols:
                lat_col = f"{col}{lat_suffix}"
                lon_col = f"{col}{lon_suffix}"
                # ensure lat/lon columns exist
                if lat_col not in df.columns:
                    df[lat_col] = ""
                if lon_col not in df.columns:
                    df[lon_col] = ""

                # convert to numeric where present
                df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
                df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

                # rows needing geocode
                mask_need = df[lat_col].isna() | df[lon_col].isna() if not force_regeocode else pd.Series(True, index=df.index)
                mask_need = mask_need & df[col].astype(bool)

                # dedupe addresses to minimize API calls
                unique_addrs = df.loc[mask_need, col].dropna().unique().tolist()
                self.log.info(f"{len(unique_addrs)} unique addresses to geocode for column '{col}'")

                for addr in unique_addrs:
                    # determine hint city if possible (try to get school hint from row sample)
                    hint_city = None
                    # attempt to find a row where this addr occurs and pull school info
                    sample_idx = df.index[df[col] == addr].tolist()
                    if sample_idx:
                        i = sample_idx[0]
                        try:
                            sname = str(df.at[i, "school_name"]).strip().lower() if "school_name" in df.columns else ""
                            sbranch = str(df.at[i, "school_branch"]).strip().lower() if "school_branch" in df.columns else ""
                            key = f"{sname}__{sbranch}"
                            if key in school_hints and school_hints[key]:
                                hint_city = school_hints[key]
                        except Exception:
                            hint_city = None

                    if do_geocode:
                        try:
                            res = self.geocode_with_cache(addr, hint_city=hint_city)
                        except TypeError:
                            # backward compat: geocode_with_cache may not accept hint_city
                            res = self.geocode_with_cache(addr)
                    else:
                        # No geocoder available - skip geocoding
                        res = None

                    # apply results to all rows that match this address
                    idxs = df.index[df[col] == addr].tolist()
                    if res and res.get("lat") is not None:
                        lat_val = float(res["lat"])
                        lon_val = float(res["lon"])
                        for i in idxs:
                            if not force_regeocode and not pd.isna(df.at[i, lat_col]) and not pd.isna(df.at[i, lon_col]):
                                continue
                    #         df.at[i, lat_col] = lat_val
                    #         df.at[i, lon_col] = lon_val
                    # else:
                    #     # log issue rows for manual review (keep them as blank lat/lon)
                    #     for i in idxs:
                    #         self.log.debug(f"Geocode failed for file={csv_path.name}, row={i}, address_col={col}, address='{addr}'")
                        per_row_school_key = school_key
                        if school_columns_map and school_columns_map.get("school_name") in df.columns:
                            try:
                                sname_r = str(df.at[i, school_columns_map.get("school_name")]).strip().lower()
                                sbranch_r = str(df.at[i, school_columns_map.get("school_branch")]).strip().lower()
                                if sname_r:
                                    per_row_school_key = f"{sname_r}__{sbranch_r}"
                                self.log.info(f"Found per-row school key: {per_row_school_key}")
                            except Exception:
                                per_row_school_key = school_key
                        # city/proximity check
                        if per_row_school_key and not self._is_within_school_area(per_row_school_key, lat_val, lon_val):
                            # issue: geocode result far from school center
                            issues.append({
                                "file": file_path.name,
                                "column": col,
                                "row_index": int(i),
                                "address": addr,
                                "geocoded_lat": lat_val,
                                "geocoded_lon": lon_val,
                                "issue": "out_of_city_radius"
                            })
                            # still write lat/lon but flagged (alternatively skip writing)
                            df.at[i, lat_col] = lat_val
                            df.at[i, lon_col] = lon_val
                        else:
                            df.at[i, lat_col] = lat_val
                            df.at[i, lon_col] = lon_val
                else:
                    # geocode failed; log issue and leave blank
                    for i in idxs:
                        issues.append({
                            "file": file_path.name,
                            "column": col,
                            "row_index": int(i),
                            "address": addr,
                            "issue": "geocode_failed"
                        })
                # save cache if updated
                try:
                    from utils.cache_loader import save_cache
                    save_cache(cache)
                except Exception:
                    pass

            # Any other processing/normalization you want can be placed here.
            # For example: generate sibling group ids from same pickup address, etc.

            self.log.info(f"Completed processing for {csv_path.name}. Returning DataFrame to write canonical file.")
            return df


        # def process_all_files(self, school_columns_map: Optional[Dict[str, str]] = None, force_regeocode: bool = False):
        #     """
        #     Process all CSV/XLSX/XLS files from the raw directory.
        #     Converts Excel to a unique temporary CSV in the system temp folder to avoid
        #     name collisions or double .tmp suffixes, then calls process_file on that temp CSV.
        #     """
        #     supported_exts = {".csv", ".xlsx", ".xls"}
        #     files = sorted([p for p in self.raw_dir.glob("*") if p.suffix.lower() in supported_exts])

        #     logging.info(f"Found {len(files)} files in raw dir: {[p.name for p in files]}")
        #     if not files:
        #         logging.warning("No CSV or Excel files found in raw dir.")
        #         return

        #     for f in files:
        #         logging.info(f"--- Processing file: {f.name} ---")
        #         tmp_csv_path = None
        #         try:
        #             # Read original file into dataframe (explicit engines for Excel)
        #             if f.suffix.lower() == ".csv":
        #                 df = pd.read_csv(f, dtype=str)
        #                 # we can use the original csv directly (no tmp created)
        #                 tmp_csv_path = f
        #             elif f.suffix.lower() == ".xlsx":
        #                 df = pd.read_excel(f, dtype=str, engine="openpyxl")
        #             elif f.suffix.lower() == ".xls":
        #                 df = pd.read_excel(f, dtype=str, engine="xlrd")
        #             else:
        #                 logging.warning(f"Skipping unsupported file type: {f.name}")
        #                 continue

        #             # If original file was not a CSV, write a unique temp CSV in system temp dir
        #             if f.suffix.lower() != ".csv":
        #                 tmp_dir = Path(tempfile.gettempdir())
        #                 # unique filename: originalstem + uuid4 + timestamp
        #                 unique_name = f"{f.stem}_{uuid.uuid4().hex[:8]}_{int(time.time())}.tmp.csv"
        #                 tmp_csv_path = tmp_dir / unique_name
        #                 df.to_csv(tmp_csv_path, index=False)
        #                 logging.info(f"Saved temporary CSV for processing: {tmp_csv_path}")

        #             # Now call process_file on a concrete csv path (tmp_csv_path points to a real file)
        #             self.process_file(Path(tmp_csv_path), school_columns_map=school_columns_map, force_regeocode=force_regeocode)

        #         except Exception as e:
        #             logging.exception(f"Failed processing {f.name}: {e}")

        #         finally:
        #             # cleanup temporary CSV if we created one in tempdir
        #             try:
        #                 # only remove if tmp file is in the system temp dir (and not the original csv)
        #                 if tmp_csv_path and tmp_csv_path.exists() and tmp_csv_path.parent == Path(tempfile.gettempdir()):
        #                     tmp_csv_path.unlink()
        #                     logging.debug(f"Removed temp file: {tmp_csv_path}")
        #             except Exception as e:
        #                 logging.debug(f"Failed to remove temp file {tmp_csv_path}: {e}")

        #     logging.info("Done processing all files.")

    def process_all_files(self,
                          school_columns_map: Optional[Dict[str, str]] = None,
                          force_regeocode: bool = False):
        """
        Process all CSV/XLSX/XLS files from the raw directory.
        For Excel files, create a unique temp CSV for processing.
        Each processed output overwrites canonical file in processed_dir:
          raw file 'Student_Master.xlsx' -> processed file 'student_master.csv'
        """
        supported_exts = {".csv", ".xlsx", ".xls"}
        files = sorted([p for p in Path(self.raw_dir).glob("*") if p.suffix.lower() in supported_exts])

        self.log.info(f"Found {len(files)} files in raw dir: {[p.name for p in files]}")
        if not files:
            self.log.warning("No CSV or Excel files found in raw dir.")
            return

        for f in files:
            self.log.info(f"--- Processing raw file: {f.name} ---")
            tmp_csv_path = None
            created_tmp = False
            try:
                # Read original file into DataFrame (explicit Excel engine)
                if f.suffix.lower() == ".csv":
                    df = pd.read_csv(f, dtype=str)
                    tmp_csv_path = f  # use the csv directly
                elif f.suffix.lower() == ".xlsx":
                    df = pd.read_excel(f, dtype=str, engine="openpyxl")
                elif f.suffix.lower() == ".xls":
                    df = pd.read_excel(f, dtype=str, engine="xlrd")
                else:
                    self.log.warning(f"Skipping unsupported file type: {f.name}")
                    continue

                # For Excel inputs, write a temp CSV in system temp dir (unique name)
                if f.suffix.lower() != ".csv":
                    tmp_dir = Path(tempfile.gettempdir())
                    unique_name = f"{f.stem}_{uuid.uuid4().hex[:8]}_{int(time.time())}.tmp.csv"
                    tmp_csv_path = tmp_dir / unique_name
                    df.to_csv(tmp_csv_path, index=False)
                    created_tmp = True
                    self.log.debug(f"Saved temporary CSV for processing: {tmp_csv_path}")

                # Now perform uniform processing from the CSV (tmp_csv_path)
                processed_df = self.process_file(Path(tmp_csv_path),
                                                school_columns_map=school_columns_map,
                                                force_regeocode=force_regeocode,
                                                original_raw_file=f)

                # Determine canonical processed filename (overwrite)
                canonical_name = f.stem.lower().replace(" ", "_") + ".csv"
                out_path = Path(self.proc_dir) / canonical_name
                Path(self.proc_dir).mkdir(parents=True, exist_ok=True)

                # Save processed dataframe (overwrite canonical file)
                processed_df.to_csv(out_path, index=False)
                self.log.info(f"Wrote canonical processed file: {out_path}")

            except Exception as e:
                self.log.exception(f"Failed processing {f.name}: {e}")

            finally:
                # Clean up temp CSV if we created one in tempdir
                try:
                    if created_tmp and tmp_csv_path and tmp_csv_path.exists():
                        tmp_csv_path.unlink()
                        self.log.debug(f"Removed temp file: {tmp_csv_path}")
                except Exception as e:
                    self.log.debug(f"Failed to remove temp file {tmp_csv_path}: {e}")

        self.log.info("Done processing all files.")

# ---------- CLI ----------
if __name__ == "__main__":
    cleaner = GeocodeCleaner()
    # Optional: provide mapping if your files use different column names for school name/branch
    # Example: {"school_name": "school_name", "school_branch": "school_branch"}
    school_map = {"School Name": "school_name", "School Branch": "school_branch"}
    cleaner.process_all_files(school_columns_map=school_map, force_regeocode=False)
