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
from utils.helpers import is_text_dtype
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
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
        logging.info(f"GeocodeCleaner initialized. use_google={self.use_google}")

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

    # ---------- Geocode with caching ----------
    # def geocode_with_cache(self, query: str) -> Optional[Dict[str, Any]]:
    #     """
    #     Geocode a query using the cache, Google Maps (if available) and Nominatim as a fallback.

    #     :param query: query to geocode
    #     :return: dict with lat, lon and provider (or None if failed)
    #     """

    #     if not isinstance(query, str) or not query.strip():
    #         return None
    #     key = query.strip().lower()
    #     if key in self.cache:
    #         return self.cache[key]
    #     # try google first if available
    #     result = None
    #     if self.use_google and self.gmaps_client:
    #         try:
    #             res = self.gmaps_client.geocode(key)
    #             if res:
    #                 loc = res[0]["geometry"]["location"]
    #                 # self.log.info(f"Google geocode result for {key}: {loc}")
    #                 result = {"lat": float(loc["lat"]), "lon": float(loc["lng"]), "provider": "google"}
    #             # self.log.info(f"Google geocode result completed")
    #         except Exception as e:
    #             # self.log.warning(f"Google geocode error for {key}: {e}")
    #             result = None
    #             raise CustomException(f"Google geocode error for {key}: {e}")
    #     # fallback to nominatim
    #     if result is None:
    #         for attempt in range(self.max_retries):
    #             try:
    #                 # RateLimiter is used; call nom_rate_limiter
    #                 loc = self.nom_rate_limiter(key)
    #                 if loc:
    #                     result = {"lat": float(loc.latitude), "lon": float(loc.longitude), "provider": "nominatim"}
    #                 # self.log.info(f"Nominatim geocode completed")
    #                 break
    #             except GeopyError as ge:
    #                 # self.log.info(f"Nominatim error (attempt {attempt+1}) for {key}: {ge}")
    #                 time.sleep(1 + attempt * 2)
    #                 continue

    #     # store in cache (even negative result)
    #     if result is None:
    #         self.cache[key] = {"lat": None, "lon": None, "provider": None}
    #     else:
    #         self.cache[key] = result
    #     # polite sleep
    #     time.sleep(self.sleep_between_calls)
    #     return self.cache[key]
    
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
                    logging.debug(f"Google geocode error for '{q}': {e}")
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
                        logging.debug(f"Google places error for '{q}': {e}")

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
                    logging.debug(f"Nominatim error for '{q}': {e}")

        # store result in cache (even negative)
        if result is None:
            self.cache[key] = {"lat": None, "lon": None, "provider": None, "tried": tried}
        else:
            self.cache[key] = {"lat": result["lat"], "lon": result["lon"], "provider": result["provider"], "tried": tried}

        try:
            from utils.cache_loader import save_cache
            save_cache(self.cache)
        except Exception as e:
            logging.debug(f"Failed to save cache: {e}")

        logging.debug(f"Geocode attempts for '{query}': {tried}; result: {self.cache[key]}")
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
        logging.info(f"Loaded {len(centers)} school centers for city-hinting.")
        return centers

    def _is_within_school_area(self, school_key: str, lat: float, lon: float) -> bool:
        """
        Check if a given lat, lon is within the area of a school (with key school_key)
        in the school master list.

        If the school center is not found in the school master list, or if the
        center does not have a valid lat/lon, this function returns True (assuming
        the point is within the school area).

        If the distance between the point and the school center is None (e.g. due
        to invalid lat/lon values), this function returns False.

        Otherwise, this function returns True if the distance between the point and
        the school center is less than or equal to the maximum city radius (in km),
        and False otherwise.

        :param school_key: key of the school to check (e.g. 'school_name__school_branch')
        :param lat: latitude of the point to check
        :param lon: longitude of the point to check
        :return: True if the point is within the school area, False otherwise
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

    # ---------- Processing a single file ----------
    def process_file(self, file_path: Path, school_columns_map: Optional[Dict[str,str]] = None, force_regeocode: bool = False):
        """
        file_path: path to CSV file to process
        school_columns_map: optional mapping to identify school name/branch columns in the file (keys: 'school_name','school_branch')
        force_regeocode: if True, geocode even if lat/lon present or in cache
        """
        self.log.info(f"Processing file: {file_path}")

        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path, dtype=str)
        elif file_path.suffix.lower() == ".xlsx":
            df = pd.read_excel(file_path, dtype=str, engine="openpyxl")
        else:
            self.log.warning(f"Skipping unsupported file type: {file_path.name}")

        # df = pd.read_excel(file_path, dtype=str)
        # normalize column names
        df.rename(
        columns={c: c.strip().lower().replace(" ", "_") for c in df.columns}, inplace=True)
        # lowercase textual columns
        for col in list(df.columns):
            if is_text_dtype(df[col]):
                df[col] = df[col].fillna("").astype(str).str.strip().str.lower()
        # detect address-like columns
        address_cols = self._find_address_cols(df)
        self.log.info(f"Found {len(address_cols)} address-like columns in {file_path}")
        # attempt to find school key for validation
        school_key = None
        if school_columns_map:
            sname_col = school_columns_map.get("school_name")
            sbranch_col = school_columns_map.get("school_branch")
            if sname_col and sbranch_col and sname_col in df.columns and sbranch_col in df.columns:
                # take first row's school name/branch for file-level validation (if multiple schools exist, validation will be per-row below)
                try:
                    sname = str(df[sname_col].iloc[0]).strip().lower()
                    sbranch = str(df[sbranch_col].iloc[0]).strip().lower()
                    school_key = f"{sname}__{sbranch}"
                    self.log.info(f"Found school key: {school_key}")

                except Exception:
                    school_key = None

        issues = []
        # for each address column, ensure lat/lon columns and geocode missing values
        for col in address_cols:
            lat_col = f"{col}{self.lat_suffix}"
            lon_col = f"{col}{self.lon_suffix}"
            # ensure lat/lon columns exist
            if lat_col not in df.columns:
                df[lat_col] = ""
            if lon_col not in df.columns:
                df[lon_col] = ""
            # convert existing lat/lon to numeric where present
            df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
            df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

            # identify rows that need geocoding:
            # - lat or lon is missing or NaN OR force_regeocode True
            mask_need = df[lat_col].isna() | df[lon_col].isna() if not force_regeocode else pd.Series(True, index=df.index)
            # but skip empty addresses
            mask_need = mask_need & (df[col].astype(bool))

            # reduce to unique addresses to minimize calls
            unique_addrs = df.loc[mask_need, col].dropna().unique().tolist()
            self.log.info(f"Geocoding {len(unique_addrs)} unique addresses for column '{col}'")

            for addr in tqdm(unique_addrs, desc=f"Geocoding {file_path.name}:{col}", leave=False):
                # res = self.geocode_with_cache(addr)

                # Determine per-row school key
                per_row_school_key = None
                if school_columns_map and school_columns_map.get("school_name") in df.columns:
                    sname_r = str(df.at[i, school_columns_map.get("school_name")]).strip().lower()
                    sbranch_r = str(df.at[i, school_columns_map.get("school_branch")]).strip().lower()
                    per_row_school_key = f"{sname_r}__{sbranch_r}"
                # get hint_city from loaded centers
                hint_city = None
                if getattr(self, "_school_centers", None) is None:
                    self._load_school_centers()
                center = self._school_centers.get(per_row_school_key) if per_row_school_key else None
                if center and center.get("location"):
                    hint_city = center["location"]

                # finally call geocoding with hint
                res = self.geocode_with_cache(addr, hint_city=hint_city)

                # apply to all rows with this addr
                idxs = df.index[df[col] == addr].tolist()
                if res and res.get("lat") is not None:
                    lat_val = float(res["lat"])
                    lon_val = float(res["lon"])
                    # city validation: if we can find school per-row, validate below; else use file-level school_key
                    for i in idxs:
                        # if row already had lat/lon and not force, skip overwrite
                        if not force_regeocode and not pd.isna(df.at[i, lat_col]) and not pd.isna(df.at[i, lon_col]):
                            continue
                        # if school columns present per-row, build per-row school_key
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
            # END for addr
        # END for address_cols

        # write processed file
        out_path = self.proc_dir / file_path.name
        df.to_csv(out_path, index=False)
        self.log.info(f"  wrote processed file: {out_path}")

        # save issues for manual review
        if issues:
            issues_df = pd.DataFrame(issues)
            issues_out = self.log_dir / f"geocode_issues__{file_path.name}"
            issues_df.to_csv(issues_out.with_suffix(".csv"), index=False)
            self.log.info(f"  issues written to {issues_out.with_suffix('.csv')}")
        # save updated cache
        save_cache(self.cache)
        return out_path



    def process_all_files(self, school_columns_map: Optional[Dict[str, str]] = None, force_regeocode: bool = False):
        """
        Process all CSV/XLSX/XLS files from the raw directory.
        Converts Excel to a unique temporary CSV in the system temp folder to avoid
        name collisions or double .tmp suffixes, then calls process_file on that temp CSV.
        """
        supported_exts = {".csv", ".xlsx", ".xls"}
        files = sorted([p for p in self.raw_dir.glob("*") if p.suffix.lower() in supported_exts])

        logging.info(f"Found {len(files)} files in raw dir: {[p.name for p in files]}")
        if not files:
            logging.warning("No CSV or Excel files found in raw dir.")
            return

        for f in files:
            logging.info(f"--- Processing file: {f.name} ---")
            tmp_csv_path = None
            try:
                # Read original file into dataframe (explicit engines for Excel)
                if f.suffix.lower() == ".csv":
                    df = pd.read_csv(f, dtype=str)
                    # we can use the original csv directly (no tmp created)
                    tmp_csv_path = f
                elif f.suffix.lower() == ".xlsx":
                    df = pd.read_excel(f, dtype=str, engine="openpyxl")
                elif f.suffix.lower() == ".xls":
                    df = pd.read_excel(f, dtype=str, engine="xlrd")
                else:
                    logging.warning(f"Skipping unsupported file type: {f.name}")
                    continue

                # If original file was not a CSV, write a unique temp CSV in system temp dir
                if f.suffix.lower() != ".csv":
                    tmp_dir = Path(tempfile.gettempdir())
                    # unique filename: originalstem + uuid4 + timestamp
                    unique_name = f"{f.stem}_{uuid.uuid4().hex[:8]}_{int(time.time())}.tmp.csv"
                    tmp_csv_path = tmp_dir / unique_name
                    df.to_csv(tmp_csv_path, index=False)
                    logging.info(f"Saved temporary CSV for processing: {tmp_csv_path}")

                # Now call process_file on a concrete csv path (tmp_csv_path points to a real file)
                self.process_file(Path(tmp_csv_path), school_columns_map=school_columns_map, force_regeocode=force_regeocode)

            except Exception as e:
                logging.exception(f"Failed processing {f.name}: {e}")

            finally:
                # cleanup temporary CSV if we created one in tempdir
                try:
                    # only remove if tmp file is in the system temp dir (and not the original csv)
                    if tmp_csv_path and tmp_csv_path.exists() and tmp_csv_path.parent == Path(tempfile.gettempdir()):
                        tmp_csv_path.unlink()
                        logging.debug(f"Removed temp file: {tmp_csv_path}")
                except Exception as e:
                    logging.debug(f"Failed to remove temp file {tmp_csv_path}: {e}")

        logging.info("Done processing all files.")


# ---------- CLI ----------
if __name__ == "__main__":
    cleaner = GeocodeCleaner()
    # Optional: provide mapping if your files use different column names for school name/branch
    # Example: {"school_name": "school_name", "school_branch": "school_branch"}
    school_map = {"School Name": "school_name", "School Branch": "school_branch"}
    cleaner.process_all_files(school_columns_map=school_map, force_regeocode=False)
