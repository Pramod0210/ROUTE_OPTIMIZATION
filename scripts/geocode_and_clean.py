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
    def geocode_with_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Geocode a query using the cache, Google Maps (if available) and Nominatim as a fallback.

        :param query: query to geocode
        :return: dict with lat, lon and provider (or None if failed)
        """

        if not isinstance(query, str) or not query.strip():
            return None
        key = query.strip().lower()
        if key in self.cache:
            return self.cache[key]
        # try google first if available
        result = None
        if self.use_google and self.gmaps_client:
            try:
                res = self.gmaps_client.geocode(key)
                if res:
                    loc = res[0]["geometry"]["location"]
                    # self.log.info(f"Google geocode result for {key}: {loc}")
                    result = {"lat": float(loc["lat"]), "lon": float(loc["lng"]), "provider": "google"}
                self.log.info(f"Google geocode result completed")
            except Exception as e:
                self.log.warning(f"Google geocode error for {key}: {e}")
                result = None
                raise CustomException(f"Google geocode error for {key}: {e}")
        # fallback to nominatim
        if result is None:
            for attempt in range(self.max_retries):
                try:
                    # RateLimiter is used; call nom_rate_limiter
                    loc = self.nom_rate_limiter(key)
                    if loc:
                        result = {"lat": float(loc.latitude), "lon": float(loc.longitude), "provider": "nominatim"}
                        self.log.info(f"Nominatim geocode result for {key}: {result}")
                    break
                except GeopyError as ge:
                    self.log.info(f"Nominatim error (attempt {attempt+1}) for {key}: {ge}")
                    time.sleep(1 + attempt * 2)
                    continue

        # store in cache (even negative result)
        if result is None:
            self.cache[key] = {"lat": None, "lon": None, "provider": None}
        else:
            self.cache[key] = result
        # polite sleep
        time.sleep(self.sleep_between_calls)
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
            key = f"{r.get('school_name','').strip().lower()}__{r.get('school_branch','').strip().lower()}"
            lat = safe_float(r.get("schoollat") or r.get("school_lat") or r.get("school_latitude"))
            lon = safe_float(r.get("schoollon") or r.get("school_lon") or r.get("school_longitude"))
            location_text = r.get("school_location", "")
            centers[key] = {"lat": lat, "lon": lon, "location": location_text}
        self.log.info(f"Loaded {len(centers)} school centers from {p}")
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
                res = self.geocode_with_cache(addr)
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

    # ---------- Batch processing ----------
    def process_all_csvs(self, school_columns_map: Optional[Dict[str,str]] = None, force_regeocode: bool = False):
        """
        Process all CSV files in the raw data directory.

        :param school_columns_map: Optional mapping of column names for school name/branch (if different from default)
        :param force_regeocode: If True, re-geocode all address columns even if they already have lat/lon values

        For each CSV file, calls process_file with the given parameters and logs any exceptions that occur.

        Returns None.
        """

        # Pick up .csv and .xlsx files
        files = sorted([p for p in self.raw_dir.glob("*") if p.suffix.lower() in [".csv", ".xlsx", ".xls"]])
        self.log.info(f"Found {len(files)} files in raw dir: {[p.name for p in files]}")
        if not files:
            self.log.warning("No CSV or Excel files found in raw dir.")
            return

        for f in files:
            self.log.info(f"--- Processing file: {f.name} ---")
            try:
                if f.suffix.lower() == ".csv":
                    df = pd.read_csv(f, dtype=str)
                elif f.suffix.lower() == ".xlsx":
                    # ensure openpyxl installed; specify engine
                    df = pd.read_excel(f, dtype=str, engine="openpyxl")
                elif f.suffix.lower() == ".xls":
                    df = pd.read_excel(f, dtype=str, engine="xlrd")
                else:
                    self.log.warning(f"Skipping unsupported file type: {f.name}")
                    continue

                # ensure temporary CSV file (same stem) is created and used for processing
                tmp_csv = (self.raw_dir / f"{f.stem}.tmp.csv")
                df.to_csv(tmp_csv, index=False)
                self.log.info(f"  wrote temporary CSV for processing: {tmp_csv.name}")

                # call process_file on the tmp csv
                self.process_file(tmp_csv, school_columns_map=school_columns_map, force_regeocode=force_regeocode)

                # remove tmp csv after processing
                try:
                    tmp_csv.unlink()
                except Exception:
                    pass

            except Exception as e:
                raise CustomException(f"Failed processing {f.name}: {e}")
                self.log.error(f"Failed processing {f.name}: {e}")
        self.log.info("Done processing all files.")



# ---------- CLI ----------
if __name__ == "__main__":
    cleaner = GeocodeCleaner()
    # Optional: provide mapping if your files use different column names for school name/branch
    # Example: {"school_name": "school_name", "school_branch": "school_branch"}
    school_map = {"School Name": "school_name", "School Branch": "school_branch"}
    cleaner.process_all_csvs(school_columns_map=school_map, force_regeocode=False)
