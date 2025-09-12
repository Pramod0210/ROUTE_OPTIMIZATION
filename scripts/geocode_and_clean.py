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
    def __init__(self, config_path: str = "config/config.yaml"):
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

    def _find_address_cols(self, df: pd.DataFrame) -> List[str]:
        cols = []
        for c in df.columns:
            low = c.lower()
            if any(k in low for k in self.address_keywords):
                cols.append(c)
        return cols

    # ---------- Geocode with caching ----------
    def geocode_with_cache(self, query: str) -> Optional[Dict[str, Any]]:
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
                    result = {"lat": float(loc["lat"]), "lon": float(loc["lng"]), "provider": "google"}
            except Exception as e:
                logging.debug(f"Google geocode error for {key}: {e}")
                result = None
        # fallback to nominatim
        if result is None:
            for attempt in range(self.max_retries):
                try:
                    # RateLimiter is used; call nom_rate_limiter
                    loc = self.nom_rate_limiter(key)
                    if loc:
                        result = {"lat": float(loc.latitude), "lon": float(loc.longitude), "provider": "nominatim"}
                    break
                except GeopyError as ge:
                    logging.debug(f"Nominatim error (attempt {attempt+1}) for {key}: {ge}")
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
        p = Path(self.school_master_file)
        if not p.exists():
            logging.warning(f"School master {p} not found; city validation disabled")
            return {}
        df = pd.read_csv(p, dtype=str).fillna("")
        # normalize columns
        df.rename(columns={c: c.strip().lower().replace(" ", "_") for c in df.columns}, inplace=True)
        centers = {}
        for _, r in df.iterrows():
            key = f"{r.get('school_name','').strip().lower()}__{r.get('school_branch','').strip().lower()}"
            lat = safe_float(r.get("schoollat") or r.get("school_lat") or r.get("school_latitude"))
            lon = safe_float(r.get("schoollon") or r.get("school_lon") or r.get("school_longitude"))
            location_text = r.get("school_location", "")
            centers[key] = {"lat": lat, "lon": lon, "location": location_text}
        return centers

    def _is_within_school_area(self, school_key: str, lat: float, lon: float) -> bool:
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
        logging.info(f"Processing file: {file_path.name}")
        df = pd.read_csv(file_path, dtype=str)
        # normalize column names
        df.rename(columns={c: c.strip().lower().replace(" ", "_"): c for c in df.columns}, inplace=True)  # map old names to normalized keys
        # lowercase textual columns
        for col in list(df.columns):
            if self.is_text_dtype(df[col]):
                df[col] = df[col].fillna("").astype(str).str.strip().str.lower()
        # detect address-like columns
        address_cols = self._find_address_cols(df)
        logging.info(f"  detected address-like columns: {address_cols}")
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
            logging.info(f"  {len(unique_addrs)} unique addresses to geocode for column '{col}'")

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
        logging.info(f"  wrote processed file: {out_path}")

        # save issues for manual review
        if issues:
            issues_df = pd.DataFrame(issues)
            issues_out = self.log_dir / f"geocode_issues__{file_path.name}"
            issues_df.to_csv(issues_out.with_suffix(".csv"), index=False)
            logging.info(f"  issues written to {issues_out.with_suffix('.csv')}")
        # save updated cache
        self.save_cache(self.cache)
        return out_path

    # ---------- Batch processing ----------
    def process_all_csvs(self, school_columns_map: Optional[Dict[str,str]] = None, force_regeocode: bool = False):
        files = sorted([p for p in self.raw_dir.glob("*.csv")])
        if not files:
            logging.warning("No CSV files found in raw dir.")
            return
        for f in files:
            try:
                self.process_file(f, school_columns_map=school_columns_map, force_regeocode=force_regeocode)
            except Exception as e:
                logging.exception(f"Failed processing {f.name}: {e}")

# ---------- CLI ----------
if __name__ == "__main__":
    cleaner = GeocodeCleaner(config_path="config/config.yaml")
    # Optional: provide mapping if your files use different column names for school name/branch
    # Example: {"school_name": "school_name", "school_branch": "school_branch"}
    school_map = {"school_name": "school_name", "school_branch": "school_branch"}
    cleaner.process_all_csvs(school_columns_map=school_map, force_regeocode=False)
