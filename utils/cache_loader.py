from pathlib import Path
import pandas as pd
import logging
from typing import Dict, Any, Optional

# Default cache file location (relative to project root)
DEFAULT_CACHE_PATH = Path("data") / "geocode_cache.csv"


def safe_float(x: Any) -> Optional[float]:
    """Safely convert a value to float, return None if not possible."""
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def load_cache(cache_path: Path = DEFAULT_CACHE_PATH) -> Dict[str, Dict[str, Any]]:
    """
    Load a geocoding cache from a CSV file.

    Expected columns:
        - query: unique string used as lookup
        - lat: latitude (float)
        - lon: longitude (float)
        - provider: geocoding provider name

    :param cache_path: Path to cache CSV file (defaults to DEFAULT_CACHE_PATH)
    :return: Dictionary with query as key and dict with lat, lon, provider as value
    """
    if not cache_path.exists():
        logging.info(f"Cache file not found at {cache_path}, starting with empty cache.")
        return {}

    try:
        df = pd.read_csv(cache_path, dtype=str).fillna("")
        cache = {}
        for _, r in df.iterrows():
            cache[r["query"]] = {
                "lat": safe_float(r.get("lat")),
                "lon": safe_float(r.get("lon")),
                "provider": r.get("provider", "")
            }
        return cache
    except Exception as e:
        logging.warning(f"Failed to load cache: {e} - starting fresh.")
        return {}


def save_cache(cache: Dict[str, Dict[str, Any]], cache_path: Path = DEFAULT_CACHE_PATH) -> None:
    """
    Save a geocoding cache dictionary to a CSV file.

    :param cache: Dictionary with query as key and dict with lat, lon, provider as value
    :param cache_path: Path to write CSV (defaults to DEFAULT_CACHE_PATH)
    """
    records = [
        {
            "query": k,
            "lat": v.get("lat"),
            "lon": v.get("lon"),
            "provider": v.get("provider", "")
        }
        for k, v in cache.items()
    ]
    df = pd.DataFrame(records)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logging.info(f"Cache saved to {cache_path} ({len(records)} records).")
