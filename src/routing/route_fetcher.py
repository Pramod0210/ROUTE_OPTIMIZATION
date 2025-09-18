# src/tripgen/route_fetcher.py
from pathlib import Path
import logging
import time
import json
from typing import Tuple, Optional
from utils.cache_loader import load_cache, save_cache
import os
from utils.haversine_distance import haversine_km
from utils.config_loader import load_config
from logger.custom_logger import CustomLogger

try:
    import googlemaps
except Exception:
    googlemaps = None

# simple haversine fallback
import math


class RouteFetcher:
    """
    Fetch driving distance (km) and duration (min) between two lat/lon points using Google Directions
    with caching fallback to haversine if API not available.

    Usage:
        rf = RouteFetcher(cfg, logger)
        km, mins = rf.get_route((lat1,lon1),(lat2,lon2))
    """
    def __init__(self):
        self.log = CustomLogger().get_logger(__name__)
        self.cfg = load_config()
        self.cache_path = Path(self.cfg.get("paths", {}).get("route_cache", "data/geocode_routes_cache.csv"))
        # load existing cache into dict
        try:
            self.cache = load_cache(self.cache_path)
        except Exception:
            self.cache = {}
        # google client
        self.use_google = bool(self.cfg.get("routing", {}).get("use_google_directions", False)) and googlemaps is not None
        self.client = None
        if self.use_google:
            # read API key from env var name in config or from GOOGLE_MAPS_API_KEY
            env_name = self.cfg.get("geocode", {}).get("google_maps_api_key_env", "GOOGLE_MAPS_API_KEY")
            key = os.getenv(env_name)
            if key:
                try:
                    self.client = googlemaps.Client(key=key)
                except Exception as e:
                    self.log.warning(f"Failed to init googlemaps client: {e}")
                    self.client = None
                    self.use_google = False
            else:
                self.log.warning(f"No Google API key in environment var '{env_name}'. Falling back to haversine.")
                self.use_google = False

        # optional network delay config
        self.sleep_between_calls = float(self.cfg.get("geocode", {}).get("sleep_between_calls", 0.2))

    def _route_key(self, o: Tuple[float,float], d: Tuple[float,float]) -> str:
        """
        Return a string key representing the route between origin and destination.
        
        The key is in the format "lat1,lon1>lat2,lon2" where lat and lon
        are rounded to 6 decimal places.

        This key is used to cache route results.
        """
        return f"{round(o[0],6)},{round(o[1],6)}->{round(d[0],6)},{round(d[1],6)}"

    def get_route(self, origin: Tuple[float,float], destination: Tuple[float,float]) -> Tuple[Optional[float], Optional[float]]:

        """
        Fetch driving distance (km) and duration (min) between two lat/lon points.

        If the Google Maps API key is available in the environment, this function
        will first try to fetch the route using Google Directions. If the API
        key is not available, or if the API call fails, this function will
        fall back to a rough estimate using the haversine formula.

        The function returns a tuple containing the driving distance (km) and
        duration (min) between the origin and destination points. If the
        function fails to fetch the route, it will return (None, None).

        The function also caches the results of the API calls and haversine
        estimates in a CSV file located at the path specified in the config
        file. The cache is stored in the format {query: {"lat": ..., "lon": ..., "provider": ...}}.

        :param origin: (lat, lon) tuple representing the origin point
        :param destination: (lat, lon) tuple representing the destination point
        :return: A tuple containing the driving distance (km) and duration (min) between the origin and destination points.
        :rtype: Tuple[Optional[float], Optional[float]]
        """
        key = self._route_key(origin, destination)
        # check cache dict format: {query: {"lat":..., "lon":..., "provider": ...}} was used by cache_loader.
        # We store route entries as provider=route and fields distance_km, duration_min serialized into 'lat'/'lon' for compatibility.
        if key in self.cache:
            ent = self.cache[key]
            # stored as JSON in provider field? we stored lat/lon earlier as floats. We now use "lat" as distance and "lon" as duration.
            dist = ent.get("lat")
            dur = ent.get("lon")
            try:
                return (float(dist), float(dur))
            except Exception:
                return (None, None)

        # Try Google Directions
        if self.use_google and self.client:
            try:
                # use 'driving' profile by default
                profile = self.cfg.get("routing", {}).get("google_profile", "driving")
                # googlemaps directions accepts origin/destination as "lat,lng"
                o_str = f"{origin[0]},{origin[1]}"
                d_str = f"{destination[0]},{destination[1]}"
                # call directions
                res = self.client.directions(origin=o_str, destination=d_str, mode=profile)
                time.sleep(self.sleep_between_calls)
                if res and len(res) > 0:
                    # sum legs if multiple
                    total_meters = 0
                    total_secs = 0
                    for leg in res[0].get("legs", []):
                        total_meters += leg.get("distance", {}).get("value", 0)
                        total_secs += leg.get("duration", {}).get("value", 0)
                    dist_km = total_meters / 1000.0
                    dur_min = total_secs / 60.0
                    # cache result
                    self.cache[key] = {"lat": dist_km, "lon": dur_min, "provider": "google_directions"}
                    try:
                        save_cache(self.cache, self.cache_path)
                    except Exception:
                        pass
                    return (dist_km, dur_min)
            except Exception as e:
                self.log.debug(f"Google Directions failed for {key}: {e}")
                # continue to fallback

        # Fallback: haversine-based estimate using avg_speed (approx)
        try:
            hk = haversine_km(origin[0], origin[1], destination[0], destination[1])
            if hk is not None:
                # rough duration using avg_speed_kmph from config
                avg_speed = float(self.cfg.get("routing", {}).get("avg_speed_kmph", 25))
                dur_min = (hk / max(0.1, avg_speed)) * 60.0
                # cache fallback as provider='haversine'
                self.cache[key] = {"lat": hk, "lon": dur_min, "provider": "haversine"}
                try:
                    save_cache(self.cache, self.cache_path)
                except Exception:
                    pass
                return (hk, dur_min)
        except Exception:
            pass

        return (None, None)
