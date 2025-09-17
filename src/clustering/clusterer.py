# src/clustering/clusterer.py
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import math
import uuid
import time
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import yaml
from logger.custom_logger import CustomLogger
from exception.custom_exception import CustomException
from utils.haversine_distance import haversine_km

# ----------------- utilities -----------------
def normalize_colname(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_")

# ----------------- Clusterer -----------------
class StopClusterer:
    """
    Create candidate stops from student pickups.

    Modes:
      - "distance": group nearby student pickup points within max_walk_meters using DBSCAN (haversine)
      - "society": group by 'society_name' column (one stop per society centroid).
          Optionally, if society_split_within_meters is set (>0), further split large societies by internal clustering.

    Configuration (from config/config.yaml):
      constraints:
        max_walk_meters: 300
      clustering:
        mode: "distance"  # or "society"
        society_split_within_meters: 0  # if >0, split large societies into sub-stops using this radius
      dwell_time_bands:
        - {min: 1, max: 2, wait_min: 2}
        - {min: 3, max: 5, wait_min: 3}
        - {min: 6, max: 9999, wait_min: 4}

    Usage:
      logger = custom_logger.get_logger(...)
      c = StopClusterer(config_path="config/config.yaml", logger=logger)
      c.run()  # reads data/processed/students_processed.csv and writes stops + students_with_stops
    """
    def __init__(self, config_path: str = "config/config.yaml", base_dir: Optional[Path] = None, logger: Optional[logging.Logger] = None):
        self.base = Path.cwd() if base_dir is None else Path(base_dir)
        self.config_path = Path(config_path)
        self.cfg = self._load_config(self.config_path)
        self.raw_students_path = self.base / Path(self.cfg["paths"].get("students_processed", "data/processed/students_processed.csv"))
        self.stops_out = self.base / Path(self.cfg["paths"].get("stops_output", "data/processed/stops.csv"))
        self.students_out = self.base / Path(self.cfg["paths"].get("students_with_stops_output", "data/processed/students_with_stops.csv"))

        # clustering params
        self.max_walk_m = float(self.cfg.get("constraints", {}).get("max_walk_meters", 300))
        clustering_cfg = self.cfg.get("clustering", {})
        self.mode = clustering_cfg.get("mode", "distance")
        self.society_split_within_m = float(clustering_cfg.get("society_split_within_meters", 0))
        # dwell time bands
        self.dwell_bands = self.cfg.get("dwell_time_bands", [
            {"min":1,"max":2,"wait_min":2},
            {"min":3,"max":5,"wait_min":3},
            {"min":6,"max":9999,"wait_min":4}
        ])
        # logger
        self.log = logger or logging.getLogger(__name__)
        self.log.info(f"StopClusterer init: mode={self.mode} max_walk_m={self.max_walk_m} society_split_within_m={self.society_split_within_m}")

    def _load_config(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Config not found at {path}")
        with open(path, "r") as fh:
            return yaml.safe_load(fh)

    def _compute_wait_time(self, num_students:int) -> int:
        for band in self.dwell_bands:
            if band["min"] <= num_students <= band["max"]:
                return int(band["wait_min"])
        # fallback
        return int(self.dwell_bands[-1]["wait_min"])

    def _cluster_coords_dbscan(self, coords:np.ndarray, eps_meters:float) -> np.ndarray:
        """Run DBSCAN on lat/lon coords with haversine metric; returns labels array."""
        if coords.shape[0] == 0:
            return np.array([], dtype=int)
        # convert degrees to radians for haversine metric
        coords_rad = np.radians(coords)
        eps_rad = eps_meters / 6371000.0  # earth radius meters
        db = DBSCAN(eps=eps_rad, min_samples=1, metric="haversine")
        labels = db.fit_predict(coords_rad)
        return labels

    def _make_stop_id(self) -> str:
        return "STOP_" + uuid.uuid4().hex[:8]

    def _cluster_group_distance(self, group_df: pd.DataFrame) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
        """
        Cluster a single group (one school_branch + batch) by distance.
        Returns (stops_list, students_with_stops_list)
        """
        stops = []
        students_map = []
        coords = group_df[["pickup_lat","pickup_lon"]].astype(float).to_numpy()
        labels = self._cluster_coords_dbscan(coords, eps_meters=self.max_walk_m)
        group_df = group_df.copy()
        group_df["stop_label"] = labels

        for lbl in sorted(set(labels)):
            cluster = group_df[group_df["stop_label"] == lbl]
            lat_c = cluster["pickup_lat"].astype(float).mean()
            lon_c = cluster["pickup_lon"].astype(float).mean()
            num_students = len(cluster)
            stop_id = self._make_stop_id()
            wait_min = self._compute_wait_time(num_students)
            stops.append({
                "stop_id": stop_id,
                "school_branch": cluster["school_branch"].iloc[0],
                "batch": cluster["batch"].iloc[0],
                "stop_lat": lat_c,
                "stop_lon": lon_c,
                "num_students": num_students,
                "wait_time_min": wait_min
            })
            for _, r in cluster.iterrows():
                students_map.append({
                    "studentid": r["studentid"],
                    "stop_id": stop_id,
                    "orig_lat": r["pickup_lat"],
                    "orig_lon": r["pickup_lon"]
                })
        return stops, students_map

    def _cluster_group_society(self, group_df: pd.DataFrame) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
        """
        Group by society_name. For each society, either create one stop at society centroid,
        or (if society_split_within_m > 0) further cluster inside the society.
        """
        stops = []
        students_map = []
        if "society_name" not in group_df.columns:
            # fallback to distance clustering
            self.log.warning("society mode requested but 'society_name' column missing — falling back to distance mode.")
            return self._cluster_group_distance(group_df)

        for society, g in group_df.groupby("society_name"):
            if pd.isna(society) or society == "":
                # treat empty society as normal distance-based clustering
                s, sm = self._cluster_group_distance(g)
                stops.extend(s); students_map.extend(sm)
                continue
            # compute centroid of society members
            lat_c = g["pickup_lat"].astype(float).mean()
            lon_c = g["pickup_lon"].astype(float).mean()
            if self.society_split_within_m and len(g) > 1:
                # further split within society using DBSCAN with eps = society_split_within_m
                coords = g[["pickup_lat","pickup_lon"]].astype(float).to_numpy()
                labels = self._cluster_coords_dbscan(coords, eps_meters=self.society_split_within_m)
                g = g.copy()
                g["sub_label"] = labels
                for sub in sorted(set(labels)):
                    subgrp = g[g["sub_label"] == sub]
                    lat_sub = subgrp["pickup_lat"].astype(float).mean()
                    lon_sub = subgrp["pickup_lon"].astype(float).mean()
                    num_students = len(subgrp)
                    stop_id = self._make_stop_id()
                    wait_min = self._compute_wait_time(num_students)
                    stops.append({
                        "stop_id": stop_id,
                        "school_branch": subgrp["school_branch"].iloc[0],
                        "batch": subgrp["batch"].iloc[0],
                        "society_name": society,
                        "stop_lat": lat_sub,
                        "stop_lon": lon_sub,
                        "num_students": num_students,
                        "wait_time_min": wait_min
                    })
                    for _, r in subgrp.iterrows():
                        students_map.append({
                            "studentid": r["studentid"],
                            "stop_id": stop_id,
                            "orig_lat": r["pickup_lat"],
                            "orig_lon": r["pickup_lon"]
                        })
            else:
                # one stop for the society
                num_students = len(g)
                stop_id = self._make_stop_id()
                wait_min = self._compute_wait_time(num_students)
                stops.append({
                    "stop_id": stop_id,
                    "school_branch": g["school_branch"].iloc[0],
                    "batch": g["batch"].iloc[0],
                    "society_name": society,
                    "stop_lat": lat_c,
                    "stop_lon": lon_c,
                    "num_students": num_students,
                    "wait_time_min": wait_min
                })
                for _, r in g.iterrows():
                    students_map.append({
                        "studentid": r["studentid"],
                        "stop_id": stop_id,
                        "orig_lat": r["pickup_lat"],
                        "orig_lon": r["pickup_lon"]
                    })
        return stops, students_map

    def run(self, save: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main entry: read students_processed.csv, cluster per school_branch+batch, create stops & student mappings.
        Returns (stops_df, students_with_stops_df) and writes CSVs when save=True
        """
        if not self.raw_students_path.exists():
            self.log.error(f"Students processed file not found at {self.raw_students_path}")
            raise FileNotFoundError(self.raw_students_path)

        df = pd.read_csv(self.raw_students_path, dtype=str).fillna("")
        # normalize column names
        df.rename(columns={c: normalize_colname(c) for c in df.columns}, inplace=True)
        required = ["studentid","pickup_lat","pickup_lon","batch","school_branch"]
        for c in required:
            if c not in df.columns:
                self.log.error(f"Missing required column '{c}' in students file.")
                raise KeyError(c)
        # convert coords and drop missing
        df["pickup_lat"] = pd.to_numeric(df["pickup_lat"], errors="coerce")
        df["pickup_lon"] = pd.to_numeric(df["pickup_lon"], errors="coerce")
        before = len(df)
        df = df.dropna(subset=["pickup_lat","pickup_lon"])
        self.log.info(f"Dropped {before-len(df)} rows missing pickup coords; {len(df)} remain")

        all_stops = []
        all_students_map = []

        group_cols = ["school_branch","batch"]
        grouped = df.groupby(group_cols)
        for (school_branch, batch), group in grouped:
            self.log.info(f"Clustering for school_branch={school_branch} batch={batch} ({len(group)} students)")
            if self.mode == "distance":
                stops, sm = self._cluster_group_distance(group)
            elif self.mode == "society":
                # ensure society_name column exists (normalized form)
                if "society_name" not in group.columns:
                    group = group.copy()
                    group["society_name"] = ""
                stops, sm = self._cluster_group_society(group)
            else:
                self.log.warning(f"Unknown clustering mode '{self.mode}'. Falling back to distance.")
                stops, sm = self._cluster_group_distance(group)

            all_stops.extend(stops)
            all_students_map.extend(sm)

        stops_df = pd.DataFrame(all_stops)
        students_map_df = pd.DataFrame(all_students_map)

        # write outputs
        if save:
            self.stops_out.parent.mkdir(parents=True, exist_ok=True)
            stops_df.to_csv(self.stops_out, index=False)
            students_map_df.to_csv(self.students_out, index=False)
            self.log.info(f"Wrote {len(stops_df)} stops to {self.stops_out}")
            self.log.info(f"Wrote {len(students_map_df)} student->stop mappings to {self.students_out}")

        return stops_df, students_map_df
