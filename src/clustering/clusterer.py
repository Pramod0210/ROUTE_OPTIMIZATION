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
from utils.config_loader import load_config


# ----------------- utilities -----------------
def normalize_colname(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_")

# ----------------- StopClusterer -----------------
class StopClusterer:
    """
    Produces pickup/drop stops from student and teacher processed files.

    Config (config/config.yaml) keys used:
      paths:
        students_processed: "data/processed/students_processed.csv"
        teachers_processed: "data/processed/teacher_processed.csv"
        stops_pickup_output: "data/processed/stops_pickup.csv"
        stops_drop_output: "data/processed/stops_drop.csv"
        participants_with_stops_pickup: "data/processed/participants_with_stops_pickup.csv"
        participants_with_stops_drop: "data/processed/participants_with_stops_drop.csv"
      constraints:
        max_walk_meters: 300
      clustering:
        mode: "distance"  # or "society"
        society_split_within_meters: 0
      dwell_time_bands:
        - {min: 1, max: 2, wait_min: 2}
        - {min: 3, max: 5, wait_min: 3}
        - {min: 6, max: 9999, wait_min: 4}
    """

    def __init__(self,
                 base_dir: Optional[Path] = None):
        self.log = CustomLogger().get_logger(__name__)
        self.cfg = load_config()
        self.base = Path.cwd() if base_dir is None else Path(base_dir)

        # paths (canonical processed file names)
        p = self.cfg.get("paths", {})
        self.students_file = self.base / Path(p.get("students_processed", "data/processed/student_master.csv"))
        self.teachers_file = self.base / Path(p.get("teachers_processed", "data/processed/teacher_master.csv"))

        # outputs
        self.stops_pickup_out = self.base / Path(p.get("stops_pickup_output", "data/processed/stops_pickup.csv"))
        self.stops_drop_out   = self.base / Path(p.get("stops_drop_output", "data/processed/stops_drop.csv"))
        self.part_map_pickup_out = self.base / Path(p.get("participants_with_stops_pickup", "data/processed/participants_with_stops_pickup.csv"))
        self.part_map_drop_out   = self.base / Path(p.get("participants_with_stops_drop", "data/processed/participants_with_stops_drop.csv"))

        # parameters
        self.max_walk_m = float(self.cfg.get("constraints", {}).get("max_walk_meters", 300))
        clustering_cfg = self.cfg.get("clustering", {})
        self.mode = clustering_cfg.get("mode", "distance")
        self.society_split_within_m = float(clustering_cfg.get("society_split_within_meters", 0))

        # load bus stop time table
        stop_time_path = self.base / Path(self.cfg["paths"].get("bus_stop_time", "data/processed/bus_stop.csv"))
        if not stop_time_path.exists():
            raise FileNotFoundError(f"Bus stop time file not found: {stop_time_path}")
        df = pd.read_csv(stop_time_path)
        df.rename(columns={c: c.strip().lower().replace(" ", "_") for c in df.columns}, inplace=True)
        # Expect columns: students, wait_time(mins)
        self.dwell_bands = (
            df[["students", "wait_time(mins)"]]
            .sort_values("students")
            .rename(columns={"students": "max", "wait_time(mins)": "wait_min"})
            .to_dict("records")
        )

        # logger
        self.log.info(f"StopClusterer initialized: mode={self.mode}, max_walk_m={self.max_walk_m}, society_split_within_m={self.society_split_within_m}")


    def _compute_wait_time(self, num_participants:int) -> int:
        """Compute wait time in minutes based on number of participants using dwell bands."""
        for band in self.dwell_bands:
            if num_participants <= band["max"]:
                return int(band["wait_min"])
        return int(self.dwell_bands[-1]["wait_min"])

    def _cluster_coords_dbscan(self, coords:np.ndarray, eps_meters:float) -> np.ndarray:
        """Run DBSCAN on lat/lon coords using haversine metric; returns labels array."""
        if coords.shape[0] == 0:
            return np.array([], dtype=int)
        coords_rad = np.radians(coords)  # convert degrees to radians
        eps_rad = eps_meters / 6371000.0
        db = DBSCAN(eps=eps_rad, min_samples=1, metric="haversine")
        labels = db.fit_predict(coords_rad)
        return labels

    def _make_stop_id(self) -> str:
        return "STOP_" + uuid.uuid4().hex[:8]

    # ---------------- data loading ----------------
    def _load_participants(self) -> pd.DataFrame:
        """
        Load students and teachers into a unified participants DataFrame.
        Columns returned: id, type, grade, society_name, batch, school_branch,
                          pickup_lat, pickup_lon, drop_lat, drop_lon, must_have_seat
        """
        parts: List[pd.DataFrame] = []

        # students
        if self.students_file.exists():
            s = pd.read_csv(self.students_file, dtype=str).fillna("")
            s.rename(columns={c: normalize_colname(c) for c in s.columns}, inplace=True)
            s_part = pd.DataFrame({
                "id": s.get("chs_number", ""),
                "type": "student",
                "grade": s.get("grade", ""),
                "society_name": s.get("society_name", ""),
                "batch": s.get("batch", ""),
                "school_branch": s.get("school_branch", ""),
                "pickup_lat": pd.to_numeric(s.get("pickup_address_lat", ""), errors="coerce"),
                "pickup_lon": pd.to_numeric(s.get("pickup_address_lon", ""), errors="coerce"),
                "drop_lat": pd.to_numeric(s.get("drop_address_lat", ""), errors="coerce"),
                "drop_lon": pd.to_numeric(s.get("drop_address_lon", ""), errors="coerce"),
                "must_have_seat": False
            })
            parts.append(s_part)
        else:
            self.log.warning(f"Students file not found: {self.students_file}")

        # teachers
        if self.teachers_file.exists():
            t = pd.read_csv(self.teachers_file, dtype=str).fillna("")
            t.rename(columns={c: normalize_colname(c) for c in t.columns}, inplace=True)
            # teachers may not have society_name or grade
            must_flag = pd.Series([False]*len(t), index=t.index)
            if "must_have_seat" in t.columns:
                must_flag = t["must_have_seat"].astype(str).str.lower().isin(["1","true","yes"])
            t_part = pd.DataFrame({
                "id": t.get("chs_number", ""),
                "type": "teacher",
                "grade": "",
                "society_name": t.get("society_name", ""),
                "batch": t.get("batch", ""),
                "school_branch": t.get("school_branch", ""),
                "pickup_lat": pd.to_numeric(t.get("pickup_address_lat", ""), errors="coerce"),
                "pickup_lon": pd.to_numeric(t.get("pickup_address_lon", ""), errors="coerce"),
                "drop_lat": pd.to_numeric(t.get("drop_address_lat", ""), errors="coerce"),
                "drop_lon": pd.to_numeric(t.get("drop_address_lon", ""), errors="coerce"),
                "must_have_seat": must_flag
            })
            parts.append(t_part)
        else:
            self.log.info("Teachers file not found; continuing with students only.")

        if not parts:
            raise FileNotFoundError("No participant files found (students/teachers).")

        df = pd.concat(parts, ignore_index=True, sort=False).fillna("")
        df["batch"] = df["batch"].astype(str).str.strip().str.lower()
        df["school_branch"] = df["school_branch"].astype(str).str.strip().str.lower()
        return df

    # ---------------- clustering per direction ----------------
    def _cluster_for_direction(self, df: pd.DataFrame, direction: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        direction: 'pickup' or 'drop'
        Returns tuple (stops_df, participants_map_df)
        """
        assert direction in ("pickup", "drop")
        lat_col = f"{direction}_lat"
        lon_col = f"{direction}_lon"

        # drop missing coords for this direction
        d = df.copy()
        d[lat_col] = pd.to_numeric(d[lat_col], errors="coerce")
        d[lon_col] = pd.to_numeric(d[lon_col], errors="coerce")
        before = len(d)
        d = d.dropna(subset=[lat_col, lon_col])
        self.log.info(f"{direction}: dropped {before-len(d)} participants missing {direction} coords; {len(d)} remain")

        all_stops: List[Dict[str, Any]] = []
        all_maps: List[Dict[str, Any]] = []

        grouped = d.groupby(["school_branch", "batch"])
        for (school_branch, batch), group in grouped:
            self.log.info(f"Clustering ({direction}) for school_branch={school_branch} batch={batch} with {len(group)} participants")

            if self.mode == "society":
                # group by society_name when available; teachers may have empty society_name
                for society, g in group.groupby("society_name"):
                    if pd.isna(society) or society == "":
                        # fallback to distance clustering for participants without society
                        coords = g[[lat_col, lon_col]].astype(float).to_numpy()
                        labels = self._cluster_coords_dbscan(coords, eps_meters=self.max_walk_m)
                        g = g.copy(); g["stop_label"] = labels
                        for lbl in sorted(set(labels)):
                            cluster = g[g["stop_label"] == lbl]
                            stop_id = self._make_stop_id()
                            lat_c = cluster[lat_col].astype(float).mean()
                            lon_c = cluster[lon_col].astype(float).mean()
                            num_participants = len(cluster)
                            wait_min = self._compute_wait_time(num_participants)
                            all_stops.append({
                                "stop_id": stop_id,
                                "direction": direction,
                                "school_branch": school_branch,
                                "batch": batch,
                                "society_name": "",
                                "stop_lat": lat_c,
                                "stop_lon": lon_c,
                                "num_participants": num_participants,
                                "wait_time_min": wait_min
                            })
                            for _, r in cluster.iterrows():
                                all_maps.append({
                                    "id": r["id"], "type": r["type"], "stop_id": stop_id,
                                    "direction": direction, "must_have_seat": bool(r.get("must_have_seat", False))
                                })
                    else:
                        soc_group = group[group["society_name"] == society]
                        if self.society_split_within_m and len(soc_group) > 1:
                            coords = soc_group[[lat_col, lon_col]].astype(float).to_numpy()
                            labels = self._cluster_coords_dbscan(coords, eps_meters=self.society_split_within_m)
                            sg = soc_group.copy(); sg["sub_label"] = labels
                            for sub in sorted(set(labels)):
                                cluster = sg[sg["sub_label"] == sub]
                                stop_id = self._make_stop_id()
                                lat_c = cluster[lat_col].astype(float).mean()
                                lon_c = cluster[lon_col].astype(float).mean()
                                num_participants = len(cluster)
                                wait_min = self._compute_wait_time(num_participants)
                                all_stops.append({
                                    "stop_id": stop_id,
                                    "direction": direction,
                                    "school_branch": school_branch,
                                    "batch": batch,
                                    "society_name": society,
                                    "stop_lat": lat_c,
                                    "stop_lon": lon_c,
                                    "num_participants": num_participants,
                                    "wait_time_min": wait_min
                                })
                                for _, r in cluster.iterrows():
                                    all_maps.append({
                                        "id": r["id"], "type": r["type"], "stop_id": stop_id,
                                        "direction": direction, "must_have_seat": bool(r.get("must_have_seat", False))
                                    })
                        else:
                            stop_id = self._make_stop_id()
                            lat_c = soc_group[lat_col].astype(float).mean()
                            lon_c = soc_group[lon_col].astype(float).mean()
                            num_participants = len(soc_group)
                            wait_min = self._compute_wait_time(num_participants)
                            all_stops.append({
                                "stop_id": stop_id,
                                "direction": direction,
                                "school_branch": school_branch,
                                "batch": batch,
                                "society_name": society,
                                "stop_lat": lat_c,
                                "stop_lon": lon_c,
                                "num_participants": num_participants,
                                "wait_time_min": wait_min
                            })
                            for _, r in soc_group.iterrows():
                                all_maps.append({
                                    "id": r["id"], "type": r["type"], "stop_id": stop_id,
                                    "direction": direction, "must_have_seat": bool(r.get("must_have_seat", False))
                                })
            else:
                # distance mode: cluster everyone together by coordinates
                coords = group[[lat_col, lon_col]].astype(float).to_numpy()
                labels = self._cluster_coords_dbscan(coords, eps_meters=self.max_walk_m)
                g = group.copy(); g["stop_label"] = labels
                for lbl in sorted(set(labels)):
                    cluster = g[g["stop_label"] == lbl]
                    stop_id = self._make_stop_id()
                    lat_c = cluster[lat_col].astype(float).mean()
                    lon_c = cluster[lon_col].astype(float).mean()
                    num_participants = len(cluster)
                    wait_min = self._compute_wait_time(num_participants)
                    all_stops.append({
                        "stop_id": stop_id,
                        "direction": direction,
                        "school_branch": school_branch,
                        "batch": batch,
                        "society_name": "",
                        "stop_lat": lat_c,
                        "stop_lon": lon_c,
                        "num_participants": num_participants,
                        "wait_time_min": wait_min
                    })
                    for _, r in cluster.iterrows():
                        all_maps.append({
                            "id": r["id"], "type": r["type"], "stop_id": stop_id,
                            "direction": direction, "must_have_seat": bool(r.get("must_have_seat", False))
                        })

        stops_df = pd.DataFrame(all_stops)
        maps_df = pd.DataFrame(all_maps)
        return stops_df, maps_df

    # ---------------- public runner ----------------
    def run_all_directions(self, save: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Run clustering for pickup and drop, write canonical outputs if save=True.
        Returns dict with dataframes for pickup/drop stops and maps.
        """
        df = self._load_participants()
        pickup_stops, pickup_map = self._cluster_for_direction(df, "pickup")
        drop_stops, drop_map     = self._cluster_for_direction(df, "drop")

        if save:
            self.stops_pickup_out.parent.mkdir(parents=True, exist_ok=True)
            self.stops_drop_out.parent.mkdir(parents=True, exist_ok=True)
            self.part_map_pickup_out.parent.mkdir(parents=True, exist_ok=True)
            self.part_map_drop_out.parent.mkdir(parents=True, exist_ok=True)

            pickup_stops.to_csv(self.stops_pickup_out, index=False)
            pickup_map.to_csv(self.part_map_pickup_out, index=False)
            drop_stops.to_csv(self.stops_drop_out, index=False)
            drop_map.to_csv(self.part_map_drop_out, index=False)

            self.log.info(f"Wrote pickup stops to {self.stops_pickup_out} ({len(pickup_stops)} stops)")
            self.log.info(f"Wrote drop stops to {self.stops_drop_out} ({len(drop_stops)} stops)")
            self.log.info(f"Wrote pickup participant map to {self.part_map_pickup_out} ({len(pickup_map)} rows)")
            self.log.info(f"Wrote drop participant map to {self.part_map_drop_out} ({len(drop_map)} rows)")

        return {
            "pickup_stops": pickup_stops,
            "pickup_map": pickup_map,
            "drop_stops": drop_stops,
            "drop_map": drop_map
        }

# ---------- CLI ----------
if __name__ == "__main__":
    c = StopClusterer()
    # Optional: provide mapping if your files use different column names for school name/branch
    # Example: {"school_name": "school_name", "school_branch": "school_branch"}
    c.run_all_directions(save=True)
