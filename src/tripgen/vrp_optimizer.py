# src/tripgen/vrp_optimizer.py
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import logging
import math
import time
import pandas as pd

# OR-Tools
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# route fetcher from earlier
from src.routing.route_fetcher import RouteFetcher
from src.optimization.fleet_optimizer import FleetOptimizer  # used later if you want MIP on top

from logger.custom_logger import CustomLogger
from utils.helpers import travel_time_min_km, listify, minutes_from_hhmm
from utils.config_loader import load_config
from utils.haversine_distance import haversine_km
from exception.custom_exception import CustomException

   

class VRPTripOptimizer:
    """
    OR-Tools VRP optimizer per (school_branch, batch).
    - Uses RouteFetcher for travel distances/times.
    - Builds time windows so students arrive within [school_start -30min, school_start +15min].
    - Enforces student_travel_time policy (flags impossible stops).
    Config keys expected in cfg:
      paths.student_travel_time -> CSV path for distance->allowed_time table
      routing.max_vehicles_per_branch, routing.search_time_limit_seconds, routing.avg_speed_kmph
      constraints.stop_wait_time (default wait per stop)
      paths.logs (where impossible stops CSV is written)
    """

    def __init__(self):
        self.cfg = load_config()
        self.log = CustomLogger().get_logger(__name__)
        self.route_fetcher = RouteFetcher()

        self.max_vehicles_per_branch = int(self.cfg.get("routing", {}).get("max_vehicles_per_branch", 20))
        self.search_time_limit = int(self.cfg.get("routing", {}).get("search_time_limit_seconds", 60))
        self.avg_speed_kmph = float(self.cfg.get("routing", {}).get("avg_speed_kmph", 25))
        self.school_days = int(self.cfg.get("routing", {}).get("school_days_per_month", 22))

        # load student_travel_time table if available
        self.student_travel_time: List[Dict[str, Any]] = []
        try:
            stt_path = Path(self.cfg.get("paths", {}).get("student_travel_time", "data/sample/Student_Travel_Time.csv"))
            if stt_path.exists():
                stt_df = pd.read_csv(stt_path)
                stt_df.columns = [c.strip().lower().replace(" ", "_") for c in stt_df.columns]
                # auto-detect distance and time columns
                dist_col = None
                time_col = None
                for c in stt_df.columns:
                    if "distance" in c:
                        dist_col = c
                    if "travel_time" in c or ("time" in c and "min" in c) or ("travel" in c and "min" in c):
                        time_col = c
                if dist_col is None or time_col is None:
                    # fallback to first two numeric columns
                    numeric_cols = [c for c in stt_df.columns if pd.api.types.is_numeric_dtype(stt_df[c])]
                    if len(numeric_cols) >= 2:
                        dist_col = dist_col or numeric_cols[0]
                        time_col = time_col or numeric_cols[1]
                if dist_col and time_col:
                    stt_clean = stt_df[[dist_col, time_col]].rename(columns={dist_col: "distance_km", time_col: "travel_time_mins"})
                    stt_clean["distance_km"] = pd.to_numeric(stt_clean["distance_km"], errors="coerce")
                    stt_clean["travel_time_mins"] = pd.to_numeric(stt_clean["travel_time_mins"], errors="coerce")
                    stt_clean = stt_clean.dropna().sort_values("distance_km").reset_index(drop=True)
                    self.student_travel_time = stt_clean.to_dict("records")
                    self.log.info(f"Loaded student_travel_time ({len(self.student_travel_time)} rows) from {stt_path}")
                else:
                    self.log.warning("student_travel_time found but columns not identified; skipping table.")
            else:
                self.log.info("student_travel_time file not found; using no per-distance limits.")
        except Exception as e:
            self.log.warning(f"Failed to load student_travel_time: {e}")
            self.student_travel_time = []

        # default stop wait time
        self.default_wait = float(self.cfg.get("constraints", {}).get("stop_wait_time", 3))

    def _allowed_ride_time_for_distance(self, dist_km: float) -> float:
        """
        Lookup allowed ride time (mins) for a given distance (km) from student_travel_time table.
        Fallback: a large default (e.g., 999) if table missing.
        """
        if not self.student_travel_time:
            return float(self.cfg.get("constraints", {}).get("max_student_ride_time", 9999))
        for row in self.student_travel_time:
            try:
                if float(dist_km) <= float(row["distance_km"]):
                    return float(row["travel_time_mins"])
            except Exception:
                continue
        # if distance larger than table, return last row's time
        return float(self.student_travel_time[-1]["travel_time_mins"])

    def solve_branch(self,
                    stops_df: pd.DataFrame,
                    parts_df: pd.DataFrame,
                    branch: str,
                    batch: str,
                    bus_types: List[Dict[str, Any]],
                    depot: Tuple[float, float],
                    school_coord: Tuple[float, float],
                    school_start_time_min: int,
                    logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
        """
        Solve VRP for one (branch, batch) with:
        - student_travel_time policy enforcement (flags impossible stops)
        - arrival window: students arrive in [school_start - 30, school_start + 15]
        Returns dict with keys:
        - routes_df, trip_records, status, time_sec, impossible_stops_df
        """
        log = logger or self.log
        start_time = time.time()
        log.info(f"VRP solve for branch={branch} batch={batch}: {len(stops_df)} stops")

        if stops_df.empty:
            return {"routes_df": pd.DataFrame(), "trip_records": [], "status": "no_stops", "impossible_stops_df": pd.DataFrame(), "time_sec": time.time() - start_time}

        # Normalize stops (reset index)
        stops = stops_df.reset_index(drop=True)

        # Build node coords: 0 = depot, 1..n = stops, n+1 = school
        node_coords = [depot] + [(float(r['stop_lat']), float(r['stop_lon'])) for _, r in stops.iterrows()] + [school_coord]
        N_full = len(node_coords)  # full nodes including school

        log.info("Building time & distance matrices (may call route_fetcher).")
        time_matrix_full = [[0] * N_full for _ in range(N_full)]
        dist_matrix_full = [[0.0] * N_full for _ in range(N_full)]

        for i in range(N_full):
            for j in range(N_full):
                if i == j:
                    time_matrix_full[i][j] = 0
                    dist_matrix_full[i][j] = 0.0
                    continue
                o = node_coords[i]
                d = node_coords[j]
                dist_km, dur_min = self.route_fetcher.get_route(o, d)
                if dist_km is None:
                    dist_km = haversine_km(o[0], o[1], d[0], d[1]) or 0.0
                if dur_min is None:
                    dur_min = (dist_km / max(0.1, self.avg_speed_kmph)) * 60.0

                # sanity checks
                if not (isinstance(dist_km, (int, float)) and dist_km >= 0 and dist_km < 1e6):
                    log.error(f"Invalid dist_km at [{i}][{j}]: {dist_km}")
                    raise ValueError(f"Invalid dist_km: {dist_km}")
                if not (isinstance(dur_min, (int, float)) and dur_min >= 0 and dur_min < 1e6):
                    log.error(f"Invalid dur_min at [{i}][{j}]: {dur_min}")
                    raise ValueError(f"Invalid dur_min: {dur_min}")

                time_matrix_full[i][j] = int(math.ceil(dur_min))
                dist_matrix_full[i][j] = float(dist_km)

        # Evaluate per-stop feasibility under student_travel_time policy and build time windows
        school_index_full = N_full - 1
        impossible_stops = []
        feasible_indices = []  # indices (1..n) of stops that can be served
        time_windows_full = []  # full for all nodes (we'll later reduce to feasible subset)
        depot_window = (0, school_start_time_min + 60)  # allow some buffer for depot
        time_windows_full.append(depot_window)

        def allowed_ride_time_for_km(km: float) -> float:
            if getattr(self, "student_travel_time", None):
                for row in self.student_travel_time:
                    try:
                        if float(km) <= float(row["distance_km"]):
                            return float(row["travel_time_mins"])
                    except Exception:
                        continue
                return float(self.student_travel_time[-1]["travel_time_mins"])
            return float(self.cfg.get("constraints", {}).get("max_student_ride_time", 9999))

        for si, r in stops.iterrows():
            node_idx = si + 1
            stop_to_school_min = time_matrix_full[node_idx][school_index_full]
            stop_to_school_km = dist_matrix_full[node_idx][school_index_full]
            allowed = allowed_ride_time_for_km(stop_to_school_km if stop_to_school_km is not None else 0.0)

            if stop_to_school_min is not None and allowed is not None and stop_to_school_min > allowed:
                log.warning(f"Stop {r['stop_id']} -> school time {stop_to_school_min:.1f}m exceeds allowed {allowed:.1f}m; flagging impossible.")
                impossible_stops.append({
                    "stop_id": r['stop_id'],
                    "stop_lat": r['stop_lat'],
                    "stop_lon": r['stop_lon'],
                    "stop_to_school_min": stop_to_school_min,
                    "stop_to_school_km": stop_to_school_km,
                    "allowed_ride_min": allowed,
                    "reason": "exceeds_allowed_ride_time"
                })
                time_windows_full.append((0, 0))
                continue

            latest_stop_arrival = int((school_start_time_min + 15) - (stop_to_school_min or 0))
            earliest_stop_arrival = int((school_start_time_min - 30) - (stop_to_school_min or 0))
            if earliest_stop_arrival < 0:
                earliest_stop_arrival = 0
            if latest_stop_arrival < earliest_stop_arrival:
                log.warning(f"Stop {r['stop_id']} has infeasible time window earliest={earliest_stop_arrival}, latest={latest_stop_arrival}; flagging.")
                impossible_stops.append({
                    "stop_id": r['stop_id'],
                    "stop_lat": r['stop_lat'],
                    "stop_lon": r['stop_lon'],
                    "stop_to_school_min": stop_to_school_min,
                    "stop_to_school_km": stop_to_school_km,
                    "earliest": earliest_stop_arrival,
                    "latest": latest_stop_arrival,
                    "reason": "infeasible_window"
                })
                time_windows_full.append((earliest_stop_arrival, earliest_stop_arrival))
                continue

            feasible_indices.append(node_idx)
            time_windows_full.append((earliest_stop_arrival, latest_stop_arrival))

        school_window = (max(0, school_start_time_min - 30), school_start_time_min + 15)
        time_windows_full.append(school_window)

        feasible_nodes = [0] + feasible_indices + [school_index_full]
        old_to_new = {old: new for new, old in enumerate(feasible_nodes)}
        N = len(feasible_nodes)

        if len(feasible_indices) == 0:
            log.warning("No feasible stops for VRP after applying student_travel_time/time-window constraints.")
            imp_df = pd.DataFrame(impossible_stops)
            return {"routes_df": pd.DataFrame(), "trip_records": [], "status": "no_feasible_stops", "impossible_stops_df": imp_df, "time_sec": time.time() - start_time}

        time_matrix = [[0] * N for _ in range(N)]
        dist_matrix = [[0.0] * N for _ in range(N)]
        for i, oi in enumerate(feasible_nodes):
            for j, oj in enumerate(feasible_nodes):
                time_matrix[i][j] = time_matrix_full[oi][oj]
                dist_matrix[i][j] = dist_matrix_full[oi][oj]

        demands = [0]
        service_times = [0]
        time_windows = [time_windows_full[0]]
        reduced_stop_rows = []
        for oi in feasible_indices:
            r = stops.iloc[oi - 1]
            demands.append(int(r['num_participants']))
            service_times.append(int(float(r.get('wait_time_min', self.cfg.get("constraints", {}).get("stop_wait_time", 3)))))
            time_windows.append(time_windows_full[oi])
            reduced_stop_rows.append(r)
        demands.append(0)
        service_times.append(0)
        time_windows.append(time_windows_full[-1])

        vehicles = []
        for bt in sorted(bus_types, key=lambda x: -x.get('usable_capacity', 0)):
            avail = int(bt.get('available_buses', 0))
            for _ in range(avail):
                vehicles.append({
                    "vendor": bt.get('vendor'),
                    "usable_capacity": int(bt.get('usable_capacity')),
                    "seating_capacity": int(bt.get('seating_capacity')),
                    "fixed_cost": float(bt.get('fixed_cost') or 0.0)
                })
                if len(vehicles) >= self.max_vehicles_per_branch:
                    break
            if len(vehicles) >= self.max_vehicles_per_branch:
                break

        if not vehicles:
            log.error("No vehicles available for VRP after expansion/capping.")
            imp_df = pd.DataFrame(impossible_stops)
            return {"routes_df": pd.DataFrame(), "trip_records": [], "status": "no_vehicles", "impossible_stops_df": imp_df, "time_sec": time.time() - start_time}

        num_vehicles = len(vehicles)
        log.info(f"Using {num_vehicles} vehicles (capped by {self.max_vehicles_per_branch}) for branch={branch} batch={batch}")

        data = {
            "time_matrix": time_matrix,
            "dist_matrix": dist_matrix,
            "demands": demands,
            "service_times": service_times,
            "time_windows": time_windows,
            "num_vehicles": num_vehicles,
            "vehicle_capacities": [v['usable_capacity'] for v in vehicles],
            "depot": 0,
            "end_index": N - 1
        }

        # Validation
        import numpy as np
        assert len(data['time_matrix']) == N, f"Expected {N} nodes, got {len(data['time_matrix'])}"
        assert all(len(row) == N for row in data['time_matrix']), "Time matrix is not square"
        assert all(all(isinstance(val, (int, float)) and val >= 0 and val < 1e6 for val in row) for row in data['time_matrix']), "Invalid time matrix values"
        assert all(all(isinstance(val, (int, float)) and val >= 0 and val < 1e6 for val in row) for row in data['dist_matrix']), "Invalid distance matrix values"
        assert len(data['demands']) == N, f"Expected {N} demands, got {len(data['demands'])}"
        assert len(data['service_times']) == N, f"Expected {N} service times, got {len(data['service_times'])}"
        assert len(data['time_windows']) == N, f"Expected {N} time windows, got {len(data['time_windows'])}"
        assert all(isinstance(d, (int, float)) and d >= 0 for d in data['demands']), "Invalid demands"
        assert all(isinstance(s, (int, float)) and s >= 0 for s in data['service_times']), "Invalid service times"
        assert all(isinstance(tw, (list, tuple)) and len(tw) == 2 and isinstance(tw[0], (int, float)) and isinstance(tw[1], (int, float)) and tw[0] <= tw[1] for tw in data['time_windows']), "Invalid time windows"
        assert len(data['vehicle_capacities']) == num_vehicles, "Vehicle capacities length mismatch"
        assert all(isinstance(c, (int, float)) and c > 0 for c in data['vehicle_capacities']), "Invalid vehicle capacities"

        log.info(f"Time matrix shape: {np.array(data['time_matrix']).shape}")
        log.info(f"Time matrix: {data['time_matrix']}")
        log.info(f"Distance matrix: {data['dist_matrix']}")
        log.info(f"Demands: {data['demands']}")
        log.info(f"Service times: {data['service_times']}")
        log.info(f"Time windows: {data['time_windows']}")
        log.info(f"Vehicle capacities: {data['vehicle_capacities']}")

        starts = [data['depot']] * data['num_vehicles']
        ends = [data['end_index']] * data['num_vehicles']
        log.info(f"Number of nodes (N): {N}")
        log.info(f"Number of vehicles: {data['num_vehicles']}")
        log.info(f"Number of demands: {len(data['demands'])}")
        log.info(f"Starts list: {starts}")
        log.info(f"Ends list: {ends}")

        try:
            manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']), data['num_vehicles'], starts, ends)
            log.info("RoutingIndexManager created successfully")
        except Exception as e:
            log.error(f"Error creating RoutingIndexManager: {str(e)}")
            raise

        try:
            routing = pywrapcp.RoutingModel(manager)
            log.info("RoutingModel created successfully")
        except Exception as e:
            log.error(f"Error creating RoutingModel: {str(e)}")
            raise

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            dist = data['dist_matrix'][from_node][to_node]
            if not (isinstance(dist, (int, float)) and dist >= 0):
                log.error(f"Invalid distance from {from_node} to {to_node}: {dist}")
                raise ValueError(f"Invalid distance: {dist}")
            return int(math.ceil(dist * 1000))

        try:
            dist_cb_idx = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(dist_cb_idx)
            log.info("Distance callback registered successfully")
        except Exception as e:
            log.error(f"Error registering distance callback: {str(e)}")
            raise

        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            demand = data['demands'][from_node]
            if not (isinstance(demand, (int, float)) and demand >= 0):
                log.error(f"Invalid demand at node {from_node}: {demand}")
                raise ValueError(f"Invalid demand: {demand}")
            return int(demand)

        try:
            demand_cb = routing.RegisterUnaryTransitCallback(demand_callback)
            routing.AddDimensionWithVehicleCapacity(demand_cb, 0, [int(c) for c in data['vehicle_capacities']], True, 'Capacity')
            log.info("Capacity dimension added successfully")
        except Exception as e:
            log.error(f"Error adding capacity dimension: {str(e)}")
            raise

        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            travel = data['time_matrix'][from_node][to_node]
            service = data['service_times'][from_node]
            if not (isinstance(travel, (int, float)) and travel >= 0):
                log.error(f"Invalid travel time from {from_node} to {to_node}: {travel}")
                raise ValueError(f"Invalid travel time: {travel}")
            if not (isinstance(service, (int, float)) and service >= 0):
                log.error(f"Invalid service time at {from_node}: {service}")
                raise ValueError(f"Invalid service time: {service}")
            return int(travel + service)

        try:
            horizon = school_start_time_min + 120
            time_cb = routing.RegisterTransitCallback(time_callback)
            routing.AddDimension(time_cb, horizon, horizon, False, 'Time')
            time_dimension = routing.GetDimensionOrDie('Time')
            log.info("Time dimension added successfully")
        except Exception as e:
            log.error(f"Error adding time dimension: {str(e)}")
            raise

        try:
            for node_idx in range(len(data['time_windows'])):
                if node_idx == data['end_index']:
                    log.info(f"Skipping time window for end node {node_idx} (school)")
                    continue
                idx = manager.NodeToIndex(node_idx)
                if idx < 0:
                    log.error(f"Invalid index for node {node_idx}: {idx}")
                    raise ValueError(f"Invalid index: {idx}")
                window = data['time_windows'][node_idx]
                log.info(f"Setting time window for node {node_idx} (index {idx}): {window}")
                time_dimension.CumulVar(idx).SetRange(int(window[0]), int(window[1]))
            log.info("Time window constraints added successfully")
        except Exception as e:
            log.error(f"Error setting time window constraints: {str(e)}")
            raise

        try:
            for v in range(data['num_vehicles']):
                sidx = routing.Start(v)
                tw = data['time_windows'][0]
                log.info(f"Setting start time window for vehicle {v} (index {sidx}): {tw}")
                time_dimension.CumulVar(sidx).SetRange(int(tw[0]), int(tw[1]))
            log.info("Vehicle start time windows set successfully")
        except Exception as e:
            log.error(f"Error setting vehicle start time windows: {str(e)}")
            raise

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.time_limit.seconds = int(min(self.search_time_limit, 30))
        search_parameters.log_search = True
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

        log.info(f"Starting OR-Tools solve for branch={branch} with time_limit={search_parameters.time_limit.seconds}s ...")
        try:
            result = routing.SolveWithParameters(search_parameters)
            log.info("Solve completed")
        except Exception as e:
            log.error(f"OR-Tools solver raised exception: {str(e)}")
            imp_df = pd.DataFrame(impossible_stops)
            return {"routes_df": pd.DataFrame(), "trip_records": [], "status": "solver_exception", "impossible_stops_df": imp_df, "time_sec": time.time() - start_time}

        if result:
            log.info("OR-Tools found a solution.")
            routes = []
            trip_records = []
            for v in range(data['num_vehicles']):
                index = routing.Start(v)
                if routing.IsEnd(index):
                    continue
                route_nodes = []
                route_km = 0.0
                route_time = 0
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    route_nodes.append(node_index)
                    previous_index = index
                    index = result.Value(routing.NextVar(index))
                    next_node = manager.IndexToNode(index) if not routing.IsEnd(index) else data['end_index']
                    route_km += data['dist_matrix'][node_index][next_node]
                    route_time += data['time_matrix'][node_index][next_node] + data['service_times'][node_index]
                route_nodes.append(data['end_index'])
                students_on_trip = sum([data['demands'][n] for n in route_nodes if n not in (0, data['end_index'])])
                if students_on_trip == 0:
                    continue
                stop_ids = []
                for n in route_nodes:
                    if n in (0, data['end_index']):
                        continue
                    reduced_idx = n - 1
                    stop_row = reduced_stop_rows[reduced_idx]
                    stop_ids.append(stop_row['stop_id'])
                trip_id = f"VRP_{branch}_{batch}_V{v}"
                routes.append({
                    "trip_id": trip_id,
                    "vehicle_index": v,
                    "stops": ",".join(stop_ids),
                    "students_on_trip": students_on_trip,
                    "est_km": round(route_km, 3),
                    "est_time_min": int(route_time)
                })
                trip_records.append({"trip_id": trip_id, "est_km": route_km, "students_on_trip": students_on_trip})

            routes_df = pd.DataFrame(routes)
            elapsed = time.time() - start_time
            log.info(f"VRP solved in {elapsed:.1f}s; produced {len(routes_df)} routes.")
            imp_df = pd.DataFrame(impossible_stops)
            return {"routes_df": routes_df, "trip_records": trip_records, "status": "solved", "impossible_stops_df": imp_df, "time_sec": elapsed}
        else:
            elapsed = time.time() - start_time
            log.warning(f"OR-Tools did not find solution within {self.search_time_limit}s; falling back.")
            imp_df = pd.DataFrame(impossible_stops)
            return {"routes_df": pd.DataFrame(), "trip_records": [], "status": "no_solution", "impossible_stops_df": imp_df, "time_sec": elapsed}


if __name__ == "__main__":
    vro = VRPTripOptimizer()
    print("VRPTripOptimizer initialized.")
    vro.solve_branch()