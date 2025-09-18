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
    OR-Tools VRP with time windows per (school_branch, batch).

    Inputs (via config and file paths):
      - stops_df: DataFrame with columns ['stop_id','stop_lat','stop_lon','num_participants','wait_time_min']
      - parts_df: participants mapping (to count riders per stop already included in num_participants)
      - bus_types: list of dicts like {'vendor','seating_capacity','available_buses','usable_capacity','fixed_cost'}
      - route_fetcher: instance of RouteFetcher (for travel times and distances)
      - config (cfg): contains routing.max_vehicles_per_branch, routing.search_time_limit_seconds, school start time etc.

    Outputs:
      - DF of trips for the branch (vehicle routes), and a trip_records list (trip_id, est_km, est_time_min, students_on_trip)
      - Writes logs via provided logger
    """

    def __init__(self):
        self.cfg = load_config()
        self.log = CustomLogger().get_logger(__name__)
        self.route_fetcher = RouteFetcher()
        self.max_vehicles_per_branch = int(self.cfg.get("routing", {}).get("max_vehicles_per_branch", 20))
        self.search_time_limit = int(self.cfg.get("routing", {}).get("search_time_limit_seconds", 60))
        self.avg_speed_kmph = float(self.cfg.get("routing", {}).get("avg_speed_kmph", 25))
        self.school_days = int(self.cfg.get("routing", {}).get("school_days_per_month", 22))
        # student travel-time table (optional preloaded structure from TripGenerator)
        self.student_travel_time = None  # set by caller if desired: list of dicts as in TripGenerator

    # ---------------------------------------------------------------------
    # public runner for a single branch+batch
    # ---------------------------------------------------------------------
    # def solve_branch(self,
    #                  stops_df: pd.DataFrame,
    #                  parts_df: pd.DataFrame,
    #                  branch: str,
    #                  batch: str,
    #                  bus_types: List[Dict[str,Any]],
    #                  depot: Tuple[float,float],
    #                  school_coord: Tuple[float,float],
    #                  school_start_time_min: int,
    #                  logger: Optional[logging.Logger]=None) -> Dict[str,Any]:
    #     """
    #     Solve VRP for one (branch, batch).
    #     - stops_df: stops in this branch+batch
    #     - parts_df: participant map entries (not used heavily since stops carry num_participants)
    #     - depot: (lat, lon)
    #     - school_coord: (lat, lon)
    #     - school_start_time_min: absolute minutes (e.g., 8:00 -> 480)
    #     Returns dictionary with keys:
    #       'routes_df' (pandas DataFrame of trips), 'trip_records' (list), 'status' (string)
    #     """
    #     log = self.log
    #     start_time = time.time()
    #     log.info(f"VRP solve for branch={branch} batch={batch}: {len(stops_df)} stops")

    #     # if zero stops, return empty
    #     if stops_df.empty:
    #         return {"routes_df": pd.DataFrame(), "trip_records": [], "status":"no_stops"}

    #     # nodes: 0 = depot, 1..N = stops, N+1 = school (end). We'll ensure vehicles end at school node.
    #     stops = stops_df.reset_index(drop=True)
    #     n_stops = len(stops)
    #     node_coords = []
    #     node_coords.append(depot)  # node 0 depot
    #     for _, r in stops.iterrows():
    #         node_coords.append((float(r['stop_lat']), float(r['stop_lon'])))
    #     node_coords.append(school_coord)  # node N+1 school

    #     # build travel time (minutes) and distance (km) matrices between all nodes
    #     N = len(node_coords)
    #     log.info("Building travel time matrix (this may call Google Directions cached).")
    #     time_matrix = [[0]*N for _ in range(N)]
    #     dist_matrix = [[0.0]*N for _ in range(N)]
    #     # we'll call route_fetcher for each pair (i,j)
    #     for i in range(N):
    #         for j in range(N):
    #             if i == j:
    #                 time_matrix[i][j] = 0
    #                 dist_matrix[i][j] = 0.0
    #                 continue
    #             o = node_coords[i]
    #             d = node_coords[j]
    #             dist_km, dur_min = self.route_fetcher.get_route(o, d)
    #             if dist_km is None:
    #                 # fallback to haversine + avg speed
    #                 dist_km = haversine_km(o[0],o[1],d[0],d[1]) or 0.0
    #                 dur_min = (dist_km / max(0.1, self.avg_speed_kmph)) * 60.0
    #             time_matrix[i][j] = int(math.ceil(dur_min))
    #             dist_matrix[i][j] = float(dist_km)

    #     # demands (node 0 and N+1 school have demand 0)
    #     demands = [0] + [int(r['num_participants']) for _, r in stops.iterrows()] + [0]

    #     # service times at nodes: depot=0, stops=wait_time_min, school = 0 (or unloading time)
    #     service_times = [0] + [int(float(r.get('wait_time_min', self.cfg.get("constraints",{}).get("stop_wait_time",3)))) for _, r in stops.iterrows()] + [0]

    #     # time windows per node: we need each stop to be visited such that students reach school before school_start_time.
    #     # We compute latest allowed arrival at stop = school_start_time - (time from stop->school) - buffer (we'll use 0)
    #     time_windows = []
    #     # depot time window wide
    #     depot_tw = (0, school_start_time_min)
    #     time_windows.append(depot_tw)
    #     # for each stop
    #     for si, r in stops.iterrows():
    #         node_index = si + 1
    #         # travel time from stop to school:
    #         stop_to_school_min = time_matrix[node_index][N-1]  # node N-1? careful: school index is N-1? Wait - we have N nodes, indexes 0..N-1, school is at index N-1
    #         # correction: school index = N-1 (since N = len(node_coords))
    #         school_idx = N-1
    #         stop_to_school_min = time_matrix[node_index][school_idx]
    #         # allowed latest arrival at stop
    #         latest_stop_arrival = school_start_time_min - stop_to_school_min
    #         # earliest arrival at stop can be 0 or optionally a configured earliest (we keep 0)
    #         earliest = 0
    #         # ensure latest >= earliest, otherwise mark stop as impossible
    #         if latest_stop_arrival < earliest:
    #             log.warning(f"Stop {r['stop_id']} impossible to serve before school start: latest_stop_arrival={latest_stop_arrival} < earliest={earliest}. It will be flagged.")
    #             latest_stop_arrival = earliest
    #         time_windows.append((earliest, int(latest_stop_arrival)))
    #     # school time window: must arrive by school_start_time with some slack (we allow up to school_start_time)
    #     time_windows.append((0, school_start_time_min))

    #     # vehicle setup: expand vehicle types into list of vehicles with capacities, but cap by max_vehicles_per_branch
    #     vehicles = []
    #     for bt in sorted(bus_types, key=lambda x: -x.get('usable_capacity',0)):
    #         avail = int(bt.get('available_buses', 0))
    #         for a in range(avail):
    #             vehicles.append({"vendor": bt.get('vendor'), "usable_capacity": int(bt.get('usable_capacity')), "seating_capacity": int(bt.get('seating_capacity')), "fixed_cost": bt.get('fixed_cost')})
    #             if len(vehicles) >= self.max_vehicles_per_branch:
    #                 break
    #         if len(vehicles) >= self.max_vehicles_per_branch:
    #             break
    #     if not vehicles:
    #         raise ValueError("No vehicles available for VRP - check bus_types / available_buses")

    #     num_vehicles = min(len(vehicles), n_stops)
    #     if num_vehicles == 0 and n_stops > 0:
    #         raise ValueError("No vehicles available for VRP with given stops.")

    #     # num_vehicles = len(vehicles)
    #     log.info(f"Using {num_vehicles} vehicles (capped by {self.max_vehicles_per_branch}) for branch={branch} batch={batch}")

    #     # build OR-Tools data model
    #     data = {}
    #     data['time_matrix'] = time_matrix
    #     data['dist_matrix'] = dist_matrix
    #     data['demands'] = demands
    #     data['service_times'] = service_times
    #     data['time_windows'] = time_windows
    #     data['num_vehicles'] = num_vehicles
    #     data['vehicle_capacities'] = [v['usable_capacity'] for v in vehicles]
    #     data['depot'] = 0
    #     # end index for vehicles should be school index (N-1). We'll set start at depot and end at school.
    #     data['end_index'] = N - 1  # school

    #     # Add this code just before the RoutingIndexManager call
    #     log.info(f"Number of nodes (N): {N}")
    #     log.info(f"Number of vehicles: {data['num_vehicles']}")
    #     log.info(f"Number of demands: {len(data['demands'])}")


    #     starts = [data['depot']] * data['num_vehicles']
    #     ends = [data['end_index']] * data['num_vehicles']

    #     log.info(f"Starts list: {starts}")
    #     log.info(f"Ends list: {ends}")
    #     # check if start and end lists are the correct length
    #     assert len(starts) == data['num_vehicles'], "Starts list size mismatch"
    #     assert len(ends) == data['num_vehicles'], "Ends list size mismatch"

    #     manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']), data['num_vehicles'], starts, ends)


    #     # Create the routing index manager with separate start and end nodes per vehicle
    #     # manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']), data['num_vehicles'], data['depot'], data['end_index'])

    #     routing = pywrapcp.RoutingModel(manager)

    #     # ---- Distance callback for objective (we'll use time as primary cost) ----
    #     def distance_callback(from_index, to_index):
    #         from_node = manager.IndexToNode(from_index)
    #         to_node = manager.IndexToNode(to_index)
    #         return int(math.ceil(data['dist_matrix'][from_node][to_node] * 1000))  # meters as integer
    #     dist_cb_idx = routing.RegisterTransitCallback(distance_callback)
    #     routing.SetArcCostEvaluatorOfAllVehicles(dist_cb_idx)

    #     # ---- Capacity dimension (demands) ----
    #     def demand_callback(from_index):
    #         from_node = manager.IndexToNode(from_index)
    #         return int(data['demands'][from_node])
    #     demand_cb = routing.RegisterUnaryTransitCallback(demand_callback)
    #     routing.AddDimensionWithVehicleCapacity(demand_cb, 0, data['vehicle_capacities'], True, 'Capacity')

    #     # ---- Time dimension (travel time + service time) with time windows ----
    #     def time_callback(from_index, to_index):
    #         from_node = manager.IndexToNode(from_index)
    #         to_node = manager.IndexToNode(to_index)
    #         travel = int(data['time_matrix'][from_node][to_node])
    #         service = int(data['service_times'][from_node])
    #         return travel + service
    #     time_cb = routing.RegisterTransitCallback(time_callback)
    #     horizon = 24 * 60  # one day in minutes
    #     routing.AddDimension(time_cb, horizon, horizon, False, 'Time')
    #     time_dimension = routing.GetDimensionOrDie('Time')

    #     # add time window constraints for each location
    #     for node_idx in range(len(data['time_windows'])):
    #         index = manager.NodeToIndex(node_idx)
    #         window = data['time_windows'][node_idx]
    #         # set time window
    #         time_dimension.CumulVar(index).SetRange(window[0], window[1])

    #     # allow waiting at nodes (already allowed by routing)
    #     # set vehicle start time windows (depot)
    #     for v in range(data['num_vehicles']):
    #         start_index = routing.Start(v)
    #         time_dimension.CumulVar(start_index).SetRange(data['time_windows'][0][0], data['time_windows'][0][1])

    #     # set search parameters
    #     search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    #     search_parameters.time_limit.seconds = int(self.search_time_limit)
    #     search_parameters.log_search = True
    #     search_parameters.max_time_seconds = int(self.search_time_limit)
    #     # prefer first solution heuristics
    #     search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    #     # local search metaheuristics
    #     search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

    #     log.info(f"Starting OR-Tools solve for branch={branch} with time_limit={self.search_time_limit}s ...")
    #     result = routing.SolveWithParameters(search_parameters)

    #     if result:
    #         log.info("OR-Tools found a solution.")
    #         # extract routes
    #         routes = []
    #         trip_records = []
    #         for v in range(data['num_vehicles']):
    #             index = routing.Start(v)
    #             if routing.IsEnd(index):
    #                 continue
    #             route_nodes = []
    #             route_load = 0
    #             route_km = 0.0
    #             route_time = 0
    #             while not routing.IsEnd(index):
    #                 node_index = manager.IndexToNode(index)
    #                 route_nodes.append(node_index)
    #                 # advance
    #                 previous_index = index
    #                 index = result.Value(routing.NextVar(index))
    #                 next_node_index = manager.IndexToNode(index) if not routing.IsEnd(index) else data['end_index']
    #                 # accumulate
    #                 route_km += data['dist_matrix'][node_index][next_node_index]
    #                 route_time += data['time_matrix'][node_index][next_node_index] + data['service_times'][node_index]
    #             # finally add the school node end
    #             route_nodes.append(data['end_index'])
    #             # compute students_on_trip as sum of demands for nodes visited (exclude depot & school)
    #             students_on_trip = sum([data['demands'][nid] for nid in route_nodes if nid not in (0, data['end_index'])])
    #             if students_on_trip == 0:
    #                 continue
    #             # convert node indexes to stop ids
    #             stop_ids = []
    #             for nid in route_nodes:
    #                 if nid == 0 or nid == data['end_index']:
    #                     continue
    #                 stop_idx = nid - 1  # mapping to stops list
    #                 stop_ids.append(stops.iloc[stop_idx]['stop_id'])
    #             trip_id = f"VRP_{branch}_{batch}_V{v}"
    #             routes.append({
    #                 "trip_id": trip_id,
    #                 "vehicle_index": v,
    #                 "stops": ",".join(stop_ids),
    #                 "students_on_trip": students_on_trip,
    #                 "est_km": round(route_km,3),
    #                 "est_time_min": int(route_time)
    #             })
    #             trip_records.append({"trip_id": trip_id, "est_km": route_km, "students_on_trip": students_on_trip})

    #         routes_df = pd.DataFrame(routes)
    #         elapsed = time.time() - start_time
    #         log.info(f"VRP solved in {elapsed:.1f}s; produced {len(routes_df)} routes.")
    #         return {"routes_df": routes_df, "trip_records": trip_records, "status":"solved", "time_sec": elapsed}
    #     else:
    #         elapsed = time.time() - start_time
    #         log.warning(f"OR-Tools did not find solution (status None) within {elapsed:.1f}s. Falling back.")
    #         return {"routes_df": pd.DataFrame(), "trip_records": [], "status":"no_solution", "time_sec": elapsed}
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
        Solve VRP for one (branch, batch).
        """
        log = logger or self.log
        start_time = time.time()
        log.info(f"VRP solve for branch={branch} batch={batch}: {len(stops_df)} stops")

        if stops_df.empty:
            return {"routes_df": pd.DataFrame(), "trip_records": [], "status": "no_stops"}

        # Nodes: 0 = depot, 1..N-2 = stops, N-1 = school
        stops = stops_df.reset_index(drop=True)
        n_stops = len(stops)
        node_coords = [depot] + [(float(r['stop_lat']), float(r['stop_lon'])) for _, r in stops.iterrows()] + [school_coord]

        # Build travel time and distance matrices
        N = len(node_coords)
        log.info("Building travel time matrix (this may call Google Directions cached).")
        time_matrix = [[0] * N for _ in range(N)]
        dist_matrix = [[0.0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                o = node_coords[i]
                d = node_coords[j]
                dist_km, dur_min = self.route_fetcher.get_route(o, d)
                if dist_km is None:
                    dist_km = haversine_km(o[0], o[1], d[0], d[1]) or 0.0
                    dur_min = (dist_km / max(0.1, self.avg_speed_kmph)) * 60.0
                if not (isinstance(dist_km, (int, float)) and dist_km >= 0 and dist_km < 1e6):
                    log.error(f"Invalid dist_km at [{i}][{j}]: {dist_km}")
                    raise ValueError(f"Invalid dist_km: {dist_km}")
                if not (isinstance(dur_min, (int, float)) and dur_min >= 0 and dur_min < 1e6):
                    log.error(f"Invalid dur_min at [{i}][{j}]: {dur_min}")
                    raise ValueError(f"Invalid dur_min: {dur_min}")
                time_matrix[i][j] = int(math.ceil(dur_min))
                dist_matrix[i][j] = float(dist_km)

        # Demands: depot and school have 0 demand
        demands = [0] + [int(r['num_participants']) for _, r in stops.iterrows()] + [0]

        # Service times: depot and school have 0 service time
        service_times = [0] + [int(float(r.get('wait_time_min', self.cfg.get("constraints", {}).get("stop_wait_time", 3)))) for _, r in stops.iterrows()] + [0]

        # Time windows with relaxed bounds
        time_windows = [(0, school_start_time_min + 60)]  # Depot
        for si, r in stops.iterrows():
            node_index = si + 1
            stop_to_school_min = time_matrix[node_index][N - 1]
            latest_stop_arrival = school_start_time_min - stop_to_school_min + 30  # Add buffer
            earliest = 0
            if latest_stop_arrival < earliest:
                log.warning(f"Stop {r['stop_id']} impossible to serve: latest_stop_arrival={latest_stop_arrival} < earliest={earliest}")
                latest_stop_arrival = earliest
            time_windows.append((earliest, int(latest_stop_arrival)))
        time_windows.append((0, school_start_time_min + 60))  # School

        # Vehicles
        vehicles = []
        for bt in sorted(bus_types, key=lambda x: -x.get('usable_capacity', 0)):
            avail = int(bt.get('available_buses', 0))
            for _ in range(avail):
                vehicles.append({
                    "vendor": bt.get('vendor'),
                    "usable_capacity": int(bt.get('usable_capacity')),
                    "seating_capacity": int(bt.get('seating_capacity')),
                    "fixed_cost": bt.get('fixed_cost')
                })
                if len(vehicles) >= self.max_vehicles_per_branch:
                    break
            if len(vehicles) >= self.max_vehicles_per_branch:
                break
        if not vehicles:
            raise ValueError("No vehicles available for VRP")

        num_vehicles = len(vehicles)
        log.info(f"Using {num_vehicles} vehicles (capped by {self.max_vehicles_per_branch}) for branch={branch} batch={batch}")

        # Build OR-Tools data model
        data = {
            'time_matrix': time_matrix,
            'dist_matrix': dist_matrix,
            'demands': demands,
            'service_times': service_times,
            'time_windows': time_windows,
            'num_vehicles': num_vehicles,
            'vehicle_capacities': [v['usable_capacity'] for v in vehicles],
            'depot': 0,
            'end_index': N - 1
        }

        # Validate inputs
        import numpy as np
        assert len(data['time_matrix']) == N, f"Expected {N} nodes, got {len(data['time_matrix'])}"
        assert all(len(row) == N for row in data['time_matrix']), "Time matrix is not square"
        assert all(all(isinstance(val, (int, float)) and val >= 0 and val < 1e6 for val in row) for row in data['time_matrix']), "Time matrix contains invalid values"
        assert all(all(isinstance(val, (int, float)) and val >= 0 and val < 1e6 for val in row) for row in data['dist_matrix']), "Distance matrix contains invalid values"
        assert len(data['demands']) == N, f"Expected {N} demands, got {len(data['demands'])}"
        assert len(data['service_times']) == N, f"Expected {N} service times, got {len(data['service_times'])}"
        assert len(data['time_windows']) == N, f"Expected {N} time windows, got {len(data['time_windows'])}"
        assert all(isinstance(d, (int, float)) and d >= 0 for d in data['demands']), "Demands contain invalid values"
        assert all(isinstance(s, (int, float)) and s >= 0 for s in data['service_times']), "Service times contain invalid values"
        assert all(isinstance(tw, (list, tuple)) and len(tw) == 2 and isinstance(tw[0], (int, float)) and isinstance(tw[1], (int, float)) and tw[0] <= tw[1] for tw in data['time_windows']), "Invalid time windows"
        assert len(data['vehicle_capacities']) == num_vehicles, "Vehicle capacities length mismatch"
        assert all(isinstance(c, (int, float)) and c > 0 for c in data['vehicle_capacities']), "Invalid vehicle capacities"

        # Log full input data
        log.info(f"Time matrix shape: {np.array(data['time_matrix']).shape}")
        log.info(f"Time matrix: {data['time_matrix']}")
        log.info(f"Distance matrix: {data['dist_matrix']}")
        log.info(f"Demands: {data['demands']}")
        log.info(f"Service times: {data['service_times']}")
        log.info(f"Time windows: {data['time_windows']}")
        log.info(f"Vehicle capacities: {data['vehicle_capacities']}")

        # Create start and end lists
        starts = [data['depot']] * data['num_vehicles']
        ends = [data['end_index']] * data['num_vehicles']
        log.info(f"Number of nodes (N): {N}")
        log.info(f"Number of vehicles: {data['num_vehicles']}")
        log.info(f"Number of demands: {len(data['demands'])}")
        log.info(f"Starts list: {starts}")
        log.info(f"Ends list: {ends}")
        assert len(starts) == num_vehicles, "Starts list size mismatch"
        assert len(ends) == num_vehicles, "Ends list size mismatch"

        # Create routing index manager
        try:
            manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']), data['num_vehicles'], starts, ends)
            log.info("RoutingIndexManager created successfully")
        except Exception as e:
            log.error(f"Error creating RoutingIndexManager: {str(e)}")
            raise

        # Create routing model
        try:
            routing = pywrapcp.RoutingModel(manager)
            log.info("RoutingModel created successfully")
        except Exception as e:
            log.error(f"Error creating RoutingModel: {str(e)}")
            raise

        # Distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            dist = data['dist_matrix'][from_node][to_node]
            if not (isinstance(dist, (int, float)) and dist >= 0):
                log.error(f"Invalid distance from {from_node} to {to_node}: {dist}")
                raise ValueError(f"Invalid distance: {dist}")
            return int(math.ceil(dist * 1000))  # Convert km to meters

        try:
            dist_cb_idx = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(dist_cb_idx)
            log.info("Distance callback registered successfully")
        except Exception as e:
            log.error(f"Error registering distance callback: {str(e)}")
            raise

        # Capacity dimension
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

        # Time dimension
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
            horizon = school_start_time_min + 120  # Increased buffer
            time_cb = routing.RegisterTransitCallback(time_callback)
            routing.AddDimension(time_cb, horizon, horizon, False, 'Time')
            time_dimension = routing.GetDimensionOrDie('Time')
            log.info("Time dimension added successfully")
        except Exception as e:
            log.error(f"Error adding time dimension: {str(e)}")
            raise

        # Time window constraints (skip end node)
        try:
            for node_idx in range(len(data['time_windows'])):
                if node_idx == data['end_index']:  # Skip school node
                    log.info(f"Skipping time window for end node {node_idx} (school)")
                    continue
                index = manager.NodeToIndex(node_idx)
                if index < 0:
                    log.error(f"Invalid index for node {node_idx}: {index}")
                    raise ValueError(f"Invalid index: {index}")
                window = data['time_windows'][node_idx]
                log.info(f"Setting time window for node {node_idx} (index {index}): {window}")
                time_dimension.CumulVar(index).SetRange(int(window[0]), int(window[1]))
            log.info("Time window constraints added successfully")
        except Exception as e:
            log.error(f"Error setting time window constraints: {str(e)}")
            raise

        # Vehicle start time windows
        try:
            for v in range(data['num_vehicles']):
                start_index = routing.Start(v)
                window = data['time_windows'][0]
                log.info(f"Setting start time window for vehicle {v} (index {start_index}): {window}")
                time_dimension.CumulVar(start_index).SetRange(int(window[0]), int(window[1]))
            log.info("Vehicle start time windows set successfully")
        except Exception as e:
            log.error(f"Error setting vehicle start time windows: {str(e)}")
            raise

        # Search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.time_limit.seconds = int(self.search_time_limit)
        search_parameters.log_search = True
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

        log.info(f"Starting OR-Tools solve for branch={branch} with time_limit={self.search_time_limit}s ...")
        try:
            result = routing.SolveWithParameters(search_parameters)
            log.info("Solve completed")
        except Exception as e:
            log.error(f"Error solving VRP: {str(e)}")
            raise

        if result:
            log.info("OR-Tools found a solution.")
            routes = []
            trip_records = []
            for v in range(data['num_vehicles']):
                index = routing.Start(v)
                if routing.IsEnd(index):
                    continue
                route_nodes = []
                route_load = 0
                route_km = 0.0
                route_time = 0
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    route_nodes.append(node_index)
                    previous_index = index
                    index = result.Value(routing.NextVar(index))
                    next_node_index = manager.NodeToIndex(index) if not routing.IsEnd(index) else data['end_index']
                    route_km += data['dist_matrix'][node_index][next_node_index]
                    route_time += data['time_matrix'][node_index][next_node_index] + data['service_times'][node_index]
                route_nodes.append(data['end_index'])
                students_on_trip = sum([data['demands'][nid] for nid in route_nodes if nid not in (0, data['end_index'])])
                if students_on_trip == 0:
                    continue
                stop_ids = [stops.iloc[nid - 1]['stop_id'] for nid in route_nodes if nid not in (0, data['end_index'])]
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
            return {"routes_df": routes_df, "trip_records": trip_records, "status": "solved", "time_sec": elapsed}
        else:
            elapsed = time.time() - start_time
            log.warning(f"OR-Tools did not find solution (status None) within {elapsed:.1f}s. Falling back.")
            return {"routes_df": pd.DataFrame(), "trip_records": [], "status": "no_solution", "time_sec": elapsed}
        
        