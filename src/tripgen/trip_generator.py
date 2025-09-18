# src/tripgen/trip_generator.py
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import math
import logging
import yaml
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from logger.custom_logger import CustomLogger
from utils.helpers import travel_time_min_km, listify
from utils.config_loader import load_config
from utils.haversine_distance import haversine_km
from exception.custom_exception import CustomException


# ------------------ TripGenerator ------------------
class TripGenerator:
    """
    Generate candidate trips (pickup/drop) and a naive fleet packing.

    """
    def __init__(self):
        self.log = CustomLogger().get_logger(__name__)
        self.cfg = load_config()

        self.base = Path.cwd()
        # paths
        p = self.cfg.get("paths", {})
        self.stops_pickup_path = self.base / Path(p.get("stops_pickup_output", "data/processed/stops_pickup.csv"))
        self.stops_drop_path   = self.base / Path(p.get("stops_drop_output", "data/processed/stops_drop.csv"))
        self.parts_pickup_path = self.base / Path(p.get("participants_with_stops_pickup", "data/processed/participants_with_stops_pickup.csv"))
        self.parts_drop_path   = self.base / Path(p.get("participants_with_stops_drop", "data/processed/participants_with_stops_drop.csv"))
        self.bus_master_path   = self.base / Path(p.get("bus_master", "data/processed/bus_master.csv"))
        self.parking_spots_path= self.base / Path(p.get("parking_spots", "data/processed/bus_parking.csv"))
        # params
        cons = self.cfg.get("constraints", {})
        self.max_occupancy = float(cons.get("max_occupancy", 0.9))
        self.max_bus_km_per_month = float(cons.get("max_bus_km_per_month", 9000))
        self.max_bus_km_per_day = float(cons.get("max_bus_km_per_day", 300))
        self.max_student_ride_time = float(cons.get("max_student_ride_time", 120))
        # routing params
        routing = self.cfg.get("routing", {})
        self.avg_speed_kmph = float(routing.get("avg_speed_kmph", 50))
        self.max_trip_time_min = float(routing.get("max_trip_time_min", 120))
        self.max_trip_km = float(routing.get("max_trip_km", 150))
        # output paths
        out = self.cfg.get("paths", {})
        self.trips_out_dir = self.base / Path(out.get("trips_output_dir", "data/processed/trips"))
        self.trips_out_dir.mkdir(parents=True, exist_ok=True)
        # load dataframes
        self.stops_pickup = pd.read_csv(self.stops_pickup_path) if self.stops_pickup_path.exists() else pd.DataFrame()
        self.parts_pickup = pd.read_csv(self.parts_pickup_path) if self.parts_pickup_path.exists() else pd.DataFrame()
        self.stops_drop = pd.read_csv(self.stops_drop_path) if self.stops_drop_path.exists() else pd.DataFrame()
        self.parts_drop = pd.read_csv(self.parts_drop_path) if self.parts_drop_path.exists() else pd.DataFrame()
        self.bus_master = pd.read_csv(self.bus_master_path) if self.bus_master_path.exists() else pd.DataFrame()
        self.parking_spots = pd.read_csv(self.parking_spots_path) if self.parking_spots_path.exists() else pd.DataFrame()

        # simple validation
        if self.stops_pickup.empty and self.stops_drop.empty:
            raise FileNotFoundError("No stops found. Run StopClusterer first and ensure stops_pickup/stops_drop exist.")

        self.log.info("TripGenerator initialized.")

    # ----------------- core greedy trip building -----------------
    def _nearest_stop(self, current_lat, current_lon, candidate_stops):
        
        """
        Find the nearest stop to the given coordinates from a list of candidate stops.
        Returns (idx, km) where idx is the index of the nearest stop in candidate_stops and km is the distance in kilometers.
        If no valid stop is found, returns (None, None).
        """
        best_idx = None
        best_km = None
        for idx, row in candidate_stops.iterrows():
            km = haversine_km(current_lat, current_lon, row['stop_lat'], row['stop_lon'])
            if km is None:
                continue
            if best_km is None or km < best_km:
                best_km = km
                best_idx = idx
        return best_idx, best_km

    def _build_trips_for_group(self, stops_df: pd.DataFrame, partmap_df: pd.DataFrame, direction: str,
                               bus_types: List[Dict[str,Any]]) -> Tuple[pd.DataFrame, List[Dict[str,Any]]]:

        """
        Build trips for a given group of stops (school_branch, batch) and participants map.
        Direction should be either 'pickup' or 'drop'.
        Returns a tuple of (trips_df, trip_records) where trips_df is a DataFrame containing trip records and trip_records is a list of dictionaries containing detailed trip information.
        """
        trips = []
        trip_records = []

        group_cols = ['school_branch', 'batch']
        grouped_stops = stops_df.groupby(group_cols)

        # build quick mapping stop_id -> participants count (and list)
        parts_lookup = {}
        for _, r in partmap_df.iterrows():
            sid = r.get('stop_id')
            parts_lookup.setdefault(sid, []).append(r)

        # determine parking/depot choices
        # if parking_spots provided and has lat/lon, use nearest for each school branch; else start at school centroid
        branch_parking = {}
        if not self.parking_spots.empty:
            # expect parking_spots to have columns 'parking_spot', 'parking_spot_lat', 'parking_spot_lon', 'buses'
            for _, br in stops_df.groupby('school_branch'):
                # use branch centroid
                lat0 = br['stop_lat'].astype(float).mean()
                lon0 = br['stop_lon'].astype(float).mean()
                best_idx, best_km = self._nearest_stop(lat0, lon0, self.parking_spots.rename(columns={'parking_spot_lat':'stop_lat','parking_spot_lon':'stop_lon'}))
                if best_idx is not None:
                    row = self.parking_spots.loc[best_idx]
                    branch_parking[br['school_branch'].iloc[0]] = (row.get('parking_spot'), row.get('parking_spot_lat'), row.get('parking_spot_lon'))
        # fallback: use school branch centroid as depot (we assume school location at centroid  of stops)
        # now per-group greedy
        for (school_branch, batch), g_stops in grouped_stops:
            candidate = g_stops.copy().reset_index(drop=True)
            # mark as unserved
            candidate['served'] = False

            # pick depot
            depot = branch_parking.get(school_branch, (None, candidate['stop_lat'].astype(float).mean(), candidate['stop_lon'].astype(float).mean()))
            depot_name, depot_lat, depot_lon = depot

            # while unserved stops exist, create new trip
            while candidate[~candidate['served']].shape[0] > 0:
                remaining = candidate[~candidate['served']].copy()
                # select starting stop: farthest from depot to reduce overlap (heuristic) OR nearest to depot?
                # We'll choose the farthest to avoid many small trips near depot later.
                remaining['dist_from_depot'] = remaining.apply(lambda r: haversine_km(depot_lat, depot_lon, r['stop_lat'], r['stop_lon']), axis=1)
                start_idx = remaining['dist_from_depot'].idxmax()
                trip_stops = []
                trip_students = 0
                trip_km = 0.0
                trip_time_min = 0.0
                cur_lat = depot_lat
                cur_lon = depot_lon

                # choose bus type greedily: pick smallest bus that can fit most students at this group's total demand?
                # compute total demand at group to pick initial bus_type
                total_demand = remaining['num_participants'].astype(int).sum()
                # pick bus_type that minimizes cost while accommodating occupancy
                chosen_bus_type = None
                for bt in sorted(bus_types, key=lambda x: x.get('seating_capacity', 50)):
                    cap = int(bt.get('seating_capacity', 50))
                    usable = math.floor(cap * self.max_occupancy)
                    if usable >= 1:
                        # if usable is >= the average cluster, accept; we'll still enforce per-trip capacity
                        if usable >= min(total_demand, remaining['num_participants'].astype(int).max()):
                            chosen_bus_type = bt
                            break
                if chosen_bus_type is None:
                    chosen_bus_type = bus_types[-1] if bus_types else {'vendor': 'unknown', 'seating_capacity': 50, 'fixed_cost': 0}

                bus_cap = int(chosen_bus_type.get('seating_capacity', 50))
                usable_cap = math.floor(bus_cap * self.max_occupancy)

                # fill trip by nearest neighbor while capacity/time constraints hold
                cur_lat, cur_lon = depot_lat, depot_lon
                while True:
                    # find nearest unserved stop
                    remaining = candidate[~candidate['served']].copy()
                    if remaining.empty:
                        break
                    idx_nearest, km_to = self._nearest_stop(cur_lat, cur_lon, remaining)
                    if idx_nearest is None:
                        break
                    row = remaining.loc[idx_nearest]
                    stop_id = row['stop_id']
                    stop_demand = int(row['num_participants'])
                    # check if adding this stop exceeds capacity
                    if trip_students + stop_demand > usable_cap:
                        # cannot add this stop; try next-nearest by removing this stop from consideration temporarily
                        # simple fallback: break trip
                        break
                    # estimate incremental time & km: cur -> stop -> school (we'll compute return to depot later)
                    leg_km = haversine_km(cur_lat, cur_lon, row['stop_lat'], row['stop_lon']) or 0.0
                    est_leg_min = travel_time_min_km(leg_km, self.avg_speed_kmph) or 0.0
                    # add stop wait time (use stop_wait_time if present in stops table)
                    wait_min = float(row.get('wait_time_min', self.cfg.get('constraints', {}).get('stop_wait_time', 3)))
                    new_trip_time = trip_time_min + est_leg_min + wait_min
                    # plus an estimate from this stop to school - approximated below when finalizing
                    if new_trip_time > self.max_trip_time_min:
                        break
                    # accept stop
                    trip_stops.append(stop_id)
                    trip_students += stop_demand
                    trip_km += leg_km
                    trip_time_min = new_trip_time
                    # mark this stop as tentatively served
                    candidate.loc[candidate['stop_id'] == stop_id, 'served'] = True
                    # move current location
                    cur_lat, cur_lon = row['stop_lat'], row['stop_lon']
                    # loop to try next stop

                # finalize trip: from last stop to school (we assume school's coord = mean of group's stops or not available)
                # approximate school at group's centroid (could be improved with school_master)
                school_lat = g_stops['stop_lat'].astype(float).mean()
                school_lon = g_stops['stop_lon'].astype(float).mean()
                last_to_school_km = haversine_km(cur_lat, cur_lon, school_lat, school_lon) or 0.0
                last_to_school_min = travel_time_min_km(last_to_school_km, self.avg_speed_kmph) or 0.0
                trip_km += last_to_school_km
                trip_time_min += last_to_school_min

                # add depot->first leg and final return to depot kms to estimate trip total
                # compute depot->first stop km if trip has stops
                if trip_stops:
                    first_stop = candidate.loc[candidate['stop_id'].isin(trip_stops)].iloc[0]
                    depot_to_first_km = haversine_km(depot_lat, depot_lon, first_stop['stop_lat'], first_stop['stop_lon']) or 0.0
                else:
                    depot_to_first_km = 0.0
                # estimate return to depot after dropping at school
                school_to_depot_km = haversine_km(school_lat, school_lon, depot_lat, depot_lon) or 0.0
                trip_km += depot_to_first_km + school_to_depot_km
                trip_time_min += travel_time_min_km(depot_to_first_km + school_to_depot_km, self.avg_speed_kmph) or 0.0

                # sanity checks: cap by max_trip_km, max_trip_time_min
                if trip_km > self.max_trip_km or trip_time_min > self.max_trip_time_min:
                    # if violates, roll back last added stops until within limits (simple)
                    # naive rollback: remove last appended stop(s) until within limits
                    while trip_stops and (trip_km > self.max_trip_km or trip_time_min > self.max_trip_time_min):
                        last = trip_stops.pop()
                        # unmark served
                        candidate.loc[candidate['stop_id'] == last, 'served'] = False
                        # recompute trip stats crudely by recomputing path from depot through trip_stops
                        trip_km = 0.0
                        trip_time_min = 0.0
                        cur_lat, cur_lon = depot_lat, depot_lon
                        for sid in trip_stops:
                            r = g_stops[g_stops['stop_id'] == sid].iloc[0]
                            lk = haversine_km(cur_lat, cur_lon, r['stop_lat'], r['stop_lon']) or 0.0
                            trip_km += lk
                            trip_time_min += travel_time_min_km(lk, self.avg_speed_kmph) or 0.0
                            trip_time_min += float(r.get('wait_time_min', self.cfg.get('constraints', {}).get('stop_wait_time', 3)))
                            cur_lat, cur_lon = r['stop_lat'], r['stop_lon']
                        # finalize legs to school & depot
                        last_to_school_km = haversine_km(cur_lat, cur_lon, school_lat, school_lon) or 0.0
                        trip_km += last_to_school_km
                        trip_time_min += travel_time_min_km(last_to_school_km, self.avg_speed_kmph) or 0.0
                        if trip_stops:
                            first_stop = g_stops[g_stops['stop_id'].isin(trip_stops)].iloc[0]
                            depot_to_first_km = haversine_km(depot_lat, depot_lon, first_stop['stop_lat'], first_stop['stop_lon']) or 0.0
                        else:
                            depot_to_first_km = 0.0
                        school_to_depot_km = haversine_km(school_lat, school_lon, depot_lat, depot_lon) or 0.0
                        trip_km += depot_to_first_km + school_to_depot_km
                        trip_time_min += travel_time_min_km(depot_to_first_km + school_to_depot_km, self.avg_speed_kmph) or 0.0
                    # final trip after rollback

                # compute bus_type_needed string and seat utilization
                seat_util_pct = trip_students / float(bus_cap) if bus_cap else 0.0
                trip_id = f"{direction.upper()}_TRIP_{len(trips)+1:05d}"
                trip_record = {
                    "trip_id": trip_id,
                    "direction": direction,
                    "school_branch": school_branch,
                    "batch": batch,
                    "depot_name": depot_name,
                    "bus_vendor": chosen_bus_type.get('vendor'),
                    "bus_seating_capacity": bus_cap,
                    "usable_capacity": usable_cap,
                    "students_on_trip": trip_students,
                    "seat_util_pct": round(seat_util_pct, 3),
                    "stops": ",".join(trip_stops),
                    "est_km": round(trip_km, 3),
                    "est_time_min": round(trip_time_min, 1)
                }
                trips.append(trip_record)
                trip_records.append({
                    "trip_id": trip_id,
                    "bus_type": chosen_bus_type,
                    "est_km": trip_km
                })
                # continue to build next trip until all stops served

        # convert to dataframe
        trips_df = pd.DataFrame(trips)
        return trips_df, trip_records

    # ----------------- fleet packing (naive) -----------------
    def pack_trips_into_fleet(self, trip_records: List[Dict[str,Any]]) -> Dict[str, Any]:
        """
        Simple first-fit decreasing bin packing by bus_km capacity (monthly).
        - trip_records: list of {"trip_id","bus_type", "est_km"}
        Returns summary: {bus_type_name: required_count, total_monthly_cost, details}
        """
        # load bus_master info into list of types
        bus_types = []
        for _, r in self.bus_master.iterrows():
            bus_types.append({
                "vendor": r.get("vendor"),
                "seating_capacity": int(r.get("seating_capacity") or 0),
                "fixed_cost": float(r.get("fixed_cost") or 0)
            })
        if not bus_types:
            # fallback: create a 50-seater default
            bus_types = [{"vendor":"default","seating_capacity":50,"fixed_cost":1000.0}]

        # group trips by bus vendor/type suggested (we used chosen_bus_type earlier)
        # For naive packing, we'll ignore vendor and pack trips into identical buses by seating capacity class.
        # Build list of trip kms
        trip_kms = [r["est_km"] for r in trip_records]
        # sort descending
        trip_kms_sorted = sorted(trip_kms, reverse=True)

        # choose bus class (largest seating capacity from bus_types)
        bus_class = max(bus_types, key=lambda x: x["seating_capacity"])
        bus_km_capacity_month = self.max_bus_km_per_month
        bus_cost = bus_class["fixed_cost"]

        # simple bin packing: first-fit decreasing with km capacity
        bins = []  # each bin is remaining_km
        trip_to_bus = []
        for km in trip_kms_sorted:
            placed = False
            for i in range(len(bins)):
                if bins[i] >= km:
                    bins[i] -= km
                    placed = True
                    trip_to_bus.append(i)
                    break
            if not placed:
                # open new bus
                bins.append(bus_km_capacity_month - km)
                trip_to_bus.append(len(bins)-1)

        required_buses = len(bins)
        total_monthly_cost = required_buses * bus_cost
        summary = {
            "bus_class_used": bus_class,
            "required_buses": required_buses,
            "total_monthly_cost": total_monthly_cost,
            "bins_remaining_km": bins
        }
        return summary

    # ----------------- run all -----------------
    def run_all(self):
        """
        Build pickup & drop trips and run naive fleet packing.
        Writes CSVs to trips_out_dir and returns a dict of results.
        """
        # get bus types from bus_master as list of dicts for use in builder
        bus_types = []
        if not self.bus_master.empty:
            for _, r in self.bus_master.iterrows():
                bus_types.append({
                    "vendor": r.get("vendor"),
                    "seating_capacity": int(r.get("seating_capacity") or 0),
                    "fixed_cost": float(r.get("fixed_cost") or 0)
                })
        else:
            # default
            bus_types = [{"vendor":"default","seating_capacity":50,"fixed_cost":0}]

        results = {}

        # pickup
        if not self.stops_pickup.empty and not self.parts_pickup.empty:
            trips_pickup_df, trip_records_pickup = self._build_trips_for_group(self.stops_pickup, self.parts_pickup, "pickup", bus_types)
            trips_pickup_df.to_csv(self.trips_out_dir / "trips_pickup.csv", index=False)
            results['trips_pickup_df'] = trips_pickup_df
            results['trip_records_pickup'] = trip_records_pickup
            # fleet packing
            results['packing_pickup'] = self.pack_trips_into_fleet(trip_records_pickup)
        else:
            self.log.info("Skipping pickup trip generation (missing stops or partmap).")

        # drop
        if not self.stops_drop.empty and not self.parts_drop.empty:
            trips_drop_df, trip_records_drop = self._build_trips_for_group(self.stops_drop, self.parts_drop, "drop", bus_types)
            trips_drop_df.to_csv(self.trips_out_dir / "trips_drop.csv", index=False)
            results['trips_drop_df'] = trips_drop_df
            results['trip_records_drop'] = trip_records_drop
            results['packing_drop'] = self.pack_trips_into_fleet(trip_records_drop)
        else:
            self.log.info("Skipping drop trip generation (missing stops or partmap).")

        # write summary
        summary = {
            "pickup_summary": results.get('packing_pickup'),
            "drop_summary": results.get('packing_drop')
        }
        pd.DataFrame([summary]).to_csv(self.trips_out_dir / "fleet_packing_summary.csv", index=False)
        self.log.info("Trip generation & naive fleet packing complete. Outputs saved to %s", str(self.trips_out_dir))
        return results


if __name__ == "__main__":
    tg = TripGenerator()
    tg.run_all()