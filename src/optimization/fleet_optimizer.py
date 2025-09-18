# src/tripgen/fleet_optimizer.py
import pulp
from typing import List, Dict, Any, Tuple
from pathlib import Path
import pandas as pd
import math
from logger.custom_logger import CustomLogger
from utils.config_loader import load_config

class FleetOptimizer:
    """
    Solve an integer MIP to choose number of buses per type and assign trips to bus types.

    Inputs:
      - trip_records: list of dicts {trip_id, est_km, students_on_trip}
      - bus_master_df: pandas.DataFrame with columns vendor,seating_capacity,available_buses,fixed_cost,empty_seats (empty_seats optional)
      - cfg: config dict (used for max_occupancy, max_bus_km_per_month, school_days_per_month)
    Output:
      - dict with assignment, buses_required, total_cost, pulp problem object (for inspection)
    """

    def __init__(self, trip_records: List[Dict[str,Any]], bus_master_df: pd.DataFrame):
        self.log = CustomLogger().get_logger(__name__)
        self.cfg = load_config()
        self.trips = trip_records
        self.bus_master = bus_master_df.copy()

        self.max_occupancy = float(self.cfg.get("constraints", {}).get("max_occupancy", 0.9))
        self.max_bus_km_per_month = float(self.cfg.get("constraints", {}).get("max_bus_km_per_month", 2500))
        self.school_days = int(self.cfg.get("routing", {}).get("school_days_per_month", 22))

        # sanitize bus_master: ensure numeric types
        self.bus_master['seating_capacity'] = pd.to_numeric(self.bus_master.get('seating_capacity', 0), errors='coerce').fillna(0).astype(int)
        self.bus_master['available_buses'] = pd.to_numeric(self.bus_master.get('available_buses', 0), errors='coerce').fillna(0).astype(int)
        self.bus_master['fixed_cost'] = pd.to_numeric(self.bus_master.get('fixed_cost', 0), errors='coerce').fillna(0).astype(float)
        # empty_seats may be missing; keep NaN for now
        if 'empty_seats' not in self.bus_master.columns:
            self.bus_master['empty_seats'] = pd.NA

    def solve(self) -> Dict[str,Any]:
        # prepare sets
        """
        Solve an integer MIP to choose number of buses per type and assign trips to bus types.

        Returns a dict with:
            - status: MIP solve status
            - total_cost: total fixed cost of buses assigned
            - buses_needed: dict of bus type indices to number of buses needed
            - bus_types: list of bus type dicts with index, vendor, seating_capacity, available_buses, empty_seats, fixed_cost, usable_capacity, buses_assigned
            - trip_assignments: dict of trip_ids to bus type dicts with index, vendor, seating_capacity, usable_capacity
            - problem: the pulp problem object for inspection
        """
        trip_ids = [t['trip_id'] for t in self.trips]
        K = list(self.bus_master.index)  # bus type indices

        # build usable capacity per type: if empty_seats present use it, else floor(capacity * max_occupancy)
        usable_cap = {}
        for k in K:
            cap = int(self.bus_master.at[k, 'seating_capacity'])
            es = self.bus_master.at[k, 'empty_seats']
            if pd.isna(es):
                usable = math.floor(cap * self.max_occupancy)
            else:
                try:
                    usable = int(cap - int(es))
                except Exception:
                    usable = math.floor(cap * self.max_occupancy)
            usable_cap[k] = max(0, usable)

        # build trip_km_month = est_km * school_days (each trip occurs once per school day)
        trip_km_month = {}
        for t in self.trips:
            trip_km_month[t['trip_id']] = float(t.get('est_km', 0.0)) * float(self.school_days)

        # create LP
        prob = pulp.LpProblem("fleet_sizing", pulp.LpMinimize)

        # decision vars:
        # y_tk = 1 if trip t assigned to bus type k
        y = pulp.LpVariable.dicts("y", ((t,k) for t in trip_ids for k in K), lowBound=0, upBound=1, cat='Binary')
        # buses_k = integer number of buses of type k
        buses = pulp.LpVariable.dicts("buses", (k for k in K), lowBound=0, cat='Integer')

        # objective: minimize sum_k buses_k * fixed_cost_k
        prob += pulp.lpSum([buses[k] * float(self.bus_master.at[k, 'fixed_cost']) for k in K])

        # constraints:
        # (1) every trip assigned exactly once
        for t in trip_ids:
            prob += pulp.lpSum([y[(t,k)] for k in K]) == 1, f"assign_{t}"

        # (2) seating capacity: y_tk = 0 if usable_cap[k] < students_on_trip
        for t in self.trips:
            req = int(t.get('students_on_trip', 0))
            for k in K:
                if usable_cap[k] < req:
                    prob += y[(t,k)] == 0, f"cap_forbid_{t}_{k}"

        # (3) monthly km capacity: for each bus type k, buses_k * cap_month >= sum_t y_tk * trip_km_month[t]
        for k in K:
            prob += buses[k] * self.max_bus_km_per_month >= pulp.lpSum([ y[(t,k)] * trip_km_month[t] for t in trip_ids ]), f"kmcap_{k}"

        # (4) available buses upper bound
        for k in K:
            prob += buses[k] <= int(self.bus_master.at[k, 'available_buses']), f"avail_{k}"

        # Solve
        solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=300, threads=0)
        result = prob.solve(solver)

        status = pulp.LpStatus[result]
        self.log.info(f"MIP solve status: {status}")

        # collect solution
        assignment = {}
        for t in trip_ids:
            for k in K:
                if pulp.value(y[(t,k)]) >= 0.5:
                    assignment[t] = int(k)
                    break

        buses_needed = {}
        for k in K:
            val = int(pulp.value(buses[k]) or 0)
            buses_needed[int(k)] = val

        total_cost = sum(buses_needed.get(k,0) * float(self.bus_master.at[k,'fixed_cost']) for k in K)

        # interpret bus type indices into readable dicts
        bus_types = []
        for k in K:
            bus_types.append({
                "index": int(k),
                "vendor": self.bus_master.at[k, 'vendor'],
                "seating_capacity": int(self.bus_master.at[k, 'seating_capacity']),
                "available_buses": int(self.bus_master.at[k, 'available_buses']),
                "empty_seats": int(self.bus_master.at[k, 'empty_seats']) if pd.notna(self.bus_master.at[k, 'empty_seats']) else None,
                "fixed_cost": float(self.bus_master.at[k, 'fixed_cost']),
                "usable_capacity": usable_cap[k],
                "buses_assigned": buses_needed.get(k, 0)
            })

        # map trip assignments to bus type descriptions
        trip_assignments = {}
        for t in trip_ids:
            k = assignment.get(t)
            if k is None:
                trip_assignments[t] = None
            else:
                trip_assignments[t] = {
                    "bus_type_index": int(k),
                    "vendor": self.bus_master.at[k, 'vendor'],
                    "seating_capacity": int(self.bus_master.at[k, 'seating_capacity']),
                    "usable_capacity": usable_cap[k]
                }

        return {
            "status": status,
            "total_cost": total_cost,
            "buses_needed": buses_needed,
            "bus_types": bus_types,
            "trip_assignments": trip_assignments,
            "problem": prob
        }
