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

    def solve(self, time_limit_seconds) -> Dict[str, Any]:
        """
        Solve the fleet sizing + trip assignment MIP.

        Expects:
        - self.trip_records: list of dicts each with keys ['trip_id','est_km','students_on_trip']
        - self.bus_master_df: DataFrame with seating_capacity, available_buses, fixed_cost, empty_seats (optional)

        Returns dict:
        - status, total_cost, buses_needed (by bus_type_index), trip_assignments (trip_id -> assigned bus_type_index), bus_types (list)
        """
        import pulp

        # --- normalize trips ---
        trips = []
        trip_by_id = {}
        for rec in self.trips:
            # required fields
            tid = str(rec.get("trip_id") or rec.get("trip") or f"trip_{len(trips)}")
            est_km = float(rec.get("est_km", 0.0))
            students = int(rec.get("students_on_trip", rec.get("students", rec.get("num_participants", 0))))
            trips.append(tid)
            trip_by_id[tid] = {"est_km": est_km, "students_on_trip": students}

        if len(trips) == 0:
            return {"status": "no_trips", "total_cost": 0.0, "buses_needed": {}, "trip_assignments": {}, "bus_types": []}

        # --- normalize bus types ---
        bus_types = []
        for _, r in self.bus_master.iterrows():
            seating = int(r.get("seating_capacity") or r.get("seating", 0))
            available = int(r.get("available_buses") or r.get("available", 0) or 0)
            fixed_cost = float(r.get("fixed_cost") or 0.0)
            empty = r.get("empty_seats", None)
            if empty is None or (isinstance(empty, float) and pd.isna(empty)) or empty == "":
                usable = int(math.floor(seating * float(self.cfg.get("constraints", {}).get("max_occupancy", 0.9))))
            else:
                usable = max(0, seating - int(empty))
            bus_types.append({
                "vendor": r.get("vendor"),
                "seating_capacity": seating,
                "available_buses": available,
                "fixed_cost": fixed_cost,
                "empty_seats": empty,
                "usable_capacity": usable
            })

        K = list(range(len(bus_types)))
        if len(K) == 0:
            return {"status": "no_bus_types", "total_cost": 0.0, "buses_needed": {}, "trip_assignments": {}, "bus_types": []}

        # --- parameters ---
        max_bus_km_per_month = float(self.cfg.get("constraints", {}).get("max_bus_km_per_month", 2500))
        school_days = int(self.cfg.get("routing", {}).get("school_days_per_month", 22))

        # monthly km required by trip t = est_km * school_days * 2? (if both pickup+drop counted separately)
        # Here we assume est_km is per-trip (one-way). If trips are round-trip, adjust accordingly.
        trip_monthly_km = {t: trip_by_id[t]["est_km"] * school_days for t in trips}
        trip_students = {t: trip_by_id[t]["students_on_trip"] for t in trips}

        # --- build problem ---
        prob = pulp.LpProblem("FleetSizingAssign", pulp.LpMinimize)

        # variables
        # buses_k: number of buses of type k (integer, 0..available)
        buses_k = {k: pulp.LpVariable(f"buses_k_{k}", lowBound=0, upBound=bus_types[k]["available_buses"], cat="Integer") for k in K}

        # y_tk: binary, 1 if trip t is assigned to bus type k
        y = {(t,k): pulp.LpVariable(f"y_{t}_{k}", cat="Binary") for t in trips for k in K}

        # objective: minimize monthly fixed cost sum_k buses_k * fixed_cost_k
        prob += pulp.lpSum([buses_k[k] * bus_types[k]["fixed_cost"] for k in K]), "TotalFixedCost"

        # constraints
        # (A) Each trip assigned to exactly one bus type
        for t in trips:
            prob += pulp.lpSum([y[(t,k)] for k in K]) == 1, f"assign_once_{t}"

        # (B) capacity: if usable_cap[k] < trip_students[t] then forbid y[t,k]==1
        for t in trips:
            req = trip_students[t]
            for k in K:
                if bus_types[k]["usable_capacity"] < req:
                    prob += y[(t,k)] == 0, f"cap_forbid_{t}_{k}"

        # (C) buses available & assignment linkage:
        # sum_t y[t,k] <= buses_k[k] * max_trips_per_bus_per_month
        # but we don't know trips-per-bus; simplest conservative approach: require buses_k[k] * max_bus_km_per_month >= sum_t y[t,k] * trip_monthly_km[t]
        for k in K:
            prob += pulp.lpSum([y[(t,k)] * trip_monthly_km[t] for t in trips]) <= buses_k[k] * max_bus_km_per_month, f"km_capacity_{k}"
            # also cannot assign more trips than buses * huge (optional)
            # prob += pulp.lpSum([y[(t,k)] for t in trips]) <= buses_k[k] * 2000, f"trip_count_upper_{k}"

        # (D) available buses upper bound is set on buses_k variable (via upBound). Optionally enforce explicit bound:
        for k in K:
            prob += buses_k[k] <= bus_types[k]["available_buses"], f"available_bound_{k}"

        # Solve
        # optional time limit for pulp CBC
        solver = None
        if time_limit_seconds:
            solver = pulp.PULP_CBC_CMD(timeLimit=int(time_limit_seconds), msg=False)
        else:
            solver = pulp.PULP_CBC_CMD(msg=False)

        prob.solve(solver)

        status = pulp.LpStatus.get(prob.status, str(prob.status))
        # gather solution
        buses_needed = {k: int(pulp.value(buses_k[k]) or 0) for k in K}
        trip_assignments = {}
        for t in trips:
            assigned = None
            for k in K:
                val = pulp.value(y[(t,k)])
                if val is not None and float(val) > 0.5:
                    assigned = int(k)
                    break
            trip_assignments[t] = {"assigned_bus_type": assigned, "assigned_bus_vendor": bus_types[assigned]["vendor"] if assigned is not None else None}

        total_cost = sum(buses_needed[k] * bus_types[k]["fixed_cost"] for k in K)

        # augment bus_types with assigned count
        bus_types_with_assign = []
        for k in K:
            bt = dict(bus_types[k])
            bt["index"] = k
            bt["buses_assigned"] = buses_needed.get(k, 0)
            bus_types_with_assign.append(bt)

        return {
            "status": status,
            "total_cost": float(total_cost),
            "buses_needed": buses_needed,
            "trip_assignments": trip_assignments,
            "bus_types": bus_types_with_assign,
            "problem": prob
        }

