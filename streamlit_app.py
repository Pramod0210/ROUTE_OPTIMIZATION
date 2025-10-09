# web/streamlit_app.py
import streamlit as st
import pandas as pd
import yaml
from pathlib import Path
import tempfile
import time
import folium
from streamlit_folium import st_folium
import json
import subprocess

# local import
from src.tripgen.vrp_optimizer import VRPTripOptimizer
from utils.config_loader import load_config



# CONFIG
ROOT = Path.cwd()
CONFIG_PATH = ROOT / "config" / "config.yaml"
STOPS_CSV = ROOT / "data" / "processed" / "stops_pickup.csv"
BUS_MASTER_CSV = ROOT / "data" / "processed" / "bus_master.csv"
VRP_REQ_DIR = ROOT / "data" / "processed" / "vrp_requests"
VRP_RUNS_DIR = ROOT / "data" / "processed" / "vrp_runs"
VRP_SCRIPT = ROOT / "scripts" / "run_vrp_branch.py"  # CLI wrapper we discussed
RUN_IN_PROCESS = True  # if True, import VRPTripOptimizer and run inline (risky), else run subprocess

VRP_REQ_DIR.mkdir(parents=True, exist_ok=True)
VRP_RUNS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(layout="wide", page_title="School Transport Optimizer")

st.title("School Transport — Run Optimization")

# load config
cfg = load_config()

# load stops (no upload)
if not STOPS_CSV.exists():
    st.error(f"Stops file not found: {STOPS_CSV}")
    st.stop()

stops_df = pd.read_csv(STOPS_CSV)
stops_df.columns = [c.strip() for c in stops_df.columns]
if "school_branch" not in stops_df.columns or "batch" not in stops_df.columns:
    st.error("stops_pickup.csv must contain columns 'school_branch' and 'batch'.")
    st.stop()

# select branch/batch
groups = stops_df.groupby(["school_branch", "batch"]).size().reset_index(name="count")
groups["label"] = groups["school_branch"] + " / " + groups["batch"] + " (" + groups["count"].astype(str) + ")"
labels = groups["label"].tolist()
labels.insert(0, "ALL")  # option to run all
sel_label = st.selectbox("Select branch / batch to run (or ALL)", labels)

# choice: run a single group or all
run_button = st.button("Run Optimization")

st.sidebar.markdown("### Job settings")
max_vehicles = st.sidebar.number_input("max_vehicles_per_branch (cap)", min_value=1, value=int(cfg.get("routing", {}).get("max_vehicles_per_branch", 6)))
search_time_limit = st.sidebar.number_input("solver time limit (s)", min_value=5, value=int(cfg.get("routing", {}).get("search_time_limit_seconds", 20)))
run_pickup = st.sidebar.checkbox("Run pickup optimization", value=True)
run_drop = st.sidebar.checkbox("Run drop optimization", value=False)

# helper: build bus_types from bus_master csv
def build_bus_types():
    if not BUS_MASTER_CSV.exists():
        st.warning("bus_master.csv not found; results may be invalid.")
        return []
    bm = pd.read_csv(BUS_MASTER_CSV)
    bus_types = []
    for _, r in bm.iterrows():
        seating = int(r.get("seating_capacity", 0))
        empty = r.get("empty_seats", None)
        if pd.notna(empty) and str(empty).strip() != "":
            usable = seating - int(empty)
        else:
            usable = int(seating * float(cfg.get("constraints", {}).get("max_occupancy", 0.9)))
        bus_types.append({
            "vendor": r.get("vendor"),
            "seating_capacity": seating,
            "available_buses": int(r.get("available_buses", 0)),
            "usable_capacity": usable,
            "fixed_cost": float(r.get("fixed_cost", 0.0))
        })
    return bus_types

bus_types_global = build_bus_types()

# helper: prepare request JSON for a given branch/batch
def prepare_req(branch, batch, out_dir=None):
    subset = stops_df[(stops_df["school_branch"]==branch) & (stops_df["batch"]==batch)].copy()
    stops_records = subset.to_dict("records")
    depot = cfg.get("paths", {}).get("default_depot", [19.070000, 72.870000])
    school_coord = cfg.get("paths", {}).get("default_school_coord", [19.075983, 72.877655])
    school_start = cfg.get("calendar", {}).get("regular_start", "08:30")
    h,m = map(int, str(school_start).split(":"))
    school_start_min = h*60 + m
    # override solver settings in config copy
    cfg_local = dict(cfg)
    cfg_local.setdefault("routing", {})["max_vehicles_per_branch"] = int(max_vehicles)
    cfg_local.setdefault("routing", {})["search_time_limit_seconds"] = int(search_time_limit)
    req = {
        "config_path": str(CONFIG_PATH),
        "branch": branch,
        "batch": batch,
        "stops": stops_records,
        "parts": [],
        "bus_types": bus_types_global,
        "depot": depot,
        "school_coord": school_coord,
        "school_start_time_min": int(school_start_min),
        "out_dir": str(out_dir) if out_dir else str(VRP_RUNS_DIR)
    }
    return req

# list previous job dirs
def list_jobs():
    jdirs = sorted([d for d in VRP_RUNS_DIR.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
    return jdirs

st.markdown("---")
st.subheader("Previous runs (most recent)")
job_dirs = list_jobs()
if job_dirs:
    for d in job_dirs[:10]:
        st.write(f"- {d.name} — {time.ctime(d.stat().st_mtime)}")
else:
    st.write("No previous runs found.")

# Main action: run optimization (subprocess)
if run_button:
    # determine which groups to run
    jobs_to_run = []
    if sel_label == "ALL":
        for _, row in groups.iterrows():
            jobs_to_run.append((row['school_branch'], row['batch']))
    else:
        sb, batchcount = sel_label.split(" / ")[0], sel_label.split(" / ")[1]
        batchname = batchcount.split(" (")[0]
        jobs_to_run.append((sb, batchname))

    # st.info(f"Scheduling {len(jobs_to_run)} job(s) — running via subprocess for isolation.")

    overall_results = []
    for branch, batch in jobs_to_run:
        # prepare request JSON file
        timestamp = int(time.time())
        jobname = f"{branch}_{batch}_{timestamp}"
        job_dir = VRP_RUNS_DIR / jobname
        job_dir.mkdir(parents=True, exist_ok=True)
        req = prepare_req(branch, batch, out_dir=str(job_dir))
        req_path = VRP_REQ_DIR / f"req_{jobname}.json"
        req_path.write_text(json.dumps(req))
        st.write(f"Prepared request for {branch}/{batch} -> {req_path}")

        if RUN_IN_PROCESS:
            # run in-process (not recommended for production)
            from src.tripgen.vrp_optimizer import VRPTripOptimizer
            vrp = VRPTripOptimizer()
            st.write("Running VRP in-process (may be unstable)...")
            try:
                result = vrp.solve_branch(pd.DataFrame(req['stops']), pd.DataFrame(), branch, batch, req['bus_types'], tuple(req['depot']), tuple(req['school_coord']), req['school_start_time_min'], logger=st.write)
                # write outputs
                routes_df = result.get("routes_df", pd.DataFrame()); routes_df.to_csv(job_dir / f"routes_{branch}_{batch}.csv", index=False)
                pd.DataFrame(result.get("trip_records", [])).to_csv(job_dir / f"trip_records_{branch}_{batch}.csv", index=False)
                if not result.get("impossible_stops_df", pd.DataFrame()).empty:
                    result.get("impossible_stops_df").to_csv(job_dir / f"impossible_{branch}_{batch}.csv", index=False)
                st.success(f"Job {jobname} completed in-process.")
            except Exception as e:
                st.error(f"In-process run failed: {e}")
        else:
            # Run the CLI script as a subprocess and stream output to the UI
            if not VRP_SCRIPT.exists():
                st.error(f"VRP script not found at {VRP_SCRIPT}. Create scripts/run_vrp_branch.py or set RUN_IN_PROCESS True.")
                break

            cmd = ["python", str(VRP_SCRIPT), "--input", str(req_path)]
            st.write("Launching subprocess:", " ".join(cmd))
            # run and stream output
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            placeholder = st.empty()
            output_lines = []
            start_t = time.time()
            try:
                # stream stdout lines
                while True:
                    line = process.stdout.readline()
                    if line:
                        output_lines.append(line)
                        # display last 20 lines
                        placeholder.code("".join(output_lines[-200:]))
                    elif process.poll() is not None:
                        # read remaining
                        rest = process.stdout.read()
                        if rest:
                            output_lines.append(rest)
                            placeholder.code("".join(output_lines[-200:]))
                        break
                    else:
                        time.sleep(0.1)
                ret = process.poll()
                elapsed = time.time() - start_t
                if ret == 0:
                    st.success(f"Subprocess completed in {elapsed:.1f}s. Job dir: {job_dir}")
                else:
                    st.error(f"Subprocess exited with code {ret}. Check logs. Job dir: {job_dir}")
                    st.code("".join(output_lines[-400:]))
            except Exception as e:
                process.kill()
                st.error(f"Subprocess failed: {e}")
                st.code("".join(output_lines[-200:]))

        # after run, show outputs if any
        routes_glob = sorted(job_dir.glob("routes_*.csv"))
        trips_glob = sorted(job_dir.glob("trip_records_*.csv"))
        imp_glob = sorted(job_dir.glob("impossible_*.csv"))
        if routes_glob:
            routes_df = pd.read_csv(routes_glob[0])
            st.write("Routes:")
            st.dataframe(routes_df)
            st.download_button("Download routes CSV", routes_df.to_csv(index=False), file_name=f"routes_{jobname}.csv", mime="text/csv")
        if trips_glob:
            trips_df = pd.read_csv(trips_glob[0])
            st.write("Trip records:")
            st.dataframe(trips_df)
            st.download_button("Download trip_records CSV", trips_df.to_csv(index=False), file_name=f"trip_records_{jobname}.csv", mime="text/csv")
        if imp_glob:
            imp_df = pd.read_csv(imp_glob[0])
            st.error("Impossible / flagged stops")
            st.dataframe(imp_df)
            st.download_button("Download impossible stops", imp_df.to_csv(index=False), file_name=f"impossible_{jobname}.csv", mime="text/csv")

        overall_results.append({"job": jobname, "dir": str(job_dir)})

    st.success("All scheduled jobs finished (or ended).")
    st.write("Jobs:", overall_results)
