"""Website Traffic Analyzer - Streamlit interface.

Upload a website access log (CSV) and the app cleans it, runs the
hand-written MapReduce pipeline from website_traffic_analyzer.py, and shows
the results.

Run with:   py -m streamlit run src/app.py
"""

import csv
import io
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from website_traffic_analyzer import (
    clean_data,
    error_rates,
    map_record,
    run_mapreduce,
    run_with_combiner,
    top_k,
)

REQUIRED_COLUMNS = ["request_id", "timestamp", "client_ip",
                    "path", "status_code", "response_ms"]
SAMPLE_FILE = Path(__file__).resolve().parent.parent / "data" / "02_website_traffic.csv"

st.set_page_config(page_title="Website Traffic Analyzer", page_icon="📊", layout="wide")

st.title("📊 Website Traffic Analyzer")
st.caption("Upload a website access log and analyse it with a hand-written "
           "MapReduce pipeline (map → shuffle → reduce).")


# ----------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------
st.sidebar.header("1. Load data")
uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
use_sample = False
if SAMPLE_FILE.exists():
    use_sample = st.sidebar.checkbox("Use the sample dataset instead")

st.sidebar.markdown(
    "**Expected columns:**  \n" + ", ".join(f"`{c}`" for c in REQUIRED_COLUMNS)
)

raw_text = None
source_name = None
if uploaded is not None:
    raw_text = uploaded.getvalue().decode("utf-8-sig", errors="replace")
    source_name = uploaded.name
elif use_sample:
    raw_text = SAMPLE_FILE.read_text(encoding="utf-8-sig")
    source_name = SAMPLE_FILE.name

if raw_text is None:
    st.info("👈 Upload a CSV file in the sidebar to begin.")
    st.stop()

# Parse the CSV into a list of dictionaries (same shape as load_data returns)
reader = csv.DictReader(io.StringIO(raw_text))
columns = [c.strip() for c in (reader.fieldnames or [])]
missing_cols = [c for c in REQUIRED_COLUMNS if c not in columns]
if missing_cols:
    st.error("This file is missing required column(s): "
             + ", ".join(f"`{c}`" for c in missing_cols))
    st.stop()

records = [{(k or "").strip(): v for k, v in row.items()} for row in reader]
if not records:
    st.error("The file has a header but no data rows.")
    st.stop()


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------
clean, stats = clean_data(records)
if not clean:
    st.error("No valid rows were left after cleaning. Check the file format.")
    st.stop()

results, mapped, groups = run_mapreduce(clean, verbose=False)

first_ts = min(r["timestamp"] for r in clean)
last_ts = max(r["timestamp"] for r in clean)
total_errors = sum(results["error_page"].values())

st.subheader(f"Results for `{source_name}`")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Valid requests", f"{stats['kept']:,}")
c2.metric("Error responses (≥ 400)", f"{total_errors:,}",
          f"{total_errors / len(clean):.1%} of requests", delta_color="off")
c3.metric("Distinct pages", len(results["page"]))
c4.metric("Time span", f"{(last_ts - first_ts).days + 1} days",
          f"{first_ts:%d %b} – {last_ts:%d %b %Y}", delta_color="off")

tabs = st.tabs(["Top pages", "Status codes", "Errors", "Hourly traffic",
                "Data cleaning", "How MapReduce worked", "Performance"])

# ---- Top pages -------------------------------------------------------
with tabs[0]:
    st.markdown("#### Five most visited pages")
    top_n = st.slider("How many pages to show", 3, max(3, len(results["page"])), 5)
    top_pages = pd.DataFrame(top_k(results["page"], top_n),
                             columns=["Page", "Requests"])
    top_pages.index = range(1, len(top_pages) + 1)
    left, right = st.columns([1, 2])
    left.dataframe(top_pages, width="stretch")
    right.bar_chart(top_pages.set_index("Page"))
    st.caption("Found with a heap-based Top-K (`heapq.nlargest`).")

# ---- Status codes ----------------------------------------------------
with tabs[1]:
    st.markdown("#### HTTP status code counts")
    status_df = pd.DataFrame(sorted(results["status"].items()),
                             columns=["Status code", "Count"])
    status_df["Share"] = (status_df["Count"] / status_df["Count"].sum()).map("{:.1%}".format)
    left, right = st.columns([1, 2])
    left.dataframe(status_df, width="stretch", hide_index=True)
    right.bar_chart(status_df.assign(**{"Status code": status_df["Status code"].astype(str)})
                    .set_index("Status code")["Count"])

# ---- Errors ----------------------------------------------------------
with tabs[2]:
    st.markdown("#### Pages generating the most errors (status ≥ 400)")
    err_counts = pd.DataFrame(top_k(results["error_page"], len(results["error_page"])),
                              columns=["Page", "Errors"])
    rates = pd.DataFrame(error_rates(results["page"], results["error_page"]),
                         columns=["Page", "Error rate", "Errors", "Requests"])
    left, right = st.columns(2)
    with left:
        st.markdown("**By number of errors**")
        st.dataframe(err_counts, width="stretch", hide_index=True)
        st.bar_chart(err_counts.set_index("Page"))
    with right:
        st.markdown("**By error rate (errors ÷ requests)**")
        shown = rates.copy()
        shown["Error rate"] = shown["Error rate"].map("{:.1%}".format)
        st.dataframe(shown, width="stretch", hide_index=True)
        st.bar_chart(rates.set_index("Page")[["Error rate"]])
    st.caption("Raw counts favour busy pages. The rate shows which page is "
               "actually the most unreliable.")

# ---- Hourly ----------------------------------------------------------
with tabs[3]:
    st.markdown("#### Traffic by hour of day")
    hourly = pd.DataFrame({"Hour": [f"{h:02d}:00" for h in range(24)],
                           "Requests": [results["hour"].get(h, 0) for h in range(24)]})
    st.bar_chart(hourly.set_index("Hour"))
    busiest = hourly.loc[hourly["Requests"].idxmax()]
    quietest = hourly.loc[hourly["Requests"].idxmin()]
    m1, m2 = st.columns(2)
    m1.metric("Busiest hour", busiest["Hour"], f"{busiest['Requests']} requests", delta_color="off")
    m2.metric("Quietest hour", quietest["Hour"], f"{quietest['Requests']} requests", delta_color="off")
    with st.expander("Show table"):
        st.dataframe(hourly, width="stretch", hide_index=True)

# ---- Data cleaning ---------------------------------------------------
with tabs[4]:
    st.markdown("#### Cleaning report")
    st.write("**One row = one HTTP request:** a client IP asked for a page "
             "path at a given time and received a status code after "
             "`response_ms` milliseconds.")
    a, b, c, d, e = st.columns(5)
    a.metric("Read", stats["read"])
    b.metric("Missing", stats["missing"])
    c.metric("Invalid", stats["invalid"])
    d.metric("Duplicate", stats["duplicate"])
    e.metric("Kept", stats["kept"])
    st.markdown("**Preview of cleaned data**")
    st.dataframe(pd.DataFrame(clean).head(20), width="stretch", hide_index=True)

# ---- How MapReduce worked --------------------------------------------
with tabs[5]:
    st.markdown("#### Follow one record through the pipeline")
    idx = st.number_input("Row number (0 = first row)", 0, len(clean) - 1, 0)
    row = clean[int(idx)]
    st.markdown("**Cleaned input row**")
    st.json({k: str(v) for k, v in row.items()})

    st.markdown("**1. Map** - this row becomes key-value pairs")
    pairs = map_record(row)
    st.dataframe(pd.DataFrame([{"Key": str(k), "Value": v} for k, v in pairs]),
                 width="stretch", hide_index=True)

    st.markdown("**2. Shuffle** - all values with the same key are grouped")
    key = pairs[0][0]
    values = groups[key]
    st.write(f"Key `{key}` collected **{len(values):,}** values: "
             f"`{values[:10]}{' ...' if len(values) > 10 else ''}`")

    st.markdown("**3. Reduce** - each group is summed")
    st.write(f"`sum(values)` = **{sum(values):,}**, so {key[1]} has "
             f"{results[key[0]][key[1]]:,} requests.")

    st.divider()
    s1, s2, s3 = st.columns(3)
    s1.metric("Rows mapped", f"{len(clean):,}")
    s2.metric("Pairs produced by mapper", f"{len(mapped):,}")
    s3.metric("Distinct keys after shuffle", f"{len(groups):,}")

# ---- Performance -----------------------------------------------------
with tabs[6]:
    st.markdown("#### Plain MapReduce vs combiner")
    st.write("The data is repeated to simulate a bigger log. The **combiner** "
             "adds up counts inside each chunk of 1,000 rows before the "
             "shuffle, so far fewer pairs are sent.")
    mult = st.select_slider("Repeat the data this many times",
                            options=[1, 10, 100, 500], value=10)
    if st.button("Run comparison"):
        big = clean * mult
        t0 = time.perf_counter()
        plain_results, plain_pairs, _ = run_mapreduce(big, verbose=False)
        plain_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        comb_results, comb_pairs = run_with_combiner(big)
        comb_time = time.perf_counter() - t0

        assert plain_results == comb_results
        perf = pd.DataFrame({
            "Method": ["Plain MapReduce", "With combiner"],
            "Rows": [len(big)] * 2,
            "Pairs sent to shuffle": [len(plain_pairs), comb_pairs],
            "Seconds": [round(plain_time, 3), round(comb_time, 3)],
        })
        st.dataframe(perf, width="stretch", hide_index=True)
        st.success(f"Results identical. The combiner sent "
                   f"{len(plain_pairs) / comb_pairs:,.0f}× fewer pairs.")
        st.caption("Timings depend on your computer and vary between runs.")

# ---- Download --------------------------------------------------------
st.sidebar.header("2. Download")
summary_rows = []
for kind, label in [("page", "Page requests"), ("status", "Status code counts"),
                    ("error_page", "Errors by page"), ("hour", "Requests by hour")]:
    for name, total in results[kind].items():
        summary_rows.append({"Result": label, "Key": name, "Count": total})
st.sidebar.download_button("Download results (CSV)",
                           pd.DataFrame(summary_rows).to_csv(index=False),
                           file_name="traffic_results.csv", mime="text/csv")