# rowcount.py — fixed settings, 2σ clipping, signed deviations, stacked charts
# Adds constrained start/end inputs (within file range) and auto-aligns end to whole hours from start.

from __future__ import annotations
from datetime import timedelta
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Muon Log Dashboard", layout="wide")
st.title("Muon Detector Log Dashboard")
st.caption("Time from column 6 (or 6–10 if split). Counts from columns 1, 3, 4, 5. 2σ clipping applied before all charts & deviations.")

# ---------------- Fixed settings (edit if your file format changes) ----------------
SEP = r"[ ]+"                      # spaces (1+)
ENGINE = "python"                  # CSV engine for regex sep
HAS_HEADER = False                 # logs have no header row
SKIP_ROWS = 0                      # >0 if your file starts with non-data lines
TIME_COL_1B = 6                    # time starts at column 6 (1-based)
C1_1B, C3_1B, C4_1B, C5_1B = 1, 3, 4, 5  # data columns (1-based)
TIMESTAMP_SPLIT_5 = True           # 'Fri | Aug | 23 | 17:02:07 | 2024'
CUSTOM_FMT = "%a %b %d %H:%M:%S %Y"

MAX_POINTS = 10_000                # cap for minute-level plotting only (performance)

# -----------------------------------------------------------------------------------

# File upload (main page)
file = st.file_uploader("Upload a .log / .txt / .csv file", type=["log", "txt", "csv"])

def load_df(_file, sep, engine, header_flag, skip_rows):
    kwargs = dict(sep=sep, engine=engine, dtype=str, skiprows=skip_rows, on_bad_lines="skip")
    if engine == "c":
        kwargs["low_memory"] = False
    return pd.read_csv(_file, header=(0 if header_flag else None), **kwargs)

def clean_numeric(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.strip()
    s2 = s2.str.replace(",", "", regex=False)
    s2 = s2.str.replace(r"\b(None|nan|NaN)\b", "", regex=True)
    s2 = s2.str.replace(r"[^0-9.\-eE]", "", regex=True)  # keep digits, dot, minus, exponent
    return pd.to_numeric(s2, errors="coerce")

def to_datetime_joined(df: pd.DataFrame, start_idx: int, fmt: str, split5: bool) -> pd.Series:
    if split5:
        parts = [df.iloc[:, start_idx + k].astype(str).str.strip() for k in range(5)]
        txt = parts[0] + " " + parts[1] + " " + parts[2] + " " + parts[3] + " " + parts[4]
    else:
        txt = df.iloc[:, start_idx].astype(str).str.strip()
    return pd.to_datetime(txt, format=fmt, errors="coerce")

def sigma_clip_series(s: pd.Series, sigma: float = 2.0) -> pd.Series:
    """Return a series where values outside mean±sigma*std are set to NaN."""
    m = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if pd.isna(m) or pd.isna(sd) or sd == 0:
        return s  # nothing to clip
    mask = (s - m).abs() <= sigma * sd
    return s.where(mask)

def align_end_to_full_hours(start_dt: pd.Timestamp, end_dt: pd.Timestamp, min_t: pd.Timestamp, max_t: pd.Timestamp) -> tuple[pd.Timestamp, bool]:
    """
    Clamp both to [min_t, max_t], then align end to start + N hours (N>=0, integer),
    preserving start's minute/second/microsecond. Returns (aligned_end, adjusted_flag).
    """
    # clamp
    start_dt = max(min_t, min(start_dt, max_t))
    end_dt_input = max(min_t, min(end_dt, max_t))

    # if end before start, move end to start
    if end_dt_input < start_dt:
        return start_dt, True

    # max allowed N hours based on file end
    max_hours = int((max_t - start_dt).total_seconds() // 3600)
    if max_hours < 0:
        max_hours = 0

    # candidate hours from user's typed end (floor to whole hours)
    desired_hours = int((end_dt_input - start_dt).total_seconds() // 3600)
    desired_hours = max(0, min(desired_hours, max_hours))

    aligned_end = start_dt + timedelta(hours=desired_hours)
    adjusted = (aligned_end != end_dt_input)
    return aligned_end, adjusted

if not file:
    with st.expander("Fixed settings (read-only)", expanded=False):
        st.write(
            {
                "separator": "Spaces (1+)",
                "time_split_across_5_cols": TIMESTAMP_SPLIT_5,
                "time_format": CUSTOM_FMT,
                "time_col_start_1based": TIME_COL_1B,
                "data_cols_1based": {"total": C1_1B, "top+mid": C3_1B, "top+bot": C4_1B, "mid+bot": C5_1B},
                "has_header": HAS_HEADER,
                "skip_rows": SKIP_ROWS,
                "max_points_minute_plots": MAX_POINTS,
            }
        )
    st.info("⬆️ Upload a log file to begin.")
    st.stop()

# ---------------- Parse file ----------------
df = load_df(file, SEP, ENGINE, HAS_HEADER, SKIP_ROWS)

# 1-based → 0-based
t0 = TIME_COL_1B - 1
i1, i3, i4, i5 = C1_1B - 1, C3_1B - 1, C4_1B - 1, C5_1B - 1

need = max(i1, i3, i4, i5, t0 if not TIMESTAMP_SPLIT_5 else t0 + 4)
if df.shape[1] <= need:
    st.error(f"File has {df.shape[1]} columns but needs index up to {need}. "
             f"Check the fixed settings at the top of this file.")
    st.stop()

time = to_datetime_joined(df, t0, CUSTOM_FMT, TIMESTAMP_SPLIT_5)
if time.isna().all():
    with st.expander("Preview of raw time columns", expanded=True):
        cols = list(range(t0, t0 + (5 if TIMESTAMP_SPLIT_5 else 1)))
        st.dataframe(df.iloc[:10, cols], use_container_width=True)
    st.error("Time parse failed. Adjust CUSTOM_FMT / TIMESTAMP_SPLIT_5 in the constants.")
    st.stop()

total     = clean_numeric(df.iloc[:, i1])
top_mid   = clean_numeric(df.iloc[:, i3])
top_bot   = clean_numeric(df.iloc[:, i4])
mid_bot   = clean_numeric(df.iloc[:, i5])

work_all = (
    pd.DataFrame(
        {
            "time":             time,
            "Total (per min)":  total,
            "Top + Middle":     top_mid,
            "Top + Bottom":     top_bot,
            "Middle + Bottom":  mid_bot,
        }
    )
    .dropna(subset=["time"])
    .sort_values("time")
    .reset_index(drop=True)
)

# ---------------- Time range inputs (no slider) ----------------
min_t, max_t = work_all["time"].min(), work_all["time"].max()
st.subheader("Time range (must be within file range and cover whole hours)")
colA, colB = st.columns(2)
start_txt = colA.text_input("Start (YYYY-MM-DD HH:MM:SS)", value=min_t.strftime("%Y-%m-%d %H:%M:%S"))
end_txt   = colB.text_input("End (YYYY-MM-DD HH:MM:SS)",   value=max_t.strftime("%Y-%m-%d %H:%M:%S"))

# Show file limits right below the inputs
st.caption(f"File time range: **{min_t}** → **{max_t}**")

# Parse user inputs
typed_start = pd.to_datetime(start_txt, errors="coerce")
typed_end   = pd.to_datetime(end_txt, errors="coerce")

# Validate start
if pd.isna(typed_start):
    st.error("Invalid start time. Please use the format YYYY-MM-DD HH:MM:SS.")
    st.stop()
if typed_start < min_t or typed_start > max_t:
    st.warning("Start was outside the file's time range and has been clamped to the nearest bound.")
typed_start = max(min_t, min(typed_start, max_t))

# Validate & align end to whole hours from start, clamped to file range
if pd.isna(typed_end):
    # if invalid, default to file end then align
    typed_end = max_t

aligned_end, adjusted = align_end_to_full_hours(typed_start, typed_end, min_t, max_t)

if adjusted:
    st.info(f"End time auto-aligned to **{aligned_end}** so the window covers a whole number of hours from the start.")

start_dt, end_dt = typed_start, aligned_end

# Final mask
mask = (work_all["time"] >= start_dt) & (work_all["time"] <= end_dt)
work_sel = work_all.loc[mask].copy()

# ---------------- 2σ clipping on counts (per series, within selection) ----------------
coinc_cols = ["Top + Middle", "Top + Bottom", "Middle + Bottom"]
work_clip = work_sel.copy()
clip_info = {}

for c in coinc_cols + ["Total (per min)"]:
    s = work_clip[c]
    before = s.notna().sum()
    s_clipped = sigma_clip_series(s, sigma=2.0)
    after = s_clipped.notna().sum()
    clip_info[c] = {"kept": int(after), "removed": int(before - after)}
    work_clip[c] = s_clipped

st.caption(
    "2σ clipping applied to counts in the selected window — "
    + "; ".join([f"{k}: kept {v['kept']}, removed {v['removed']}" for k, v in clip_info.items() if k != "Total (per min)"])
)

# ---------------- Deviations (computed from CLIPPED data) ----------------
# Minute-level signed deviation vs overall mean (within selection, using clipped counts)
minute_means = {c: work_clip[c].mean(skipna=True) for c in coinc_cols}
for c in coinc_cols:
    m = minute_means[c]
    work_clip[f"{c} Δ% (min vs overall)"] = ((work_clip[c] - m) / m * 100.0) if (pd.notna(m) and m != 0) else pd.NA

# Hourly mean (from clipped minute data, within selection)
hourly = (
    work_clip.set_index("time")
             .resample("H")
             .mean(numeric_only=True)
             .dropna(how="all")
)

# Hourly signed deviation: hour mean vs overall hourly mean (from clipped hourly)
hourly_dev = pd.DataFrame(index=hourly.index)
for c in coinc_cols:
    overall = hourly[c].mean(skipna=True)
    hourly_dev[f"{c} Δ% (hour mean vs overall)"] = (
        (hourly[c] - overall) / overall * 100.0 if (pd.notna(overall) and overall != 0) else pd.NA
    )

# ---------------- Downsample minute-level plots only (after clipping) ----------------
n = len(work_clip)
stride = max(1, n // MAX_POINTS)
plot_min = work_clip.iloc[::stride].copy()

# ---------------- Tabs (each chart stacked) ----------------
tabs = st.tabs([
    "Counts — minute (2σ-clipped)",
    "Counts — hourly mean (from clipped)",
    "Deviation % — minute (vs overall mean, clipped)",
    "Deviation % — hourly (hour mean vs overall mean, clipped)"
])

# ---- Counts (minute) ----
with tabs[0]:
    st.subheader("Top + Middle (per minute)")
    st.line_chart(plot_min.set_index("time")[["Top + Middle"]])
    st.subheader("Top + Bottom (per minute)")
    st.line_chart(plot_min.set_index("time")[["Top + Bottom"]])
    st.subheader("Middle + Bottom (per minute)")
    st.line_chart(plot_min.set_index("time")[["Middle + Bottom"]])

# ---- Counts (hourly mean) ----
with tabs[1]:
    st.subheader("Top + Middle (hourly mean)")
    st.line_chart(hourly[["Top + Middle"]])
    st.subheader("Top + Bottom (hourly mean)")
    st.line_chart(hourly[["Top + Bottom"]])
    st.subheader("Middle + Bottom (hourly mean)")
    st.line_chart(hourly[["Middle + Bottom"]])

# ---- Deviation % (minute vs overall mean) ----
with tabs[2]:
    st.subheader("Top + Middle — Δ% (minute vs overall mean)")
    st.line_chart(plot_min.set_index("time")[["Top + Middle Δ% (min vs overall)"]])
    st.subheader("Top + Bottom — Δ% (minute vs overall mean)")
    st.line_chart(plot_min.set_index("time")[["Top + Bottom Δ% (min vs overall)"]])
    st.subheader("Middle + Bottom — Δ% (minute vs overall mean)")
    st.line_chart(plot_min.set_index("time")[["Middle + Bottom Δ% (min vs overall)"]])

# ---- Deviation % (hour mean vs overall hourly mean) ----
with tabs[3]:
    st.subheader("Top + Middle — Δ% (hour mean vs overall mean)")
    st.line_chart(hourly_dev[[c for c in hourly_dev.columns if c.startswith("Top + Middle")]])
    st.subheader("Top + Bottom — Δ% (hour mean vs overall mean)")
    st.line_chart(hourly_dev[[c for c in hourly_dev.columns if c.startswith("Top + Bottom")]])
    st.subheader("Middle + Bottom — Δ% (hour mean vs overall mean)")
    st.line_chart(hourly_dev[[c for c in hourly_dev.columns if c.startswith("Middle + Bottom")]])

# ---- Downloads (optional) ----
with st.expander("Downloads"):
    st.download_button(
        "Download minute-level CLIPPED CSV (includes minute Δ% columns)",
        data=work_clip.to_csv(index=False).encode(),
        file_name="muon_minute_clipped_with_deviation.csv",
        mime="text/csv",
    )
    hcsv = hourly.reset_index().rename(columns={"index": "time"})
    st.download_button(
        "Download hourly mean (from clipped) CSV",
        data=hcsv.to_csv(index=False).encode(),
        file_name="muon_hourly_mean_clipped.csv",
        mime="text/csv",
    )
    dcsv = hourly_dev.reset_index().rename(columns={"index": "time"})
    st.download_button(
        "Download hourly deviation CSV (from clipped)",
        data=dcsv.to_csv(index=False).encode(),
        file_name="muon_hourly_deviation_clipped.csv",
        mime="text/csv",
    )
