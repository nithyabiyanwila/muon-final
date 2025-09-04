# rowcount.py — fixed settings, 2σ clipping, signed deviations, stacked charts
# Start/end inputs are clamped to file range; end auto-aligns to whole hours from start.

from __future__ import annotations
from datetime import timedelta
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Muon Log Dashboard", layout="wide")
st.title("Muon Detector Log Dashboard")

# ---------------- Fixed settings ----------------
SEP = r"[ ]+"                      # spaces (1+)
ENGINE = "python"                  # CSV engine for regex sep
HAS_HEADER = False
SKIP_ROWS = 0
TIME_COL_1B = 6
C1_1B, C3_1B, C4_1B, C5_1B = 1, 3, 4, 5
TIMESTAMP_SPLIT_5 = True
CUSTOM_FMT = "%a %b %d %H:%M:%S %Y"
MAX_POINTS = 10_000                # minute-level plotting cap

# ---------------- Helpers ----------------
def load_df(_file, sep, engine, header_flag, skip_rows):
    kwargs = dict(sep=sep, engine=engine, dtype=str, skiprows=skip_rows, on_bad_lines="skip")
    if engine == "c":
        kwargs["low_memory"] = False
    return pd.read_csv(_file, header=(0 if header_flag else None), **kwargs)

def clean_numeric(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.strip()
    s2 = s2.str.replace(",", "", regex=False)
    s2 = s2.str.replace(r"\b(None|nan|NaN)\b", "", regex=True)
    s2 = s2.str.replace(r"[^0-9.\-eE]", "", regex=True)
    return pd.to_numeric(s2, errors="coerce")

def to_datetime_joined(df: pd.DataFrame, start_idx: int, fmt: str, split5: bool) -> pd.Series:
    if split5:
        parts = [df.iloc[:, start_idx + k].astype(str).str.strip() for k in range(5)]
        txt = parts[0] + " " + parts[1] + " " + parts[2] + " " + parts[3] + " " + parts[4]
    else:
        txt = df.iloc[:, start_idx].astype(str).str.strip()
    return pd.to_datetime(txt, format=fmt, errors="coerce")

def sigma_clip_series(s: pd.Series, sigma: float = 2.0) -> pd.Series:
    m = s.mean(skipna=True); sd = s.std(skipna=True)
    if pd.isna(m) or pd.isna(sd) or sd == 0: return s
    return s.where((s - m).abs() <= sigma * sd)

def align_end_to_full_hours(start_dt: pd.Timestamp, end_dt: pd.Timestamp,
                            min_t: pd.Timestamp, max_t: pd.Timestamp) -> pd.Timestamp:
    # Clamp both to [min_t, max_t], then set end = start + floor((end-start)/1h)*1h
    start_dt = max(min_t, min(start_dt, max_t))
    end_dt   = max(min_t, min(end_dt,   max_t))
    if end_dt < start_dt: end_dt = start_dt
    hours = int((end_dt - start_dt).total_seconds() // 3600)
    hours = max(0, min(hours, int((max_t - start_dt).total_seconds() // 3600)))
    return start_dt + timedelta(hours=hours)

# ---------------- File upload ----------------
file = st.file_uploader("Upload a .log / .txt / .csv file", type=["log", "txt", "csv"])
if not file:
    st.stop()

# ---------------- Parse file ----------------
df = load_df(file, SEP, ENGINE, HAS_HEADER, SKIP_ROWS)

t0 = TIME_COL_1B - 1
i1, i3, i4, i5 = C1_1B - 1, C3_1B - 1, C4_1B - 1, C5_1B - 1
need = max(i1, i3, i4, i5, t0 if not TIMESTAMP_SPLIT_5 else t0 + 4)
if df.shape[1] <= need:
    st.error(f"File has {df.shape[1]} columns but needs index up to {need}.")
    st.stop()

time = to_datetime_joined(df, t0, CUSTOM_FMT, TIMESTAMP_SPLIT_5)
if time.isna().all():
    st.error("Time parse failed. Check CUSTOM_FMT / TIMESTAMP_SPLIT_5 in the code.")
    st.stop()

total   = clean_numeric(df.iloc[:, i1])
top_mid = clean_numeric(df.iloc[:, i3])
top_bot = clean_numeric(df.iloc[:, i4])
mid_bot = clean_numeric(df.iloc[:, i5])

work_all = (pd.DataFrame({
    "time": time,
    "Total (per min)": total,
    "Top + Middle":    top_mid,
    "Top + Bottom":    top_bot,
    "Middle + Bottom": mid_bot,
}).dropna(subset=["time"]).sort_values("time").reset_index(drop=True))

# ---------------- Time range (typed only) ----------------
min_t, max_t = work_all["time"].min(), work_all["time"].max()

st.subheader("Time range")
c1, c2 = st.columns(2)
start_txt = c1.text_input("Start (YYYY-MM-DD HH:MM:SS)", value=min_t.strftime("%Y-%m-%d %H:%M:%S"))
end_txt   = c2.text_input("End (YYYY-MM-DD HH:MM:SS)",   value=max_t.strftime("%Y-%m-%d %H:%M:%S"))
st.caption(f"File time range: **{min_t}** → **{max_t}**")

typed_start = pd.to_datetime(start_txt, errors="coerce")
typed_end   = pd.to_datetime(end_txt,   errors="coerce")
if pd.isna(typed_start):
    st.error("Invalid start time. Use YYYY-MM-DD HH:MM:SS.")
    st.stop()
if pd.isna(typed_end):
    st.error("Invalid end time. Use YYYY-MM-DD HH:MM:SS.")
    st.stop()

showed_err = False
if typed_start < min_t or typed_start > max_t:
    st.error("Start time was outside the file's time range and has been aligned to the nearest bound.")
    showed_err = True
if typed_end < min_t or typed_end > max_t:
    st.error("End time was outside the file's time range and has been aligned to the nearest bound.")
    showed_err = True

# Clamp & align end to whole hours from start
start_dt = max(min_t, min(typed_start, max_t))
aligned_end = align_end_to_full_hours(start_dt, typed_end, min_t, max_t)
end_dt = aligned_end

mask = (work_all["time"] >= start_dt) & (work_all["time"] <= end_dt)
work_sel = work_all.loc[mask].copy()

# ---------------- 2σ clipping (per series, within selection) ----------------
coinc_cols = ["Top + Middle", "Top + Bottom", "Middle + Bottom"]
work_clip = work_sel.copy()
for c in coinc_cols + ["Total (per min)"]:
    work_clip[c] = sigma_clip_series(work_clip[c], sigma=2.0)

st.caption("2σ clipping applied to counts in the selected window")

# ---------------- Deviations (from CLIPPED data) ----------------
# Minute-level signed deviation vs overall mean (selected window)
minute_means = {c: work_clip[c].mean(skipna=True) for c in coinc_cols}
for c in coinc_cols:
    m = minute_means[c]
    work_clip[f"{c} Δ% (min vs overall)"] = ((work_clip[c] - m) / m * 100.0) if (pd.notna(m) and m != 0) else pd.NA

# Hourly mean (from clipped minute data)
hourly = (work_clip.set_index("time").resample("H").mean(numeric_only=True).dropna(how="all"))

# Hourly deviation: hour mean vs overall hourly mean
hourly_dev = pd.DataFrame(index=hourly.index)
for c in coinc_cols:
    overall = hourly[c].mean(skipna=True)
    hourly_dev[f"{c} Δ% (hour mean vs overall)"] = (
        (hourly[c] - overall) / overall * 100.0 if (pd.notna(overall) and overall != 0) else pd.NA
    )

# Downsample minute plots (after clipping)
n = len(work_clip)
stride = max(1, n // MAX_POINTS)
plot_min = work_clip.iloc[::stride].copy()

# ---------------- Tabs (stacked charts) ----------------
tabs = st.tabs([
    "counts per minute",
    "counts per hour",
    "percentage change per minute",
    "percentage change per hour",
])

# counts per minute
with tabs[0]:
    st.subheader("top and middle layers")
    st.line_chart(plot_min.set_index("time")[["Top + Middle"]])
    st.subheader("top and bottom layers")
    st.line_chart(plot_min.set_index("time")[["Top + Bottom"]])
    st.subheader("middle and bottom layers")
    st.line_chart(plot_min.set_index("time")[["Middle + Bottom"]])

# counts per hour
with tabs[1]:
    st.subheader("top and middle layers")
    st.line_chart(hourly[["Top + Middle"]])
    st.subheader("top and bottom layers")
    st.line_chart(hourly[["Top + Bottom"]])
    st.subheader("middle and bottom layers")
    st.line_chart(hourly[["Middle + Bottom"]])

# percentage change per minute
with tabs[2]:
    st.subheader("top and middle layers")
    st.line_chart(plot_min.set_index("time")[["Top + Middle Δ% (min vs overall)"]])
    st.subheader("top and bottom layers")
    st.line_chart(plot_min.set_index("time")[["Top + Bottom Δ% (min vs overall)"]])
    st.subheader("middle and bottom layers")
    st.line_chart(plot_min.set_index("time")[["Middle + Bottom Δ% (min vs overall)"]])

# percentage change per hour
with tabs[3]:
    st.subheader("top and middle layers")
    st.line_chart(hourly_dev[[c for c in hourly_dev.columns if c.startswith("Top + Middle")]])
    st.subheader("top and bottom layers")
    st.line_chart(hourly_dev[[c for c in hourly_dev.columns if c.startswith("Top + Bottom")]])
    st.subheader("middle and bottom layers")
    st.line_chart(hourly_dev[[c for c in hourly_dev.columns if c.startswith("Middle + Bottom")]])
