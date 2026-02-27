# -*- coding: utf-8 -*-
"""
CSV -> KML (pins) for BIOMASS SCS RFI summary

- Reads footprint as WKT POLYGON
- Computes centroid -> lat/lon
- Clusters nearby products (DBSCAN haversine)
- Applies radial layout inside each cluster
- Styles pins by rfiFMFraction (from qualityParameters_HH)
- Popup shows ALL fields from the CSV row (pretty tables)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import simplekml
from shapely import wkt
from sklearn.cluster import DBSCAN

# ============================================================
# INPUT
# ============================================================

CSV_FILE = r"E:\BIOMASS\02_DATA\14_RFI\RFI_INFORMATION_FROM_ANNOTATION\rfi_RFI_OFF.csv"
OUT_KML  = r"E:\BIOMASS\02_DATA\14_RFI\RFI_INFORMATION_FROM_ANNOTATION\rfi_RFI_OFF.kml"



# Spatial clustering + radial layout tuning
EARTH_RADIUS_KM = 6371.0
CLUSTER_KM      = 25.0     # <-- products within ~25 km are grouped
BASE_RADIUS_M   = 11000.0  # <-- radius for cluster of size 1-2
STEP_RADIUS_M   = 1500.0   # <-- +2 km per extra point


# ============================================================
# HELPERS
# ============================================================

def load_json(v) -> dict:
    if isinstance(v, dict):
        return v
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return {}
    s = str(v).strip()
    if not s:
        return {}
    return json.loads(s)

def centroid_from_wkt_polygon(poly_wkt: str) -> tuple[float, float]:
    """Return (lat, lon) centroid from WKT POLYGON."""
    g = wkt.loads(poly_wkt)
    c = g.centroid
    return float(c.y), float(c.x)

def radial_offset_m(base_lat: float, base_lon: float, i: int, n: int, radius_m: float) -> tuple[float, float]:
    """Distribute n points around a circle of radius_m centered at base lat/lon."""
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(base_lat))

    angle = 2.0 * math.pi * (i / max(n, 1))
    dlat = (radius_m * math.cos(angle)) / m_per_deg_lat
    dlon = (radius_m * math.sin(angle)) / m_per_deg_lon

    return base_lat + dlat, base_lon + dlon

def kml_color_from_rfi_fm_fraction(v: float) -> str:
    """
    KML color is AABBGGRR.

    Requested scale:
      green   [0.00, 0.05)
      yellow  [0.05, 0.10)
      orange  [0.10, 0.15)
      red     [0.15, 0.20)
      black   >=0.20
    """
    v = float(v)

    # opaque colors (AA=ff)
    if 0.0 <= v < 0.05:
        return "ff00ff00"  # green
    elif 0.05 <= v < 0.10:
        return "ff00ffff"  # yellow (BGR for yellow is 00ffff)
    elif 0.10 <= v < 0.15:
        return "ff00a5ff"  # orange-ish
    elif 0.15 <= v < 0.20:
        return "ff0000ff"  # red
    else:
        return "ff000000"  # black

def html_escape(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))

def dict_to_table(title: str, d: dict) -> str:
    if not d:
        return f"<h4 style='margin:6px 0 2px 0;'>{html_escape(title)}</h4><i>empty</i>"

    rows = ""
    for k, v in d.items():
        rows += (
            "<tr>"
            f"<td style='padding:2px 6px; white-space:nowrap;'><b>{html_escape(str(k))}</b></td>"
            f"<td style='padding:2px 6px;'>{html_escape(str(v))}</td>"
            "</tr>"
        )

    return f"""
    <h4 style="margin:6px 0 2px 0;">{html_escape(title)}</h4>
    <table border="1" cellpadding="2" style="border-collapse:collapse; font-size:11px;">
      {rows}
    </table>
    """

def build_grouped_popup(row: pd.Series) -> str:

    pols = ["HH", "HV", "VH", "VV"]

    rfi_iso = {p: load_json(row[f"rfiIsolated_{p}"]) for p in pols}
    rfi_per = {p: load_json(row[f"rfiPersistent_{p}"]) for p in pols}
    qpar    = {p: load_json(row[f"qualityParameters_{p}"]) for p in pols}

    iso_keys = ["percentageAffectedLines", "maxPercentageAffectedBW", "avgPercentageAffectedBW"]
    per_keys = ["percentageAffectedLines", "maxPercentageAffectedBW", "avgPercentageAffectedBW"]
    q_keys   = ["rfiTMFraction", "maxRFITMPercentage", "rfiFMFraction"]

    TABLE_STYLE = """
    style="
      border-collapse:collapse;
      font-size:11px;
      background:#e9f6e9;
    "
    """

    TH_STYLE = "style='background:#cfeccc;padding:4px;'"
    TD_STYLE = "style='padding:3px;'"

    def table_metrics(title, keys, data):

        header = "".join(f"<th {TH_STYLE}>{p}</th>" for p in pols)

        rows = ""
        for k in keys:
            vals = "".join(
                f"<td {TD_STYLE}>{data[p].get(k, float('nan')):.6f}</td>"
                for p in pols
            )
            rows += f"<tr><td {TD_STYLE}><b>{k}</b></td>{vals}</tr>"

        return f"""
        <h4>{title}</h4>
        <table border="1" {TABLE_STYLE}>
        <tr><th {TH_STYLE}>Metric</th>{header}</tr>
        {rows}
        </table>
        """

    def table_quality(title, data):
    
        q_keys = ["rfiTMFraction", "maxRFITMPercentage", "rfiFMFraction"]
        pols = ["HH", "HV", "VH", "VV"]
    
        header = "".join(f"<th {TH_STYLE}>{p}</th>" for p in pols)
    
        rows = ""
        for k in q_keys:
            vals = "".join(
                f"<td {TD_STYLE}>{data[p].get(k, float('nan')):.6f}</td>"
                for p in pols
            )
            rows += f"<tr><td {TD_STYLE}><b>{k}</b></td>{vals}</tr>"
    
        return f"""
        <h4>{title}</h4>
        <table border="1" {TABLE_STYLE}>
        <tr><th {TH_STYLE}>Metric</th>{header}</tr>
        {rows}
        </table>
        """

    legend = """
    <h4>RFI FM Fraction – color scale</h4>
    <table border="1" cellpadding="3" style="border-collapse:collapse;font-size:11px;">
      <tr><td style="background:#00ff00;">0 – 0.05</td><td>Low</td></tr>
      <tr><td style="background:#ffff00;">0.05 – 0.10</td><td>Moderate</td></tr>
      <tr><td style="background:#ffa500;">0.10 – 0.15</td><td>High</td></tr>
      <tr><td style="background:#ff0000;color:white;">0.15 – 0.20</td><td>Very high</td></tr>
      <tr><td style="background:#000000;color:white;">> 0.20</td><td>Extreme</td></tr>
    </table>
    """

    popup = f"""
    <div style="font-family:Arial;font-size:11px;">
    <h3>{row["SCS_filename"]}</h3>

    {table_metrics("rfiIsolated", iso_keys, rfi_iso)}
    {table_metrics("rfiPersistent", per_keys, rfi_per)}
    {table_quality("qualityParameters", qpar)}

    {legend}

    </div>
    """

    return popup


# ============================================================
# MAIN
# ============================================================

df = pd.read_csv(CSV_FILE, sep=None, engine="python")  # auto-detect separator

required = {"SCS_filename", "footprint", "qualityParameters_HH"}
missing = sorted(required - set(df.columns))
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")

# Centroids from footprint polygon
df[["lat", "lon"]] = df["footprint"].apply(lambda s: pd.Series(centroid_from_wkt_polygon(s)))

# rfiFMFraction from HH only (as requested)
def extract_rfi_fm_fraction(row: pd.Series) -> float:
    q = load_json(row["qualityParameters_HH"])
    # expected key: "rfiFMFraction"
    return float(q.get("rfiFMFraction", float("nan")))

df["rfiFMFraction_HH"] = df.apply(extract_rfi_fm_fraction, axis=1)

# Cluster nearby points using haversine
coords_rad = np.radians(df[["lat", "lon"]].values)
eps = CLUSTER_KM / EARTH_RADIUS_KM

db = DBSCAN(
    eps=eps,
    min_samples=1,
    algorithm="ball_tree",
    metric="haversine"
)
df["spatial_cluster"] = db.fit_predict(coords_rad)

# KML
kml = simplekml.Kml()

def add_pin(kml_obj: simplekml.Kml, lon: float, lat: float, value: float, popup_html: str):
    p = kml_obj.newpoint(
        name=f"{value:.6f}",        # label = HH rfiFMFraction
        coords=[(lon, lat)]
    )
    p.description = popup_html

    p.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/paddle/wht-blank.png"
    p.style.iconstyle.color = kml_color_from_rfi_fm_fraction(value)
    p.style.iconstyle.scale = 0.8
    p.style.labelstyle.scale = 0.8

# Radial layout inside each cluster
for _, group in df.groupby("spatial_cluster", sort=False):

    base_lat = float(group["lat"].mean())
    base_lon = float(group["lon"].mean())
    n = int(len(group))

    radius_m = BASE_RADIUS_M + STEP_RADIUS_M * max(n - 1, 0)

    for i, (_, row) in enumerate(group.iterrows()):
        if n == 1:
            lat, lon = base_lat, base_lon
        else:
            lat, lon = radial_offset_m(base_lat, base_lon, i, n, radius_m=radius_m)

        value = float(row["rfiFMFraction_HH"])
        popup = build_grouped_popup(row)

        add_pin(kml, lon, lat, value, popup)

# Save
Path(OUT_KML).parent.mkdir(parents=True, exist_ok=True)
kml.save(OUT_KML)

print(f"OK - generated: {OUT_KML}")