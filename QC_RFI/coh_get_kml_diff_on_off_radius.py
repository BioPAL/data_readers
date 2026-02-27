# -*- coding: utf-8 -*-
"""
KML for BIOMASS RFI coherence comparison
PERCENTUAL diff map with full popup stats

@author: GPALUMBO
"""

import pandas as pd
import json
import math
from shapely import wkb
import simplekml

# ============================================================
# INPUT
# ============================================================

CSV_FILE = r"E:\BIOMASS\02_DATA\14_RFI\COV_kml_and_CSV_PTC\stacks_rfi_comparison_pct_clean.csv"
OUT_KML  = r"E:\BIOMASS\02_DATA\14_RFI\COV_kml_and_CSV_PTC\stacks_rfi_comparison_pct_radial.kml"



# ============================================================
# GEOMETRY
# ============================================================

def parse_point(val):

    if pd.isna(val):
        return None, None

    if isinstance(val, float):
        return None, None

    geom = wkb.loads(bytes.fromhex(val))
    return geom.y, geom.x


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(CSV_FILE)

df[["lat","lon"]] = df["centerofpos"].apply(
    lambda x: pd.Series(parse_point(x))
)

GRID = 0.00005  

df["cluster_lat"] = (df["lat"] / GRID).round().astype(int)
df["cluster_lon"] = (df["lon"] / GRID).round().astype(int)

df["spatial_cluster"] = (
    df["cluster_lat"].astype(str) + "_" + df["cluster_lon"].astype(str)
)

# ============================================================
# SAFE JSON
# ============================================================

def load_json(val):
    if isinstance(val, dict):
        return val
    if pd.isna(val):
        return {"HH":0,"VV":0,"XP":0}
    if isinstance(val, (float, int)):
        return {"HH":float(val),"VV":float(val),"XP":float(val)}
    return json.loads(val)

# ============================================================
# COLOR SCALE — PERCENTUAL
# ============================================================

def diff_color_pct(delta):

    delta = float(delta)

    if delta >10:
        return "ff00ff00"
    elif 5 < delta <= 10:
        return "ffff0000"
    elif 1 < delta <= 5:
        return "ffffcc00"
    elif -1 <= delta <= 1:
        return "ff00ffff"
    elif -5 <= delta < -1:
        return "ff00a5ff"
    elif -10 <= delta < -5:
        return "ff0000ff"
    else:
        return "ff000000"

# ============================================================
# HTML TABLES (unchanged)
# ============================================================

def make_stats_table(title, mean, std, minv, maxv, median):

    def row(name, d):
        return f"""
        <tr>
          <td><b>{name}</b></td>
          <td>{d["HH"]:.6f}</td>
          <td>{d["VV"]:.6f}</td>
          <td>{d["XP"]:.6f}</td>
        </tr>
        """

    return f"""
    <h3>{title}</h3>
    <table border="1" cellpadding="4" cellspacing="0"
           style="background:#e8f5e9;border-collapse:collapse;">
      <tr>
        <th>Stat</th><th>HH</th><th>VV</th><th>XP</th>
      </tr>
      {row("Mean", mean)}
      {row("Std", std)}
      {row("Min", minv)}
      {row("Max", maxv)}
      {row("Median", median)}
    </table>
    """

def stats_table(title, stats):

    rows = ""
    for k in ["HH","VV","XP"]:
        rows += f"<tr><td>{k}</td><td>{stats[k]:.6f}</td></tr>"

    return f"""
    <h4>{title}</h4>
    <table border="1" cellpadding="3">
    <tr><th>Pol</th><th>Value</th></tr>
    {rows}
    </table>
    """

# ============================================================
# KML
# ============================================================

kml = simplekml.Kml()

# ============================================================
# METRICS
# ============================================================

df["vv_abs"] = df["coh_mean_diff"].apply(lambda x: load_json(x)["VV"])
df["vv_pct"] = df["coh_mean_diff_pct"].apply(lambda x: load_json(x)["VV"])

df["severity"] = df["vv_abs"]
df = df.sort_values("severity")

# ============================================================
# RADIAL OFFSET
# ============================================================

def radial_offset_m(base_lat, base_lon, i, n, radius_m=80.0):

    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(base_lat))

    angle = 2.0 * math.pi * i / n

    dlat = (radius_m * math.cos(angle)) / m_per_deg_lat
    dlon = (radius_m * math.sin(angle)) / m_per_deg_lon

    return base_lat + dlat, base_lon + dlon


# ============================================================
# LOOP
# ============================================================

for _, group in df.groupby("spatial_cluster"):

    base_lat = group["lat"].mean()
    base_lon = group["lon"].mean()
    n = len(group)

    for i, (_, row) in enumerate(group.iterrows()):

        RADIUS_M = 15000.0 + 2000.0 * (n - 1)
        lat, lon = radial_offset_m(base_lat, base_lon, i, n, RADIUS_M)

        mean_off = load_json(row["coh_mean_off"])
        mean_on  = load_json(row["coh_mean_on"])

        std_off  = load_json(row["coh_std_off"])
        std_on   = load_json(row["coh_std_on"])

        min_off  = load_json(row["coh_min_off"])
        min_on   = load_json(row["coh_min_on"])

        max_off  = load_json(row["coh_max_off"])
        max_on   = load_json(row["coh_max_on"])

        med_off  = load_json(row["coh_median_off"])
        med_on   = load_json(row["coh_median_on"])

        mean_diff = load_json(row["coh_mean_diff"])
        std_diff  = load_json(row["coh_std_diff"])
        min_diff  = load_json(row["coh_min_diff"])
        max_diff  = load_json(row["coh_max_diff"])
        med_diff  = load_json(row["coh_median_diff"])
        
        mean_diff_pct = load_json(row["coh_mean_diff_pct"])
        std_diff_pct  = load_json(row["coh_std_diff_pct"])
        min_diff_pct  = load_json(row["coh_min_diff_pct"])
        max_diff_pct  = load_json(row["coh_max_diff_pct"])
        med_diff_pct  = load_json(row["coh_median_diff_pct"])

        pct_diff  = load_json(row["coh_mean_diff_pct"])

        vv_abs = mean_diff["VV"]
        vv_pct = pct_diff["VV"]

        color = diff_color_pct(vv_pct)

        stackID = row["stackid"][:-10]

        html = f"""
        <b>stackID</b><br>{stackID}<br><br>

        <b>stackstandard_off</b><br>{row['stackstandard_off']}<br><br>

        <b>stackstandard_on</b><br>{row['stackstandard_on']}<br><br>

        <h3>VV diff (RFI_ON − RFI_OFF) = {vv_abs:.6f}</h3>
        <h4>VV diff % = {vv_pct:.2f} %  &nbsp; ((RFI_ON − RFI_OFF) / RFI_OFF × 100)</h4>

        {make_stats_table("RFI OFF",
            mean_off, std_off, min_off, max_off, med_off)}

        <br>

        {make_stats_table("RFI ON",
            mean_on, std_on, min_on, max_on, med_on)}

        {make_stats_table("DIFF (RFI_ON − RFI_OFF)",
            mean_diff, std_diff, min_diff, max_diff, med_diff)}

        <br>
        
        {make_stats_table(
            "DIFF % ((RFI_ON − RFI_OFF) / RFI_OFF × 100)",
            mean_diff_pct,
            std_diff_pct,
            min_diff_pct,
            max_diff_pct,
            med_diff_pct
        )}
        
        <br>

        <b>Color scale (PERCENTUAL diff - based on VV mean):</b>
        <ul>
        <li style="color:green;"> > 10 %</li>
        <li style="color:blue;"> 5 – 10 %</li>
        <li style="color:cyan;"> 1 – 5 %</li>
        <li style="color:gold;"> -1 – 1 %</li>
        <li style="color:orange;"> -5 – -1 %</li>
        <li style="color:red;"> -10 – -5 %</li>
        <li style="color:black;"> < -10 %</li>
        </ul>
        """

        p = kml.newpoint(
            name=f"{vv_pct:.2f}%",
            coords=[(lon, lat)]
        )

        p.description = html
        p.style.iconstyle.icon.href = \
            "http://maps.google.com/mapfiles/kml/paddle/wht-blank.png"
        p.style.iconstyle.color = color
        p.style.iconstyle.scale = 1.0
        p.style.labelstyle.scale = 0.8

# ============================================================
# SAVE
# ============================================================

kml.save(OUT_KML)

print("KML generated:", OUT_KML)


