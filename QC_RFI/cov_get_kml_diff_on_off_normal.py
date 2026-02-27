# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 08:58:53 2026

@author: GPALUMBO
"""

# -*- coding: utf-8 -*-

import pandas as pd
import json
import simplekml
from shapely import wkb

# ============================================================
# INPUT
# ============================================================

CSV_FILE = r"E:\BIOMASS\02_DATA\14_RFI\COV_Kmls_and_csv\COV_RFI_COMPARISON_ON_OFF.csv"
OUT_DIR  = r"E:\BIOMASS\02_DATA\14_RFI\COV_Kmls_and_csv"

df = pd.read_csv(CSV_FILE)

channels = ["rllr","hh-hv","hh-vv","hv-vh","vv-vh"]


# ============================================================
# HELPERS
# ============================================================

def parse_point(wkb_hex):
    g = wkb.loads(bytes.fromhex(wkb_hex))
    return g.y, g.x

def load_json(v):
    if isinstance(v, dict): return v
    if pd.isna(v): return {}
    return json.loads(v)

# ============================================================
# COLOR SCALE
# ============================================================

def diff_color(d):
    d = float(d)

    if d > 0.10: return "ff00ff00"
    elif 0.05 < d <= 0.10: return "ffff0000"
    elif 0.01 < d <= 0.05: return "ffffcc00"
    elif -0.01 <= d <= 0.01: return "ff00ffff"
    elif -0.05 <= d < -0.01: return "ff00a5ff"
    elif -0.10 <= d < -0.05: return "ff0000ff"
    else: return "ff000000"

# ============================================================
# POPUP TABLE
# ============================================================

def stats_table(title, d):
    rows = ""
    for k in channels:
        v = d.get(k, float("nan"))
        rows += f"<tr><td>{k}</td><td>{v:.6f}</td></tr>"

    return f"""
    <h3>{title}</h3>
    <table border="1" cellpadding="4">
    <tr><th>Channel</th><th>Value</th></tr>
    {rows}
    </table>
    """

# ============================================================
# ORDER: worst degradation first (ABS + PHASE)
# ============================================================

df["severity"] = df.apply(
    lambda r: min(
        min(load_json(r["abs_mean_diff"]).values()),
        min(load_json(r["phase_mean_diff"]).values())
    ),
    axis=1
)

df = df.sort_values("severity")

# ============================================================
# CREATE 10 KML FILES
# ============================================================

kml_abs   = {ch: simplekml.Kml() for ch in channels}
kml_phase = {ch: simplekml.Kml() for ch in channels}

def add_ball(kml, lon, lat, delta, popup):

    p = kml.newpoint(
        name=f"{delta:+.4f}",   # ← numero sopra il pin
        coords=[(lon, lat)]
    )

    p.description = popup

    p.style.iconstyle.icon.href = \
        "http://maps.google.com/mapfiles/kml/paddle/wht-blank.png"

    p.style.iconstyle.color = diff_color(delta)
    p.style.iconstyle.scale = 0.8

    # label visible but small
    p.style.labelstyle.scale = 0.8


stats = ["mean","median","std","min","max"]

def full_stats_table(title, mean, median, std, vmin, vmax):

    rows = ""
    for ch in channels:
        rows += f"""
        <tr>
        <td>{ch}</td>
        <td>{mean[ch]:.6f}</td>
        <td>{median[ch]:.6f}</td>
        <td>{std[ch]:.6f}</td>
        <td>{vmin[ch]:.6f}</td>
        <td>{vmax[ch]:.6f}</td>
        </tr>
        """

    return f"""
    <h4 style="margin:4px 0;">{title}</h4>
    <table border="1" cellpadding="2" style="font-size:10px;border-collapse:collapse;">
    <tr>
    <th>Channel</th>
    <th>Mean</th>
    <th>Median</th>
    <th>Std</th>
    <th>Min</th>
    <th>Max</th>
    </tr>
    {rows}
    </table>
    """


# ============================================================
# LOOP
# ============================================================

for _, row in df.iterrows():

    lat, lon = parse_point(row["centerofpos"])

    abs_mean_off   = load_json(row["abs_mean_off"])
    abs_median_off = load_json(row["abs_median_off"])
    abs_std_off    = load_json(row["abs_std_off"])
    abs_min_off    = load_json(row["abs_min_off"])
    abs_max_off    = load_json(row["abs_max_off"])

    abs_mean_on   = load_json(row["abs_mean_on"])
    abs_median_on = load_json(row["abs_median_on"])
    abs_std_on    = load_json(row["abs_std_on"])
    abs_min_on    = load_json(row["abs_min_on"])
    abs_max_on    = load_json(row["abs_max_on"])

    abs_mean_diff   = load_json(row["abs_mean_diff"])
    abs_median_diff = load_json(row["abs_median_diff"])
    abs_std_diff    = load_json(row["abs_std_diff"])
    abs_min_diff    = load_json(row["abs_min_diff"])
    abs_max_diff    = load_json(row["abs_max_diff"])

    phase_mean_off   = load_json(row["phase_mean_off"])
    phase_median_off = load_json(row["phase_median_off"])
    phase_std_off    = load_json(row["phase_std_off"])
    phase_min_off    = load_json(row["phase_min_off"])
    phase_max_off    = load_json(row["phase_max_off"])

    phase_mean_on   = load_json(row["phase_mean_on"])
    phase_median_on = load_json(row["phase_median_on"])
    phase_std_on    = load_json(row["phase_std_on"])
    phase_min_on    = load_json(row["phase_min_on"])
    phase_max_on    = load_json(row["phase_max_on"])

    phase_mean_diff   = load_json(row["phase_mean_diff"])
    phase_median_diff = load_json(row["phase_median_diff"])
    phase_std_diff    = load_json(row["phase_std_diff"])
    phase_min_diff    = load_json(row["phase_min_diff"])
    phase_max_diff    = load_json(row["phase_max_diff"])

    for ch in channels:

        # ==========================
        # ABS POPUP
        # ==========================
        popup_abs = f"""
        <div style="font-size:11px;">

        <h2 style="color:green;">ABS CHANNEL: {ch}</h2>

        <b>scs_product_off:</b><br>{row["scs_product_off"]}<br><br>
        <b>scs_product_on:</b><br>{row["scs_product_on"]}<br><br>

        <table border="1" cellpadding="3" style="font-size:10px;">
        <tr><th></th><th>Mean</th><th>Median</th><th>Std</th><th>Min</th><th>Max</th></tr>

        <tr>
        <td><b>OFF</b></td>
        <td>{abs_mean_off[ch]:.6f}</td>
        <td>{abs_median_off[ch]:.6f}</td>
        <td>{abs_std_off[ch]:.6f}</td>
        <td>{abs_min_off[ch]:.6f}</td>
        <td>{abs_max_off[ch]:.6f}</td>
        </tr>

        <tr>
        <td><b>ON</b></td>
        <td>{abs_mean_on[ch]:.6f}</td>
        <td>{abs_median_on[ch]:.6f}</td>
        <td>{abs_std_on[ch]:.6f}</td>
        <td>{abs_min_on[ch]:.6f}</td>
        <td>{abs_max_on[ch]:.6f}</td>
        </tr>

        <tr>
        <td><b>DIFF</b></td>
        <td>{abs_mean_diff[ch]:.6f}</td>
        <td>{abs_median_diff[ch]:.6f}</td>
        <td>{abs_std_diff[ch]:.6f}</td>
        <td>{abs_min_diff[ch]:.6f}</td>
        <td>{abs_max_diff[ch]:.6f}</td>
        </tr>

        </table>

        </div>
        """

        add_ball(
            kml_abs[ch],
            lon,
            lat,
            abs_mean_diff[ch],
            popup_abs
        )

        # ==========================
        # PHASE POPUP
        # ==========================
        popup_phase = f"""
        <div style="font-size:11px;">

        <h2 style="color:green;">PHASE CHANNEL: {ch}</h2>

        <b>scs_product_off:</b><br>{row["scs_product_off"]}<br><br>
        <b>scs_product_on:</b><br>{row["scs_product_on"]}<br><br>

        <table border="1" cellpadding="3" style="font-size:10px;">
        <tr><th></th><th>Mean</th><th>Median</th><th>Std</th><th>Min</th><th>Max</th></tr>

        <tr>
        <td><b>OFF</b></td>
        <td>{phase_mean_off[ch]:.6f}</td>
        <td>{phase_median_off[ch]:.6f}</td>
        <td>{phase_std_off[ch]:.6f}</td>
        <td>{phase_min_off[ch]:.6f}</td>
        <td>{phase_max_off[ch]:.6f}</td>
        </tr>

        <tr>
        <td><b>ON</b></td>
        <td>{phase_mean_on[ch]:.6f}</td>
        <td>{phase_median_on[ch]:.6f}</td>
        <td>{phase_std_on[ch]:.6f}</td>
        <td>{phase_min_on[ch]:.6f}</td>
        <td>{phase_max_on[ch]:.6f}</td>
        </tr>

        <tr>
        <td><b>DIFF</b></td>
        <td>{phase_mean_diff[ch]:.6f}</td>
        <td>{phase_median_diff[ch]:.6f}</td>
        <td>{phase_std_diff[ch]:.6f}</td>
        <td>{phase_min_diff[ch]:.6f}</td>
        <td>{phase_max_diff[ch]:.6f}</td>
        </tr>

        </table>
        
        
        <br>
        <b>Color scale (signed diff):</b>
        <ul>
        <li style="color:green;"> > 0.10</li>
        <li style="color:blue;"> 0.05 – 0.10</li>
        <li style="color:cyan;"> 0.01 – 0.05</li>
        <li style="color:gold;"> -0.01 – 0.01</li>
        <li style="color:orange;"> -0.05 – -0.01</li>
        <li style="color:red;"> -0.10 – -0.05</li>
        <li style="color:black;"> < -0.10</li>
        </ul>

        </div>
        """

        add_ball(
            kml_phase[ch],
            lon,
            lat,
            phase_mean_diff[ch],
            popup_phase
        )

# ============================================================
# SAVE
# ============================================================

for ch in channels:
    kml_abs[ch].save(f"{OUT_DIR}\\COV_ABS_{ch}.kml")
    kml_phase[ch].save(f"{OUT_DIR}\\COV_PHASE_{ch}.kml")

print("Generated 10 KML files successfully")