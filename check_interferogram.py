# -*- coding: utf-8 -*-
"""
Copyright 2025, European Space Agency (ESA)
Licensed under ESA Software Community Licence Permissive (Type 3) - v2.4
"""
import BiomassProduct
import xml.etree.ElementTree as ET  #needed to fetch  the data from annotations
import rasterio
import netCDF4
import numpy as np
import numpy.typing as npt
import scipy
from PIL import Image
from pathlib import Path
import os
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import glob
import pprint
from reportlab.lib.styles import ParagraphStyle
import zipfile
from reportlab.platypus import Image as RLImage, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,Image as RLImage, Table, TableStyle)
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors
from PIL import Image as PILImage
from osgeo import gdal
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import shutil
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.signal import convolve2d


# --- COHERENCE COLOR THRESHOLD CONFIGURATION ---
COH_LOW_THRESHOLD  = 0.35   
COH_HIGH_THRESHOLD = 0.80




# --- CRITICAL BASELINE (meters) ---
BC_BY_MISSION_PHASE = {
    "TOM": 5.7887e3,
    "INT": 4.9725e3,
    "COM": 5.7887e3,   # COMMISSIONING 
}



def generate_interferogram_pdf_report(
    output_pdf: Path,
    primary_name: str,
    secondary_name: str,
    coh_stats_global: dict,
    pol_plots: dict,
    sta_plots: dict,
    flatten: str,
    skpPhaseCalibrationFlag: bool,
    skpPhaseCorrectionFlag: bool,
    skpPhaseCorrectionFlatteningOnlyFlag: bool,):
    """
    PDF report:
      - Title with primary / secondary product names
      - ONE table mirroring the KML popup content (VV reference)
      - Coherence plots (VV)
      - Interferograms per polarization
      - STA phase screens
      - Phase correction metadata
    """

    output_pdf = Path(output_pdf)

    # ------------------------------------------------------------------
    # STYLES
    # ------------------------------------------------------------------
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="TitleStyle",
        fontSize=16,
        leading=20,
        spaceAfter=10,
        alignment=TA_LEFT
    ))

    styles.add(ParagraphStyle(
        name="SectionStyle",
        fontSize=13,
        leading=16,
        spaceBefore=14,
        spaceAfter=8,
        textColor=colors.darkgreen
    ))

    styles.add(ParagraphStyle(
        name="NormalSmall",
        fontSize=9,
        leading=11
    ))

    param_style = ParagraphStyle(
        name="ParamStyle",
        fontSize=9,
        leading=11,
        wordWrap="CJK"
    )

    styles.add(ParagraphStyle(
        name="CourierSmall",
        fontName="Courier",
        fontSize=8.5,
        leading=10,
        spaceBefore=4,
        spaceAfter=2
    ))

    # ------------------------------------------------------------------
    # DOCUMENT
    # ------------------------------------------------------------------
    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm
    )

    elements = []

    # ------------------------------------------------------------------
    # TITLE
    # ------------------------------------------------------------------
    elements.append(Paragraph(
        "Interferogram Coherence Analysis",
        styles["TitleStyle"]
    ))

    elements.append(Paragraph(
        f"<b>Primary:</b> {primary_name}<br/>"
        f"<b>Secondary:</b> {secondary_name}",
        styles["NormalSmall"]
    ))

    # ------------------------------------------------------------------
    # PARAMETERS TABLE (VV reference)
    # ------------------------------------------------------------------
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Parameters (VV reference)", styles["SectionStyle"]))

    table_data = [[
        Paragraph("<b>Parameter</b>", styles["NormalSmall"]),
        Paragraph("<b>Value</b>", styles["NormalSmall"])
    ]]

    for k, v in coh_stats_global.items():
    
        # Caso 1: statistiche per polarizzazione (dict)
        if isinstance(v, dict):
            parts = []
            for pol, val in v.items():
                try:
                    parts.append(f"{pol} {float(val):.4f}")
                except Exception:
                    parts.append(f"{pol} {val}")
            v_fmt = " ; ".join(parts)
    
        # Caso 2: valore scalare
        else:
            v_fmt = format_value(v, ndigits=4)
    
        table_data.append([
            Paragraph(str(k), param_style),
            Paragraph(v_fmt, styles["NormalSmall"])
        ])

    table = Table(
        table_data,
        colWidths=[7.5 * cm, 6.5 * cm],
        repeatRows=1
    )

    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))

    elements.append(table)

    # ------------------------------------------------------------------
    # IMAGE LOADER
    # ------------------------------------------------------------------
    def load_img(path: Path, max_width_cm: float):
        if not path or not Path(path).exists():
            return None

        with PILImage.open(path) as im:
            w, h = im.size

        aspect = h / float(w)
        width = max_width_cm * cm
        height = width * aspect

        return RLImage(str(path), width=width, height=height)

    # ------------------------------------------------------------------
    # COHERENCE PLOTS (ALL POLARIZATIONS)
    # ------------------------------------------------------------------
    elements.append(Spacer(1, 16))
    elements.append(Paragraph("Coherence", styles["SectionStyle"]))
    
    for pol, pol_data in pol_plots.items():
    
        elements.append(Spacer(1, 10))
        elements.append(Paragraph(f"Polarization: {pol}", styles["NormalSmall"]))
    
        plots = pol_data.get("plots", {})
    
        for label, key in [
            ("Coherence amplitude", "coh_amp"),
            ("Coherence phase", "coh_phase"),
        ]:
            img_path = plots.get(key)
            img = load_img(img_path, 16)
            if img:
                elements.append(Spacer(1, 4))
                elements.append(Paragraph(label, styles["NormalSmall"]))
                elements.append(img)

    # ------------------------------------------------------------------
    # INTERFEROGRAMS PER POLARIZATION
    # ------------------------------------------------------------------
    elements.append(Spacer(1, 18))
    elements.append(Paragraph("Interferograms", styles["SectionStyle"]))

    for pol, pol_dict in pol_plots.items():
        plots = pol_dict["plots"]

        elements.append(Spacer(1, 10))
        elements.append(Paragraph(f"Polarization: {pol}", styles["NormalSmall"]))

        for label, key in [
            ("Interferogram phase", "interferogram_phase"),
            ("Flattened interferogram phase", "interferogram_phase_flat"),
        ]:
            img = load_img(plots.get(key), 16)
            if img:
                elements.append(Spacer(1, 4))
                elements.append(Paragraph(label, styles["NormalSmall"]))
                elements.append(img)

    # ------------------------------------------------------------------
    # STA PHASE SCREENS
    # ------------------------------------------------------------------
    elements.append(Spacer(1, 18))
    elements.append(Paragraph("STA Phase Screens", styles["SectionStyle"]))

    for label, path in sta_plots.items():
        img = load_img(path, 16)
        if img:
            elements.append(Spacer(1, 6))
            elements.append(Paragraph(label, styles["NormalSmall"]))
            elements.append(img)

    # ------------------------------------------------------------------
    # PHASE CORRECTION METADATA
    # ------------------------------------------------------------------
    elements.append(Spacer(1, 16))
    elements.append(Paragraph("Phase correction configuration", styles["SectionStyle"]))

    elements.append(Paragraph(
        f"""
        <font face="Courier">
        flatten = {flatten}<br/>
        skpPhaseCalibrationFlag              = {skpPhaseCalibrationFlag}<br/>
        skpPhaseCorrectionFlag               = {skpPhaseCorrectionFlag}<br/>
        skpPhaseCorrectionFlatteningOnlyFlag = {skpPhaseCorrectionFlatteningOnlyFlag}
        </font>
        """,
        styles["CourierSmall"]
    ))

    # ------------------------------------------------------------------
    # BUILD (ONE TIME ONLY)
    # ------------------------------------------------------------------
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc.build(elements)

    print(f"[OK] PDF report generated: {output_pdf}")




def save_clean_image(data, cmap='RdBu', out_path=None, vmin=None, vmax=None):

    
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    plt.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)

def format_value(v, ndigits=4):
    # None / vuoti
    if v is None:
        return ""

    # numpy scalari -> python scalari
    if isinstance(v, (np.generic,)):
        v = v.item()

    # bool
    if isinstance(v, bool):
        return "true" if v else "false"

    # int
    if isinstance(v, int):
        return str(v)

    # float
    if isinstance(v, float):
        if np.isfinite(v) and float(v).is_integer():
            return str(int(v))
        return f"{v:.{ndigits}f}"

    # stringhe: prova a capire se sono numeri
    if isinstance(v, str):
        s = v.strip()
        try:
            fv = float(s)
            if np.isfinite(fv) and fv.is_integer():
                return str(int(fv))
            return f"{fv:.{ndigits}f}"
        except Exception:
            return s

    # fallback
    return str(v)

def save_colorbar_png(out_path,
                      cmap='viridis',
                      vmin=0.0, vmax=1.0,
                      ticks=(0.0, 0.5, 1.0),
                      label='Coherence amplitude'):

    fig, ax = plt.subplots(figsize=(1.0, 4.0))
    norm = Normalize(vmin=vmin, vmax=vmax)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap)),
                      cax=ax, orientation='vertical')
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:.1f}" for t in ticks])
    cb.set_label(label)
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def make_overlay_kmz_with_quad(
    kmz_out,
    png_path,                      # coherence abs (reference, e.g. VV)
    preview_kml_file,
    pol_plots,                     # dict {pol: {"plots": {...}, "stats": {...}}}
    coh_stats=None,
    coh_low_thr=0.35,
    coh_high_thr=0.80,
    flatteningPhaseScreen_co_file=None,
    skpCalibrationPhaseScreen_co_file=None,
    primary_name=None,
    secondary_name=None,):
    """
    KMZ with:
      - GroundOverlay (reference coherence)
      - Pin colored by mean coherence
      - Popup with:
          * global info table
          * tabs per polarization (coherence + interferogram)
          * common STA phase screens
    """


    ns = {
        "kml": "http://www.opengis.net/kml/2.2",
        "gx": "http://www.google.com/kml/ext/2.2",
    }

    png_path = Path(png_path)
    preview_kml_file = Path(preview_kml_file)

    # ------------------------------------------------------------------
    # 1) Read LatLonQuad from preview KML
    # ------------------------------------------------------------------
    root = ET.parse(preview_kml_file).getroot()
    quad = root.find(".//gx:LatLonQuad", ns)
    if quad is None:
        raise ValueError(f"gx:LatLonQuad not found in {preview_kml_file}")

    coords_text = quad.find("kml:coordinates", ns) or quad.find("coordinates")
    coords_list = coords_text.text.strip().split()

    # ------------------------------------------------------------------
    # 2) Geometric center
    # ------------------------------------------------------------------
    def parse_coord(c):
        lon, lat, *_ = c.split(",")
        return float(lon), float(lat)

    lon_list, lat_list = zip(*[parse_coord(c) for c in coords_list])
    center_coord = f"{sum(lon_list)/len(lon_list)},{sum(lat_list)/len(lat_list)},0"

    # ------------------------------------------------------------------
    # 3) Pin color from mean coherence
    # ------------------------------------------------------------------
    mean_coh = None
    if coh_stats and "Mean coh abs" in coh_stats:
        mean_dict = coh_stats["Mean coh abs"]
        if isinstance(mean_dict, dict) and "VV" in mean_dict:
            mean_coh = mean_dict["VV"]
    
    label_text = ""

    try:
        mean_dict = coh_stats.get("Mean coh abs", {})
        mean_coh_vv = mean_dict.get("VV")

        baseline_pct = coh_stats.get("baselinePercentage_secondary")

        if mean_coh_vv is not None and baseline_pct is not None:
            label_text = (
                f"VV |coh|={mean_coh_vv:.2f}\n"
                f"Bc={baseline_pct:.1f}%"
            )
    except Exception:
        label_text = ""
    
    
    
    def coh_to_kml_argb(v, low, high):
        try:
            v = float(v)
        except Exception:
            v = 0.0
        if v <= low:
            t = 0.0
        elif v >= high:
            t = 1.0
        else:
            t = (v - low) / (high - low)
        r = int(255 * (1 - t))
        g = int(255 * t)
        return f"ff00{g:02x}{r:02x}"

    pin_color = coh_to_kml_argb(mean_coh, coh_low_thr, coh_high_thr)

    # ------------------------------------------------------------------
    # 4) HTML POPUP
    # ------------------------------------------------------------------
    desc_html = "<h3>Interferogram Coherence Analysis</h3>"
    
    desc_html += """
    <style>
    /* ---- TABLE STYLING ---- */
    table.kml-table {
    border-collapse: collapse;
    border: 2px solid #2e8b57;   /* verde scuro */
    margin-bottom: 12px;
    font-size: 12px;
      }

    table.kml-table th {
    background-color: #e6f4ea;  /* verde chiarissimo */
    color: #1f5f3a;
    border: 1px solid #2e8b57;
    padding: 4px 8px;
    text-align: left;
    }

    table.kml-table td {
    border: 1px solid #2e8b57;
    padding: 4px 8px;
    vertical-align: top;
    }

    table.kml-table tr:nth-child(even) td {
    background-color: #f4fbf7;  /* zebra leggerissima */
    }
    </style>
    """
    
    
    

    if primary_name or secondary_name:
        desc_html += "<p>"
        if primary_name:
            desc_html += f"<b>Primary:</b> {primary_name}<br/>"
        if secondary_name:
            desc_html += f"<b>Secondary:</b> {secondary_name}"
        desc_html += "</p>"

    # ---- GLOBAL INFO TABLES (PRIMARY / SECONDARY / COMMON) ----------
    if coh_stats:

      primary_items = {}
      secondary_items = {}
      common_items = {}

      for k, v in coh_stats.items():
          if k.endswith("_primary"):
            primary_items[k] = v
          elif k.endswith("_secondary"):
            secondary_items[k] = v
          else:
            common_items[k] = v

      def render_table(title, items_dict):
        nonlocal desc_html
        if not items_dict:
            return
        
        desc_html += f"<h4 style='color:#1f5f3a'>{title}</h4>"
        desc_html += "<table class='kml-table'>"
        for k, v in items_dict.items():
            v_fmt = format_value(v, ndigits=4)
            desc_html += f"<tr><td><b>{k}</b></td><td>{v_fmt}</td></tr>"
        desc_html += "</table>"

      render_table("Primary parameters", primary_items)
      render_table("Secondary parameters", secondary_items)
      render_table("Common / global parameters", common_items)

    # ---- TAB CSS ----------------------------------------------------
    desc_html += """
    <style>
    .tabset > input { display: none; }
    .tabset > label {
      padding: 6px 12px;
      border: 1px solid #aaa;
      border-bottom: none;
      cursor: pointer;
      font-weight: bold;
      background: #eee;
    }
    .tabset > input:checked + label {
      background: #fff;
      border-bottom: 1px solid white;
    }
    .tab-panel {
      display: none;
      border: 1px solid #aaa;
      padding: 10px;
    }
    """

    for pol in pol_plots:
        desc_html += f"""
        #tab-{pol}:checked ~ .content #content-{pol} {{
            display: block;
        }}
        """

    desc_html += "</style>"

    # ---- TAB HEADERS ------------------------------------------------
    desc_html += "<div class='tabset'>"

    first = True
    for pol in pol_plots:
        checked = "checked" if first else ""
        desc_html += f"""
        <input type="radio" name="tabset" id="tab-{pol}" {checked}>
        <label for="tab-{pol}">{pol.upper()}</label>
        """
        first = False

    desc_html += "<div class='content'>"

    # ------------------------------------------------------------------
    # Attachments + helper
    # ------------------------------------------------------------------
    attachments = [png_path]

    def img_block(title, path):
        nonlocal attachments
        if path and Path(path).exists():
            p = Path(path)
            attachments.append(p)
            return f"<h5>{title}</h5><img src='{p.name}' width='650'/><br/>"
        return ""

    # ------------------------------------------------------------------
    # TAB CONTENT (per polarization)
    # ------------------------------------------------------------------
    for pol, pol_data in pol_plots.items():

        plots = pol_data.get("plots", {})

        desc_html += f"<div id='content-{pol}' class='tab-panel'>"
        desc_html += f"<h3>Polarization {pol.upper()}</h3>"

        desc_html += "<h4>Coherence</h4>"
        desc_html += img_block("|coh|", plots.get("coh_amp"))
        desc_html += img_block("arg(coh) [deg]", plots.get("coh_phase"))

        desc_html += "<h4>Interferogram</h4>"
        desc_html += img_block("Phase [deg]", plots.get("interferogram_phase"))
        desc_html += img_block("Flattened phase [deg]", plots.get("interferogram_phase_flat"))

        desc_html += "</div>"

    desc_html += "</div></div>"

    # ---- STATIC STA SCREENS -----------------------------------------
    desc_html += "<h4>STA Phase Screens (common)</h4>"
    desc_html += img_block("Flattening phase screen (secondary)", flatteningPhaseScreen_co_file)
    desc_html += img_block("SKP calibration phase screen (secondary)", skpCalibrationPhaseScreen_co_file)

    # ------------------------------------------------------------------
    # 5) BUILD KML
    # ------------------------------------------------------------------
    kml = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2">',
        "<Document>",
        f"<name>{png_path.stem}</name>",
        "<Style id='pin'>"
        f"<IconStyle><color>{pin_color}</color>"
        "<Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>"
        "</IconStyle></Style>",
        "<GroundOverlay>",
        f"<Icon><href>{png_path.name}</href></Icon>",
        "<gx:LatLonQuad>",
        f"<coordinates>{' '.join(coords_list)}</coordinates>",
        "</gx:LatLonQuad></GroundOverlay>",
        "<Placemark>",
        f"<name>{label_text}</name>",
        "<styleUrl>#pin</styleUrl>",
        f"<description><![CDATA[{desc_html}]]></description>",
        f"<Point><coordinates>{center_coord}</coordinates></Point>",
        "</Placemark>",
        "</Document></kml>",
    ]

    # ------------------------------------------------------------------
    # 6) WRITE KMZ
    # ------------------------------------------------------------------
    with zipfile.ZipFile(kmz_out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", "\n".join(kml))
        for p in attachments:
            if p.exists():
                zf.write(p, arcname=p.name)

    print(f"[OK] KMZ created: {kmz_out}")







def upsample_phase_via_complex(phase_in: np.ndarray,
                               axes_in: tuple[np.ndarray, np.ndarray],
                               axes_out: tuple[np.ndarray, np.ndarray],
                               bbox: list[float],
                               kx: int = 1, ky: int = 1, s: float = 0.0,
                               nodata: float = -9999.0) -> np.ndarray:
    """
     Upsampling of a phase (in radians) using interpolation on cosine and sine, 
     to avoid artifacts caused by phase wrapping.
    """
    
    if nodata is not None:
        mask = (phase_in == nodata) | ~np.isfinite(phase_in)
    else:
        mask = ~np.isfinite(phase_in)

    # Convert to real and imaginary parts of e^(j*phase)
    # e^{j phi} -> (cos, sin); dove mask, metti NaN per non far "tirare" l'interpolazione
    real_part = np.cos(phase_in).astype(np.float64)
    imag_part = np.sin(phase_in).astype(np.float64)
    real_part[mask] = np.nan
    imag_part[mask] = np.nan
    
    real_part = np.nan_to_num(real_part, nan=0.0)
    imag_part = np.nan_to_num(imag_part, nan=0.0)

    interp_re = scipy.interpolate.RectBivariateSpline(
        axes_in[0], axes_in[1], real_part,
        bbox=bbox, kx=kx, ky=ky, s=s
    )
    interp_im = scipy.interpolate.RectBivariateSpline(
        axes_in[0], axes_in[1], imag_part,
        bbox=bbox, kx=kx, ky=ky, s=s
    )

    re_up = interp_re(axes_out[0], axes_out[1])
    im_up = interp_im(axes_out[0], axes_out[1])

    # Reconstruct the complex number e^(jφ) and extract its angle
    mag = np.hypot(re_up, im_up)
    eps = 1e-12
    re_up /= (mag + eps)
    im_up /= (mag + eps)

    # Ricostruisci la fase
    phase_up = np.arctan2(im_up, re_up)
    return phase_up



def kernel_generation(shape: tuple[int, ...]) -> np.ndarray:
    """
    Generates a kernel with the given shape, where each element is the reciprocal of the product of the shape dimensions.

    Parameters:
    shape (tuple[int, ...]): The shape of the kernel.

    Returns:
    np.ndarray: A numpy array representing the kernel.
    """
    return np.full(shape, 1 / np.prod(shape))

def plot_2d(x, y, zplot, cb='rainbow', fs=12,
            vmin=None, vmax=None, title='',
            y_title='', x_title='', cb_title='',
            n_levels=None,
            samp_level=0.5, max_n_levels=4, level_format='%1.1f',
            contour=False,
            y_axis_format='%.0f', x_axis_format='%.1f',
            cb_orientation='vertical', aspect='auto',
            no_interp=False, origin='lower',
            nocolorbar=False, dpi=150,
            mask_color='black', x_nticks=None,
            show=False, file_2_save=None):
    """Plot 2D

    Args:
        x (ndarray): x axis
        y (ndarray): y axis
        zplot (ndarray): data
        cb (str): color map to use
        vmin (float): minimum value to show
        vmax (float): maximum value to show
        fs (int): font size
        title (str): label
        x_title (str): xlabel
        y_title (str): ylabel
        cb_title (str): colorbar ylabel
        n_levels (int): number of levels to contour
        samp_level (float): samp of levels
        max_n_levels (int): max. number of levels to plot
        coutour (bool): plot or not contour
        level_format (str): formatting of levels
        x_axis_format (str): formatting of xaxis labels
        y_axis_format (str): formatting of yaxis labels
        cb_orientation (str): defines colorbar orientation
        aspect (str): plot aspect
        no_interp (bool): dont perform interpolation of plot
        origin (str): origin of plot
        nocolorbar (bool): dont plot colorbar
        dpi (int): density of pixels of saved plot
        mask_color (str): color of invalids
        x_nticks (int): number of ticks of axis
        show (bool, optional): show plots. Default to False
        file_2_save (str): file to save
    """

    # set NaNs to black
    current_cmap = plt.colormaps[cb].copy()
    current_cmap.set_bad(color=mask_color)

    if file_2_save:
        dir = os.path.dirname(file_2_save)
        if not os.path.exists(dir) and dir != '':
            os.makedirs(dir)

    if not vmin:
        vmin = np.nanmean(zplot) - 3 * np.nanstd(zplot)
    if not vmax:
        vmax = np.nanmean(zplot) + 3 * np.nanstd(zplot)

    if contour:
        if not n_levels:
            n_levels = np.round((vmax - vmin) / samp_level)
            n_levels = np.min([n_levels, max_n_levels])
        levels = MaxNLocator(nbins=n_levels).tick_values(vmin, vmax)
        levels = np.where(levels < np.min(zplot), np.min(zplot), levels)
        levels = np.where(levels > np.max(zplot), np.max(zplot), levels)

    if aspect == 'auto':
        dim = zplot.shape
        fig, ax = plt.subplots(figsize=plt.figaspect(2))
    else:
        fig, ax = plt.subplots()

    ax.yaxis.set_major_formatter(FormatStrFormatter(y_axis_format))
    ax.xaxis.set_major_formatter(FormatStrFormatter(x_axis_format))
    ax.tick_params(axis='both', labelsize=fs)

    if origin == 'lower':
        ext = [y.min(), y.max(), x.min(), x.max()]
    else:
        ext = [y.min(), y.max(), x.max(), x.min()]

    if x[0] > x[-1]:
        if no_interp:
            im = ax.imshow(np.flip(zplot, axis=0), aspect=aspect, origin=origin, extent=ext,
                           cmap=cb, interpolation='none', vmin=vmin, vmax=vmax)

        else:
            im = ax.imshow(np.flip(zplot, axis=0), aspect=aspect, origin=origin, extent=ext,
                           cmap=cb, interpolation='bilinear', vmin=vmin, vmax=vmax)
    else:
        if no_interp:
            im = ax.imshow(zplot, aspect=aspect, origin=origin, extent=ext,
                           cmap=cb, interpolation='none', vmin=vmin, vmax=vmax)

        else:
            im = ax.imshow(zplot, aspect=aspect, origin=origin, extent=ext,
                           cmap=cb, interpolation='bilinear', vmin=vmin, vmax=vmax)

    if contour:
        if n_levels > 1 and zplot.shape[0] > 1 and zplot.shape[1] > 1:
            CS = ax.contour(zplot, levels)
            ax.clabel(CS, levels, inline=True, fmt=level_format, fontsize=fs)

    # make a colorbar for the image
    if nocolorbar == False:
        CBI = fig.colorbar(im, orientation=cb_orientation, shrink=1)
        CBI.set_label(cb_title, fontsize=fs)
        CBI.ax.tick_params(labelsize=fs)
    ax.set_title(title, fontsize=fs)
    ax.set_ylabel(x_title, fontsize=fs)
    ax.set_xlabel(y_title, fontsize=fs)
    if x_nticks:
        plt.xticks(np.arange(x.min(), x.max(), (x.max() - x.min()) / x_nticks))
    fig.tight_layout()
    if file_2_save:
        plt.savefig(file_2_save, dpi=dpi, bbox_inches='tight')
        plt.close('all')
    if show:
        plt.show()

    del fig, im
    return

def plot_2d_with_hist(
    x, y, zplot,
    cb='rainbow', fs=12,
    vmin=None, vmax=None,
    title='',
    y_title='', x_title='', cb_title='',
    y_axis_format='%.0f', x_axis_format='%.1f',
    cb_orientation='vertical',
    no_interp=False, origin='lower',
    nocolorbar=False, dpi=150,
    mask_color='black',
    height_compression=0.1,
    nodata=None,
    show=False,
    file_2_save=None
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    # ------------------------------------------------------------------
    # 1) NORMALIZE INPUT (ONE SINGLE SOURCE OF TRUTH)
    # ------------------------------------------------------------------
    if isinstance(zplot, np.ma.MaskedArray):
        data = zplot.filled(np.nan)
    else:
        data = np.array(zplot, dtype=np.float64, copy=True)

    if nodata is not None:
        data[data == nodata] = np.nan

    # ------------------------------------------------------------------
    # 2) AUTO RANGE (ROBUST, NODATA SAFE)
    # ------------------------------------------------------------------
    if vmin is None or vmax is None:
        finite_vals = data[np.isfinite(data)]
        if finite_vals.size == 0:
            raise ValueError("No valid data available after nodata masking")

        if vmin is None:
            vmin = np.nanmean(finite_vals) - 3 * np.nanstd(finite_vals)
        if vmax is None:
            vmax = np.nanmean(finite_vals) + 3 * np.nanstd(finite_vals)

    # ------------------------------------------------------------------
    # 3) FIGURE LAYOUT
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(
        1, 2,
        figsize=(14, 7),
        gridspec_kw={'width_ratios': [2.3, 1]}
    )
    ax_map, ax_hist = ax

    # ------------------------------------------------------------------
    # 4) COLORMAP
    # ------------------------------------------------------------------
    current_cmap = plt.colormaps[cb].copy()
    current_cmap.set_bad(color=mask_color)

    # ------------------------------------------------------------------
    # 5) EXTENT & ASPECT RATIO
    # ------------------------------------------------------------------
    if origin == "lower":
        extent = [y.min(), y.max(), x.min(), x.max()]
    else:
        extent = [y.min(), y.max(), x.max(), x.min()]

    H, W = data.shape
    natural_aspect = H / W
    aspect_ratio = natural_aspect * height_compression

    # ------------------------------------------------------------------
    # 6) PREPARE MAP DATA
    # ------------------------------------------------------------------
    data_plot = np.flip(data, axis=0) if x[0] > x[-1] else data
    data_plot = np.ma.masked_invalid(data_plot)

    interp_opt = 'none' if no_interp else 'bilinear'

    # ------------------------------------------------------------------
    # 7) MAP
    # ------------------------------------------------------------------
    im = ax_map.imshow(
        data_plot,
        origin=origin,
        extent=extent,
        cmap=current_cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation=interp_opt,
        aspect=aspect_ratio
    )

    ax_map.set_title(title, fontsize=fs)
    ax_map.set_xlabel(y_title, fontsize=fs)
    ax_map.set_ylabel(x_title, fontsize=fs)
    ax_map.xaxis.set_major_formatter(FormatStrFormatter(x_axis_format))
    ax_map.yaxis.set_major_formatter(FormatStrFormatter(y_axis_format))

    if not nocolorbar:
        cb_obj = fig.colorbar(im, ax=ax_map, orientation=cb_orientation)
        cb_obj.set_label(cb_title, fontsize=fs)

    # ------------------------------------------------------------------
    # 8) HISTOGRAM + STATISTICS (NODATA SAFE)
    # ------------------------------------------------------------------
    vals = data[np.isfinite(data)]

    ax_hist.hist(vals, bins=300, color="steelblue", edgecolor="blue")
    ax_hist.grid(True)
    ax_hist.set_title(f"{title} histogram", fontsize=fs)
    ax_hist.set_xlabel("Value")
    ax_hist.set_ylabel("Count")

    stats_txt = (
        f"count:  {vals.size}\n"
        f"min:    {np.min(vals):.3f}\n"
        f"max:    {np.max(vals):.3f}\n"
        f"mean:   {np.mean(vals):.3f}\n"
        f"median: {np.median(vals):.3f}\n"
        f"std:    {np.std(vals):.3f}"
    )

    ax_hist.text(
        0.98, 0.95,
        stats_txt,
        ha="right", va="top",
        transform=ax_hist.transAxes,
        fontsize=fs,
        bbox=dict(facecolor="white", alpha=0.8)
    )

    # ------------------------------------------------------------------
    # 9) SAVE / SHOW
    # ------------------------------------------------------------------
    fig.tight_layout()

    if file_2_save:
        fig.savefig(file_2_save, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    if show:
        plt.show()




def plot_2d_with_hist_old(
    x, y, zplot,
    cb='rainbow', fs=12,
    vmin=None, vmax=None,
    title='',
    y_title='', x_title='', cb_title='',
    y_axis_format='%.0f', x_axis_format='%.1f',
    cb_orientation='vertical',
    no_interp=False, origin='lower',
    nocolorbar=False, dpi=150,
    mask_color='black',
    height_compression=0.1,
    nodata=None,
    show=False,
    file_2_save=None
):


    # ------------------------------------------------------------------
    # 1) NORMALIZE INPUT (ONE SINGLE SOURCE OF TRUTH)
    # ------------------------------------------------------------------
    if isinstance(zplot, np.ma.MaskedArray):
        data = zplot.filled(np.nan)
    else:
        data = np.array(zplot, dtype=np.float64, copy=True)

    if nodata is not None:
        data[data == nodata] = np.nan

    # ------------------------------------------------------------------
    # 2) AUTO RANGE (ROBUST, NODATA SAFE)
    # ------------------------------------------------------------------
    if vmin is None or vmax is None:
        finite_vals = data[np.isfinite(data)]
        if finite_vals.size == 0:
            raise ValueError("No valid data available after nodata masking")

        if vmin is None:
            vmin = np.nanmean(finite_vals) - 3 * np.nanstd(finite_vals)
        if vmax is None:
            vmax = np.nanmean(finite_vals) + 3 * np.nanstd(finite_vals)

    # ------------------------------------------------------------------
    # 3) FIGURE LAYOUT
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(
        1, 2,
        figsize=(14, 7),
        gridspec_kw={'width_ratios': [2.3, 1]}
    )
    ax_map, ax_hist = ax

    # ------------------------------------------------------------------
    # 4) COLORMAP
    # ------------------------------------------------------------------
    current_cmap = plt.colormaps[cb].copy()
    current_cmap.set_bad(color=mask_color)

    # ------------------------------------------------------------------
    # 5) EXTENT & ASPECT RATIO
    # ------------------------------------------------------------------
    if origin == "lower":
        extent = [y.min(), y.max(), x.min(), x.max()]
    else:
        extent = [y.min(), y.max(), x.max(), x.min()]

    H, W = data.shape
    natural_aspect = H / W
    aspect_ratio = natural_aspect * height_compression

    # ------------------------------------------------------------------
    # 6) PREPARE MAP DATA
    # ------------------------------------------------------------------
    data_plot = np.flip(data, axis=0) if x[0] > x[-1] else data
    data_plot = np.ma.masked_invalid(data_plot)

    interp_opt = 'none' if no_interp else 'bilinear'

    # ------------------------------------------------------------------
    # 7) MAP
    # ------------------------------------------------------------------
    im = ax_map.imshow(
        data_plot,
        origin=origin,
        extent=extent,
        cmap=current_cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation=interp_opt,
        aspect=aspect_ratio
    )

    ax_map.set_title(title, fontsize=fs)
    ax_map.set_xlabel(y_title, fontsize=fs)
    ax_map.set_ylabel(x_title, fontsize=fs)
    ax_map.xaxis.set_major_formatter(FormatStrFormatter(x_axis_format))
    ax_map.yaxis.set_major_formatter(FormatStrFormatter(y_axis_format))

    if not nocolorbar:
        cb_obj = fig.colorbar(im, ax=ax_map, orientation=cb_orientation)
        cb_obj.set_label(cb_title, fontsize=fs)

    # ------------------------------------------------------------------
    # 8) HISTOGRAM + STATISTICS (NODATA SAFE)
    # ------------------------------------------------------------------
    vals = data[np.isfinite(data)]

    ax_hist.hist(vals, bins=300, color="steelblue", edgecolor="blue")
    ax_hist.grid(True)
    ax_hist.set_title(f"{title} histogram", fontsize=fs)
    ax_hist.set_xlabel("Value")
    ax_hist.set_ylabel("Count")

    stats_txt = (
        f"count:  {vals.size}\n"
        f"min:    {np.min(vals):.3f}\n"
        f"max:    {np.max(vals):.3f}\n"
        f"mean:   {np.mean(vals):.3f}\n"
        f"median: {np.median(vals):.3f}\n"
        f"std:    {np.std(vals):.3f}"
    )

    ax_hist.text(
        0.98, 0.95,
        stats_txt,
        ha="right", va="top",
        transform=ax_hist.transAxes,
        fontsize=fs,
        bbox=dict(facecolor="white", alpha=0.8)
    )

    # ------------------------------------------------------------------
    # 9) SAVE / SHOW
    # ------------------------------------------------------------------
    fig.tight_layout()

    if file_2_save:
        fig.savefig(file_2_save, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    if show:
        plt.show()



def single_baseline_single_pol_coh_old(primary_complex: npt.NDArray[complex],
                                   secondary_complex: npt.NDArray[complex],
                                   avg_kernel_shape: tuple[int, ...],
                                   ) -> npt.NDArray[complex]:
    """
    Compute the coherence map (complex).

    The coherence map at an azimuth/range pixel (a, r) is defined as:

                                E[S(a, r) * conj(P(a, r))]
       Coh{P, S}(a, r) :=  -----------------------------------
                            sqrt(Var[P(a, r)] * Var[S(a, r)])

    Parameters:
    primary_complex (npt.NDArray[complex]): Complex array of the primary image.
    secondary_complex (npt.NDArray[complex]): Complex array of the secondary image.
    avg_kernel_shape (tuple[int, ...]): Shape of the averaging kernel.
    flag_avg (bool): If False, only the Hermitian product is applied without averaging. Default is True.

    Returns:
    npt.NDArray[complex]: The [Nazm x Nrng] coherence map.

    Raises:
    ValueError: If the shapes of primary_complex and secondary_complex do not match.
    """
    if primary_complex.shape != secondary_complex.shape:
        raise ValueError(f"Coh inputs have different shapes {primary_complex.shape} != {secondary_complex.shape}")

    kernel = kernel_generation(avg_kernel_shape)

    covariance_primary_secondary = (primary_complex * np.conj(secondary_complex)).astype(np.complex64)
    covariance_primary_secondary = scipy.signal.convolve2d(covariance_primary_secondary,
                                                            kernel,
                                                            boundary="symm",
                                                            mode="same")

    variance_primary = np.abs(primary_complex)**2
    variance_primary = scipy.signal.convolve2d(variance_primary,
                                                kernel,
                                                boundary="symm",
                                                mode="same")

    variance_secondary = np.abs(secondary_complex)**2
    variance_secondary = scipy.signal.convolve2d(variance_secondary,
                                                    kernel,
                                                    boundary="symm",
                                                    mode="same")

    variance_primary_variance_secondary = variance_primary * variance_secondary
    valid = variance_primary_variance_secondary > 0.0

    coherence = np.empty_like(covariance_primary_secondary)
    coherence[valid] = covariance_primary_secondary[valid] / np.sqrt(
        variance_primary_variance_secondary[valid]
    )

    coherence[~valid] = 0
    coherence[np.isnan(coherence)] = 0 + 0j

    return coherence

def single_baseline_single_pol_coh(
    slc_primary: np.ndarray,
    slc_secondary: np.ndarray,
    window: tuple[int, int] = (5, 5),
    valid_mask: np.ndarray | None = None,
    min_valid_frac: float = 0.8
):

    wy, wx = window
    kernel = np.ones((wy, wx), dtype=float)
    win_area = wy * wx

    # ------------------------------------------------------------
    # 0) Valid mask
    # ------------------------------------------------------------
    if valid_mask is None:
        valid_mask = np.ones(slc_primary.shape, dtype=bool)

    valid_mask = valid_mask.astype(float)

    # ------------------------------------------------------------
    # 1) Zero-out invalid samples
    # ------------------------------------------------------------
    p = slc_primary.copy()
    s = slc_secondary.copy()

    p[~valid_mask.astype(bool)] = 0.0
    s[~valid_mask.astype(bool)] = 0.0

    # ------------------------------------------------------------
    # 2) Core interferometric products
    # ------------------------------------------------------------
    num = p * np.conj(s)
    den_p = np.abs(p) ** 2
    den_s = np.abs(s) ** 2

    # ------------------------------------------------------------
    # 3) Convolutions (only valid samples contribute)
    # ------------------------------------------------------------
    num_f = convolve2d(num, kernel, mode="same", boundary="fill", fillvalue=0.0)
    den_p_f = convolve2d(den_p, kernel, mode="same", boundary="fill", fillvalue=0.0)
    den_s_f = convolve2d(den_s, kernel, mode="same", boundary="fill", fillvalue=0.0)

    # Count valid samples per window
    n_valid = convolve2d(valid_mask, kernel, mode="same", boundary="fill", fillvalue=0.0)

    # ------------------------------------------------------------
    # 4) Coherence
    # ------------------------------------------------------------
    denom = np.sqrt(den_p_f * den_s_f)
    coh = np.zeros_like(num_f, dtype=np.complex128)

    with np.errstate(divide="ignore", invalid="ignore"):
        coh = num_f / denom

    coh_abs = np.abs(coh)
    coh_phase = np.angle(coh)

    # ------------------------------------------------------------
    # 5) Enforce window validity criterion
    # ------------------------------------------------------------
    min_valid = min_valid_frac * win_area
    valid_coh = n_valid >= min_valid

    coh_abs[~valid_coh] = np.nan
    coh_phase[~valid_coh] = np.nan

    return coh_abs, coh_phase

def extract_date_from_sta_name(sta_path):
    """
    Extract the acquisition date (YYYYMMDD) from the STA product name.
    Assumes format: BIO_S1_STA__1S_YYYYMMDDTHHMMSS_...
    """
    sta_path = Path(sta_path)
    name = sta_path.name
    print(f"[DEBUG] sta_path.name: {name}")

    parts = name.split("_")
    #print(f"[DEBUG] parts: {parts}")

    # Look for the first part that looks like a timestamp
    for part in parts:
        if part.startswith("20") and "T" in part:
            date_part = part[:15]
            print(f"[DEBUG] Extracted date: {date_part}")
            return date_part

    raise ValueError(f"Cannot find date in STA name: {name}")

def save_phase_map_lut(arr: np.ndarray,
                       title: str,
                              date: str,
                              out_file: Path,
                              cmap: str = "rainbow",
                              aspect_ratio: float = 0.2,
                              colorbar_fraction: float = 0.05,
                              colorbar_pad: float = 0.04,
                              nodata: float = -9999.0,
                              fontsize: int = 6) -> None:

    data = np.ma.masked_equal(arr, nodata)
    valid = data.compressed()
    if valid.size == 0:
        print(f"[WARN] No valid values for {out_file.name}")
        return

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(data,
                   cmap=cmap,
                   aspect=aspect_ratio,
                   interpolation="none")
    ax.set_title(f"{title}", fontsize=fontsize)
    fig.colorbar(im, ax=ax, orientation="vertical",
                 fraction=colorbar_fraction, pad=colorbar_pad)

    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Image saved to: {out_file}")
     
    
        
def save_phase_map_slc(arr: np.ndarray,
                       title: str, date: str, out_file: Path,
                       nodata: float = -9999.0,
                       std_multiplier: float = 3.0,
                       cmap: str = "rainbow",
                       aspect: str = 0.2) -> None:

    data = np.ma.masked_equal(arr.astype(np.float64), nodata)
    valid = data.compressed()
    if valid.size == 0:
        print(f"[WARN] No valid values to plot for {out_file}")
        return
    vmin = np.nanpercentile(valid, 2)
    vmax = np.nanpercentile(valid, 98)
    nan_value = 255
    std_multiplier = 3
    aspect_ratio =0.2
    colorbar_fraction = 0.05
    colorbar_pad = 0.04
    

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(data, cmap=cmap, aspect=aspect_ratio, interpolation="none",
                   vmin=vmin, vmax=vmax)
    ax.set_title(f"{title} ", fontsize=6)
    fig.colorbar(im, ax=ax, orientation='vertical', fraction=colorbar_fraction, pad=colorbar_pad)
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Image saved: {out_file}")   
    

def getHistogram(arr, nodata=-9999, title="Histogram", out_file=None, bins=100):
    """
    Crea e salva un istogramma semplice con statistiche a fianco.
    """
    # maschera valori validi
    mask = np.isfinite(arr) & (arr != nodata)
    vals = arr[mask]

    if vals.size == 0:
        print("[WARN] No valid values for histogram.")
        return

    # statistiche
    stats = {
        "count": int(vals.size),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
    }

    # plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=bins, color="steelblue", edgecolor="black")  # bins ora definito
    ax.set_title(title)
    ax.set_xlabel("Value [rad]")
    ax.set_ylabel("Count")

    # formattazione dinamica: scientifica se molto piccoli
    def fmt(x):
        return f"{x:.3e}" if abs(x) < 1e-3 else f"{x:.3f}"

    box = (
        f"count: {stats['count']}\n"
        f"min: {fmt(stats['min'])}\n"
        f"max: {fmt(stats['max'])}\n"
        f"mean: {fmt(stats['mean'])}\n"
        f"std: {fmt(stats['std'])}"
    )
    ax.text(
        0.98, 0.98, box,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    fig.tight_layout()
    if out_file:
        fig.savefig(out_file, dpi=200)
        print(f"[INFO] Histogram saved to: {out_file}")
    plt.close(fig)

    return stats





    
def _safe_text(root, xpath, default=""):
    node = root.find(xpath)
    return (node.text or "").strip() if node is not None and node.text is not None else default

def get_info_baseline(product_primary, product_secondary,listinfobaseline):
    """
    Extracts from the STA product all the required baseline information
    and appends it as a dictionary.
    Returns listinfobaseline (a list of dictionaries).
    """


    info_pri = {
        

        "orbitNumber_primary":                        int(product_primary.orbitNumber),  #getattr(product_primary, "orbitNumber", ""),
        "orbitDirection_primary":                     (product_primary.orbitDirection),#getattr(product_primary, "orbitDirection", ""),
        
        "startTimeFromAscendingNode_primary":         product_primary.startTimeFromAscendingNode,#getattr(product_primary, "startTimeFromAscendingNode", ""),
        "completionTimeFromAscendingNode_primary":    product_primary.completionTimeFromAscendingNode,#getattr(product_primary, "completionTimeFromAscendingNode", ""),
        "dataTakeID_primary":                         int(product_primary.dataTakeID),#getattr(product_primary, "dataTakeID", ""),
        "overallProductQualityIndex_STA_primary" :      product_primary.overallProductQualityIndex
    }
    
    annotation_file_sec = product_secondary.annotation_coregistered_xml_file
    root_sec  = ET.parse(annotation_file_sec).getroot()
    
    
    mission_phase = product_secondary.missionPhaseID
    bc_critical = BC_BY_MISSION_PHASE.get(mission_phase)

    normal_baseline = float(_safe_text(root_sec, ".//normalBaseline", "0.0"))

    baseline_percentage = None
    if bc_critical and bc_critical > 0:
        baseline_percentage = abs(normal_baseline) / bc_critical * 100.0
    
    
    
    
    info_sec = {
        
        "primaryImageSelectionInformation_secondary":  _safe_text(root_sec, ".//primaryImageSelectionInformation"),
        "normalBaseline":                     normal_baseline,
        "criticalBaseline":                  bc_critical,
        "baselinePercentage":                baseline_percentage,
        "averageRangeCoregistrationShift_secondary":    _safe_text(root_sec, ".//averageRangeCoregistrationShift"),
        "averageAzimuthCoregistrationShift_secondary": _safe_text(root_sec, ".//averageAzimuthCoregistrationShift"),
    
        "orbitNumber_secondary":                      int(product_secondary.orbitNumber) ,#getattr(product_secondary, "orbitNumber", ""),
        "orbitDirection_secondary":                   (product_secondary.orbitDirection), #getattr(product_secondary, "orbitDirection", ""),
        
        "startTimeFromAscendingNode_secondary":       product_secondary.startTimeFromAscendingNode,  #getattr(product_secondary, "startTimeFromAscendingNode", ""),
        "completionTimeFromAscendingNode_secondary":  product_secondary.completionTimeFromAscendingNode, #getattr(product_secondary, "completionTimeFromAscendingNode", ""),        
        "dataTakeID_secondary":                       int(product_secondary.dataTakeID), #getattr(product_secondary, "dataTakeID", ""),        
        "centerLat_secondary":                        product_secondary.center_lat, #getattr(product_secondary, "center_lat", None),
        "centerLon_secondary":                        product_secondary.center_lon, #getattr(product_secondary, "center_lon", None), 
        "missionPhaseID":                             product_secondary.missionPhaseID,
        "coregistrationMethod_secondary":             product_secondary.coregistrationMethod,
        "rfiDetectionFlag_secondary" :                product_secondary.rfiDetectionFlag,
        "rfiCorrectionFlag_secondary":                product_secondary.rfiCorrectionFlag,
        "rfiMitigationMethod_secondary":              product_secondary.rfiMitigationMethod,
        "rfiMask_secondary"   :                       product_secondary.rfiMask,
        "rfiMaskGenerationMethod_secondary" :         product_secondary.rfiMaskGenerationMethod,
        "overallProductQualityIndex_STA_secondary" :      product_secondary.overallProductQualityIndex
   }        
    
    
    
    print(
    f"[DEBUG] normalBaseline = {normal_baseline:.2f} m | "
    f"Bc = {bc_critical:.1f} m | "
    f"Baseline % = {baseline_percentage:.2f} %"
     )
    # Merge e append
    merged = {}
    merged.update(info_pri)
    merged.update(info_sec)
    listinfobaseline.append(merged)
    return listinfobaseline  
    
def add_alpha_border(png_path, border=10):
    """Adds a transparent alpha channel along the edges of the PNG image."""
    img = Image.open(png_path).convert("RGBA")
    arr = np.array(img)
    alpha = np.ones((arr.shape[0], arr.shape[1]), dtype=np.uint8) * 255
    # rende trasparenti i bordi
    alpha[:border, :] = 0
    alpha[-border:, :] = 0
    alpha[:, :border] = 0
    alpha[:, -border:] = 0
    arr[:, :, 3] = alpha
    Image.fromarray(arr).save(png_path)
    
def make_white_transparent(png_path, threshold=250):
    """
    Makes white or near-white pixels transparent (RGBA).
    threshold = level above which a pixel is considered 'white'.
    """
    img = Image.open(png_path).convert("RGBA")
    data = np.array(img)

    # Mask: all channels (R, G, B) > threshold and alpha > 0
    mask = (data[:, :, 0] >= threshold) & \
           (data[:, :, 1] >= threshold) & \
           (data[:, :, 2] >= threshold) & \
           (data[:, :, 3] > 0)

    
    data[mask, 3] = 0

    
    Image.fromarray(data).save(png_path)
    print(f"[OK] Trasparenza applicata a: {png_path}")        


def get_pol_to_band_index(tiff_path):
    pol_to_band = {}

    with rasterio.open(tiff_path) as ds:
        for band_idx in ds.indexes:   # 1-based
            tags = ds.tags(band_idx)
            pol = tags.get("POLARIMETRIC_INTERP")

            if pol:
                pol_to_band[pol] = band_idx

    return pol_to_band

def get_gcps_from_reference(tiff_ref_path):
    ds = gdal.Open(str(tiff_ref_path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open reference TIFF: {tiff_ref_path}")

    gcps = ds.GetGCPs()
    gcp_proj = ds.GetGCPProjection()
    ds = None
    return gcps, gcp_proj

def save_tiff_with_gcps( out_path, array, ref_tiff, dtype=gdal.GDT_Float32): 
    
    gcps, gcp_proj = get_gcps_from_reference(ref_tiff)

    driver = gdal.GetDriverByName("GTiff")
    ny, nx = array.shape

    ds_out = driver.Create(
        str(out_path),
        nx,
        ny,
        1,
        dtype,
        options=["COMPRESS=DEFLATE"]
    )

    ds_out.GetRasterBand(1).WriteArray(array)
    ds_out.GetRasterBand(1).SetNoDataValue(-9999)

    if gcps:
        ds_out.SetGCPs(gcps, gcp_proj)

    ds_out.FlushCache()
    ds_out = None

def write_interferogram_annotation_xml(out_dir: Path,product_name: str, global_info: dict, pol_plots: dict, ref_pol: str):
    


    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    annot_name = product_name.lower() + "_annot.xml"
    out_xml = out_dir / annot_name

    root = ET.Element("biomassInterferogramAnnotation")

    # ------------------------------------------------------------------
    # Identification
    # ------------------------------------------------------------------
    ident = ET.SubElement(root, "identification")

    ET.SubElement(ident, "primaryProduct").text = global_info.get("primaryProduct", "")
    ET.SubElement(ident, "secondaryProduct").text = global_info.get("secondaryProduct", "")
    ET.SubElement(ident, "frame").text = str(global_info.get("frame", ""))
    ET.SubElement(ident, "phaseCorrectionMode").text = str(
        global_info.get("phaseCorrectionMode", "")
    )

    # ------------------------------------------------------------------
    # Dates
    # ------------------------------------------------------------------
    dates = ET.SubElement(root, "dates")
    ET.SubElement(dates, "primary").text = global_info.get("date_primary", "")
    ET.SubElement(dates, "secondary").text = global_info.get("date_secondary", "")

    # ------------------------------------------------------------------
    # Baseline & orbit info (everything else in global_info)
    # ------------------------------------------------------------------
    baseline = ET.SubElement(root, "baselineAndOrbit")

    skip_keys = {
        "frame", "phaseCorrectionMode",
        "date_primary", "date_secondary",
        "primaryProduct", "secondaryProduct"
    }

    for k, v in global_info.items():
        if k in skip_keys:
            continue
        ET.SubElement(baseline, k).text = str(v)

    # ------------------------------------------------------------------
    # Coherence statistics
    # ------------------------------------------------------------------
    coh = ET.SubElement(root, "coherenceStatistics")
    ET.SubElement(coh, "referencePolarization").text = ref_pol

    for pol, pol_data in pol_plots.items():
        pol_el = ET.SubElement(coh, "polarization", name=pol)

        stats = pol_data.get("stats", {})
        for k, v in stats.items():
            ET.SubElement(pol_el, k).text = f"{v:.6f}"

    # ------------------------------------------------------------------
    # Write file
    # ------------------------------------------------------------------
    tree = ET.ElementTree(root)
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)

    print(f"[OK] Annotation XML written: {out_xml}")
    
    
def build_safe_spline(field_in: np.ndarray,
                      axes_in: tuple[np.ndarray, np.ndarray],
                      bbox,
                      kx=1, ky=1, s=0.0, nodata=-9999.0):
    """
    Build a RectBivariateSpline that is robust to NaNs.
    Returns: interpolator, validity_mask (boolean)
    """
    field = np.asarray(field_in, dtype=np.float64)

    mask = ~np.isfinite(field)| (field == nodata)
    if mask.all():
        raise ValueError("Input field contains only NaNs — spline interpolation impossible.")

    mean_val = np.nanmean(field)
    field_filled = field.copy()
    field_filled[mask] = mean_val
     
    spline = sp.interpolate.RectBivariateSpline(
        axes_in[0],
        axes_in[1],
        field_filled,
        bbox=bbox,
        kx=kx,
        ky=ky,
        s=s
    )

    # interpolator for the mask (1 = valid, 0 = invalid)
    mask_interp = sp.interpolate.RectBivariateSpline(
        axes_in[0],
        axes_in[1],
        (~mask).astype(float),
        kx=1,
        ky=1,
        s=0.0
    )

    return spline, mask_interp

def canonical_pol_name(pol: str) -> str:
    """
    Map raw polarization names to canonical output names.

    HV / VH  -> XP
    HH, VV   -> unchanged
    """
    pol = pol.upper()
    if pol in ("HV", "VH"):
        return "XP"
    return pol

def quick_stats(name, arr):
        print(f"{name}:")
        print("  shape =", arr.shape)
        print("  min   =", np.nanmin(arr))
        print("  max   =", np.nanmax(arr))
        print()

def check_interferogram(path_primary, path_secondary, flatten, is_light, number_frame,coh_low_thr, coh_high_thr ,
                        skpPhaseCalibrationFlag,skpPhaseCorrectionFlag, skpPhaseCorrectionFlatteningOnlyFlag, polarizations_requested ):


    path_primary = Path(path_primary)
    path_secondary = Path(path_secondary)

    print(path_primary)
    print(path_secondary)
    print(flatten)
    # -- Open products (uses your BiomassProduct class)
    product_primary=    BiomassProduct.BiomassProductSTA(path_primary)
    product_secondary=  BiomassProduct.BiomassProductSTA(path_secondary)
    listinfobaseline=list()
    get_info_baseline(product_primary, product_secondary,listinfobaseline)
    
    # -- Dates for naming
    date_primary = extract_date_from_sta_name(path_primary)
    date_secondary = extract_date_from_sta_name(path_secondary)
    print(f"[DEBUG] date_primary: {date_primary}")
    print(f"[DEBUG] date_secondary: {date_secondary}")

    # -- Output folder
    parent_folder = path_primary.parent 
    mode_label = "M" if is_light else "S"
    output_folder_check_interferogram = parent_folder / f"BIO_STA_CHK_INT_{mode_label}_{date_primary}_{date_secondary}_F{number_frame}_{flatten.upper()}"
    output_folder_check_interferogram.mkdir(exist_ok=True)
    # ------------------------------------------------------------------
    # Subfolders
    # ------------------------------------------------------------------
    preview_dir = output_folder_check_interferogram / "preview"
    annotation_dir = output_folder_check_interferogram / "annotation"
    
    preview_dir.mkdir(exist_ok=True)

    annotation_dir.mkdir(exist_ok=True)
    
    measurement_dir = None
    if not is_light:   # mode S
        measurement_dir = output_folder_check_interferogram / "measurement"
        measurement_dir.mkdir(exist_ok=True)
    

    print(f"[INFO] Output root      : {output_folder_check_interferogram}")
    print(f"[INFO] Preview folder   : {preview_dir}")
    print(f"[INFO] Annotation folder: {annotation_dir}")
    if measurement_dir:
        print(f"[INFO] Measurement folder: {measurement_dir}")
    print(f"[INFO] Saving figures to: {output_folder_check_interferogram}")
    

    data_primary_abs =      product_primary.measurement_abs_file   
    preview_kml_pri =       product_primary.preview_kml_file
    data_primary_phase =    product_primary.measurement_phase_file       
    path_lut_primary =      product_primary.annotation_coregistered_lut_file   
    path_main_ann_primary = product_primary.annotation_coregistered_xml_file   
    
    
    

    data_secondary_abs =         product_secondary.measurement_abs_file  
    preview_kml_sec  =           product_secondary.preview_kml_file
    data_secondary_phase =       product_secondary.measurement_phase_file        
    path_lut_coregistered =      product_secondary.annotation_coregistered_lut_file  
    path_main_ann_coregistered = product_secondary.annotation_coregistered_xml_file  
    
    print("\n[INFO] PRIMARY PRODUCT:")
    print(f"  - Measurement ABS:     {data_primary_abs}")
    print(f"  - Measurement PHASE:   {data_primary_phase}")
    print(f"  - LUT (coreg):         {path_lut_primary}")
    print(f"  - Annotation XML:      {path_main_ann_primary}")

    print("\n[INFO] SECONDARY PRODUCT:")
    print(f"  - Measurement ABS:     {data_secondary_abs}")
    print(f"  - Measurement PHASE:   {data_secondary_phase}")
    print(f"  - LUT (coreg):         {path_lut_coregistered}")
    print(f"  - Annotation XML:      {path_main_ann_coregistered}")
                
    # --------------------------------------------------
    # Read STA primary measurement structural info
    # --------------------------------------------------
    sta_info = read_sta_primary_measurement_info(data_primary_abs)
    nodata = float(sta_info['nodata'])
    print("[INFO] STA primary measurement info:")
    print(f"  Polarizations: {sta_info['polarizations']}")
    print(f"  NoData value: {sta_info['nodata']}")
    print(f"  Software: {sta_info['software']}")
    print(f"  Number of GCPs: {len(sta_info['gcps'])}")
    
    # --------------------------------------------------
    # Polarizations handling
    # --------------------------------------------------
    polarizations_product = sta_info["polarizations"]
    
    if polarizations_requested is None:
        # default: use all polarizations from the product
        polarizations = polarizations_product
    else:
        # user-requested subset
        polarizations = [
            p for p in polarizations_product
            if p.upper() in polarizations_requested
        ]
    
        if not polarizations:
            raise ValueError(
                f"Requested polarizations {polarizations_requested} "
                f"are not available in the product. "
                f"Available polarizations: {polarizations_product}"
            )
    
    print(f"[INFO] Polarizations to process: {polarizations}")
    
    
    # -------------------------
    # 2) Read LUTs and axes
    # -------------------------
    lut_co = netCDF4.Dataset(path_lut_coregistered)
    lut_pri = netCDF4.Dataset(path_lut_primary)
    
    # input LUT axes (coarse grid)
    relativeAzimuthTime_pri = lut_pri['relativeAzimuthTime'][:].astype(np.float64)
    slantRangeTime_pri = lut_pri['slantRangeTime'][:].astype(np.float64)
    lut_az_axes_pri = (relativeAzimuthTime_pri - relativeAzimuthTime_pri[0]).astype(np.float64)
    lut_range_axis_pri = slantRangeTime_pri - slantRangeTime_pri[0]
    
    # output SLC axes (fine grid) from annotation XML
    main_ann_primary = ET.parse(path_main_ann_primary)
    main_ann_primary_root = main_ann_primary.getroot()
    sarImage_pri = main_ann_primary_root.findall('sarImage')
    #range coreg
    firstSampleSlantRangeTime_pri = np.float64(sarImage_pri[0].findall("firstSampleSlantRangeTime")[0].text)
    rangeTimeInterval_pri = np.float64(sarImage_pri[0].findall("rangeTimeInterval")[0].text)
    numberOfSamples_pri = np.int64(sarImage_pri[0].findall("numberOfSamples")[0].text)
    #az coreg
    azimuthTimeInterval_pri = np.float64(sarImage_pri[0].findall("azimuthTimeInterval")[0].text)
    firstLineAzimuthTime_pri = np.datetime64(sarImage_pri[0].findall("firstLineAzimuthTime")[0].text)
    numberOfLines_pri = np.int64(sarImage_pri[0].findall("numberOfLines")[0].text)
    
    
    staProcessingParameters_pri = main_ann_primary_root.findall('staProcessingParameters')
    print('----------------------------------------coregistration Metod')
    
    
    coreg_el = staProcessingParameters_pri[0].find("coregistrationMethod")
    coregistrationMethod_pri = " ".join(coreg_el.itertext()).strip().replace("\n", " ")
    print(coregistrationMethod_pri)
    
    coreg_el = staProcessingParameters_pri[0].find("coregistrationMethod")

    print("RAW .text repr:", repr(coreg_el.text))
    for i, t in enumerate(coreg_el.itertext()):
        print(f"itertext[{i}]:", repr(t))
    
    #primary az axis
    axis = 0
    roi_pri =  [0, 0,numberOfLines_pri, numberOfSamples_pri]
    time_step_pri = azimuthTimeInterval_pri
    time_start_pri = 0
    az_slc_axis_pri_stac = np.arange(roi_pri[axis + 2], dtype=np.float64) * time_step_pri
    
    
    
    #pri rg axis
    axis = 1
    roi_pri =  [0, 0,numberOfLines_pri, numberOfSamples_pri]
    time_step_pri = rangeTimeInterval_pri
    time_start_pri = firstSampleSlantRangeTime_pri
    range_slc_axis_pri_stac = np.arange(roi_pri[axis + 2], dtype=np.float64) * time_step_pri
    
    # Print result clearly
    print("\n[INFO] SLC Primary Range Axis:")
    print(f"  - Number of samples : {numberOfSamples_pri}")
    print(f"  - Range spacing     : {rangeTimeInterval_pri} seconds")
    print(f"  - Max range axis    : {np.max(range_slc_axis_pri_stac):.6f} seconds")
    
    # -------------------------
    # 3) Read phase screens on LUT grid
    # -------------------------

    
    # flattening phase (coreg group)
    flatteningPhaseScreen_co = lut_co.groups['coregistration']['flatteningPhaseScreen'][:].astype(np.float64)
    flatteningPhaseScreen_pri =lut_pri.groups['coregistration']['flatteningPhaseScreen'][:].astype(np.float64)
    
    
    # SKP calibration phase (skpPhaseCalibration group) 
    skpCalibrationPhaseScreen_co = lut_co.groups['skpPhaseCalibration']['skpCalibrationPhaseScreen'][:].astype(np.float64)
    skpCalibrationPhaseScreen_pri =lut_pri.groups['skpPhaseCalibration']['skpCalibrationPhaseScreen'][:].astype(np.float64)
    
    
   
    #######plot flatten ################
    
    
    save_phase_map_lut(
        flatteningPhaseScreen_pri,
        title=f"flatteningPhaseScreen_pri [rad] - {date_primary}",
        date=date_primary,
        out_file=preview_dir / f"flatteningPhaseScreen_pri_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png",)


    save_phase_map_lut(
        flatteningPhaseScreen_co,
        title=f"flatteningPhaseScreen_co [rad]-{date_secondary}",
        date=date_secondary,
        out_file=preview_dir / f"flatteningPhaseScreen_co_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png",)

    #######plot skp ################
    save_phase_map_lut(
            skpCalibrationPhaseScreen_pri,
            title=f"SKP Calibration Phase pri [rad] - {date_primary}",
            date=date_primary,
            out_file=preview_dir / f"skpCalibrationPhaseScreen_pri_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png",)


    save_phase_map_lut(
            skpCalibrationPhaseScreen_co,
            title=f"SKP Calibration Phase Co [rad]-{date_secondary}",
            date=date_secondary,
            out_file=preview_dir / f"skpCalibrationPhaseScreen_co_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png",)
    
    

   
    #preparation of the interpolators
    axes_in = (lut_az_axes_pri, lut_range_axis_pri) # as showed before you can use also the *co axes since they are equal as expteced 
    axes_out = (az_slc_axis_pri_stac, range_slc_axis_pri_stac) # as showed before you can use also the *co axes since they are equal as expteced 
    degree_x = 1
    degree_y = 1
    smoother = 0.0
    bbox=[
                min(np.min(axes_in[0]), np.min(axes_out[0])),
                max(np.max(axes_in[0]), np.max(axes_out[0])),
                max(np.min(axes_in[1]), np.min(axes_out[1])),
                max(np.max(axes_in[1]), np.max(axes_out[1])),
            ]

    # -------------------------------
    # Coregistered flattening screen
    # -------------------------------
    fps_co = flatteningPhaseScreen_co.values if hasattr(flatteningPhaseScreen_co, "values") else flatteningPhaseScreen_co    
    fps_co_spline, fps_co_mask_interp = build_safe_spline(fps_co,axes_in,bbox=bbox, kx=degree_x,ky=degree_y,s=smoother,nodata=nodata)   
    flatteningPhaseScreen_co_upsampled = fps_co_spline(axes_out[0], axes_out[1])    
    validity_co = fps_co_mask_interp(axes_out[0], axes_out[1])
    flatteningPhaseScreen_co_upsampled[validity_co < 0.5] = np.nan
    
    
    # -------------------------
    # Primary flattening screen
    # -------------------------
    fps_pri = flatteningPhaseScreen_pri.values if hasattr(flatteningPhaseScreen_pri, "values") else flatteningPhaseScreen_pri    
    fps_pri_spline, fps_pri_mask_interp = build_safe_spline(fps_pri,axes_in, bbox=bbox, kx=degree_x, ky=degree_y,s=smoother, nodata=nodata)    
    flatteningPhaseScreen_pri_upsampled = fps_pri_spline(axes_out[0], axes_out[1])    
    validity_pri = fps_pri_mask_interp(axes_out[0], axes_out[1])
    
    # Keep only pixels where the interpolated validity is sufficiently high:
    # validity >= 0.5  -> accept the interpolated value as reliable
    # validity <  0.5  -> consider the value unreliable and mask it as NaN
    flatteningPhaseScreen_pri_upsampled[validity_pri < 0.5] = np.nan
    
    

    
    quick_stats("flatteningPhaseScreen_co", fps_co)
    quick_stats("flatteningPhaseScreen_co_upsampled", flatteningPhaseScreen_co_upsampled)
    
    quick_stats("flatteningPhaseScreen_pri", fps_pri)
    quick_stats("flatteningPhaseScreen_pri_upsampled", flatteningPhaseScreen_pri_upsampled)

    # --- upsample SKP come exp(j·skp) ---
    skpCalibrationPhaseScreen_co_upsampled = upsample_phase_via_complex(
        skpCalibrationPhaseScreen_co,
        axes_in=(lut_az_axes_pri, lut_range_axis_pri),
        axes_out=(az_slc_axis_pri_stac, range_slc_axis_pri_stac),
        bbox=bbox, kx=degree_x, ky=degree_y, s=smoother, nodata=nodata
        )

    skpCalibrationPhaseScreen_pri_upsampled = upsample_phase_via_complex(
        skpCalibrationPhaseScreen_pri,
        axes_in=(lut_az_axes_pri, lut_range_axis_pri),
        axes_out=(az_slc_axis_pri_stac, range_slc_axis_pri_stac),
        bbox=bbox, kx=degree_x, ky=degree_y, s=smoother, nodata=nodata
        )
    
    
    
    save_phase_map_slc(flatteningPhaseScreen_pri_upsampled,  f"flatteningPhaseScreen\npri_upsampled [rad] - {date_primary}", date_primary,
                       preview_dir / f"flatteningPhaseScreen_pri_upsampled_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png")
    
    save_phase_map_slc(flatteningPhaseScreen_co_upsampled,   f"flatteningPhaseScreen\nco_upsampled [rad] - {date_secondary}",  date_secondary,
                       preview_dir / f"flatteningPhaseScreen_co_upsampled_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png")
    
    save_phase_map_slc(skpCalibrationPhaseScreen_pri_upsampled,  f"skpCalibrationPhaseScreen\npri_upsampled [rad] - {date_primary}", date_primary,
                       preview_dir / f"skpCalibrationPhaseScreen_pri_upsampled_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png")
    
    save_phase_map_slc(skpCalibrationPhaseScreen_co_upsampled,   f"skpCalibrationPhaseScreen\nco_upsampled [rad] - {date_secondary}",  date_secondary,
                       preview_dir / f"skpCalibrationPhaseScreen_co_upsampled_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png")
    
    
    vmin = np.nanpercentile(flatteningPhaseScreen_co_upsampled, 2)
    vmax = np.nanpercentile(flatteningPhaseScreen_co_upsampled, 98)
    
    flatteningPhaseScreen_co_upsampled = np.ma.masked_equal(flatteningPhaseScreen_co_upsampled.astype(np.float64), nodata)
    
    
    plot_2d_with_hist(
    x=np.arange(flatteningPhaseScreen_co_upsampled.shape[0]),
    y=np.arange(flatteningPhaseScreen_co_upsampled.shape[1]),
    zplot=flatteningPhaseScreen_co_upsampled,   # oppure phase_deg
    cb='rainbow',
    vmin=vmin, vmax=vmax,             # se rad
    title=f"flatteningPhaseScreen\nco_upsampled [rad] - {date_secondary}",
    x_title="Azimuth [pixels]",
    y_title="Range [pixels]",
    cb_title="[rad]",
    height_compression=0.01,  
    nodata=nodata,           
    file_2_save=preview_dir / f"flatteningPhaseScreen_co_upsampled_panel_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png",
    show=False
    ) 
    

    vmin = np.nanpercentile(flatteningPhaseScreen_pri_upsampled, 2)
    vmax = np.nanpercentile(flatteningPhaseScreen_pri_upsampled, 98)
    
    
    flatteningPhaseScreen_pri_upsampled = np.ma.masked_equal(flatteningPhaseScreen_pri_upsampled.astype(np.float64), nodata)    
    plot_2d_with_hist(
    x=np.arange(flatteningPhaseScreen_pri_upsampled.shape[0]),
    y=np.arange(flatteningPhaseScreen_pri_upsampled.shape[1]),
    zplot=flatteningPhaseScreen_pri_upsampled,   # oppure phase_deg
    cb='rainbow',
    vmin=vmin, vmax=vmax,            
    title= f"flatteningPhaseScreen\npri_upsampled [rad] - {date_primary}",
    x_title="Azimuth [pixels]",
    y_title="Range [pixels]",
    cb_title="[rad]",
    height_compression=0.01, 
    nodata=nodata,            
    file_2_save=preview_dir / f"flatteningPhaseScreen_pri_upsampled_panel_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png",
    show=False
    ) 
    
    
    plot_2d_with_hist(
    x=np.arange(skpCalibrationPhaseScreen_co_upsampled.shape[0]),
    y=np.arange(skpCalibrationPhaseScreen_co_upsampled.shape[1]),
    zplot=skpCalibrationPhaseScreen_co_upsampled,   # oppure phase_deg
    cb='rainbow',
    vmin=-np.pi, vmax=np.pi,             # se rad
    # vmin=-180, vmax=180,               # se deg
    title="SKP Calibration Phase Co_upsampled [rad]",
    x_title="Azimuth [pixels]",
    y_title="Range [pixels]",
    cb_title="[rad]",
    height_compression=0.01, 
    nodata=nodata,             
    file_2_save=preview_dir /
        f"skpCalibrationPhaseScreen_co_upsampled_panel_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png",
    show=False
    ) 
    
    
    plot_2d_with_hist(
    x=np.arange(skpCalibrationPhaseScreen_pri_upsampled.shape[0]),
    y=np.arange(skpCalibrationPhaseScreen_pri_upsampled.shape[1]),
    zplot=skpCalibrationPhaseScreen_pri_upsampled,   # oppure phase_deg
    cb='rainbow',
    vmin=-np.pi, vmax=np.pi,             # se rad
    # vmin=-180, vmax=180,               # se deg
    title="SKP Calibration Phase pri upsampled [rad]",
    x_title="Azimuth [pixels]",
    y_title="Range [pixels]",
    cb_title="[rad]",
    height_compression=0.01,  
    nodata=nodata,            
    file_2_save=preview_dir /
        f"skpCalibrationPhaseScreen_pri_upsampled_panel_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png",
    show=False
    ) 
    
 
    ###########################################################################
        
    # -------------------------
    # 6) Apply requested correction policy
    # -------------------------
    # Build total per-image corrections 'corr_pri' and 'corr_co'
    
    flatteningPhaseScreen_pri_upsampled.min()
    flatteningPhaseScreen_pri_upsampled.max()

    corr_pri = 0.0
    corr_co  = 0.0
    
    if flatten == "None":
        # processor did NOT apply any screen -> we must compensate (flattening + skp)
        corr_pri = flatteningPhaseScreen_pri_upsampled + skpCalibrationPhaseScreen_pri_upsampled
        corr_co  = flatteningPhaseScreen_co_upsampled  + skpCalibrationPhaseScreen_co_upsampled
        
        print("[INFO] phaseCorrection: None -> applying (flattening + skp)")
        
    elif flatten == "geometry":
        # processor applied only the geometry/DSI screen -> we add SKP only)
        corr_pri = skpCalibrationPhaseScreen_pri_upsampled
        corr_co  = skpCalibrationPhaseScreen_co_upsampled
        
        
        print("[INFO] phaseCorrection: Flattening-only -> applying SKP only")
        
    elif flatten == "skp":
        # processor applied the full screen -> no correction here
        corr_pri = 0.0
        corr_co  = 0.0
        print("[INFO] phaseCorrection: Ground Phase (full) -> no additional correction")
    else:
        raise ValueError(f"Unknown flatten option: {flatten}")    



    # -------------------------
    # 1) Load STAs (amp & phase) and build complex images
    # -------------------------
    nan_value = sta_info["nodata"]
    pol_to_band_index = get_pol_to_band_index(data_primary_abs)
    
    # ------------------------------------------------------------------
    # Preview subfolders per polarization
    # ------------------------------------------------------------------
    preview_pol_dirs = {}
       
    pol_plots = {}    
    for pol_raw in polarizations:
        
        pol_out = canonical_pol_name(pol_raw)
        pol_dir = preview_dir / pol_out
        pol_dir.mkdir(exist_ok=True)
        preview_pol_dirs[pol_out] = pol_dir
        print(f"[INFO] Preview folder for {pol_out}: {pol_dir}")
        pol_plots[pol_out] = {}
        pol_plots[pol_out]["plots"] = {}
        pol_plots[pol_out]["stats"] = {}
        channel = pol_to_band_index[pol_raw]   
        print(f"\n[INFO] Processing polarization {pol_out} (band {channel})")
        
       
        # -------------------------
        # Load PRIMARY SLC
        # -------------------------
        with rasterio.open(data_primary_abs) as ds:
            amp_pri = ds.read(channel)
            amp_pri = np.ma.masked_equal(amp_pri, nan_value)
    
        with rasterio.open(data_primary_phase) as ds:
            phase_pri = ds.read(channel)
            phase_pri = np.ma.masked_equal(phase_pri, nan_value)
    
        # -------------------------
        # Load SECONDARY SLC
        # -------------------------
        with rasterio.open(data_secondary_abs) as ds:
            amp_sec = ds.read(channel)
            amp_sec = np.ma.masked_equal(amp_sec, nan_value)
    
        with rasterio.open(data_secondary_phase) as ds:
            phase_sec = ds.read(channel)
            phase_sec = np.ma.masked_equal(phase_sec, nan_value)
            
            
            
        valid_pri = np.isfinite(amp_pri) & np.isfinite(phase_pri)
        valid_pri &= (amp_pri != nan_value) & (phase_pri != nan_value)
        
        valid_sec = np.isfinite(amp_sec) & np.isfinite(phase_sec)
        valid_sec &= (amp_sec != nan_value) & (phase_sec != nan_value)
        
        valid_slc = valid_pri & valid_sec   # pixel validi in entrambe
    
        # -------------------------
        # Build complex images
        # -------------------------
        
        primary   = (amp_pri * np.exp(1j * phase_pri)).astype(np.complex64)
        secondary = (amp_sec * np.exp(1j * phase_sec)).astype(np.complex64)
        
        primary[~valid_slc] = np.nan + 1j*np.nan
        secondary[~valid_slc] = np.nan + 1j*np.nan
        
        # -------------------------
        # Debug stats
        # -------------------------
        print(f"[INFO] PRIMARY ({pol_raw})")
        print(f"  - Max amplitude: {np.nanmax(amp_pri)}")
        print(f"  - Max phase:     {np.nanmax(phase_pri)}")
    
        print(f"[INFO] SECONDARY ({pol_raw})")
        print(f"  - Max amplitude: {np.nanmax(amp_sec)}")
        print(f"  - Max phase:     {np.nanmax(phase_sec)}")
        
        # -------------------------
        # 7) Interferogram & Coherence
        # -------------------------
        
        interferogram = primary * np.conj(secondary)
        interferogram_flat = (primary  * np.exp(1j * corr_pri)) * np.conj(secondary * np.exp(1j * corr_co))
    
        phase_deg = np.angle(interferogram, deg=True)
        phase_flat_deg = np.angle(interferogram_flat, deg=True)    
        
        #########################################################################

        shape = phase_deg.shape
        x = np.arange(shape[0])
        y = np.arange(shape[1])
    
        # --------------------------------------------
        # Phase before flattening with histogram
        # --------------------------------------------
        plot_2d_with_hist(
            x=x,
            y=y,
            zplot=phase_deg,
            cb='rainbow',
            vmin=-180,
            vmax=180,
            title='Interferometric Phase [deg]',
            x_title='Azimuth [pixels]',
            y_title='Range [pixels]',
            cb_title='[deg]',
            height_compression=0.01,
            file_2_save=pol_dir /
                f"interferogram_phase_deg_{date_primary}_{date_secondary}_{number_frame}_{flatten}_{pol_out}.png",
            show=False
        )
        pol_plots[pol_out]["plots"]["interferogram_phase"] = (pol_dir /f"interferogram_phase_deg_{date_primary}_{date_secondary}_{number_frame}_{flatten}_{pol_out}.png")            
        # --------------------------------------------
        # Phase after flattening with histogram
        # --------------------------------------------
        plot_2d_with_hist(
            x=x,
            y=y,
            zplot=phase_flat_deg,
            cb='rainbow',
            vmin=-180,
            vmax=180,
            title='Interferometric Flattened Phase [deg]',
            x_title='Azimuth [pixels]',
            y_title='Range [pixels]',
            cb_title='[deg]',
            height_compression=0.01,
            file_2_save=pol_dir /
                f"interferogram_phase_flat_deg_{date_primary}_{date_secondary}_{number_frame}_{flatten}_{pol_out}.png",
            show=False
           )
        pol_plots[pol_out]["plots"]["interferogram_phase_flat"] = (pol_dir /f"interferogram_phase_flat_deg_{date_primary}_{date_secondary}_{number_frame}_{flatten}_{pol_out}.png")            
    
        
        #########################################################################
        
        valid_slc = ( np.isfinite(primary) & np.isfinite(secondary) & (primary != 0) & (secondary != 0))
        #coh = single_baseline_single_pol_coh(primary,secondary * np.exp(1j * (corr_co - corr_pri)),(5,5))   
        
        coh_abs, coh_phase = single_baseline_single_pol_coh( primary,secondary * np.exp(1j * (corr_co - corr_pri)), window=(5,5), valid_mask=valid_slc, min_valid_frac=0.8) 
        coh_phase = np.rad2deg(coh_phase)
        #coh_abs = np.abs(coh)
        #coh_phase = np.angle(coh, deg=True)

        vals_stats = coh_abs[np.isfinite(coh_abs)]

        pol_plots[pol_out]["stats"] = {
            "coh_mean":   float(np.mean(vals_stats)),
            "coh_std":    float(np.std(vals_stats)),
            "coh_min":    float(np.min(vals_stats)),
            "coh_max":    float(np.max(vals_stats)),
            "coh_median": float(np.median(vals_stats)),
        }
        
        
        '''
        pol_plots[pol]["stats"] = {
            "coh_mean":   float(np.nanmean(coh_abs)),
            "coh_std":    float(np.nanstd(coh_abs)),
            "coh_min":    float(np.nanmin(coh_abs)),
            "coh_max":    float(np.nanmax(coh_abs)),
            "coh_median": float(np.nanmedian(coh_abs)),}
        '''
        plot_2d_with_hist(
            np.arange(coh_abs.shape[0]),
            np.arange(coh_abs.shape[1]),
            coh_abs,
            cb='Greys_r',
            vmin=0.00001, vmax=0.99999,
            title='|coh|',
            x_title='Azimuth [pixels]',
            y_title='Range [pixels]',
            cb_title='Amplitude',
            height_compression=0.01,   
            file_2_save=pol_dir / f"coh_amp_{date_primary}_{date_secondary}_{number_frame}_{flatten}_{pol_out}.png"
        )
    
        pol_plots[pol_out]["plots"]["coh_amp"] = (pol_dir /f"coh_amp_{date_primary}_{date_secondary}_{number_frame}_{flatten}_{pol_out}.png") 
        
        
        plot_2d_with_hist(
            np.arange(coh_phase.shape[0]),
            np.arange(coh_phase.shape[1]),
            coh_phase,
            cb='rainbow',
            vmin=-180, vmax=180,
            title='arg(coh) [deg]',
            x_title='Azimuth [pixels]',
            y_title='Range [pixels]',
            cb_title='[deg]',
            height_compression=0.01,
            file_2_save=pol_dir / f"coh_phase_{date_primary}_{date_secondary}_{number_frame}_{flatten}_{pol_out}.png"
        )
       
        pol_plots[pol_out]["plots"]["coh_phase"] = (pol_dir /f"coh_phase_{date_primary}_{date_secondary}_{number_frame}_{flatten}_{pol_out}.png") 
        
        
        print(np.mean(coh_abs), np.median(coh_abs))
    
    

        if not is_light:

            # riferimento STA (mantiene GCP corretti)
            ref_tiff = data_primary_abs
        
            # --- Coherence amplitude ---
            coh_abs_out = measurement_dir / f"coh_abs_{date_primary}_{date_secondary}_{number_frame}_{flatten}_{pol_out}.tif"
            save_tiff_with_gcps(
                coh_abs_out,
                coh_abs.astype(np.float32),
                ref_tiff
            )
        
            # --- Coherence phase ---
            coh_phase_out = measurement_dir / f"coh_phase_{date_primary}_{date_secondary}_{number_frame}_{flatten}_{pol_out}.tif"
            save_tiff_with_gcps(
                coh_phase_out,
                coh_phase.astype(np.float32),
                ref_tiff
            )
        
         # -------------------------------------------------------------------------
         # 4. Save "clean" PNG maps (without axes or colorbar)
         # -------------------------------------------------------------------------
        print("[INFO] Saving clean PNG maps...")
     
        coh_abs_png = pol_dir / f"coherence_abs_{date_primary}_{date_secondary}_{number_frame}_{flatten}_kml_{pol_out}.png"
        pol_plots[pol_out]["plots"]["coh_abs_png"] = (pol_dir /f"coherence_abs_{date_primary}_{date_secondary}_{number_frame}_{flatten}_kml_{pol_out}.png")
        save_clean_image(coh_abs, cmap='Greys_r', out_path=coh_abs_png, vmin=0, vmax=np.nanmax(coh_abs))
        make_white_transparent(coh_abs_png)


    # ======================================================================
    # GLOBAL INFO (baseline + metadata run)
    # ======================================================================
    baseline_info = listinfobaseline[0] if listinfobaseline else {}
    
    global_info = {}
    global_info.update(baseline_info)
    global_info["coregistrationMethod_primary"] = coregistrationMethod_pri
    global_info["frame"] = int(number_frame)
    global_info["phaseCorrectionMode"] = flatten
    global_info["date_primary"] = date_primary
    global_info["date_secondary"] = date_secondary
    
    # ======================================================================
    # Reference polarization (prefer VV)
    # ======================================================================
    ref_pol = "VV" if "VV" in pol_plots else list(pol_plots.keys())[0]
    vv_stats = pol_plots[ref_pol]["stats"]
    
    # ======================================================================
    # Popup statistics = coherence stats for ALL polarizations + global info
    # ======================================================================
    coh_stats_popup = {
        "Mean coh abs":   {},
        "Std coh abs":    {},
        "Min coh abs":    {},
        "Max coh abs":    {},
        "Median coh abs": {}
    }

    for pol, pol_data in pol_plots.items():
        stats = pol_data["stats"]

        coh_stats_popup["Mean coh abs"][pol]   = stats["coh_mean"]
        coh_stats_popup["Std coh abs"][pol]    = stats["coh_std"]
        coh_stats_popup["Min coh abs"][pol]    = stats["coh_min"]
        coh_stats_popup["Max coh abs"][pol]    = stats["coh_max"]
        coh_stats_popup["Median coh abs"][pol] = stats["coh_median"]

    # append global (non-polarimetric) info
    for k, v in global_info.items():
        coh_stats_popup[k] = v
    
    # ======================================================================
    # KML
    # ======================================================================
    kmz_overlay = preview_dir / \
        f"{pol_plots[ref_pol]['plots']['coh_abs_png'].stem}.kmz"
    
    make_overlay_kmz_with_quad(
        kmz_out=kmz_overlay,
        png_path=pol_plots[ref_pol]["plots"]["coh_abs_png"],
        preview_kml_file=preview_kml_sec,
        pol_plots=pol_plots,
        coh_stats=coh_stats_popup,
        flatteningPhaseScreen_co_file=      preview_dir /f"flatteningPhaseScreen_co_upsampled_panel_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png",
        skpCalibrationPhaseScreen_co_file=  preview_dir /f"skpCalibrationPhaseScreen_co_upsampled_panel_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png",
        coh_low_thr=coh_low_thr,
        coh_high_thr=coh_high_thr,
        primary_name=Path(path_primary).name,
        secondary_name=Path(path_secondary).name,
    )
    
    print(f"[OK] KML created: {kmz_overlay}")
    
    # ======================================================================
    # PDF
    # ======================================================================
    sta_plots = {
        "SKP phase screen primary":                     preview_dir / f"skpCalibrationPhaseScreen_pri_upsampled_panel_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png",
        "SKP phase screen secondary":                   preview_dir / f"skpCalibrationPhaseScreen_co_upsampled_panel_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png",
        "Flattening phase screen primary":              preview_dir / f"flatteningPhaseScreen_pri_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png",
        "Flattening phase screen secondary":            preview_dir / f"flatteningPhaseScreen_co_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png",
        "Flattening phase screen primary unsampled":    preview_dir / f"flatteningPhaseScreen_pri_upsampled_panel_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png",
        "Flattening phase screen secondary unsampled":  preview_dir / f"flatteningPhaseScreen_co_upsampled_panel_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png",
    }
    
    generate_interferogram_pdf_report(
        output_pdf=         annotation_dir /f"{output_folder_check_interferogram.name}.pdf",
        primary_name=       Path(path_primary).name,
        secondary_name=     Path(path_secondary).name,
        coh_stats_global=   coh_stats_popup,
        pol_plots=          pol_plots,
        sta_plots=          sta_plots,
        flatten=            flatten,
        skpPhaseCalibrationFlag=skpPhaseCalibrationFlag,
        skpPhaseCorrectionFlag= skpPhaseCorrectionFlag,
        skpPhaseCorrectionFlatteningOnlyFlag=skpPhaseCorrectionFlatteningOnlyFlag,
    )
    
    global_info["primaryProduct"] = Path(path_primary).name
    global_info["secondaryProduct"] = Path(path_secondary).name
    
    write_interferogram_annotation_xml(
        out_dir=annotation_dir,
        product_name=output_folder_check_interferogram.name,
        global_info=global_info,
        pol_plots=pol_plots,
        ref_pol=ref_pol)    
  
    # --------------------------------------------------
    # Copy secondary STA annotation XML (same filename)
    # --------------------------------------------------
    try:
        sec_ann_src = Path(path_main_ann_coregistered)
        sec_ann_dst = annotation_dir / sec_ann_src.name

        shutil.copy(sec_ann_src, sec_ann_dst)

        print(f"[OK] Secondary STA annotation copied: {sec_ann_dst}")

    except Exception as e:
        print(f"[WARN] Failed to copy secondary STA annotation: {e}")
    
    
    
    
      
    try:
         lut_co.close()
    except Exception:
         pass
    try:
         lut_pri.close()
    except Exception:
         pass




def read_sta_primary_measurement_info(tiff_path):
    """
    Read structural information from the STA primary measurement TIFF.
    Returns a dictionary with:
      - polarizations
      - band_to_pol
      - nodata
      - software
      - gcps
    """

    
    ds = gdal.Open(str(tiff_path), gdal.GA_ReadOnly)

    if ds is None:
        raise RuntimeError(f"Cannot open STA measurement TIFF: {tiff_path}")

    info = {}

    # --------------------------------------------------
    # Global metadata
    # --------------------------------------------------
    md = ds.GetMetadata()
    info["software"] = md.get("TIFFTAG_SOFTWARE", "UNKNOWN")

    # --------------------------------------------------
    # NoData (assumiamo coerente su tutte le bande)
    # --------------------------------------------------
    b1 = ds.GetRasterBand(1)
    info["nodata"] = b1.GetNoDataValue()

    # --------------------------------------------------
    # Polarizations per banda
    # --------------------------------------------------
    band_to_pol = {}
    polarizations = []

    for i in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(i)
        bmd = band.GetMetadata()
        pol = bmd.get("POLARIMETRIC_INTERP")

        if pol:
            band_to_pol[i] = pol
            polarizations.append(pol)

    info["band_to_pol"] = band_to_pol
    info["polarizations"] = sorted(set(polarizations))

    # --------------------------------------------------
    # GCPs
    # --------------------------------------------------
    gcps = ds.GetGCPs()
    gcp_list = []

    for g in gcps:
        gcp_list.append({
            "pixel": g.GCPPixel,
            "line": g.GCPLine,
            "lon": g.GCPX,
            "lat": g.GCPY,
            "height": g.GCPZ,
        })

    info["gcps"] = gcp_list
    info["gcp_projection"] = ds.GetGCPProjection()

    ds = None
    return info


def get_primary(path_stacks_folder):
    pattern = os.path.join(path_stacks_folder, "*STA__1S*")
    for p in sorted(glob.glob(pattern)):
        sta_path = Path(p)
        sta_product = BiomassProduct.BiomassProductSTA(sta_path)
        if sta_product.isCoregistrationPrimary:
            print(f"[INFO] Found primary STA: {sta_path}")
            return sta_path

def main(path_stacks_folder,
         mode="light",
         coh_low_thr=COH_LOW_THRESHOLD,
         coh_high_thr=COH_HIGH_THRESHOLD,
         polarizations_requested=None):

    is_light = mode.lower() == "light"

    print("----------------------------------------------------------------------------")
    print(f"Mode selected: {'light' if is_light else 'all'}")
    print("----------------------------------------------------------------------------")

    root = Path(path_stacks_folder)

    # --------------------------------------------------
    # Find all STA products
    # --------------------------------------------------
    pattern = str(root / "*STA__1S*")
    sta_paths = sorted(Path(p) for p in glob.glob(pattern))

    print(f"Found {len(sta_paths)} STA products:\n")
    pprint.pprint([p.name for p in sta_paths])

    if not sta_paths:
        raise RuntimeError("No STA products found.")

    # --------------------------------------------------
    # Identify PRIMARY
    # --------------------------------------------------
    sta_primary = get_primary(path_stacks_folder)
    print(f"[INFO] Using PRIMARY STA: {sta_primary.name}")

    primary_product = BiomassProduct.BiomassProductSTA(sta_primary)
    number_frame = primary_product.wrsLatitudeGrid

    # --------------------------------------------------
    # Loop over SECONDARIES
    # --------------------------------------------------
    for sta_secondary in sta_paths:

        if sta_secondary == sta_primary:
            continue   # skip primary

        print("----------------------------------------------------------------------------")
        print(f"Processing secondary: {sta_secondary.name}")
        print("----------------------------------------------------------------------------")

        secondary_product = BiomassProduct.BiomassProductSTA(sta_secondary)

        skpPhaseCalibrationFlag = secondary_product.skpPhaseCalibrationFlag
        skpPhaseCorrectionFlag = secondary_product.skpPhaseCorrectionFlag
        skpPhaseCorrectionFlatteningOnlyFlag = (
            secondary_product.skpPhaseCorrectionFlatteningOnlyFlag
        )

        # --------------------------------------------------
        # Phase correction mode
        # --------------------------------------------------
        if skpPhaseCorrectionFlag:
            if skpPhaseCorrectionFlatteningOnlyFlag:
                flatten = "geometry"
            else:
                flatten = "skp"
        else:
            flatten = "None"

        print(f"[INFO] phaseCorrection mode = {flatten}")

        print(
            f"[OK] Matches found:\n"
            f" - PRIMARY:   {sta_primary.name}\n"
            f" - SECONDARY: {sta_secondary.name}"
        )

        # --------------------------------------------------
        # Run interferogram
        # --------------------------------------------------
        check_interferogram(
            sta_primary,
            sta_secondary,
            flatten,
            is_light,
            number_frame,
            coh_low_thr,
            coh_high_thr,
            skpPhaseCalibrationFlag,
            skpPhaseCorrectionFlag,
            skpPhaseCorrectionFlatteningOnlyFlag,
            polarizations_requested)

    



def print_help():
    """
    Print command-line usage instructions for the script.
    """
    help_message = """
USAGE
    python check_interferogram.py <path_stacks_folder> [options]

DESCRIPTION
    For each BIOMASS STA product in the input folder, the script identifies
    the primary and secondary STA images and generates interferometric
    and coherence products.

    Processing can be restricted to specific polarizations.
    If no polarization is specified, all available polarizations
    in the product are processed.

ARGUMENTS
    <path_stacks_folder>
        Path to the folder containing STA product directories.

OPTIONS
    --mode {light,all}
        Processing mode.
        light : generate PNG previews only (default)
        all   : generate PNG previews and GeoTIFF outputs

    --coh-low <float>
        Low threshold for coherence color mapping.
        Default: 0.35

    --coh-high <float>
        High threshold for coherence color mapping.
        Default: 0.80

    --pol, --polarizations <list>
        Comma-separated list of polarizations to process.
        Example: VV or VV,HH
        If omitted, all polarizations are processed.

EXAMPLES
    # Default processing (all polarizations, light mode)
    python check_interferogram.py /data/biomass/STA

    # Only VV polarization
    python check_interferogram.py /data/biomass/STA --pol VV

    # VV and HH polarizations
    python check_interferogram.py /data/biomass/STA --pol VV,HH

    # Full processing, VV only
    python check_interferogram.py /data/biomass/STA --mode all --pol VV
"""
    print(help_message)
    
    
    
    



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Interferogram and coherence analysis for BIOMASS STA products"
    )

    parser.add_argument(
        "path_stacks_folder",
        help="Path to the folder containing STA product directories"
    )

    parser.add_argument(
        "--mode",
        choices=["light", "all"],
        default="light",
        help="Processing mode (default: light)"
    )

    parser.add_argument(
        "--coh-low",
        type=float,
        default=COH_LOW_THRESHOLD,
        help="Low coherence threshold (default: 0.35)"
    )

    parser.add_argument(
        "--coh-high",
        type=float,
        default=COH_HIGH_THRESHOLD,
        help="High coherence threshold (default: 0.80)"
    )

    parser.add_argument(
        "--pol",
        "--polarizations",
        dest="polarizations",
        help="Comma-separated list of polarizations to process (e.g. VV or VV,HH, XP)"
    )
    

    args = parser.parse_args()

    # --- polarization parsing ---
    if args.polarizations:
        pol_list = [p.strip().upper() for p in args.polarizations.split(",")]
        print(f"[INFO] Requested polarizations: {pol_list}")
    else:
        pol_list = None

    print(f"[INFO] Coherence thresholds: LOW={args.coh_low}, HIGH={args.coh_high}")

    main(
        args.path_stacks_folder,
        args.mode,
        args.coh_low,
        args.coh_high,
        pol_list
    )






