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
import matplotlib.pyplot as plt
import numpy.typing as npt
import scipy
import sys
from PIL import Image
from pathlib import Path
import copy
import os
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import re
import glob
import pprint

import shutil
import subprocess


import zipfile

# --- COHERENCE COLOR THRESHOLD CONFIGURATION ---
COH_LOW_THRESHOLD  = 0.35   
COH_HIGH_THRESHOLD = 0.80

def save_colorbar_png(out_path,
                      cmap='viridis',
                      vmin=0.0, vmax=1.0,
                      ticks=(0.0, 0.5, 1.0),
                      label='Coherence amplitude'):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    fig, ax = plt.subplots(figsize=(1.0, 4.0))
    norm = Normalize(vmin=vmin, vmax=vmax)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap)),
                      cax=ax, orientation='vertical')
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:.1f}" for t in ticks])
    cb.set_label(label)
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def save_clean_image(data, cmap='RdBu', out_path=None, vmin=None, vmax=None):

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    plt.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)


def make_overlay_kmz_with_quad(kmz_out,
                               png_path,
                               preview_kml_file,
                               colorbar_png_path=None,
                               coh_stats=None,
                               coh_phase_file=None,
                               Hist_skpCalibrationPhaseScreen_co_file=None,
                               Hist_skpCalibrationPhaseScreen_pri_file=None,
                               flatteningPhaseScreen_co_file=None,
                               skpCalibrationPhaseScreen_co_file=None,
                               phase_flat_deg_file=None,
                               coh_low_thr=COH_LOW_THRESHOLD,
                               coh_high_thr=COH_HIGH_THRESHOLD):
    """
    Creates a KMZ containing:
      - A georeferenced GroundOverlay (gx:LatLonQuad)
      - A round pin at the center, colored according to the mean coherence
      - A popup with statistics and side-by-side images arranged in tables
      - All resources packaged inside the KMZ
    """


    pprint.pprint(coh_stats)
    ns = {"kml": "http://www.opengis.net/kml/2.2",
          "gx": "http://www.google.com/kml/ext/2.2"}

    png_path = Path(png_path)
    preview_kml_file = Path(preview_kml_file)
    if colorbar_png_path:
        colorbar_png_path = Path(colorbar_png_path)

    # -------------------------------------------------------------------------
    # 1) Read the quad from the preview KML
    # -------------------------------------------------------------------------
    root = ET.parse(preview_kml_file).getroot()
    quad = root.find(".//gx:LatLonQuad", ns)
    if quad is None:
        raise ValueError(f"gx:LatLonQuad non trovato in {preview_kml_file}")

    coords_text = quad.find("kml:coordinates", ns) or quad.find("coordinates")
    coords_str = coords_text.text.strip()

    coords_list = [c for c in coords_str.split()]
    if coords_list[0] != coords_list[-1]:
        coords_poly = " ".join(coords_list + [coords_list[0]])
    else:
        coords_poly = " ".join(coords_list)

    # -------------------------------------------------------------------------
    # 2) Compute the geometric center of the quadrilateral
    # -------------------------------------------------------------------------

    def parse_coord(c):
        lon, lat, *_ = c.split(",")
        return float(lon), float(lat)

    lon_list, lat_list = zip(*[parse_coord(c) for c in coords_list])
    center_lon = sum(lon_list) / len(lon_list)
    center_lat = sum(lat_list) / len(lat_list)
    center_coord = f"{center_lon},{center_lat},0"

    # -------------------------------------------------------------------------
    # 3) Compute the pin color based on the mean coherence
    # ------------------------------------------------------------------------
    mean_coh = None
    if coh_stats and "Mean coh abs" in coh_stats:
        try:
            
            mean_coh = float(coh_stats["Mean coh abs"])
        except Exception:
            mean_coh = None

    def interpolate_color(value):
        """Interpolazione da viola (0) a verde (1)."""
        v = max(0.0, min(1.0, value))
        r = int((1 - v) * 128)   # meno rosso
        g = int(v * 255)         # più verde
        b = int((1 - v) * 255)   # da viola a verde
        # formato AABBGGRR (AA=trasparenza)
        return f"ff{b:02x}{g:02x}{r:02x}"

    pin_color = interpolate_color(mean_coh) if mean_coh is not None else "ffffffff"

    # -------------------------------------------------------------------------
    # 4) Create HTML of popup
    # -------------------------------------------------------------------------
    desc_html = "<h3>Coherence Analysis</h3>"
    if coh_stats:
        desc_html += "<ul>"
        for k, v in coh_stats.items():
            try:
                val = f"{float(v):.4f}"
            except Exception:
                val = str(v)
            desc_html += f"<li><b>{k}</b>: {val}</li>"
        desc_html += "</ul>"

    attachments = [png_path]

    # --- Coherence phase ---
    if coh_phase_file and Path(coh_phase_file).exists():
        attachments.append(Path(coh_phase_file))
        desc_html += f'<p><b>Coherence phase</b><br><img src="{Path(coh_phase_file).name}" width="420"/></p>'

    # --- Side-by-side Istogrammi  ---
    if Hist_skpCalibrationPhaseScreen_co_file and Path(Hist_skpCalibrationPhaseScreen_co_file).exists() and \
       Hist_skpCalibrationPhaseScreen_pri_file and Path(Hist_skpCalibrationPhaseScreen_pri_file).exists():
        attachments.extend([Path(Hist_skpCalibrationPhaseScreen_co_file), Path(Hist_skpCalibrationPhaseScreen_pri_file)])
        desc_html += (
            '<h4>SKP Calibration Phase Screen Histograms</h4>'
            '<table style="border-collapse:collapse; text-align:center;"><tr>'
            f'<td><img src="{Path(Hist_skpCalibrationPhaseScreen_co_file).name}" width="300"/></td>'
            f'<td><img src="{Path(Hist_skpCalibrationPhaseScreen_pri_file).name}" width="300"/></td>'
            '</tr></table>'
        )

    # --- Side-by-side phase screens ---
    if flatteningPhaseScreen_co_file and Path(flatteningPhaseScreen_co_file).exists() and \
       skpCalibrationPhaseScreen_co_file and Path(skpCalibrationPhaseScreen_co_file).exists():
        attachments.extend([Path(flatteningPhaseScreen_co_file), Path(skpCalibrationPhaseScreen_co_file)])
        desc_html += (
            '<h4>Flattening & SKP Calibration Phase Screens</h4>'
            '<table style="border-collapse:collapse; text-align:center;"><tr>'
            f'<td><img src="{Path(flatteningPhaseScreen_co_file).name}" width="300"/></td>'
            f'<td><img src="{Path(skpCalibrationPhaseScreen_co_file).name}" width="300"/></td>'
            '</tr></table>'
        )

    # --- Phase flat (deg) ---
    if phase_flat_deg_file and Path(phase_flat_deg_file).exists():
        attachments.append(Path(phase_flat_deg_file))
        desc_html += f'<p><b>Phase flat (deg)</b><br><img src="{Path(phase_flat_deg_file).name}" width="420"/></p>'

    # -------------------------------------------------------------------------
    # 5) build KML
    # -------------------------------------------------------------------------
    kml_parts = []
    kml_parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    kml_parts.append('<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2">')
    kml_parts.append('  <Document>')
    kml_parts.append(f'    <name>{png_path.stem}</name>')


    
    def coh_to_kml_argb(mean_coh,
                        low_thr=coh_low_thr,
                        high_thr=coh_high_thr):
        """
        Restituisce un colore KML (AABBGGRR) in base alla coerenza media (sfumata da rosso→verde).
        low_thr = limite inferiore (rosso)
        high_thr = limite superiore (verde)
        """
    
        try:
            v = float(mean_coh)
        except Exception:
            print(f"[WARN] valore coerenza non numerico: {mean_coh!r}, imposto 0.0")
            v = 0.0
    
        
        print(f"[DEBUG] mean_coh={v:.3f}  low_thr={low_thr}  high_thr={high_thr}")
    
        # Normalize to [0, 1] between low_thr and high_thr
        if v <= low_thr:
            t = 0.0
        elif v >= high_thr:
            t = 1.0
        else:
            t = (v - low_thr) / (high_thr - low_thr)
    
        print(f"[DEBUG] t normalizzato = {t:.3f}")
    
        # RGB interpolation from red (255, 0, 0) to green (0, 255, 0)
        r = int(255 * (1 - t))
        g = int(255 * t)
        b = 0
        a = 255
    
        # KML usa AABBGGRR (ordine invertito)
        color = f"{a:02x}{b:02x}{g:02x}{r:02x}"
        print(f"[DEBUG] colore KML = {color}")
        return color
        
    color_hex = coh_to_kml_argb(mean_coh or 0.0)
    
    kml_parts.append('    <Style id="colored_pin">')
    kml_parts.append('      <IconStyle>')
    kml_parts.append(f'        <color>{color_hex}</color>')
    kml_parts.append('        <scale>1.8</scale>')
    kml_parts.append('        <Icon>')
    kml_parts.append('          <href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href>')
    kml_parts.append('        </Icon>')
    kml_parts.append('      </IconStyle>')
    kml_parts.append('    </Style>')

    # --- Transparent polygon style ---
    kml_parts.append('    <Style id="poly_clickable">')
    kml_parts.append('      <LineStyle><color>00ffffff</color><width>0</width></LineStyle>')
    kml_parts.append('      <PolyStyle><color>01ffffff</color><fill>1</fill><outline>0</outline></PolyStyle>')
    kml_parts.append('      <BalloonStyle><text><![CDATA[$[description]]]></text></BalloonStyle>')
    kml_parts.append('    </Style>')

    # --- GroundOverlay ---
    kml_parts.append('    <GroundOverlay>')
    kml_parts.append(f'      <name>{png_path.stem}</name>')
    kml_parts.append('      <drawOrder>-1</drawOrder>')
    kml_parts.append('      <Icon>')
    kml_parts.append(f'        <href>{png_path.name}</href>')
    kml_parts.append('      </Icon>')
    kml_parts.append('      <gx:LatLonQuad>')
    kml_parts.append(f'        <coordinates>{coords_str}</coordinates>')
    kml_parts.append('      </gx:LatLonQuad>')
    kml_parts.append('    </GroundOverlay>')

    # --- Placemark with centered and colored pin ---
    kml_parts.append('    <Placemark>')
    kml_parts.append('      <visibility>1</visibility>')
    pprint.pprint(coh_stats)
    mean_coh_val = coh_stats.get("Mean coh abs") if coh_stats else None
    norm_baseline_val = coh_stats.get("normalBaseline_secondary") if coh_stats else None
    print(mean_coh_val)
    print(norm_baseline_val)
    mean_coh_val = f"{float(mean_coh_val):.3f}"
    norm_baseline_val = f"{float(norm_baseline_val):.1f} m"
    pin_label = f"{mean_coh_val} | {norm_baseline_val}"
    kml_parts.append(f'      <name>{pin_label}</name>')
    
    kml_parts.append('      <styleUrl>#colored_pin</styleUrl>')
    kml_parts.append(f'      <description><![CDATA[{desc_html}]]></description>')
    kml_parts.append('      <Point>')
    kml_parts.append(f'        <coordinates>{center_coord}</coordinates>')
    kml_parts.append('      </Point>')
    kml_parts.append('    </Placemark>')

    # --- Clickable polygon for the popup area ---
    kml_parts.append('    <Placemark>')
    kml_parts.append(f'      <name>{png_path.stem}_area</name>')
    kml_parts.append('      <styleUrl>#poly_clickable</styleUrl>')
    kml_parts.append(f'      <description><![CDATA[{desc_html}]]></description>')
    kml_parts.append('      <Polygon>')
    kml_parts.append('        <altitudeMode>clampToGround</altitudeMode>')
    kml_parts.append('        <outerBoundaryIs><LinearRing><tessellate>1</tessellate>')
    kml_parts.append(f'          <coordinates>{coords_poly}</coordinates>')
    kml_parts.append('        </LinearRing></outerBoundaryIs>')
    kml_parts.append('      </Polygon>')
    kml_parts.append('    </Placemark>')


    if colorbar_png_path and colorbar_png_path.exists():
        attachments.append(colorbar_png_path)
        kml_parts.append('    <ScreenOverlay>')
        kml_parts.append('      <name>Colorbar</name>')
        kml_parts.append(f'      <Icon><href>{colorbar_png_path.name}</href></Icon>')
        kml_parts.append('      <overlayXY x="0" y="0.5" xunits="fraction" yunits="fraction"/>')
        kml_parts.append('      <screenXY  x="0.98" y="0.5" xunits="fraction" yunits="fraction"/>')
        kml_parts.append('      <size x="-1" y="0.4" xunits="fraction" yunits="fraction"/>')
        kml_parts.append('    </ScreenOverlay>')

    kml_parts.append('  </Document>')
    kml_parts.append('</kml>')

    # -------------------------------------------------------------------------
    # 6) write KMZ
    # -------------------------------------------------------------------------
    kml_text = "\n".join(kml_parts)
    with zipfile.ZipFile(kmz_out, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml_text)
        seen = set()
        for p in attachments:
            if p and Path(p).exists() and p.name not in seen:
                zf.write(p, arcname=p.name)
                seen.add(p.name)

    print(f"[OK] KMZ creato: {kmz_out}")

def upsample_phase_via_complex_old(phase_in: np.ndarray,
                               axes_in: tuple[np.ndarray, np.ndarray],
                               axes_out: tuple[np.ndarray, np.ndarray],
                               bbox: list[float],
                               kx: int = 1, ky: int = 1, s: float = 0.0,
                               nodata: float = -9999.0) -> np.ndarray:
    """
     Upsampling of a phase (in radians) using interpolation on cosine and sine, 
     to avoid artifacts caused by phase wrapping.
    """

    # Convert to real and imaginary parts of e^(j*phase)
    real_part = np.cos(phase_in)
    imag_part = np.sin(phase_in)

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
    comp = re_up + 1j * im_up
    phase_up = np.angle(comp)

    return phase_up
    
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

def plot_2d(x, y, zplot, cb='jet', fs=12,
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
            show=True, file_2_save=None):
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

def single_baseline_single_pol_coh(primary_complex: npt.NDArray[complex],
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
    vmin = np.percentile(valid, 2)
    vmax = np.percentile(valid, 98)
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
    }
    
    annotation_file_sec = product_secondary.annotation_coregistered_xml_file
    root_sec  = ET.parse(annotation_file_sec).getroot()

    info_sec = {
        "primaryImageSelectionInformation_secondary":  _safe_text(root_sec, ".//primaryImageSelectionInformation"),
        "normalBaseline_secondary":                     _safe_text(root_sec, ".//normalBaseline"),
        "averageRangeCoregistrationShift_secondary":    _safe_text(root_sec, ".//averageRangeCoregistrationShift"),
        "averageAzimuthCoregistrationShift_secondary":  _safe_text(root_sec, ".//averageAzimuthCoregistrationShift"),
    
        "orbitNumber_secondary":                        int(product_secondary.orbitNumber) ,#getattr(product_secondary, "orbitNumber", ""),
        "orbitDirection_secondary":                     (product_secondary.orbitDirection), #getattr(product_secondary, "orbitDirection", ""),
        
        "startTimeFromAscendingNode_secondary":        product_secondary.startTimeFromAscendingNode,  #getattr(product_secondary, "startTimeFromAscendingNode", ""),
        "completionTimeFromAscendingNode_secondary":    product_secondary.completionTimeFromAscendingNode, #getattr(product_secondary, "completionTimeFromAscendingNode", ""),        
        "dataTakeID_secondary":                         int(product_secondary.dataTakeID), #getattr(product_secondary, "dataTakeID", ""),        
        "centerLat_secondary":                         product_secondary.center_lat, #getattr(product_secondary, "center_lat", None),
        "centerLon_secondary":                        product_secondary.center_lon, #getattr(product_secondary, "center_lon", None), 
   }        
    
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
    
    
    

def check_interferogram(path_primary, path_secondary, flatten, is_light, number_frame,coh_low_thr, coh_high_thr):

    print(path_primary)
    print(path_secondary)
    print(flatten)
    path_primary = Path(path_primary)
    path_secondary = Path(path_secondary)


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
    mode_label = "light" if is_light else "all"
    output_folder_check_interferogram = parent_folder / f"check_interferogram_{date_primary}_{date_secondary}_{number_frame}_{flatten}_{mode_label}"
    output_folder_check_interferogram.mkdir(exist_ok=True)

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
    
    # -------------------------
    # 1) Load SLCs (amp & phase) and build complex images
    # -------------------------
       
    channel = 3
    nan_value = -9999
    with rasterio.open(data_primary_abs) as amp_raster_pri: #get the amp channels
        amp_hh_pri = amp_raster_pri.read(channel) #nupy.ndarray object
        amp_hh_pri = np.ma.masked_equal(amp_hh_pri, nan_value)
            
    
    with rasterio.open(data_primary_phase) as phase_hh_pri: #get the amp channels
        phase_hh_pri = phase_hh_pri.read(channel) #nupy.ndarray object
        phase_hh_pri = np.ma.masked_equal(phase_hh_pri, nan_value)
      
    
    with rasterio.open(data_secondary_abs) as amp_raster_sec: #get the amp channels
        amp_hh_sec = amp_raster_sec.read(channel) #nupy.ndarray object
        amp_hh_sec = np.ma.masked_equal(amp_hh_sec, nan_value)
    
    with rasterio.open(data_secondary_phase) as phase_hh_sec: #get the amp channels
        phase_hh_sec = phase_hh_sec.read(channel) #nupy.ndarray object
        phase_hh_sec = np.ma.masked_equal(phase_hh_sec, nan_value)
        
    primary = (amp_hh_pri * np.exp(1j * phase_hh_pri)).astype(np.complex64)
    secondary = (amp_hh_sec * np.exp(1j * phase_hh_sec)).astype(np.complex64)
    
    # Print stats
    print("\n[INFO] PRIMARY PRODUCT:")
    print(f"  - Max amplitude (HH): {np.max(amp_hh_pri)}")
    print(f"  - Max phase (HH):     {np.max(phase_hh_pri)}")
    
    print("\n[INFO] SECONDARY PRODUCT:")
    print(f"  - Max amplitude (HH): {np.max(amp_hh_sec)}")
    print(f"  - Max phase (HH):     {np.max(phase_hh_sec)}")
    
    
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

    lut_co = netCDF4.Dataset(path_lut_coregistered)
    lut_pri = netCDF4.Dataset(path_lut_primary)
    
    # flattening phase (coreg group)
    flatteningPhaseScreen_co = lut_co.groups['coregistration']['flatteningPhaseScreen'][:].astype(np.float64)
    flatteningPhaseScreen_pri =lut_pri.groups['coregistration']['flatteningPhaseScreen'][:].astype(np.float64)
    
    
    # SKP calibration phase (skpPhaseCalibration group) 
    skpCalibrationPhaseScreen_co = lut_co.groups['skpPhaseCalibration']['skpCalibrationPhaseScreen'][:].astype(np.float64)
    skpCalibrationPhaseScreen_pri =lut_pri.groups['skpPhaseCalibration']['skpCalibrationPhaseScreen'][:].astype(np.float64)
    
    
    #get histogram of  skpCalibrationPhaseScreen_co 
    output_filename = f"Hist_skpCalibrationPhaseScreen_co_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png"
    output_path = output_folder_check_interferogram / output_filename    
    getHistogram(skpCalibrationPhaseScreen_co, title="Histogram SKP Calibration Phase Screen (co)",  out_file=output_path,bins=200)
    
    #get histogram of skpCalibrationPhaseScreen_pri
    output_filename = f"Hist_skpCalibrationPhaseScreen_pri_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png"
    output_path = output_folder_check_interferogram / output_filename 
    getHistogram(skpCalibrationPhaseScreen_pri, title="Histogram SKP Calibration Phase Screen (pri)", out_file=output_path,bins=200)

    
    #######################
    
    if not is_light:
        save_phase_map_lut(
            flatteningPhaseScreen_pri,
            title=f"flatteningPhaseScreen_pri [rad] - {date_primary}",
            date=date_primary,
            out_file=output_folder_check_interferogram / f"flatteningPhaseScreen_pri_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png",
        )
    
        save_phase_map_lut(
            skpCalibrationPhaseScreen_pri,
            title=f"skpCalibrationPhaseScreen_pri [rad] - {date_primary}",
            date=date_primary,
            out_file=output_folder_check_interferogram / f"skpCalibrationPhaseScreen_pri_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png",
        )
    
        save_phase_map_lut(
            flatteningPhaseScreen_co,
            title=f"flatteningPhaseScreen_co [rad]-{date_secondary}",
            date=date_secondary,
            out_file=output_folder_check_interferogram / f"flatteningPhaseScreen_co_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png",
        )
    
        save_phase_map_lut(
            skpCalibrationPhaseScreen_co,
            title=f"skpCalibrationPhaseScreen_co [rad]-{date_secondary}",
            date=date_secondary,
            out_file=output_folder_check_interferogram / f"skpCalibrationPhaseScreen_co_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png",
        )
   
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
    
    flatteningPhaseScreen_co_interpolator = scipy.interpolate.RectBivariateSpline(    
            axes_in[0],
            axes_in[1],
            flatteningPhaseScreen_co,
            bbox= bbox,
        kx=degree_x,
            ky=degree_y,
            s=smoother
         )
    
    flatteningPhaseScreen_pri_interpolator = scipy.interpolate.RectBivariateSpline(    
            axes_in[0],
            axes_in[1],
            flatteningPhaseScreen_pri,
            bbox= bbox,
        kx=degree_x,
            ky=degree_y,
            s=smoother
         )
    

    
    flatteningPhaseScreen_co_upsampled = flatteningPhaseScreen_co_interpolator(axes_out[0], axes_out[1])
    flatteningPhaseScreen_pri_upsampled = flatteningPhaseScreen_pri_interpolator(axes_out[0], axes_out[1])
    

    # --- upsample SKP come exp(j·skp) ---
    skpCalibrationPhaseScreen_co_upsampled = upsample_phase_via_complex(
        skpCalibrationPhaseScreen_co,
        axes_in=(lut_az_axes_pri, lut_range_axis_pri),
        axes_out=(az_slc_axis_pri_stac, range_slc_axis_pri_stac),
        bbox=bbox, kx=degree_x, ky=degree_y, s=smoother, nodata=-9999.0
        )

    skpCalibrationPhaseScreen_pri_upsampled = upsample_phase_via_complex(
        skpCalibrationPhaseScreen_pri,
        axes_in=(lut_az_axes_pri, lut_range_axis_pri),
        axes_out=(az_slc_axis_pri_stac, range_slc_axis_pri_stac),
        bbox=bbox, kx=degree_x, ky=degree_y, s=smoother, nodata=-9999.0
        )
    
    
    if not is_light:
        save_phase_map_slc(flatteningPhaseScreen_pri_upsampled,  f"flatteningPhaseScreen\npri_upsampled [rad] - {date_primary}", date_primary,
                           output_folder_check_interferogram / f"flatteningPhaseScreen_pri_upsampled_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png")
        
        save_phase_map_slc(flatteningPhaseScreen_co_upsampled,   f"flatteningPhaseScreen\nco_upsampled [rad] - {date_secondary}",  date_secondary,
                           output_folder_check_interferogram / f"flatteningPhaseScreen_co_upsampled_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png")
        
        save_phase_map_slc(skpCalibrationPhaseScreen_pri_upsampled, f"skpCalibrationPhaseScreen\npri_upsampled [rad] - {date_secondary}", date_primary,
                           output_folder_check_interferogram / f"skpCalibrationPhaseScreen_pri_upsampled_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png")
        
        save_phase_map_slc(skpCalibrationPhaseScreen_co_upsampled,  f"skpCalibrationPhaseScreen\nco_upsampled [rad] - {date_secondary}",  date_secondary,
                           output_folder_check_interferogram / f"skpCalibrationPhaseScreen_co_upsampled_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png")

    
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
    # 7) Interferogram & Coherence
    # -------------------------
    
    interferogram = primary * np.conj(secondary)
    interferogram_flat = (primary  * np.exp(1j * corr_pri)) * np.conj(secondary * np.exp(1j * corr_co))

    phase_deg = np.angle(interferogram, deg=True)
    phase_flat_deg = np.angle(interferogram_flat, deg=True)    
    
    #########################################################################
    if not is_light:
        std_multiplier = 3 
        aspect_ratio = "auto"
        colorbar_fraction = 0.05
        colorbar_pad = 0.04
        
        fig, axes = plt.subplots(1,2) 
        
        ax2 = axes[0].imshow( 
            phase_deg , 
            cmap='rainbow',
            aspect=aspect_ratio,
            vmin = -180 ,#vmin =  interferogram_xx_avg - std_multiplier* interferogram_xx_std,
            vmax =  180, #vmax =  interferogram_xx_avg + std_multiplier* interferogram_xx_std,
            interpolation="nearest"
        )
        axes[0].set_title(f'phase_deg [deg]  ')
        
        
        ax5 = axes[1].imshow( 
            phase_flat_deg, 
            cmap='rainbow',
            aspect=aspect_ratio,
            vmin = -180 ,#vmin =  interferogram_xx_with_screens_avg - std_multiplier* interferogram_xx_with_screens_std,
            vmax =  180, # vmax =  interferogram_xx_with_screens_avg + std_multiplier* interferogram_xx_with_screens_std,
            interpolation="nearest"
        )
        axes[1].set_title(f'phase_flat_deg [deg]  flat')
        fig.colorbar(ax2, ax=axes[0], orientation='vertical', fraction=colorbar_fraction, pad=colorbar_pad)    
        fig.colorbar(ax5, ax=axes[1], orientation='vertical', fraction=colorbar_fraction, pad=colorbar_pad)   
        fig.tight_layout()
        
        output_filename = f"phase_flat_deg_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png"
        output_path = output_folder_check_interferogram / output_filename
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Image saved to: {output_path}")
    
    #########################################################################
    coh = single_baseline_single_pol_coh(primary,secondary * np.exp(1j * (corr_co - corr_pri)),(5,5))   
    print(np.min(coh))
    print(np.max(coh))
    
    coh_abs = np.abs(coh)
    coh_phase = np.angle(coh, deg=True)

    print(np.mean(coh_abs), np.median(coh_abs))
       
    shape = coh_abs.shape
    print(shape)
    

    plot_2d(
        np.arange(shape[0]),
        np.arange(shape[1]),
        coh_phase,
        y_title='Range [pixels]',
        x_title='Azimuth [pixels]',
        vmax=-180,
        vmin=180,
        cb_title='[deg]',
        no_interp=True,
        file_2_save=output_folder_check_interferogram / f"cho_phase_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png")
    shape = coh_abs.shape
    plot_2d(
        np.arange(shape[0]),
        np.arange(shape[1]),
        coh_abs,
        y_title='Range [pixels]',
        x_title='Azimuth [pixels]',
        vmax=1,
        vmin=0.001,
        no_interp=False,
        cb='Greys_r', file_2_save=output_folder_check_interferogram / f"cho_amp_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png")


    '''
    if not is_light:
        # Coherence amplitude
        img = Image.fromarray(coh_abs)
        new_size = (img.width // 2, img.height // 2)  # 50% in both dimensions
        img_resized = img.resize(new_size, resample=Image.BILINEAR)
        img_resized.save(output_folder_check_interferogram / f"coh_abs_{date_primary}_{date_secondary}_{number_frame}_{flatten}.tif")
        
        # Coherence phase
        img = Image.fromarray(coh_phase)
        new_size = (img.width // 2, img.height // 2)
        img_resized = img.resize(new_size, resample=Image.BILINEAR)
        img_resized.save(output_folder_check_interferogram / f"coh_phase_{date_primary}_{date_secondary}_{number_frame}_{flatten}.tif")
    '''     
        
    # -------------------------------------------------------------------------
    # 4. Save "clean" PNG maps (without axes or colorbar)
    # -------------------------------------------------------------------------
    print("[INFO] Saving clean PNG maps...")

    coh_phase_png = output_folder_check_interferogram / f"coherence_phase_{date_primary}_{date_secondary}_{number_frame}_{flatten}_kml.png"
    coh_abs_png = output_folder_check_interferogram / f"coherence_amp_{date_primary}_{date_secondary}_{number_frame}_{flatten}_kml.png"

    save_clean_image(coh_phase, cmap='RdBu', out_path=coh_phase_png, vmin=-np.pi, vmax=np.pi)
    save_clean_image(coh_abs, cmap='Greys_r', out_path=coh_abs_png, vmin=0, vmax=np.nanmax(coh_abs))

    make_white_transparent(coh_phase_png)
    make_white_transparent(coh_abs_png)
    
    print(f"[OK] Saved: {coh_phase_png.name}")
    print(f"[OK] Saved: {coh_abs_png.name}")
    # Coherence stats required
    coh_mean   = float(np.nanmean(coh_abs))
    coh_min    = float(np.nanmin(coh_abs))
    coh_max    = float(np.nanmax(coh_abs))
    coh_median = float(np.nanmedian(coh_abs))

    row = {
    "coh_mean": coh_mean,
    "coh_std"    :float(np.nanstd(coh_abs)),
    "coh_max":  coh_max,
    "coh_min":  coh_min,
    "coh_median": coh_median
    }
 
    # NB: get_info_baseline ha fatto listinfobaseline.append(info) -> lista con un solo dict
    baseline_info = listinfobaseline[0] if listinfobaseline else {}
    
    # Unisci i metadati baseline dentro "row"
    row.update(baseline_info)
    # -------------------------------------------------------------------------
    # 5. Prepara il KML con popup e colorbar laterale
    # -------------------------------------------------------------------------
    preview_kml_sec = Path(preview_kml_sec) 
    
    if not preview_kml_sec.exists():
        print(f"[WARN] Preview KML not found: {preview_kml_sec}")
    else:
        # Costruisci il dict che va nel popup del KML
        # (prende le metriche da row e, se presenti, aggiunge i campi baseline)
        coh_stats = {
            "Mean coh abs":   row["coh_mean"],
            "Std coh abs":    row["coh_std"],
            "Min coh abs":    row["coh_min"],
            "Max coh abs":    row["coh_max"],
            "Median coh abs": row["coh_median"],
        }
    
        # Chiavi baseline che vuoi mostrare (verifica che esistano prima di aggiungerle)
        fieldnames = [
        # identificativi STA
        "primary_sta", "secondary_sta",

        # --- PRIMARY (da annotation del primary) ---

        "orbitNumber_primary",
        "orbitDirection_primary",
        "startTimeFromAscendingNode_primary",
        "completionTimeFromAscendingNode_primary",
        "dataTakeID_primary",


        # --- SECONDARY (da annotation del secondary) ---

        "primaryImageSelectionInformation_secondary",
        "normalBaseline_secondary",
        "averageRangeCoregistrationShift_secondary",
        "averageAzimuthCoregistrationShift_secondary",
        "orbitNumber_secondary",
        "orbitDirection_secondary",
        "startTimeFromAscendingNode_secondary",
        "completionTimeFromAscendingNode_secondary",
            "dataTakeID_secondary",
        "centerLat_secondary",
        "centerLon_secondary", 

          # --- STATISTICHE COERENZA ---
        "mean(|coh|)", "min(|coh|)", "max(|coh|)", "median(|coh|)",

          # --- METADATI RUN ---
          "frame", "phaseCorrectionMode", "date_primary", "date_secondary",
         ]
        for k in fieldnames:
            if k in row:
                coh_stats[k] = row[k]
        '''
        # Colorbar verticale 0–1 per la coerenza ABS (come immagine separata)
        colorbar_png_path = output_folder_check_interferogram / f"colorbar_coh_abs_{date_primary}_{date_secondary}.png"
        save_colorbar_png(
            colorbar_png_path,
            cmap='Greys_r', vmin=0.0, vmax=1.0,
            ticks=(0.0, 0.5, 1.0),
            label='Coherence amplitude'
        )
        '''
        # Crea il KML (usa il PNG "pulito" della coerenza ABS)
        kmz_overlay = output_folder_check_interferogram / f"{coh_abs_png.stem}.kmz"
        make_overlay_kmz_with_quad(
                kmz_out=kmz_overlay,
            png_path=coh_abs_png,
            preview_kml_file=preview_kml_sec,
            coh_stats=coh_stats,
            coh_phase_file=                           output_folder_check_interferogram / f"cho_phase_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png",
            Hist_skpCalibrationPhaseScreen_co_file=   output_folder_check_interferogram / f"Hist_skpCalibrationPhaseScreen_co_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png",
            Hist_skpCalibrationPhaseScreen_pri_file=  output_folder_check_interferogram / f"Hist_skpCalibrationPhaseScreen_pri_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png",
            flatteningPhaseScreen_co_file=            output_folder_check_interferogram / f"flatteningPhaseScreen_co_upsampled_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png",
            skpCalibrationPhaseScreen_co_file=        output_folder_check_interferogram / f"skpCalibrationPhaseScreen_co_upsampled_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png",
            phase_flat_deg_file=                      output_folder_check_interferogram / f"phase_flat_deg_{date_primary}_{date_secondary}_{number_frame}_{flatten}.png",
            coh_low_thr=coh_low_thr, coh_high_thr=coh_high_thr
        )
        print(f"[OK] KML created with popup + side colorbar: {kmz_overlay}")
    
    
    
    # --- Coherence stats ---
    coh_mean   = float(np.nanmean(coh_abs))
    coh_min    = float(np.nanmin(coh_abs))
    coh_max    = float(np.nanmax(coh_abs))
    coh_median = float(np.nanmedian(coh_abs))
    

    # ---------------- CSV per la cartella di questo stack ----------------
    stack_folder = path_primary.parent.parent
    summary_csv = stack_folder / "coherence_summary.csv"

    # recupero info baseline aggiunta poco fa (relativa al PRIMARY)
    baseline_info = listinfobaseline[-1] if listinfobaseline else {}

    # nomi prodotti STA (fallback se .root non esiste)
    primary_sta_name   = Path(getattr(product_primary,  "root", path_primary)).name
    secondary_sta_name = Path(getattr(product_secondary,"root", path_secondary)).name





    
    
    
    # componi la riga: unisci baseline + stats + metadati
    row_dict = {
        **{k: baseline_info.get(k, "") for k in baseline_info.keys()}, 
      "primary_sta": primary_sta_name,
      "secondary_sta": secondary_sta_name,

      "mean(|coh|)":   float(coh_mean),
      "min(|coh|)":    float(coh_min),
      "max(|coh|)":    float(coh_max),
      "median(|coh|)": float(coh_median),

      "frame":              str(number_frame),
      "phaseCorrectionMode": str(flatten),
      "date_primary":        date_primary,
      "date_secondary":      date_secondary,
    }


    # normalizza eventuali chiavi mancanti (evita KeyError)
    for k in fieldnames:
        row_dict.setdefault(k, "")
    
    import csv
    write_header = not summary_csv.exists()
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row_dict)

    print(f"[INFO] CSV updated: {summary_csv}")        
    try:
      lut_co.close()
    except Exception:
      pass
    try:
      lut_pri.close()
    except Exception:
      pass    
    
        


def extract_date_from_scs_name(scs_name):
    """
    Extract only the date (YYYYMMDD) from the first timestamp in the SCS product name.
    """
    match = re.search(r"(20\d{6})T\d{6}", scs_name)
    if not match:
        raise ValueError(f"Cannot extract date from: {scs_name}")
    return match.group(1)

def find_matching_sta(start_time, sta_list):
    """
    Find the STA product whose name contains the given start time.
    """
    
    for sta in sta_list:
        if start_time in sta.name:
            return sta
    return None


def run_check_interferogram(path_stacks_folder, mode="light" ,coh_low_thr=COH_LOW_THRESHOLD,
         coh_high_thr=COH_HIGH_THRESHOLD):
    is_light = mode.lower() == "light"
    print(coh_low_thr)
    print(coh_high_thr)
    print ('----------------------------------------------------------------------------')
    print(f"Mode selected: {'light' if is_light else 'all'}")
    print ('----------------------------------------------------------------------------')
    
    
    flatten = None    
    # Find all folders containing 'STA__1S' in their name
    pattern = os.path.join(path_stacks_folder, "*STA__1S*")
    sta_paths = sorted([Path(p) for p in glob.glob(pattern)])

    print(f"Found {len(sta_paths)} STA products:\n")
    pprint.pprint([p.name for p in sta_paths])
    
    
    # Loop over each STA product and extract the related SCS pair
    for sta_path in sta_paths:
        print ('----------------------------------------------------------------------------')
        print (sta_path)
        print ('----------------------------------------------------------------------------')
        # Load the STA product using your custom class
        sta_product = BiomassProduct.BiomassProductSTA(sta_path)
        number_frame=sta_product.wrsLatitudeGrid  
        annotation_file = sta_product.annotation_coregistered_xml_file
        
        def _get_bool(root, xpath):
            node = root.find(xpath)
            return (node is not None) and (node.text or "").strip().lower() == "true"
        
        
        

        
        # Parse XML and extract primary/secondary SCS names
        tree = ET.parse(annotation_file)
        root = tree.getroot()
        scs_primary = root.find(".//primaryImage").text
        scs_secondary = root.find(".//secondaryImage").text

        skpPhaseCalibrationFlag = _get_bool(root, ".//skpPhaseCalibrationFlag")
        skpPhaseCorrectionFlag = _get_bool(root, ".//skpPhaseCorrectionFlag")        
        skpPhaseCorrectionFlatteningOnlyFlag = _get_bool( root, ".//skpPhaseCorrectionFlatteningOnlyFlag")
        
        
        print(f"Primary image: {scs_primary}")
        print(f"Secondary image: {scs_secondary}")
        print(f"skpPhaseCalibrationFlag: {skpPhaseCalibrationFlag}")
        print(f"skpPhaseCorrectionFlag: {skpPhaseCorrectionFlag}")
        print(f"skpPhaseCorrectionFlatteningOnlyFlag: {skpPhaseCorrectionFlatteningOnlyFlag}")


        
        if skpPhaseCorrectionFlag:
            if skpPhaseCorrectionFlatteningOnlyFlag:
                print( "Flattening Phase Screen")
                flatten = 'geometry'
            else:
                print("Ground Phase Screen")
                flatten = 'skp'
        else:
            print("No Phase Screen")
            flatten = "None"
        
        print(f"[INFO] phaseCorrection mode = {flatten}")
        print (scs_primary)
        print(scs_secondary)
        
        time_primary = extract_date_from_scs_name(scs_primary)
        time_secondary = extract_date_from_scs_name(scs_secondary)  
        print (time_primary)
        print(time_secondary)       
        
        if time_primary!=time_secondary:
            # Match the extracted times to actual STA product folders
            sta_primary = find_matching_sta(time_primary, sta_paths)
            sta_secondary = find_matching_sta(time_secondary, sta_paths)       

            print(f"[OK] Matches found:\n - PRIMARY:  {sta_primary.name}\n - SECONDARY:{sta_secondary.name}")
            check_interferogram(sta_primary, sta_secondary, flatten, is_light, number_frame,coh_low_thr, coh_high_thr)






def extract_timestamp_from_name(name: str):
    """Estrae il primo timestamp nel formato 20251006T214636 dal nome del prodotto"""
    match = re.search(r"(20\d{6}T\d{6})", name)
    return match.group(1) if match else None

def extract_frame_from_name(name: str):
    """Estrae il frame number (es. F173)"""
    match = re.search(r"(F\d{3})", name)
    return match.group(1) if match else None

def create_folder_name(index, inputs):
    """Crea il nome della folder per lo stack"""
    timestamps = [extract_timestamp_from_name(i) for i in inputs]
    timestamps = [t for t in timestamps if t]
    frame = extract_frame_from_name(inputs[0]) or "FXXX"
    start = min(timestamps)
    stop = max(timestamps)
    folder_name = f"{index:02d}_{start}_{stop}_{frame}"
    return folder_name

def cleanup_sta_products(stack_folder: Path, dry_run: bool = False) -> None:
    """
    Rimuove le directory dei prodotti STA estratti dentro stack_folder,
    lasciando intatte le cartelle di output 'check_interferogram_*' e file CSV.
    """
    if not isinstance(stack_folder, Path):
        stack_folder = Path(stack_folder)

    to_delete = []
    for p in stack_folder.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        # Mantieni risultati e altre cartelle non-STA
        if name.startswith("check_interferogram_"):
            continue
        # Candidati: directory dei prodotti STA
        if "STA__1S" in name:
            to_delete.append(p)

    if not to_delete:
        print("[CLEANUP] Nessuna directory STA da rimuovere.")
        return

    print("[CLEANUP] Rimuovo le seguenti directory STA:")
    for d in to_delete:
        print(f"  - {d}")

    if dry_run:
        print("[CLEANUP] DRY-RUN attivo: nessuna rimozione eseguita.")
        return

    for d in to_delete:
        try:
            shutil.rmtree(d)
            print(f"[CLEANUP] Rimossa: {d}")
        except Exception as e:
            print(f"[CLEANUP][ERRORE] Impossibile rimuovere {d}: {e}")



def main(XML_FILE):
    print('----------------------------------------------------------------------------')
    print('MAIN')
    print('----------------------------------------------------------------------------')
    coh_low_thr=0.35
    coh_high_thr=0.8
    mode="all"
    xml_path = Path(XML_FILE).resolve()  # converte in percorso assoluto
    BASE_OUTPUT_DIR = xml_path.parent.parent / "RESULTS"  # aggiunge la cartella "output"

    print(f"XML path:         {xml_path}")
    print(f"Output directory: {BASE_OUTPUT_DIR}")

    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # crea la cartella se non esiste
    print(f"[INFO] Parsing XML file: {XML_FILE}")
    tree = ET.parse(XML_FILE)
    root = tree.getroot()

    stacks = root.findall("stack")
  
    CLEANUP_ON_FAILURE=True
    #per ogni gruppo di stack 
    for idx, stack in enumerate(stacks, start=1):
        inputs = [inp.text.strip() for inp in stack.findall("input")]
        folder_name = create_folder_name(idx, inputs)
        stack_folder = BASE_OUTPUT_DIR / folder_name

        # crea la folder
        if not stack_folder.exists():
            stack_folder.mkdir(parents=True)
            print(f"[INFO] Created folder: {stack_folder}")
           
        # scarica i prodotti
        for istack in inputs:
            print(f"[INFO] Downloading product: {istack}")
            cmd = [
                "runBioDownloadSingle",  # comando bash
                "BiomassLevel1cIOC",    # nome della collezione
                istack,                 # nome del prodotto
                str(stack_folder)       # cartella di destinazione
            ]

            # esegui il comando
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Download failed for {istack}: {e}")
                
            # Trova i file zip appena scaricati
        for zip_file in stack_folder.glob("*.zip"):
            print(f"[INFO] Unzipping {zip_file.name}")
            try:
                    with zipfile.ZipFile(zip_file, 'r') as zf:
                        zf.extractall(stack_folder)
                    print(f"[OK] Extracted {zip_file.name}")
                    zip_file.unlink()  # elimina lo zip
                    print(f"[OK] Deleted {zip_file.name}")
            except Exception as e:
                    print(f"[ERROR] Failed to unzip {zip_file.name}: {e}")
        #    lanciare check_inteferogram su quella folder
        # check_interferogram con protezione: se fallisce, continua col prossimo stack
        try:
            run_check_interferogram(stack_folder, mode,coh_low_thr, coh_high_thr)
        
        except Exception as e:

            print(f"[ERROR] check_interferogram FAILED for stack {stack_folder.name}: {e}")

            try:
                (stack_folder / "check_interferogram_error.txt").write_text(str(e))
            except Exception:
                pass
            if CLEANUP_ON_FAILURE:
                try:
                    cleanup_sta_products(stack_folder, dry_run=False)
                except Exception as ce:
                    print(f"[CLEANUP][WARN] Cleanup after failure raised: {ce}")
            continue  # passa al prossimo stack

        # se il check è andato a buon fine, fai la pulizia
        try:
            cleanup_sta_products(stack_folder, dry_run=False)
        except Exception as ce:
            print(f"[CLEANUP][WARN] Cleanup raised: {ce}")
            


def print_help():

    """
    Print command-line usage instructions for the script.
    """
    help_message = """
-------------------------------------------------------------------------------
BIOMASS – Interferogram Checker
-------------------------------------------------------------------------------

USAGE:
    python check_interferogram.py <PATH_TO_XML_STACK_LIST>

DESCRIPTION:
    The script reads an XML file containing multiple <stack> entries.
    Each stack lists a set of BIOMASS Level-1 STA products.
    For each stack the script:

      1. Creates an output folder inside RESULTS/
      2. Downloads all products listed in the <stack>
      3. Unzips them
      4. Runs the interferogram and coherence analysis
      5. Generates:
         - coherence amplitude/phase PNGs
         - flattened phase maps
         - LUT and histogram images
         - a KMZ overlay with popup statistics
      6. Appends summary coherence statistics to coherence_summary.csv
      7. Removes the extracted STA product folders (cleanup)

ARGUMENTS:
    <PATH_TO_XML_STACK_LIST>   Path to the XML file containing:
                               <stack>
                                 <input>PRODUCT_NAME</input>
                                 <input>PRODUCT_NAME</input>
                                 ...
                               </stack>

OUTPUT STRUCTURE:
    RESULTS/
        01_YYYYMMDD_YYYYMMDD_Fxxx/
            check_interferogram_*
            coherence_summary.csv
            (KMZ, PNG, histograms, phase maps, etc.)

EXAMPLES:
    python check_interferogram.py stacks_test.xml

    python check_interferogram.py /home/user/stacks_list.xml

NOTES:
    - The script calls 'runBioDownloadSingle', so the tool must be available
      in your environment.
    - All intermediate STA folders are cleaned up unless CLEANUP_ON_FAILURE=False.


-------------------------------------------------------------------------------
"""

    print(help_message)


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print_help()
    else:
        path_xml= sys.argv[1]
        main(path_xml)







