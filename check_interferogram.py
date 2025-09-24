# -*- coding: utf-8 -*-
"""
Created on Wed May 28 21:11:09 2025

"""

import xml.etree.ElementTree as ET  #needed to fetch  the data from annotations
import rasterio
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy
import sys
from PIL import Image
from pathlib import Path
import BiomassProduct
import copy
import os
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import re

import os
import glob
import pprint


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

def check_interferogram(path_primary, path_secondary, flatten, is_light, number_frame):

    print(path_primary)
    print(path_secondary)
    print(flatten)
    path_primary = Path(path_primary)
    path_secondary = Path(path_secondary)


    # -- Open products (uses your BiomassProduct class)
    product_primary=    BiomassProduct.BiomassProductSTA(path_primary)
    product_secondary=  BiomassProduct.BiomassProductSTA(path_secondary)
        
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
    data_primary_phase =    product_primary.measurement_phase_file   
    path_lut_primary =      product_primary.annotation_coregistered_lut_file   
    path_main_ann_primary = product_primary.annotation_coregistered_xml_file   

    data_secondary_abs =         product_secondary.measurement_abs_file        
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


def main(path_stacks_folder, mode="light"):
    is_light = mode.lower() == "light"
    
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
        #<skpPhaseCalibrationFlag>false</skpPhaseCalibrationFlag>
        #<skpPhaseCorrectionFlag>true</skpPhaseCorrectionFlag>
 
        #ideallly it would be triggered by this product metadata

        
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
            check_interferogram(sta_primary, sta_secondary, flatten, is_light, number_frame)
    #scs_primary=first_stack.
    
    #annotation_coregistered.primaryImage  questo è il prodotto primary ed è un scs quindi bisogna prendere la data e confrontarlo con gli sta che sono nella folder.
    #quindi biosgna calcolare check interferogram per ogni coppia primary -secondary 
    
    
    #check_interferogram(path_primary, path_secondary, flatten = True):
    


def print_help():
    """
    Print command-line usage instructions for the script.
    """
    help_message = """
[USAGE]
    python check_interferogram.py <path_stacks_folder> [mode]

[DESCRIPTION]
    For each STA product,
    it identifies the primary and secondary SCS images from the annotation,
    finds the corresponding STA folders, and generates:

        - Flattening phase maps (original and upsampled)
        - Interferometric phase
        - Coherence amplitude and phase 

    Results are saved in a dedicated folder named:
        check_interferogram_<primary_date>_<secondary_date>[_flatten]

[ARGUMENTS]
    <path_stacks_folder>   : Path to the folder containing STA product directories.
    <mode>                 : Optional. Set to "all" to generate just coh_amp and coh_phase png. Default is "light".

[EXAMPLES]
    # With light mode
    python check_interferogram.py /data/biomass/stack_folder light

    # Full processing 
    python check_interferogram.py /data/biomass/stack_folder all
"""
    print(help_message)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_help()
    else:
        path_stacks_folder = sys.argv[1]
        mode = sys.argv[2] if len(sys.argv) > 2 else "light"
        main(path_stacks_folder, mode)







