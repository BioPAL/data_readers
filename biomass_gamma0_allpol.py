# Copyright 2025 Serco Italia S.p.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

"""
BIOMASS L1 Radiometric Calibration  —  γ⁰  (HH, HV, VH, VV)
==============================================================
Compute Gamma-Nought for all four BIOMASS polarizations
(HH, HV, VH and VV).

The implementation follows the BIOMASS Product Format Definition
(PFD v1.6.1) and the Level-1 Algorithm Theoretical Basis Document
(ATBD v1.2.4).
  β⁰ = amp²
  γ⁰ = β⁰ · gammaNought_LUT    (LUT already in intensity units)

Output: one GeoTIFF per polarisation
  *_HH_gamma0.tif / *_HH_gamma0_dB.tif
  *_HV_gamma0.tif / *_HV_gamma0_dB.tif
  *_VH_gamma0.tif / *_VH_gamma0_dB.tif
  *_VV_gamma0.tif / *_VV_gamma0_dB.tif
    
"""

import os
import glob
import numpy as np
import netCDF4 as nc
import rasterio

import xml.etree.ElementTree as ET
from scipy.interpolate import RectBivariateSpline
from datetime import datetime


# ============================================================
# CONFIGURAZIONE
# ============================================================

PRODUCT_DIR = r"D:\BIOMASS\03_SCRIPTS\SNAP\35_SNAP\Desert\BIO_S2_SCS__1S_20260625T185853_20260625T185913_T_G01_M03_CDR_T012_F137_02_DTLXZL"

OUTPUT_DIR  = None   

SAVE_LINEAR = True
SAVE_DB     = True

FILL_VALUE  = -9999.0

POLARISATIONS = ["HH", "HV", "VH", "VV"]  

# ============================================================


def find_files(product_dir):
    """
    Locate the mandatory files of a BIOMASS L1 product.
    
    Returns
    -------
    abs_tiff : str
        Multi-band amplitude image.
    
    lut_nc : str
        Radiometric calibration LUT.
    
    annot_xml : str
        Main Annotation ADS.
    """
    meas_dir = os.path.join(product_dir, "measurement")
    ann_dir  = os.path.join(product_dir, "annotation")
 
    abs_matches   = glob.glob(os.path.join(meas_dir, "*_i_abs.tiff"))
    lut_matches   = glob.glob(os.path.join(ann_dir,  "*_lut.nc"))
    annot_matches = glob.glob(os.path.join(ann_dir,  "*_annot.xml"))
 
    if not abs_matches:
        raise FileNotFoundError( f"No *_i_abs.tiff found in: {meas_dir}")
    if not lut_matches:
        raise FileNotFoundError(f"No *_lut.nc found in: {ann_dir}")
    if not annot_matches:
        raise FileNotFoundError(f"No *_annot.xml found in: {ann_dir}")
 
    print(f"  Amplitude   : {os.path.basename(abs_matches[0])}")
    print(f"  LUT       : {os.path.basename(lut_matches[0])}")
    print(f"  Annotation: {os.path.basename(annot_matches[0])}")
    return abs_matches[0], lut_matches[0], annot_matches[0]

def read_amplitude(abs_tiff):
    """
    Read the BIOMASS amplitude image.
    
    The input GeoTIFF contains four float32 bands storing the square root
    of Beta-Nought (√β⁰) in linear units.
    
    The band order is:
    
        Band 1 -> HH
        Band 2 -> HV
        Band 3 -> VH
        Band 4 -> VV
    
    Rasterio returns arrays in the form:
    
        (bands, azimuth_lines, range_samples)
    
    Masked pixels are converted to NaN so they remain invalid throughout
    the radiometric processing chain.
    
    The function also preserves all geolocation information (GCPs),
    metadata and band tags required to reproduce the original product.
    """
    with rasterio.open(abs_tiff) as src:
        if src.count != 4:
            raise ValueError( f"Expected 4 bands, found {src.count}.")

        amp = (src.read(masked=True) .astype(np.float32) .filled(np.nan))


        profile = src.profile.copy()
        profile.update(count=1, dtype="float32")

        gcps, gcp_crs = src.gcps
        file_tags = src.tags().copy()

        band_tags = { i: src.tags(i).copy()
            for i in range(1, src.count + 1)}

        nodata = src.nodata
        if nodata is None:
            nodata = FILL_VALUE

    print(f"  Shape     : {amp.shape}  (bands, azimuth, range)")
    print(f"  N GCPs    : {len(gcps)}")
    print(f"  nodata    : {nodata}")
    print(f"  Invalid   : {np.count_nonzero(~np.isfinite(amp))}")

    return (amp, profile,gcps,gcp_crs, file_tags, band_tags,nodata,)


def read_lut_geometry(lut_nc):
    """
    Read the Gamma-Nought calibration LUT.
    
    The BIOMASS radiometric LUT is defined on its own temporal grid:
    
        relativeAzimuthTimeRGC
        slantRangeTimeRGC
    
    These coordinates are expressed in the same physical reference system
    used by the SAR image geometry.
    
    
    Returns
    -------
    gamma_lut : ndarray
        Gamma-Nought calibration factors.
    az_axis : ndarray
        Relative azimuth time coordinates [s].
    rg_axis : ndarray
        Slant-range time coordinates [s].
    reference_azimuth_time : str
        UTC reference used by relativeAzimuthTimeRGC.
 
    """
    with nc.Dataset(lut_nc) as ds:
        if "radiometry" not in ds.groups:
            raise KeyError("NetCDF group 'radiometry' not found.")

        rad = ds.groups["radiometry"]

        if "gammaNought" not in rad.variables:
            raise KeyError("Variable 'gammaNought' not found.")

        gamma_var = rad.variables["gammaNought"]
        gamma_dims = gamma_var.dimensions

        gamma_lut = np.ma.filled( gamma_var[:].astype(np.float64),np.nan, )
        az_axis = np.asarray(ds.variables["relativeAzimuthTimeRGC"][:], dtype=np.float64, )

        rg_axis = np.asarray(  ds.variables["slantRangeTimeRGC"][:], dtype=np.float64,)

        reference_azimuth_time = ds.getncattr("referenceAzimuthTime" )

    expected_dims = ( "relativeAzimuthTimeRGC","slantRangeTimeRGC",)

    if gamma_dims != expected_dims:
        raise ValueError( f"Unexpected Gamma-Nought dimensions: {gamma_dims}" )

    if gamma_lut.shape != (len(az_axis), len(rg_axis)):
        raise ValueError(
    f"LUT shape {gamma_lut.shape} is inconsistent "
    f"with coordinate axes {(len(az_axis), len(rg_axis))}."
        )

    if not np.all(np.diff(az_axis) > 0):
        raise ValueError(
            "relativeAzimuthTimeRGC axis is not strictly increasing."
        )

    if not np.all(np.diff(rg_axis) > 0):
        raise ValueError(
            "slantRangeTimeRGC axis is not strictly increasing."
        )

    print(f"  Gamma LUT shape : {gamma_lut.shape}")
    print(f"  Azimuth LUT     : " f"{az_axis[0]:.12f} .. {az_axis[-1]:.12f} s")
    print( f"  Range LUT       : "f"{rg_axis[0]:.12f} .. {rg_axis[-1]:.12f} s")
    print(f"  Invalid LUT     : " f"{np.count_nonzero(~np.isfinite(gamma_lut))}")

    return (gamma_lut, az_axis, rg_axis, reference_azimuth_time, )



def parse_utc(value):
    """
    Convert an ISO-8601 UTC string into a datetime object.
    """
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))

def read_sar_image_geometry(annot_xml, reference_azimuth_time):
    """
    Build the temporal coordinates of every SAR image pixel.
    
    The image grid is reconstructed from the Main Annotation ADS using:
    
        firstSampleSlantRangeTime
        rangeTimeInterval
    
        firstLineAzimuthTime
        azimuthTimeInterval
    
    The resulting coordinates are expressed in the same temporal reference
    system used by the calibration LUT.
    
    Returns
    -------
    target_az_axis : ndarray
        Relative azimuth time for every image line.
    target_rg_axis : ndarray
        Slant-range time for every image sample.
    """
    root = ET.parse(annot_xml).getroot()
    sar_image = root.findall(".//sarImage")[0]
 
    
    first_rg = float(sar_image.findtext("firstSampleSlantRangeTime"))
    last_rg = float(sar_image.findtext("lastSampleSlantRangeTime"))


    range_time_interval   = float(sar_image.find("rangeTimeInterval").text)
    n_samples             = int(sar_image.find("numberOfSamples").text)
    first_az = parse_utc(sar_image.findtext( "firstLineAzimuthTime"))
    last_az = parse_utc(sar_image.findtext("lastLineAzimuthTime"))
    
    azimuth_time_interval = float(sar_image.find("azimuthTimeInterval").text)
    n_lines                = int(sar_image.find("numberOfLines").text)
 
    reference_az = parse_utc(reference_azimuth_time)
    first_az_relative = (first_az - reference_az).total_seconds()
    last_az_relative = (last_az - reference_az).total_seconds()
    target_rg_axis = (first_rg + np.arange( n_samples,dtype=np.float64,) * range_time_interval)

    target_az_axis = (first_az_relative+ np.arange(n_lines,dtype=np.float64,) * azimuth_time_interval)

    if not np.isclose(target_rg_axis[-1], last_rg, atol=1e-12,rtol=0.0,):
        raise ValueError(" Inconsistent slant-range axis: computed last sample does not match the annotation.")

    if not np.isclose(target_az_axis[-1], last_az_relative, atol=1e-6,rtol=0.0,):
        raise ValueError(
            "Inconsistent azimuth axis: computed last line "
            "does not match the annotation."
        )

    return (target_az_axis, target_rg_axis,n_lines,n_samples,)

def check_lut_coverage(lut_az_axis,lut_rg_axis,target_az_axis,target_rg_axis,):
    """
    Verify that the SAR image is fully covered by the calibration LUT.

    
    This check guarantees that all image pixels are calibrated using
    interpolation only, without extrapolation outside the LUT domain.
    
    Raises
    ------
    ValueError
        If any image coordinate lies outside the LUT domain.
    """
    checks = {"azimuth start": (target_az_axis[0]>= lut_az_axis[0]),
        "azimuth stop": (target_az_axis[-1]<= lut_az_axis[-1] ),
        "range start": (target_rg_axis[0]>= lut_rg_axis[0]),
        "range stop": ( target_rg_axis[-1]<= lut_rg_axis[-1]),}

    failed = [name for name, passed in checks.items() if not passed]

    if failed:
        raise ValueError(    "The SAR image is not fully covered "
    f"by the Gamma-Nought LUT: {failed}")


def interpolate_lut(lut, lut_az_axis, lut_rg_axis, target_az_axis, target_rg_axis):
    """
    Interpolate the Gamma-Nought LUT onto the SAR image grid.
    
    A bilinear interpolation (RectBivariateSpline with kx=1, ky=1)
    
    The interpolation uses the physical temporal coordinates rather than
    pixel indices, ensuring correct alignment between the SAR image and
    the calibration LUT.
    
    The function refuses to interpolate LUTs containing invalid values.
    """
    invalid_count = np.count_nonzero(
        ~np.isfinite(lut))
    
    if invalid_count:
        raise ValueError(
            f"Gamma-Nought LUT contains "
            f"{invalid_count} invalid values.")  
 
 
    spline = RectBivariateSpline(lut_az_axis, lut_rg_axis, lut, kx=1, ky=1, s=0.0)

    gamma_full = spline( target_az_axis,target_rg_axis, grid=True, )
    return gamma_full.astype(np.float32)

def compute_gamma0(amp_band, gamma_full):
    """
    Compute Gamma-Nought.
    
    The BIOMASS PFD defines:
    
        β⁰ = amplitude²
    
    Gamma-Nought is then computed by applying the Gamma-Nought
    conversion factor provided in the product LUT:
    
        γ⁰ = β⁰ × GammaNought_LUT
    
    
    -------
    ndarray
        Gamma-Nought image in linear units.
    """

    if amp_band.shape != gamma_full.shape:
        raise ValueError(
            f"Amplitude image shape {amp_band.shape} "
            f"does not match LUT shape {gamma_full.shape}."
        )

    beta0 = np.square(amp_band, dtype=np.float32, )
    gamma0 = beta0 * gamma_full

    return gamma0.astype(np.float32, copy=False)


def save_tiff(data, profile, gcps, gcp_crs, file_tags, band_tags, nodata, path, pol, quantity):
    """
    Write a single-band Gamma-Nought GeoTIFF.
    
    The output preserves:
    
        • GCPs
        • GCP CRS
        • source metadata
        • band metadata
        • NoData value
    
    NaN values generated during processing are converted back to the
    original NoData value before writing the file.
    The output GeoTIFF is intended as a scientific reference product
    and preserves the geolocation and metadata of the source image.
    
    """
 
    out_profile = profile.copy()
    out_profile.update({
        "driver"  : "GTiff",
        "dtype"   : "float32",
        "count"   : 1,
        "compress": "deflate",
        "nodata"  : nodata,
        "crs"     : None,       
    })
 
    data_out = np.where(np.isfinite(data), data, nodata).astype(np.float32)
 
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(data_out, 1)
        dst.gcps = (gcps, gcp_crs)
        dst.update_tags(**file_tags)
        dst.update_tags(
            QUANTITY    = quantity,
            POLARISATION= pol,
            REFERENCE   = "PFD v1.6.1 Sec.4.3.2 | ATBD v1.2.4 Sec.4.8",
        )
        out_band_tags = band_tags.copy()
        out_band_tags["POLARIMETRIC_INTERP"] = pol
        out_band_tags["QUANTITY"] = quantity
        dst.update_tags(1, **out_band_tags)
 
    print(f"  -> {os.path.basename(path)}")
    
def to_db(data):
    """
    Convert Gamma-Nought from linear scale to decibels.
    
    Only strictly positive finite pixels are converted.
    
    Invalid pixels remain NaN.
    Returns
    -------
    ndarray
        Gamma-Nought expressed in decibels.
    """
    result = np.full( data.shape,np.nan, dtype=np.float32,)

    valid = np.isfinite(data) & (data > 0)

    result[valid] = ( 10.0 * np.log10(data[valid]) ).astype(np.float32)
    return result


# ============================================================
# PIPELINE
# ============================================================
 
def run(product_dir,output_dir=None,save_linear=True,save_db=True,):
    """
    Execute the complete Gamma-Nought processing workflow.
    
    Processing steps
    ----------------
    
    1. Locate BIOMASS product files.
    2. Read the amplitude image.
    3. Read the radiometric LUT.
    4. Build the SAR image temporal grid.
    5. Verify LUT coverage.
    6. Interpolate Gamma-Nought.
    7. Compute Gamma-Nought for each polarization.
    8. Export linear and dB GeoTIFF products. 
    """
    
    print(f"\n{'=' * 60}")
    print(f"Product : {os.path.basename(product_dir)}")
    print(f"{'=' * 60}")

    out_dir = output_dir if output_dir else product_dir
    os.makedirs(out_dir, exist_ok=True)
    abs_tiff, lut_nc, annot_xml = find_files(product_dir)
    basename = os.path.basename(abs_tiff).replace("_i_abs.tiff", "",)

    (amp,profile, gcps, gcp_crs, file_tags, band_tags, nodata, ) = read_amplitude(abs_tiff)
    
    print("Reading radiometric LUT...")
    (gamma_lut, lut_az_axis, lut_rg_axis, reference_azimuth_time,) = read_lut_geometry(lut_nc)

    print("Reading SAR image geometry...")
    ( target_az_axis, target_rg_axis, n_lines, n_samples,) = read_sar_image_geometry(annot_xml, reference_azimuth_time, )

    if amp.shape[1:] != (n_lines, n_samples):
        raise ValueError(
            f"GeoTIFF dimensions {amp.shape[1:]} "
            f"do not match the Main Annotation ADS "
            f"({n_lines}, {n_samples})."
        )

    check_lut_coverage( lut_az_axis, lut_rg_axis, target_az_axis, target_rg_axis, )

    print(
        f"  [CHECK] LUT azimuth: "
        f"{lut_az_axis[0]:.9f} .. "
        f"{lut_az_axis[-1]:.9f} s"
    )
    print(
        f"  [CHECK] Image azimuth: "
        f"{target_az_axis[0]:.9f} .. "
        f"{target_az_axis[-1]:.9f} s"
    )
    print(
        f"  [CHECK] LUT range: "
        f"{lut_rg_axis[0]:.12f} .. "
        f"{lut_rg_axis[-1]:.12f} s"
    )
    print(
        f"  [CHECK] Image range: "
        f"{target_rg_axis[0]:.12f} .. "
        f"{target_rg_axis[-1]:.12f} s"
    )

    print(
        "  Interpolating Gamma-Nought LUT "
    )

    gamma_full = interpolate_lut( gamma_lut,lut_az_axis, lut_rg_axis, target_az_axis, target_rg_axis,)
    
    save_tiff( gamma_full,profile, gcps, gcp_crs, file_tags,band_tags[1],    nodata,
              os.path.join(out_dir, f"{basename}_GammaLUT_resampled.tif"), pol="NONE", quantity="GammaNought_LUT")

    if gamma_full.shape != amp.shape[1:]:
        raise ValueError(
            f"Interpolated LUT shape {gamma_full.shape} "
            f"does not match image shape {amp.shape[1:]}.")

    for band_idx, pol in enumerate( POLARISATIONS, start=1, ):
        print(
            f"\n  Polarization {pol} "
            f"(band {band_idx})..." )

        gamma0 = compute_gamma0( amp[band_idx - 1], gamma_full, )
        valid = np.isfinite(gamma0) & (gamma0 > 0)

        if not np.any(valid):
            raise ValueError(
                f"No valid Gamma-Nought pixels found for {pol}.")

        mean_lin = np.mean(gamma0[valid])
        db_of_mean = 10.0 * np.log10(mean_lin)
        mean_db = np.mean( 10.0 * np.log10(gamma0[valid]))

        print(
            f"  γ⁰ {pol}: "
            f"mean linear={mean_lin:.6e}, "
            f"10log10(mean)={db_of_mean:.3f} dB, "
            f"mean dB={mean_db:.3f} dB" )
        
        gamma0_db = to_db(gamma0)
        valid_db = np.isfinite(gamma0_db)        
        min_db = np.min(gamma0_db[valid_db])
        max_db = np.max(gamma0_db[valid_db])
        mean_db = np.mean(gamma0_db[valid_db])
        std_db = np.std(gamma0_db[valid_db])
        median_db = np.median(gamma0_db[valid_db])        
        print(
            f"  γ⁰ {pol}: "
            f"min={min_db:.6f} dB, "
            f"max={max_db:.6f} dB, "
            f"mean={mean_db:.6f} dB, "
            f"std={std_db:.6f} dB, "
            f"median={median_db:.6f} dB"
        )

        def out(suffix):
            return os.path.join( out_dir,f"{basename}_{pol}_{suffix}.tif", )

        args = ( profile, gcps, gcp_crs, file_tags, band_tags[band_idx],  nodata, )

        if save_linear: 
            save_tiff( gamma0, *args,out("gamma0"), pol,"gamma0", )

        if save_db:
            save_tiff(to_db(gamma0),*args, out("gamma0_dB"), pol, "gamma0_dB", )

    print("Processing completed.")
# ============================================================
if __name__ == "__main__":
    run(
        product_dir = PRODUCT_DIR,
        output_dir  = OUTPUT_DIR,
        save_linear = SAVE_LINEAR,
        save_db     = SAVE_DB,
    )