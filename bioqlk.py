#!/usr/bin/env python3

# Copyright 2025, European Space Agency (ESA)
# Licensed under ESA Software Community License Permissive (Type 3) – v2.4
#
# Script Dependencies:
#   numpy
#   scipy
#   rasterio
#   matplotlib   # quick plotting (optional)
#   argcomplete  # cli (optional)
#
# NOTE: the tool also used the "gdalwarp" and "gdal_translate" programs.
#
# Version history:
#
# * v1.6 (WIP):
#     - Specify script dependencies in a PEP 722 compliant way
#       (https://peps.python.org/pep-0722)
#     - New functions for polarimetric RLLR coherence computation
#     - Use ".tif" instead of ".tiff" as extension for GeoTIFF files
#       (to make the MAAP viewer happy)
#     - New "wrap_to_cog" function
#     - Now "wrap_to_cog" and "save_kmz" exploit the same core function
#       ("_warp_image")
#     - In case the "quick-test" option is specified for "make_rgm",
#       the KMZ is never generated
# * v1.5:
#     - expose the "quick_test" option of the "make_rgb" function to CLI
#     - add the "quick_test" option to the "make_polarimetric_coherence"
#       function and the corresponding CLI sub-command
# * v1.4:
#     - new cpx_coherence function
#     - quick-look generation is now a sub-command of the main CLI
#     - new "coh" sub-command for polarimetric coherence generation
#     - add support for HSV representation
# * v1.3:
#     - new scaling method based on mean value
#     - new option for selecting the data scale: linear amplitude,
#       linear power or dB
#     - move the computation of scaling extrema (for different scaling
#       methods) to dedicated functions
#     - add copyright statement
#     - always express phase corrections in radians
#     - save the data in floating point format before any smoothing and scaling
#     - switch to the new reader function exploiting the GDAL VRT file for
#       complex data
#     - the KMZ resolution is now configurable
# * v1.2:
#     - added option for lexicographic representation
#     - fix GDAL warnings about affine transform
#     - add option to save PNG
#     - add logging of scaling thresholds
#     - change default values for phase corrections to [0, 0, 0, 0]
#     - add parameters to CLI
#     - perform smoothing before scaling
#     - allow to pass scaling parameter via command line
# * v1.1:
#     - add the capability to fix phase imbalances coming from the iCal
#     - generated TIFF files are now compressed
#     - KMZ resolution is reduced by a factor 10 (currently hardcoded)
#       w.r.t. the original image. The resampling method is now "average"
#     - default logging level set to "WARNING"
#     - suppress warnings about missing affine transforms coming from rasterio
#     - fix a bug with alpha channel
# * v1.0:
#     Initial release
#
# PYTHON_ARGCOMPLETE_OK

"""Tools for the generation of quick-looks images for BIOMASS.

Currently the Pauli (RGB and HSV) and Lexicographic (RGB) representations
are supported for quick-looks ("ql" sub-command).
Several normalization options are available.

Generation of polarimetric coherence images is supported via "coh" sub-command.

Generation of polarimetric RL-LR  coherence is supported via "rllr"
sub-command.

Data can be saved in different formats, including KMZ.
"""

import os
import enum
import logging
import pathlib
import argparse
import warnings
import subprocess

import numpy as np
import rasterio as rio
import rasterio.warp

from scipy import ndimage


__version__ = "1.6.dev0"
_log = logging.getLogger(__name__)


DEFAULT_THRESHOLD = 0.01
DEFAULT_CSCALE = 2.5
DEFAULT_WINDOW_SIZE = (15, 3)
TIFF_EXT = ".tif"  # ".tiff" is not OK for the MAAP viewer


class EScalingMethod(enum.Enum):
    """Methods for RGB channels scaling."""

    QUANTILE = "QUANTILE"
    MEAN = "MEAN"
    MANUAL = "MANUAL"

    def __str__(self):
        return self.name


class EScale(enum.Enum):
    """Data scale."""

    LINEAR_AMPLITUDE = "LINEAR_AMPLITUDE"
    LINEAR_POWER = "LINEAR_POWER"
    DB = "DB"

    def __str__(self):
        return self.name


def _first(iterable):
    for item in iterable:
        return item
    raise ValueError("empty iterable")


def load_data(product_path: str, *, return_metadata: bool = False):
    """Load data from a BIOMASS products.

    Limitations:
    * currently only works for L1A products (4 polarimetric channels).

    Parameters
    ----------

    product_path : str
        path to the BIOMASS product (folder)
    return_metadata : bool
        return metadata and GCPs in addition to the 4 polarizations
    """

    basepath = pathlib.Path(product_path)

    try:
        file_path = _first(basepath.glob("measurement/bio_s1_scs__1s_*_i.vrt"))
    except ValueError:
        raise FileNotFoundError(
            f"unable to find 'bio_s1_scs__1s_*_i.vrt' in "
            f"{basepath / 'measurement'}"
        )

    with rio.open(file_path) as src:
        data = src.read()
        metadata = src.meta.copy()
        gcps = src.get_gcps()

    hh, hv, vh, vv = data

    if return_metadata:
        return hh, hv, vh, vv, metadata, gcps
    else:
        return hh, hv, vh, vv


def _smooth_cpx(
    data,
    winsize: int | tuple[int, int],
    *,
    decimate: bool | int | tuple[int, int] = False,
):
    # arr_filled = np.nan_to_num(arr, nan=0.0)
    real_smoothed = ndimage.uniform_filter(data.real, size=winsize)
    imag_smoothed = ndimage.uniform_filter(data.imag, size=winsize)
    smoothed = real_smoothed + 1j * imag_smoothed
    
    if decimate:
        if decimate is True:
            dy, dx = winsize
        else:
            dy, dx = decimate
        assert dx > 0 and dy > 0
        smoothed = smoothed[::dy, ::dx]

    return smoothed


def fix_ical_phase(hh, hv, vh, vv, pc_hh, pc_hv, pc_vh, pc_vv=0.0):
    """Correct phase imbalances between polarimetric channels.

    pc_hh, pc_hv, pc_vh and pc_vv are phase corrections expressed in radians.
    """
    hh_out = hh * np.exp(1j * pc_hh)
    hv_out = hv * np.exp(1j * pc_hv)
    vh_out = vh * np.exp(1j * pc_vh)
    vv_out = vv * np.exp(1j * pc_vv)
    
    return hh_out, hv_out, vh_out, vv_out


# https://step.esa.int/main/wp-content/help/versions/12.0.0/snap-toolboxes/org.csa.rstb.rstb.op.polarimetric.tools.ui/operators/PolarimetricDecompositionOp.html
def pauli(
    hh,
    hv,
    vh,
    vv,
    *,
    scale: EScale = EScale.DB,
    winsize: tuple[int, int] | None = None,
    decimate: bool | int | tuple[int, int] = False,
):
    """Compute the Pauli representation.

    * Red band: 0.5 * |hh - vv| ** 2
    * Green band: 0.5 * |hv + vh| ** 2 (for L1A) or |hv| ** 2 (for L1C)
    * Blue band: 0.5 * |hh + vv| ** 2
    """
    if winsize is not None and max(winsize) > 1:
        assert winsize > 0
        red = 1 / np.sqrt(2) * np.abs(
            _smooth_cpx(hh - vv, winsize, decimate=decimate)
        )
        green = 1 / np.sqrt(2) * np.abs(
            _smooth_cpx(hv + vh, winsize, decimate=decimate)
        )
        blue = 1 / np.sqrt(2) * np.abs(
            _smooth_cpx(hh + vv, winsize, decimate=decimate)
        )
    else:
        red = 1 / np.sqrt(2) * np.abs(hh - vv)
        green = 1 / np.sqrt(2) * np.abs(hv + vh)
        blue = 1 / np.sqrt(2) * np.abs(hh + vv)

    if scale == EScale.LINEAR_POWER:
        red = red**2
        green = green**2
        blue = blue**2
    elif scale == EScale.DB:
        red = 20 * np.ma.log10(red) 
        green = 20 * np.ma.log10(green)
        blue = 20 * np.ma.log10(blue) 
        
    return red, green, blue


def hsv(
    hh,
    hv,
    vh,
    vv,
    *,
    scale: EScale = EScale.DB,
    kscale: float = 2.5,
    winsize: tuple[int, int] | None = None,
    decimate: bool | int | tuple[int, int] = False,
):
    """Compute the HSV representation based on Pauli decomposition."""
    if scale != EScale.LINEAR_AMPLITUDE:
        raise NotImplementedError(
            "only linear amplitude scaling is supported by hsv representation"
        )
    p2, p3, p1 = pauli(
        hh, hv, vh, vv, scale=scale, winsize=winsize, decimate=decimate
    )

    amplitude = np.sqrt(p1**2 + p2**2 + p3**2)
    alpha = np.arccos(p1 / amplitude) / (np.pi / 2.0)
    saturation = np.ones_like(alpha)

    vmax = kscale * np.mean(amplitude)
    amplitude = np.clip(amplitude, 0, vmax) / vmax

    hue, saturation, value = alpha, saturation, amplitude

    return hue, saturation, value


def lexicographic(
    hh,
    hv,
    vh,
    vv,
    *,
    scale: EScale = EScale.LINEAR_AMPLITUDE,
    winsize: tuple[int, int] | None = None,
    decimate: bool | int | tuple[int, int] = False,
):
    """Compute the Lexicographic representation.

    * Red channel: |hh| ** 2
    * Green channel: 0.5 * (|hv| ** 2 + |vh| ** 2)
    * Blue channel: |vv| ** 2
    """
    if winsize is not None:
        red = np.abs(_smooth_cpx(hh, winsize, decimate=decimate))
        # green = 1 / np.sqrt(2) * (
        #     np.abs(_smooth_cpx(hv, winsz))
        #     + np.abs(_smooth_cpx(vh, winsz))
        # )
        green = 1 / np.sqrt(2) * np.abs(
            _smooth_cpx(hv + vh, winsize, decimate=decimate)
        )
        blue = np.abs(_smooth_cpx(hh + vv, winsize, decimate=decimate))
    else:
        red = np.abs(hh)
        # green = 1 / np.sqrt(2) * (np.abs(hv) + np.abs(vh))
        green = 1 / np.sqrt(2) * np.abs(hv + vh)
        blue = np.abs(vv)

    if scale == EScale.LINEAR_POWER:
        red = red**2
        green = green**2
        blue = blue**2
    elif scale == EScale.DB:
        red = 20 * np.ma.log10(red)
        green = 20 * np.ma.log10(green)
        blue = 20 * np.ma.log10(blue)

    return red, green, blue


def _bulk_rllr(hh, hv, vh, vv):
    """Computation of polarimetric RL-LR product.

    The RL-LR product computed as follow::

        sym = hv - vh
        ant = hh + vv
        rl = (-sym + 1j * ant) / 2
        lr = (+sym + 1j * ant) / 2
        rllr = rl * np.conj(lr)
    """
    sym = hv - vh
    ant = hh + vv
    rl = (-sym + 1j * ant) / 2
    lr = (+sym + 1j * ant) / 2
    return rl, lr


def rllr_coherence(
    hh,
    hv,
    vh,
    vv,
    *,
    winsize: int | tuple[int, int] = DEFAULT_WINDOW_SIZE,
    decimate: bool | int | tuple[int, int] = False,
):
    """Computation of polarimetric RL-LR product.

    The RL-LR product computed as follow::

        sym = hv - vh
        ant = hh + vv
        rl = (-sym + 1j * ant) / 2
        lr = (+sym + 1j * ant) / 2
        rllr = rl * np.conj(lr)
    """
    rl, lr = _bulk_rllr(hh, hv, vh, vv)
    return cpx_coherence(rl, lr, winsize=winsize, decimate=decimate)


def smooth_nan(
    data,
    winsize: int | tuple[int, int],
    *,
    decimate: bool | int | tuple[int, int] = False,
    mode: str = "nearest",
):
    """Smooth the input 2D array.

    The uniform filter of `scipy.ndimage`:

    * has no NaNn handling ->  np.nan_to_num
    * mode parameter: `nearest`: (a a a a | a b c d | d d d d)
      -> The input is extended by replicating the last pixel.
    """
    nan_indices = np.where(np.isnan(data))
    data = np.nan_to_num(data, nan=0.0)
    smoothed = ndimage.uniform_filter(data, size=winsize, mode=mode)
    smoothed[nan_indices] = np.nan
    if decimate:
        if decimate is True:
            dy, dx = winsize
        else:
            dy, dx = decimate
        assert dx > 0 and dy > 0
        smoothed = smoothed[::dy, ::dx]
    return smoothed


def cpx_coherence(
    d1: np.ndarray,
    d2: np.ndarray,
    *,
    phase: np.ndarray | None = None,
    winsize: int | tuple[int, int] = DEFAULT_WINDOW_SIZE,
    decimate: bool | int | tuple[int, int] = False,
) -> np.ndarray:
    """Coherence computation.

    Parameters
    ----------
    d1 : np.ndarray
        first data channel
    d2 : np.ndarray
        second data channel
    phase : np.ndarray | None
        optional correction phase in radians
    winsize : int | tuple[int, int], optional
        size of the boxcar filter, if a single integer is provided, then the
        the same size is assume d for both dimensions of teh filter kernel.
    decimate : bool | int | tuple[int, int]
        enable data decimation after coherence computation.
        If decimate is "True" then the decimation factor is equal to "winzise".
        if a single positive integer is provided, then both the image
        dimensions are decimated with teh same value.

    Returns
    -------
    np.ndarray
        coherence array
    """
    num = d1 * d2.conj()
    if phase is not None:
        assert d1.shape == phase.shape
        num = num * np.exp(1j * phase)
    num = ndimage.uniform_filter(num, size=winsize, mode="constant", cval=0.0)
    den1 = ndimage.uniform_filter(
        np.abs(d1) ** 2, size=winsize, mode="constant", cval=0.0
    )
    den2 = ndimage.uniform_filter(
        np.abs(d2) ** 2, size=winsize, mode="constant", cval=0.0
    )

    den = np.ma.masked_less_equal(den1 * den2, 0)
    coherence = np.ma.masked_invalid(num / np.sqrt(den))
    coherence = np.ma.masked_where(np.abs(coherence) > 1, coherence)

    if decimate:
        if decimate is True:
            dy, dx = winsize
        else:
            dy, dx = decimate
        assert dx > 0 and dy > 0
        coherence = coherence[::dy, ::dx]

    return coherence


def quantile_scaling(data, threshold: float = DEFAULT_THRESHOLD):
    """Compute percentiles for the given threshold."""
    thresholds = [threshold, 1 - threshold]
    low, high = np.quantile(data.ravel(), thresholds)
    return low, high


def mean_scaling(data, cscale_factor: float = DEFAULT_CSCALE):
    """
    Compute the upper value according to the data mean and the provided factor.
    """
    low = 0
    high = cscale_factor * np.mean(data)

    return low, high


def scale_to_8bits(data, vmin: float, vmax: float):
    """Scale the data to be representable with 8bits (unsigned) integers."""
    normalized_image = 255 * (data - vmin) / (vmax - vmin)
    return normalized_image


def hsv_to_rgb(h, s, v, colormap: str = "turbo"):
    from matplotlib import pyplot as plt

    cmap = plt.colormaps.get(colormap)

    rgb = v[..., np.newaxis] * (
        (1 - s)[..., np.newaxis] * np.ones(3)
        + s[..., np.newaxis] * cmap(h)[..., :3]
    )
    return rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]


def to_rgb_array(red, green, blue, *, alpha: int | bool | None = None):
    """COnvert the input in a single uint8) 3D array for RGB.

    The convention is: "pixel interleaved".
    """
    assert red.shape == green.shape == blue.shape

    if alpha is False or alpha is None:
        n_channels = 3
    else:
        n_channels = 4
    shape = [*red.shape, n_channels]

    rgb = np.zeros(shape, dtype=np.uint8)
    rgb[:, :, 0] = np.clip(red, 0, 255)  # Red: Double-bounce
    rgb[:, :, 1] = np.clip(green, 0, 255)  # Green: Volume
    rgb[:, :, 2] = np.clip(blue, 0, 255)  # Blue: Surface

    if n_channels == 4:
        if alpha is True:
            alpha = 255
        assert 0 <= alpha <= 255

        rgb[:, :, 3] = np.full_like(rgb[:, :, 0], alpha)  # alpha
        black_areas = (
            (rgb[:, :, 0] == 0) & (rgb[:, :, 1] == 0) & (rgb[:, :, 2] == 0)
        )
        rgb[:, :, 3][black_areas] = 0

    return rgb


def save_rgb(rgb, outfile: str, *, metadata=None, gcps=None):
    """Save the RGB data into a (Geo)TIFF file."""
    if metadata is not None:
        kwargs = metadata.copy()
        kwargs.pop("nodata", None)
        kwargs.pop("crs", None)
        kwargs.pop("transform", None)
    else:
        kwargs = {}

    kwargs["dtype"] = rio.uint8
    kwargs["count"] = rgb.shape[-1]
    kwargs["driver"] = "gtiff"
    # kwargs["driver"] = "COG"  # TODO: check why this breaks GCPs

    # if gcps is not None:
    #     gcps, crs = gcps
    #     transform = rio.transform.GCPTransformer(gcps)
    # else:
    #     transform = None
    #     crs = None

    pathlib.Path(outfile).parent.mkdir(exist_ok=True, parents=True)

    with rio.open(
        outfile,
        "w",
        # transform=transform,
        # crs=crs,
        tiled=True,
        compress="deflate",
        **kwargs,
    ) as dst:
        if gcps is not None:
            dst.gcps = gcps

        dst.write(rgb[:, :, 0], 1)
        dst.write(rgb[:, :, 1], 2)
        dst.write(rgb[:, :, 2], 3)
        if rgb.shape[-1] == 4:
            dst.write(rgb[:, :, 3], 4)


def save_gtiff(data, outfile: str, *, metadata=None, gcps=None):
    """Save the data into a (Geo)TIFF file."""
    if metadata is not None:
        kwargs = metadata.copy()
        kwargs.pop("nodata", None)
        kwargs.pop("crs", None)
        kwargs.pop("transform", None)
    else:
        kwargs = {}

    pathlib.Path(outfile).parent.mkdir(exist_ok=True, parents=True)

    data = np.asarray(data)
    kwargs["dtype"] = data.dtype
    assert data.ndim == 2 or (data.ndim == 3 and len(data) <= 4)
    kwargs["count"] = len(data) if data.ndim == 3 else 1
    kwargs["driver"] = "gtiff"
    # kwargs["driver"] = "COG"  # TODO: check why this breaks GCPs

    # if gcps is not None:
    #     gcps, crs = gcps
    #     transform = rio.transform.GCPTransformer(gcps)
    # else:
    #     transform = None
    #     crs = None

    with rio.open(
        outfile,
        "w",
        # transform=transform,
        # crs=crs,
        tiled=True,
        compress="deflate",
        **kwargs,
    ) as dst:
        if gcps is not None:
            dst.gcps = gcps
        if data.ndim == 2:
            dst.write(data, 1)
        else:
            dst.write(data)


def _warp_image(
    filename,
    outfile,
    *,
    driver_name: str | None = None,
    t_srs: str = "EPSG:4326",
    resolution: float | None = None,
    resampling_mode: str | None = "average",
    compress: str | None = "DEFLATE",
    extra_opts: list[str] | None = None,
) -> None:
    """Wrapper for gdalwarp.

    This function resamples the input image, that must have geo-referencing
    information attached, onto a regular grid int the target SRS
    specified by the user (default: WGS84 - EPSG:4326).

    Parameters
    ----------

    filename : str
        input image file with geo-referencing information
    outfile : str
        output file
    resolution : float
        resolution of the output KMZ file expressed in degrees.
        If not provided the output resolution is automatically computed
        to match the one of the input file.
        Note: a value of 0.001deg corresponds to about 110m at equator
        and 20m at 80deg latitude.
    """
    if pathlib.Path(outfile).exists():
        raise FileExistsError(outfile)
        # warnings.warn(f"'{outfile}' already exists, it will be overwritten.")
        # os.unlink(outfile)

    cmd = ["gdalwarp"]

    # outout format
    if driver_name is None:
        try:
            driver_name = rio.driver_from_extension(outfile)
        except ValueError:
            driver_name = None
    if driver_name is not None:
        cmd.extend(["-of", driver_name])

    # target SRS
    cmd.extend(["-t_srs", t_srs])

    # resampling algorithm
    if resampling_mode is not None:
        cmd.extend(["-r", resampling_mode])

    # compression
    if compress is not None:
        cmd.extend(["-co", f"COMPRESS={compress}"])

    # target spacing
    if resolution is not None:
        cmd.extend(["-tr", str(resolution), str(resolution)])

    # pass through extra option
    if extra_opts is None:
        # NOTE: no check is performed on extra options
        # TODO:
        #   * validate user provided extra options
        #   * raise warning and discard if options overlapping with other
        #     parameters are provided
        cmd.extend(extra_opts)

    # input and output files
    cmd.extend([str(filename), str(outfile)])

    subprocess.run(cmd, check=True)


# TODO: implement this using directly the rasterio API
def save_kmz(
    filename: str,
    outfile: str | None = None,
    *,
    resolution: float | None = None,
) -> pathlib.Path:
    """Generate a geocoded KMZ file from an RGB GeoTIFF.

    Parameters
    ----------

    filename : str
        input image file with geo-referencing information
    outfile : str
        output KMZ file
    resolution : float
        resolution of the output KMZ file expressed in degrees.
        If not provided the output resolution is automatically computed
        to match the one of the input file.
        Note: a value of 0.001deg corresponds to about 110m at equator
        and 20m at 80deg latitude.
    """
    filename = pathlib.Path(filename)
    if not outfile:
        outfile = pathlib.Path(filename.name).with_suffix(".kmz")
    else:
        outfile = pathlib.Path(outfile)

    # TODO: remove this
    if outfile.exists():
        warnings.warn(f"'{outfile}' already exists, it will be overwritten.")
        os.unlink(outfile)

    extra_opts = []
    extra_opts.extend(["-co", "format=png"])
    extra_opts.extend(["-wo", "OPTIMIZE_SIZE=YES"])
    # extra_opts.append("-tps")

    _warp_image(
        filename,
        outfile,
        resolution=resolution,
        driver_name="KMLSUPEROVERLAY",
        extra_opts=extra_opts,
    )
    return outfile


def warp_to_cog(
    filename,
    outfile: str | None = None,
    t_srs: str = "EPSG:4326",
    resolution: float | None = None,
    compress: str | None = "DEFLATE",
) -> pathlib.Path:
    """Generate a geocoded COG file from an RGB GeoTIFF.

    Parameters
    ----------

    filename : str
        input image file with geo-referencing information
    outfile : str
        output KMZ file
    t_srs: str (optional)
        target spatial reference (default: EPSG:4326 a.k.a. WGS84)
    resolution : float (optional)
        resolution of the output COG file expressed in target coordinates.
        If not provided the output resolution is automatically computed
        to match the one of the input file.
        Note: a value of 0.001deg corresponds to about 110m at equator
        and 20m at 80deg latitude.
    compress: str (optional)
        compression scheme (default: "DEFLATE")
        If it is set to `None`, no compression is performed.
    """
    filename = pathlib.Path(filename)
    if outfile is None:
        outfile = filename.with_name(f"{filename.stem}_warped{TIFF_EXT}")
    else:
        outfile = pathlib.Path(outfile)

    # TODO: remove this
    if outfile.exists():
        warnings.warn(f"'{outfile}' already exists, it will be overwritten.")
        os.unlink(outfile)

    _warp_image(
        filename,
        outfile,
        driver_name="COG",
        t_srs=t_srs,
        compress=compress,
    )

    return outfile


def save_png(
    filename: str,
    outfile: str | None = None,
    *,
    outsize: tuple[int, int] | None = None,
):
    cmd = [
        "gdal_translate",
        "-of",
        "PNG",
        "-outsize",
    ]
    if outsize is not None:
        cmd.extend(
            [
                f"{outsize[0]}%",
                f"{outsize[0]}%",
            ]
        )
    cmd.extend(
        [
            filename,
            outfile,
        ]
    )
    subprocess.run(cmd)


def save_png_rgb(rgb, filename: str):
    import matplotlib.pyplot as plt

    # plt.figure(figsize=plt.figaspect(2))
    plt.figure(figsize=(3, 9))
    plt.imshow(rgb, aspect="auto")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close()


def make_rgb(
    product_path: str,
    outfile: str | None = None,
    *,
    representation="pauli",
    scale: EScale = EScale.DB,
    alpha: int | bool | None = None,
    scaling_method: EScalingMethod = EScalingMethod.MEAN,
    kscale: float | None = None,
    scaling_ranges: dict[str, float] | None = None,
    winsize: tuple[int, int] = (20, 4),
    decimate: bool = False,
    quick_test: bool = False,
    phase_corrections: tuple[float, float, float, float] = (0, 0, 0, 0),
    float_file: str | None = None,
    kmz_file: str | None = None,
    kmz_res: float = 0.001,
    png_file: str | None = None,
    outdir: str | None = None,
):
    """Read a BIOMASS L1A product and generate an RGB.

    Pauli or lexicographic representations can be used.

    Known limitation:
    * only L1A (SLC) product are supported so far (no L1B and no L1C)
    * pauli/lexicographical channels are scaled independently
    """
    product_path = pathlib.Path(product_path)
    outdir = pathlib.Path(outdir) if outdir else None
    scaling_method = EScalingMethod(scaling_method)
    scale = EScale(scale)

    kwargs = {}
    if isinstance(representation, str):
        representation_name = representation
        if representation_name in globals():
            representation = globals()[representation_name]
        else:
            raise ValueError(f"invalid decomposition: {representation_name!r}")
        if representation_name == "hsv":
            kwargs["kscale"] = kscale if kscale is not None else DEFAULT_CSCALE
    else:
        representation_name = representation.__name__

    _log.info("Loading: %s", product_path)
    hh, hv, vh, vv, metadata, gcps = load_data(
        product_path, return_metadata=True
    )

    if tuple(phase_corrections) != (0.0, 0.0, 0.0, 0.0):
        assert len(phase_corrections) == 4  # TODO: make proper error control
        _log.info("phase correction: %s", phase_corrections)
        hh, hv, vh, vv = fix_ical_phase(hh, hv, vh, vv, *phase_corrections)

    _log.info("%s representation", representation_name)
    if representation_name == "hsv":
        hue, saturations, value = representation(
            hh, hv, vh, vv, scale=scale, **kwargs
        )
        red, green, blue = hsv_to_rgb(hue, saturations, value)
    else:
        red, green, blue = representation(hh, hv, vh, vv, scale=scale)

    # smoothing
    if max(winsize) > 1:
        _log.info("Smoothing (winzise=%s)", winsize)
        assert min(winsize) >= 1
        red = smooth_nan(red, winsize, decimate=decimate)
        green = smooth_nan(green, winsize, decimate=decimate)
        blue = smooth_nan(blue, winsize, decimate=decimate)

    # IMPORTANT:
    #   the function for saving unscaled data must be called before
    #   overwriting the red, green and blue channels with the smoothed
    #   version of the data
    if float_file:
        if float_file is True:
            float_file = (
                f"{product_path.name}_{representation_name}{TIFF_EXT}"
            )
            if outdir is not None:
                float_file = outdir / float_file

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", rio.rasterio.errors.NotGeoreferencedWarning
            )
            data = np.asarray([red, green, blue], dtype=np.float32)
            save_gtiff(data, float_file, metadata=metadata, gcps=gcps)
            del data

    # scaling
    _log.info("scaling method: %s", scaling_method.name)
    computed_scaling_ranges = {}
    if representation_name == "hsv":
        # normalization is inside hsv()
        low, high = 0, 1
        computed_scaling_ranges["red_low"] = low
        computed_scaling_ranges["red_high"] = high
        computed_scaling_ranges["green_low"] = low
        computed_scaling_ranges["green_high"] = high
        computed_scaling_ranges["blue_low"] = low
        computed_scaling_ranges["blue_high"] = high

    elif scaling_method == EScalingMethod.QUANTILE:
        assert scaling_ranges is None
        threshold = kscale if kscale is not None else DEFAULT_THRESHOLD
        _log.info("threshold: %s", threshold)

        low, high = quantile_scaling(red, threshold=threshold)
        computed_scaling_ranges["red_low"] = low
        computed_scaling_ranges["red_high"] = high

        low, high = quantile_scaling(green, threshold=threshold)
        computed_scaling_ranges["green_low"] = low
        computed_scaling_ranges["green_high"] = high

        low, high = quantile_scaling(blue, threshold=threshold)
        computed_scaling_ranges["blue_low"] = low
        computed_scaling_ranges["blue_high"] = high

    elif scaling_method == EScalingMethod.MEAN:
        assert scaling_ranges is None
        cscale_factor = kscale if kscale is not None else DEFAULT_CSCALE
        _log.info("cscale_factor: %s", cscale_factor)

        low, high = mean_scaling(red, cscale_factor=cscale_factor)
        computed_scaling_ranges["red_low"] = low
        computed_scaling_ranges["red_high"] = high

        low, high = mean_scaling(green, cscale_factor=cscale_factor)
        computed_scaling_ranges["green_low"] = low
        computed_scaling_ranges["green_high"] = high

        low, high = mean_scaling(blue, cscale_factor=cscale_factor)
        computed_scaling_ranges["blue_low"] = low
        computed_scaling_ranges["blue_high"] = high

    elif scaling_method == EScalingMethod.MANUAL:
        if scaling_ranges is None:
            computed_scaling_ranges["red_low"] = red.min()
            computed_scaling_ranges["red_high"] = red.max()

            computed_scaling_ranges["green_low"] = green.min()
            computed_scaling_ranges["green_high"] = green.max()

            computed_scaling_ranges["blue_low"] = blue.min()
            computed_scaling_ranges["blue_high"] = blue.max()
        else:
            computed_scaling_ranges = scaling_ranges

    else:
        raise ValueError(f"invalid scaling method: {scaling_method}")

    scaling_ranges = computed_scaling_ranges

    vmin = scaling_ranges["red_low"]
    vmax = scaling_ranges["red_high"]
    _log.info("R: vmin=%f, vmax=%f", vmin, vmax)
    red = scale_to_8bits(red, vmin=vmin, vmax=vmax)

    vmin = scaling_ranges["green_low"]
    vmax = scaling_ranges["green_high"]
    _log.info("G: vmin=%f, vmax=%f", vmin, vmax)
    green = scale_to_8bits(green, vmin=vmin, vmax=vmax)

    vmin = scaling_ranges["blue_low"]
    vmax = scaling_ranges["blue_high"]
    _log.info("B: vmin=%f, vmax=%f", vmin, vmax)
    blue = scale_to_8bits(blue, vmin=vmin, vmax=vmax)

    print(scaling_ranges)

    rgb = to_rgb_array(red, green, blue, alpha=alpha)

    # TODO: consistent scaling
    # s = np.stack([red, green, blue])
    # red, green, blue = bioqlk.scale_to_8bits(s, threshold=threshold)
    # del s

    if not quick_test:
        if not outfile:
            outfile = (
                f"{product_path.name}_{representation_name}_rgb{TIFF_EXT}"
            )
            if outdir is not None:
                outfile = outdir / outfile

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", rio.rasterio.errors.NotGeoreferencedWarning
            )
            save_rgb(rgb, outfile, metadata=metadata, gcps=gcps)

        if kmz_file:
            if kmz_file is True:
                kmz_file = f"{product_path.name}_{representation_name}_rgb.kmz"
                if outdir is not None:
                    kmz_file = outdir / kmz_file

            save_kmz(outfile, kmz_file, resolution=kmz_res)

        if png_file:
            if png_file is True:
                png_file = f"{product_path.name}_{representation_name}_rgb.png"
                if outdir is not None:
                    png_file = outdir / png_file

            outsize = [20, 20]

            save_png(outfile, png_file, outsize=outsize)

        return outfile

    else:
        if not png_file or png_file is True:
            png_file = f"{product_path.name}_{representation_name}_rgb.png"
            if outdir is not None:
                png_file = outdir / png_file

        save_png_rgb(rgb, png_file)

        return png_file


def plot_coherence_statistics(coh, outfile: str | None = None):
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=[10, 8])

    coh_abs = np.abs(coh)

    img = ax[0, 0].imshow(coh_abs, vmin=0, vmax=1, aspect="auto", cmap="gray")
    ax[0, 0].grid()
    ax[0, 0].set_xlabel("samples")
    ax[0, 0].set_ylabel("lines")
    ax[0, 0].set_title("|coh|")
    fig.colorbar(img, ax=ax[0, 0])

    h, b, _ = ax[0, 1].hist(
        coh_abs.ravel(),
        bins=500,
        density=True,
        label="|coh|",
        range=[0, 1],
    )
    ax[0, 1].grid()
    ax[0, 1].set_title("|coh| histogram")
    b = (b[:-1] + b[1:]) / 2
    mode = b[np.argmax(h)]
    ax[0, 1].text(
        0.02,
        0.95,
        f"mode: {mode:.02f}",
        transform=ax[0, 1].transAxes,
    )

    del coh_abs, h, b, _
    coh_ph = np.rad2deg(np.angle(coh))

    img = ax[1, 0].imshow(coh_ph, aspect="auto", interpolation="nearest")
    ax[1, 0].grid()
    ax[1, 0].set_xlabel("samples")
    ax[1, 0].set_ylabel("lines")
    ax[1, 0].set_title(r"$\angle coh$")
    cb = fig.colorbar(img, ax=ax[1, 0])
    cb.set_label("deg")

    ax[1, 1].hist(
        coh_ph.ravel(),
        bins=500,
        density=True,
        label="angle(coh)",
        range=[-180, +180],
    )
    ax[1, 1].grid()
    ax[1, 1].set_xlabel("deg")
    ax[1, 1].set_title("phase histogram")

    mean = coh_ph.mean()
    std = coh_ph.std()

    ax[1, 1].text(
        0.02,
        0.95,
        f"mean: {mean:.02f}",
        transform=ax[1, 1].transAxes,
    )
    ax[1, 1].text(
        0.02,
        0.90,
        f"std: {std:.02f}",
        transform=ax[1, 1].transAxes,
    )

    del coh_ph

    plt.tight_layout()

    if outfile is not None:
        fig.savefig(outfile)

    plt.close(fig)


def make_polarimetric_coherence(
    product_path: str,
    outfile: str | None = None,
    *,
    channels: str = "hh-vv",
    winsize: tuple[int, int] = (10, 2),
    phase_corrections: tuple[float, float, float, float] = (0, 0, 0, 0),
    kmz_file: str | None = None,
    kmz_res: float = 0.001,
    png_file: str | None = None,
    quick_test: bool = False,
    outdir: str | None = None,
):
    """Read a BIOMASS L1A product and generate a polarimetric coherence map.

    Known limitation:
    * only L1A (SLC) product are supported so far (no L1C).
    """
    product_path = pathlib.Path(product_path)
    outdir = pathlib.Path(outdir) if outdir else None

    _log.info("Loading: %s", product_path)
    hh, hv, vh, vv, metadata, gcps = load_data(
        product_path, return_metadata=True
    )

    if tuple(phase_corrections) != (0.0, 0.0, 0.0, 0.0):
        assert len(phase_corrections) == 4  # TODO: make proper error control
        _log.info("phase correction: %s", phase_corrections)
        hh, hv, vh, vv = fix_ical_phase(hh, hv, vh, vv, *phase_corrections)

    channels_str = channels
    if channels_str == "hh-vv":
        channels = (hh, vv)
    elif channels_str == "hh-hv":
        channels = (hh, hv)
    elif channels_str == "vv-vh":
        channels = (vv, vh)
    elif channels_str == "hv-vh":
        channels = (hv, vh)
    else:
        raise ValueError(f"unexpected channels combination: {channels}")

    _log.info("compute polarimetric coherence on channels: %s", channels_str)
    coh = cpx_coherence(*channels, winsize=winsize)

    if not quick_test:
        if not outfile:
            outfile = f"{product_path.name}_coh_{channels_str}{TIFF_EXT}"
            if outdir is not None:
                outfile = outdir / outfile

        save_gtiff(
            coh.astype(np.complex64), outfile, metadata=metadata, gcps=gcps
        )
    # else:
    #     if not png_file or png_file is True:
    #         png_file = f"{product_path.name}_coh_{channels_str}.png"
    #         if outdir is not None:
    #             png_file = outdir / png_file
    #
    #     save_png_rgb(np.abs(coh), png_file)

    if kmz_file:
        if kmz_file is True:
            kmz_file = f"{product_path.name}_coh_{channels_str}.kmz"
            if outdir is not None:
                kmz_file = outdir / kmz_file

        dst_resolution = [kmz_res, kmz_res]
        dst_crs = "WGS84"

        acoh = np.abs(coh)
        acoh = scale_to_8bits(acoh, 0, 1)
        acoh = np.round(acoh).clip(0, 255).astype(np.uint8)

        otransform, owidth, oheight = rio.warp.calculate_default_transform(
            src_crs=gcps[1],
            width=metadata["width"],
            height=metadata["height"],
            gcps=gcps[0],
            dst_crs=dst_crs,
            resolution=dst_resolution,
        )

        with rio.open(
            kmz_file,
            "w",
            driver="KMLSUPEROVERLAY",
            width=owidth,
            height=oheight,
            count=1,
            dtype=rio.uint8,
            transform=otransform,
            format="png",
            nodata=0,
            # **kwargs,
        ) as dst:
            rasterio.warp.reproject(
                source=acoh.data,
                destination=rio.band(dst, 1),
                src_crs=gcps[1],
                gcps=gcps[0],
                dst_crs=dst_crs,
                dst_resolution=dst_resolution,
                resampling=rio.enums.Resampling.average,
            )
        del acoh

    if png_file or quick_test:
        if png_file is True or (quick_test and not png_file):
            png_file = f"{product_path.name}_coh_{channels_str}.png"
            if outdir is not None:
                png_file = outdir / png_file

        plot_coherence_statistics(coh, outfile=png_file)

    return outfile


def make_rllr_coherence(
    product_path: str,
    outfile: str | None = None,
    *,
    winsize: tuple[int, int] = DEFAULT_WINDOW_SIZE,
    phase_corrections: tuple[float, float, float, float] = (0, 0, 0, 0),
    kmz_file: str | None = None,
    kmz_res: float = 0.001,
    png_file: str | None = None,
    quick_test: bool = False,
    outdir: str | None = None,
):
    """Read a BIOMASS L1A product and generate a RL-LR coherence map.

    Known limitation:
    * only L1A (SLC) product are supported so far (no L1C).
    """
    product_path = pathlib.Path(product_path)
    outdir = pathlib.Path(outdir) if outdir else None

    _log.info("Loading: %s", product_path)
    hh, hv, vh, vv, metadata, gcps = load_data(
        product_path, return_metadata=True
    )

    if tuple(phase_corrections) != (0.0, 0.0, 0.0, 0.0):
        assert len(phase_corrections) == 4  # TODO: make proper error control
        _log.info("phase correction: %s", phase_corrections)
        hh, hv, vh, vv = fix_ical_phase(hh, hv, vh, vv, *phase_corrections)

    _log.info("compute RL-LR coherence")
    rllr_coh = rllr_coherence(hh, hv, vh, vv, winsize=winsize)

    if not quick_test:
        if not outfile:
            outfile = f"{product_path.name}_rllr-coh{TIFF_EXT}"
            if outdir is not None:
                outfile = outdir / outfile

        save_gtiff(
            rllr_coh.astype(np.complex64),
            outfile,
            metadata=metadata,
            gcps=gcps,
        )
    # else:
    #     if not png_file or png_file is True:
    #         png_file = f"{product_path.name}_coh_{channels_str}.png"
    #         if outdir is not None:
    #             png_file = outdir / png_file
    #
    #     save_png_rgb(np.abs(coh), png_file)

    if kmz_file:
        if kmz_file is True:
            kmz_file = f"{product_path.name}_rllr-coh.kmz"
            if outdir is not None:
                kmz_file = outdir / kmz_file

        dst_resolution = [kmz_res, kmz_res]
        dst_crs = "WGS84"

        acoh = np.abs(rllr_coh)
        acoh = scale_to_8bits(acoh, 0, 1)
        acoh = np.round(acoh).clip(0, 255).astype(np.uint8)

        otransform, owidth, oheight = rio.warp.calculate_default_transform(
            src_crs=gcps[1],
            width=metadata["width"],
            height=metadata["height"],
            gcps=gcps[0],
            dst_crs=dst_crs,
            resolution=dst_resolution,
        )

        with rio.open(
            kmz_file,
            "w",
            driver="KMLSUPEROVERLAY",
            width=owidth,
            height=oheight,
            count=1,
            dtype=rio.uint8,
            transform=otransform,
            format="png",
            nodata=0,
            # **kwargs,
        ) as dst:
            rasterio.warp.reproject(
                source=acoh.data,
                destination=rio.band(dst, 1),
                src_crs=gcps[1],
                gcps=gcps[0],
                dst_crs=dst_crs,
                dst_resolution=dst_resolution,
                resampling=rio.enums.Resampling.average,
            )
        del acoh

    if png_file or quick_test:
        if png_file is True or (quick_test and not png_file):
            png_file = f"{product_path.name}_rllr-coh.png"
            if outdir is not None:
                png_file = outdir / png_file

        plot_coherence_statistics(rllr_coh, outfile=png_file)

    return outfile


# === CLI =====================================================================

EX_OK = 0
EX_FAILURE = 1
EX_INTERRUPT = 130

PROG = pathlib.Path(__file__).stem
# LOGFMT = "%(levelname)s: %(message)s"
LOGFMT = "%(asctime)s %(levelname)-8s -- %(message)s"
DEFAULT_LOGLEVEL = "INFO"


def _autocomplete(parser: argparse.ArgumentParser) -> None:
    try:
        import argcomplete
    except ImportError:
        pass
    else:
        argcomplete.autocomplete(parser)


def _add_logging_control_args(
    parser: argparse.ArgumentParser, default_loglevel: str = DEFAULT_LOGLEVEL
) -> argparse.ArgumentParser:
    """Add command line options for logging control."""
    loglevels = [logging.getLevelName(level) for level in range(10, 60, 10)]

    parser.add_argument(
        "--loglevel",
        default=default_loglevel,
        choices=loglevels,
        help="logging level (default: %(default)s)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        dest="loglevel",
        action="store_const",
        const="ERROR",
        help=(
            "suppress standard output messages, "
            "only errors are printed to screen (set 'loglevel' to 'ERROR')"
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        action="store_const",
        const="INFO",
        help="print verbose output messages (set 'loglevel' to 'INFO')",
    )
    parser.add_argument(
        "-d",
        "--debug",
        dest="loglevel",
        action="store_const",
        const="DEBUG",
        help="print debug messages (set 'loglevel' to 'DEBUG')",
    )

    return parser


def _get_ql_parser(subparsers) -> argparse.ArgumentParser:
    """Set up the argument parser for the "ql" sub-command."""
    name = "ql"
    synopsis = "generate RGB quick-looks images for BIOMASS"
    doc = """Tool for the generation of RGB quick-looks images for BIOMASS.

    Currently the Pauli and Lexicographic representations are supported.
    Data can be saved in different formats, including KMZ.
    Several normalization options are available.
    """
    parser = subparsers.add_parser(name, description=doc, help=synopsis)

    parser.set_defaults(func=make_rgb)

    # command line options
    parser.add_argument(
        "-r",
        "--representation",
        default="pauli",
        help=(
            "polarimetric representation (default: %(default)s). "
            "Possible options are 'pauli', 'lexicographic' amd 'hsv'"
        ),
    )
    parser.add_argument(
        "--scale",
        type=EScale.__getitem__,
        default=EScale.LINEAR_AMPLITUDE,
        choices=EScale,
        help="data scale (default: %(default)s)",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        action="store_true",
        default=True,
        help="add an alpha channel to the RGB (default: True)",
    )
    parser.add_argument(
        "--no-alpha",
        action="store_false",
        dest="alpha",
        help=(
            "do not add an alpha channel to the RGB "
            "(the alpha channel is added by default)"
        ),
    )
    parser.add_argument(
        "--scaling-method",
        type=EScalingMethod.__getitem__,
        default=EScalingMethod.MEAN,
        choices=EScalingMethod,
        help=(
            "method used for scaling the data of each RGB channel "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--kscale",
        type=float,
        help=(
            "value used for the computation of the scaling range. "
            "If 'QUANTILE' is selected as scaling method then the two "
            "quantiles are determined as follows: "
            "q1 = 'kscale', q2 = (1 - 'kscale'). "
            f"The default value is {DEFAULT_THRESHOLD}. "
            "If 'MEAN' is selected as scaling method then the scaling ranges "
            "is determined as follows: [0, 'kscale' * mean(data)]. "
            f"(default: {DEFAULT_CSCALE})."
        ),
    )
    parser.add_argument(
        "--scaling-ranges",
        type=float,
        nargs=6,
        help=(
            "scaling extrema for the three channels (red, green and blue). "
            "The values are used value stretching the channels data to fit "
            "the dynamic range of 8 bit integers "
            "Example: --scaling-ranges "
            "red_low red_high green_low green_high blue_low blue_high"
        ),
    )
    parser.add_argument(
        "--winsize",
        type=int,
        nargs=2,
        default=[10, 2],
        help="Size of the smoothing window [lines, samples] "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--phase-corrections",
        nargs=4,
        type=float,
        default=(0.0, 0.0, 0.0, 0.0),
        help=(
            "phase corrections (in radians) to be applied to polarimetric "
            "channels (default: %(default)s). "
            "iCal factors: (-0.4145157, 1.98025057, -1.56556034, 0.0) radians "
            "corresponding to (-23.75, 113.46, -89.7, 0.0) degrees"
        ),
    )
    parser.add_argument(
        "--float",
        action="store_true",
        default=False,
        help=(
            "generate a TIFF file with floating point channels in "
            "addition to the RGB"
        ),
    )
    parser.add_argument(
        "--kmz",
        action="store_true",
        default=False,
        help="generate a KMZ file in addition to the RGB",
    )
    parser.add_argument(
        "--kmz-res",
        type=float,
        default=0.001,
        help=(
            "resolution (in degrees) of the generated KMZ file. "
            "It is only used if also the '--kmz' option is specified"
        ),
    )
    parser.add_argument(
        "--png",
        action="store_true",
        default=False,
        help="generate a PNG file in addition to the RGB",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=False,
        dest="quick_test",
        help=(
            "skip the generation of the full resolution GeoTIFF "
            "and only save a PNG quick-look"
        ),
    )

    parser.add_argument(
        "--outdir",
        help=(
            "path to the output directory (only used if filenames for "
            "output files are not provided)"
        ),
    )

    # positional arguments
    parser.add_argument(
        "product_path", help="Path to the BIOMASS product directory"
    )
    parser.add_argument(
        "outfile", nargs="?", default=None, help="path to the output TIFF file"
    )

    return parser


def _get_coh_parser(subparsers) -> argparse.ArgumentParser:
    """Set up the argument parser for the "coh" sub-command."""
    name = "coh"
    synopsis = "generate polarimetdic coherence images for BIOMASS"
    doc = """Tool for the generation of polarimetric coherence images
    for BIOMASS.
    """

    parser = subparsers.add_parser(name, description=doc, help=synopsis)

    parser.set_defaults(func=make_polarimetric_coherence)

    # command line options
    parser.add_argument(
        "-c",
        "--channels",
        choices=["hh-vv", "hh-hv", "vv-vh", "hv-vh"],
        default="hh-vv",
        help="Polarimetric chnannels to be use dfor the coherence computation"
        "(default: %(default)s)",
    )

    parser.add_argument(
        "--winsize",
        type=int,
        nargs=2,
        default=[10, 2],
        help="Size of the smoothing window [lines, samples] "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--phase-corrections",
        nargs=4,
        type=float,
        default=(0.0, 0.0, 0.0, 0.0),
        help=(
            "phase corrections (in radians) to be applied to polarimetric "
            "channels (default: %(default)s). "
            "iCal factors: (-0.4145157, 1.98025057, -1.56556034, 0.0) radians "
            "corresponding to (-23.75, 113.46, -89.7, 0.0) degrees"
        ),
    )
    parser.add_argument(
        "--kmz",
        action="store_true",
        default=False,
        help="generate a KMZ file in addition to the RGB",
    )
    parser.add_argument(
        "--kmz-res",
        type=float,
        default=0.001,
        help=(
            "resolution (in degrees) of the generated KMZ file. "
            "It is only used if also the '--kmz' option is specified"
        ),
    )
    parser.add_argument(
        "--png",
        action="store_true",
        default=False,
        help="generate a PNG file in addition to the RGB",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=False,
        dest="quick_test",
        help=(
            "skip the generation of the full resolution GeoTIFF "
            "and only save a PNG quick-look"
        ),
    )

    parser.add_argument(
        "--outdir",
        help=(
            "path to the output directory (only used if filenames for "
            "output files are not provided)"
        ),
    )

    # positional arguments
    parser.add_argument(
        "product_path", help="Path to the BIOMASS product directory"
    )
    parser.add_argument(
        "outfile", nargs="?", default=None, help="path to the output TIFF file"
    )

    return parser


def _get_rllr_parser(subparsers) -> argparse.ArgumentParser:
    """Set up the argument parser for the "rllr" sub-command."""
    name = "rllr"
    synopsis = "generate polarimetdic RL-LR coherence images for BIOMASS"
    doc = """Tool for the generation of polarimetric RL-LR coherence images
    (computed in circular basis) for BIOMASS.
    """

    parser = subparsers.add_parser(name, description=doc, help=synopsis)

    parser.set_defaults(func=make_rllr_coherence)

    # command line options
    parser.add_argument(
        "--winsize",
        type=int,
        nargs=2,
        default=[10, 2],
        help="Size of the smoothing window [lines, samples] "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--phase-corrections",
        nargs=4,
        type=float,
        default=(0.0, 0.0, 0.0, 0.0),
        help=(
            "phase corrections (in radians) to be applied to polarimetric "
            "channels (default: %(default)s). "
            "iCal factors: (-0.4145157, 1.98025057, -1.56556034, 0.0) radians "
            "corresponding to (-23.75, 113.46, -89.7, 0.0) degrees"
        ),
    )
    parser.add_argument(
        "--kmz",
        action="store_true",
        default=False,
        help="generate a KMZ file in addition to the RGB",
    )
    parser.add_argument(
        "--kmz-res",
        type=float,
        default=0.001,
        help=(
            "resolution (in degrees) of the generated KMZ file. "
            "It is only used if also the '--kmz' option is specified"
        ),
    )
    parser.add_argument(
        "--png",
        action="store_true",
        default=False,
        help="generate a PNG file in addition to the RGB",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=False,
        dest="quick_test",
        help=(
            "skip the generation of the full resolution GeoTIFF "
            "and only save a PNG quick-look"
        ),
    )

    parser.add_argument(
        "--outdir",
        help=(
            "path to the output directory (only used if filenames for "
            "output files are not provided)"
        ),
    )

    # positional arguments
    parser.add_argument(
        "product_path", help="Path to the BIOMASS product directory"
    )
    parser.add_argument(
        "outfile", nargs="?", default=None, help="path to the output TIFF file"
    )

    return parser


def _get_parser(subparsers=None) -> argparse.ArgumentParser:
    """Instantiate the command line argument (sub-)parser."""
    name = PROG
    extra_doc = """
    The program is organizae in sub-commands.
    Additional help can be obtained by using the '-h'/'--help' option
    on the specific sub-command.
    """
    doc = __doc__ + extra_doc

    parser = argparse.ArgumentParser(prog=name, description=doc)
    parser.add_argument(
        "--version", action="version", version="%(prog)s v" + __version__
    )

    # Command line options
    _add_logging_control_args(parser)

    # Sub-command management
    if subparsers is None:
        sp = parser.add_subparsers(
            title="sub-commands",
            # metavar="",
            # dest="func",
        )
        _get_ql_parser(sp)
        _get_coh_parser(sp)
        _get_rllr_parser(sp)

    if subparsers is None:
        _autocomplete(parser)

    return parser


def parse_args(args=None, namespace=None, parser=None):
    """Parse command line arguments."""
    if parser is None:
        parser = _get_parser()

    args = parser.parse_args(args, namespace)

    if args.func is make_rgb:
        if args.scale is not None:
            args.scale = EScale(args.scale)

        if args.representation == "hsv":
            if (
                args.scaling_method is not EScalingMethod.MEAN
                or args.scale is not EScale.LINEAR_AMPLITUDE
            ):
                parser.error(
                    "'hsv' representation only supports 'LINEAR_AMPLITUDE' "
                    "scale and 'MEAN' scaling method"
                )
        if (
            args.scaling_method is EScalingMethod.MEAN
            and args.scale is EScale.DB
        ):
            parser.error(
                "'MEAN' scaling method is incompatible with 'scale' 'DB'"
            )

        if args.scaling_ranges is not None:
            if args.scaling_method is EScalingMethod.MANUAL:
                parser.error(
                    "'scaling_ranges' parameter provided for a scaling method "
                    "different from 'MANUAL'"
                )

            (red_low, red_high, green_low, green_high, blue_low, blue_high) = (
                args.scaling_ranges
            )
            args.scaling_ranges = {
                "red_low": red_low,
                "red_high": red_high,
                "green_low": green_low,
                "green_high": green_high,
                "blue_low": blue_low,
                "blue_high": blue_high,
            }
        if args.quick_test and args.kmz:
            warnings.warn(
                "the '--quick' and the '--kmz' optionas are incompatible: "
                "ignoring '--kmz'",
                stacklevel=2,
            )

    return args


def _get_kwargs(args) -> dict:
    """Convert an argparse.Namespace into a dictionary.

    The "loglevel" and "func" arguments are never included in the output
    dictionary.
    """
    kwargs = vars(args).copy()
    kwargs.pop("func", None)
    kwargs.pop("loglevel", None)
    return kwargs


def main(*argv):
    """Implement the main CLI interface."""
    # setup logging
    logging.basicConfig(format=LOGFMT, level=DEFAULT_LOGLEVEL)
    logging.captureWarnings(True)
    log = logging.getLogger(__name__)

    # parse cmd line arguments
    args = parse_args(argv if argv else None)

    exit_code = EX_OK
    try:
        log.setLevel(args.loglevel)

        func = args.func
        kwargs = _get_kwargs(args)
        kmz_file = kwargs.pop("kmz")
        png_file = kwargs.pop("png")

        if func is make_rgb:
            float_file = kwargs.pop("float")
            make_rgb(
                **kwargs,
                float_file=float_file,
                kmz_file=kmz_file,
                png_file=png_file,
            )
        elif func is make_polarimetric_coherence:
            make_polarimetric_coherence(
                **kwargs,
                kmz_file=kmz_file,
                png_file=png_file,
            )
        elif func is make_rllr_coherence:
            make_rllr_coherence(
                **kwargs,
                kmz_file=kmz_file,
                png_file=png_file,
            )
        else:
            raise ValueError(f"unexpected function: {func}")
    except Exception as exc:  # noqa: B902
        log.critical(
            "unexpected exception caught: %r %s", type(exc).__name__, exc
        )
        log.debug("stacktrace:", exc_info=True)
        exit_code = EX_FAILURE
    except KeyboardInterrupt:
        log.warning("Keyboard interrupt received: exit the program")
        exit_code = EX_INTERRUPT

    return exit_code


if __name__ == "__main__":
    import sys

    sys.exit(main())
