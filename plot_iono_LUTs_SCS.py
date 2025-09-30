# -*- coding: utf-8 -*-
"""
Copyright 2025, European Space Agency (ESA)
Licensed under ESA Software Community Licence Permissive (Type 3) - v2.4
"""

import BiomassProduct
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from rasterio.transform import from_origin
import rasterio

from matplotlib.colors import Normalize


def export_ionosphere_geotiff(scs, output_dir, nodata_value=-9999):
    """
    Export ionosphereCorrection variables as GeoTIFFs (EPSG:4326), clipped using mean ± 2*std.
    Automatically flips data if needed based on latitude order.

    Parameters:
    - scs: BiomassProductSCS instance.
    - output_dir: target folder for GeoTIFFs.
    - nodata_value: NoData value (default -9999).
    """
    if not hasattr(scs, "geometry_latitude") or not hasattr(scs, "geometry_longitude"):
        print("[ERROR] geometry/latitude or geometry/longitude not found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    lat = scs.geometry_latitude.values
    lon = scs.geometry_longitude.values

    if lat.shape != lon.shape:
        print("[ERROR] Latitude and longitude grids have mismatched shapes.")
        return

    flip_ud = lat[0, 0] < lat[-1, 0]
    lat_res = abs(lat[1, 0] - lat[0, 0])
    lon_res = abs(lon[0, 1] - lon[0, 0])
    top_left_lat = max(lat[0, 0], lat[-1, 0])
    top_left_lon = min(lon[0, 0], lon[0, -1])

    transform = from_origin(top_left_lon, top_left_lat, lon_res, lat_res)

    iono_vars = [attr for attr in scs.__dict__ if attr.startswith("ionosphereCorrection_")]
    if not iono_vars:
        print("[WARNING] No ionospheric correction variables found.")
        return

    for var_name in iono_vars:
        data = getattr(scs, var_name).values.astype("float32")
        name = var_name.replace("ionosphereCorrection_", "")
        out_path = os.path.join(output_dir, f"{name}.tif")

        # Mask and compute stats
        data_masked = np.where(data == nodata_value, np.nan, data)
        valid_data = data_masked[~np.isnan(data_masked)]

        if valid_data.size == 0:
            print(f"[WARNING] No valid data for {name}, skipping.")
            continue

        mean = np.nanmean(valid_data)
        std = np.nanstd(valid_data)
        vmin = mean - 2 * std
        vmax = mean + 2 * std

        # Clip to balanced range
        data_clipped = np.clip(data_masked, vmin, vmax)
        data_filled = np.nan_to_num(data_clipped, nan=nodata_value)

        # Flip vertically if needed
        if flip_ud:
            data_filled = np.flipud(data_filled)

        with rasterio.open(
            out_path,
            'w',
            driver='GTiff',
            height=data_filled.shape[0],
            width=data_filled.shape[1],
            count=1,
            dtype='float32',
            crs='EPSG:4326',
            transform=transform,
            nodata=nodata_value
        ) as dst:
            dst.write(data_filled, 1)

        print(f"[INFO] GeoTIFF saved: {out_path}")

def export_ionosphere_png(scs, output_dir, nodata_value=-9999):
    """
    Export each ionosphereCorrection variable as a compact PNG image with colorbar.
    The color scale is based on mean ± 2*std. NaNs are shown as white.

    Parameters:
    - scs: BiomassProductSCS instance.
    - output_dir: folder where PNGs will be saved.
    - nodata_value: value to be masked (usually -9999).
    """

    if not hasattr(scs, "geometry_latitude") or not hasattr(scs, "geometry_longitude"):
        print("[ERROR] geometry/latitude or geometry/longitude not found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    iono_vars = [attr for attr in scs.__dict__ if attr.startswith("ionosphereCorrection_")]
    if not iono_vars:
        print("[WARNING] No ionospheric correction variables found.")
        return

    for var_name in iono_vars:
        data = getattr(scs, var_name).values.astype("float32")
        name = var_name.replace("ionosphereCorrection_", "")
        out_path = os.path.join(output_dir, f"{name}.png")

        # Mask noData
        data_masked = np.where(data == nodata_value, np.nan, data)
        valid_data = data_masked[~np.isnan(data_masked)]

        if valid_data.size == 0:
            print(f"[WARNING] No valid data for {name}, skipping.")
            continue

        # Color scale based on mean ± 2 std
        mean = np.nanmean(valid_data)
        std = np.nanstd(valid_data)
        vmin = mean - 2 * std
        vmax = mean + 2 * std
        norm = Normalize(vmin=vmin, vmax=vmax)

        # Set up color map with white for NaNs
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color='white')

        # Set figure size based on aspect ratio
        height, width = data_masked.shape
        fig_width = 5
        fig_height = fig_width * (height / width)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        im = ax.imshow(data_masked, cmap=cmap, norm=norm)
        ax.axis("off")

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(name, fontsize=10)

        # Save image tightly
        fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        print(f"[INFO] PNG saved: {out_path}")


def main(input_folder, type_format):
    if not os.path.isdir(input_folder):
        print(f"[ERROR] Input folder does not exist: {input_folder}")
        sys.exit(1)
    
    
    scs = BiomassProduct.BiomassProductSCS(input_folder)
    

    output_dir = os.path.join(input_folder, "ionosphere_export")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.isdir(output_dir):
        print(f"[ERROR] Input folder does not exist: {input_folder}")
        sys.exit(1)

    # Esporta
    if type_format == 'GEOTIFF':
        export_ionosphere_geotiff(scs, output_dir)
    
    elif type_format == 'PNG':
        export_ionosphere_png(scs, output_dir)

def print_help():
    print("Usage:")
    print("  python plot_iono_LUTs_SCS.py <input_folder> <output_format>")
    print("Arguments:")
    print("  <input_folder>   Path to the SCS product folder.")
    print("  <output_format>  Format to export: 'PNG' or  'GEOTIFF'")

if __name__ == "__main__":
    print('-----------------------------------')
    if len(sys.argv) < 3:
        print_help()
        sys.exit(1)

    input_folder = sys.argv[1]
    type_format = sys.argv[2].upper()

    if type_format not in ["PNG", "GEOTIFF"]:
        print(f"[ERROR] Unsupported format: {type_format}")
        print_help()
        sys.exit(1)
        
    main(input_folder, type_format)    
        
        
        