# -*- coding: utf-8 -*-
"""
(c) European Space Agency (ESA) Licensed under ESA Software Community Licence Permissive (Type 3) – v2.4

BiomassProduct.py
------------------
This module contains classes and utilities to read and process BIOMASS satellite products,
including Level 1A SCS and Level 2 STA formats. It provides structured access to annotation files,
measurement data, LUT variables.

"""

import folium
import numpy as np
import rasterio
from pathlib import Path
import xml.etree.ElementTree as ET
import netCDF4
from folium.raster_layers import ImageOverlay
import xarray as xr
import matplotlib.pyplot as plt
from rasterio.transform import from_origin
from matplotlib import cm
from matplotlib.colors import Normalize
from PIL import Image
import io
from rasterio.enums import Resampling
import base64
from folium.plugins import DualMap
from IPython.display import display
import os
import datetime
from folium import Map, LayerControl
from shapely.geometry import Polygon
from xml.etree import ElementTree as ET
from pathlib import Path

class BiomassProductL1VFRA:
     def __init__(self, eof_path):
         print('__init__')

         self.ns = {'cfi': "http://eop-cfi.esa.int/CFI"}
         self.parse(eof_path)
    
     def parse(self, xml_path):
         tree = ET.parse(xml_path)
         root = tree.getroot()
    
         # Fixed Header
         fh = root.find("cfi:Earth_Explorer_Header/cfi:Fixed_Header", self.ns)
    
         self.file_name = fh.findtext("cfi:File_Name", default="", namespaces=self.ns)
         self.file_description = fh.findtext("cfi:File_Description", default="", namespaces=self.ns)
         self.mission = fh.findtext("cfi:Mission", default="", namespaces=self.ns)
         self.file_class = fh.findtext("cfi:File_Class", default="", namespaces=self.ns)
         self.file_type = fh.findtext("cfi:File_Type", default="", namespaces=self.ns)
         self.file_version = fh.findtext("cfi:File_Version", default="", namespaces=self.ns)
    
         validity = fh.find("cfi:Validity_Period", self.ns)
         self.validity_start = validity.findtext("cfi:Validity_Start", default="", namespaces=self.ns)
         self.validity_stop = validity.findtext("cfi:Validity_Stop", default="", namespaces=self.ns)
    
         source = fh.find("cfi:Source", self.ns)
         self.system = source.findtext("cfi:System", default="", namespaces=self.ns)
         self.creator = source.findtext("cfi:Creator", default="", namespaces=self.ns)
         self.creator_version = source.findtext("cfi:Creator_Version", default="", namespaces=self.ns)
         self.creation_date = source.findtext("cfi:Creation_Date", default="", namespaces=self.ns)
    
         # Data Block
         db = root.find("cfi:Data_Block", self.ns)
    
         self.source_L0S = db.findtext("cfi:source_L0S", default="", namespaces=self.ns)
         self.source_L0M = db.findtext("cfi:source_L0M", default="", namespaces=self.ns)
         self.source_AUX_ORB = db.findtext("cfi:source_AUX_ORB", default="", namespaces=self.ns)
         self.frame_id = db.findtext("cfi:frame_id", default="", namespaces=self.ns)
         self.frame_status = db.findtext("cfi:frame_status", default="", namespaces=self.ns)
         
         #self.frame_start_time = db.findtext("cfi:frame_start_time", default="", namespaces=self.ns)
         #self.frame_stop_time = db.findtext("cfi:frame_stop_time", default="", namespaces=self.ns)
         # Raw values from XML (example)
         raw_start = db.findtext("cfi:frame_start_time", default="", namespaces=self.ns)
         raw_stop = db.findtext("cfi:frame_stop_time", default="", namespaces=self.ns)

         # Remove 'UTC=' prefix
         self.frame_start_time = raw_start.replace("UTC=", "")
         self.frame_stop_time = raw_stop.replace("UTC=", "")

         # Convert to datetime objects
         #dt_start = datetime.strptime(raw_start, "%Y-%m-%dT%H:%M:%S.%f")
         #dt_stop = datetime.strptime(raw_stop, "%Y-%m-%dT%H:%M:%S.%f")

         # Convert back to string with fixed format (auto-pads microseconds to 6 digits)
         #self.frame_start_time = dt_start.isoformat()
         #self.frame_stop_time = dt_stop.isoformat()
         
         
         self.frame_status = db.findtext("cfi:frame_status", default="", namespaces=self.ns)
         self.ops_angle_start = db.findtext("cfi:ops_angle_start", default="", namespaces=self.ns)
         self.ops_angle_stop = db.findtext("cfi:ops_angle_stop", default="", namespaces=self.ns)

    

class BiomassProductRAWS:
    def __init__(self, path):
        print('__init__')
        self.path = Path(path)
        
        
        #search file in  measurement
        self.annotation_file = next(self.path.glob("*.xml"), None)
        self.measurement_ia_rxh_file =  next(self.path.glob("*_ia_rxh.dat"), None)
        self.measurement_ia_rxv_file =  next(self.path.glob("*_ia_rxv.dat"), None)
        self.measurement_idx_rxh_file = next(self.path.glob("*_idx_rxh.dat"), None)
        self.measurement_idx_rxv_file = next(self.path.glob("*_idx_rxv.dat"), None)
        self.measurement_rxh_file = next((f for f in self.path.glob("*_rxh.dat") if "_idx_" not in f.name and "_ia_" not in f.name),None)
        self.measurement_rxv_file = next((f for f in self.path.glob("*_rxv.dat") if "_idx_" not in f.name and "_ia_" not in f.name),None)

        self.parse_annotation_file()

    def parse_annotation_file(self):
        
        self.ns = {  # namespace abbreviati
                   "bio": "http://earth.esa.int/biomass/1.0",
                   "gml": "http://www.opengis.net/gml/3.2",
                   "om": "http://www.opengis.net/om/2.0",
                   "eop": "http://www.opengis.net/eop/2.1",
                   "sar": "http://www.opengis.net/sar/2.1",
                   "xlink": "http://www.w3.org/1999/xlink",
                   "ows": "http://www.opengis.net/ows/2.0",
                   "xsi": "http://www.w3.org/2001/XMLSchema-instance"
                   }
        
        
        tree = ET.parse(self.annotation_file)
        root = tree.getroot()

        # gml:id del root
        self.product_id = root.attrib.get("{http://www.opengis.net/gml/3.2}id")

        # om:phenomenonTime
        time_period = root.find(".//om:phenomenonTime/gml:TimePeriod", self.ns)
        if time_period is not None:
            self.start_time = time_period.find("gml:beginPosition", self.ns).text
            self.end_time = time_period.find("gml:endPosition", self.ns).text

        # om:validTime
        valid_time = root.find(".//om:validTime/gml:TimePeriod", self.ns)
        if valid_time is not None:
            self.valid_start_time = valid_time.find("gml:beginPosition", self.ns).text
            self.valid_end_time = valid_time.find("gml:endPosition", self.ns).text

        # platform, instrument, sensor
        self.platform_name = root.find(".//eop:platform/eop:Platform/eop:shortName", self.ns).text
        self.instrument_name = root.find(".//eop:instrument/eop:Instrument/eop:shortName", self.ns).text
        self.sensor_type = root.find(".//eop:sensor/eop:Sensor/eop:sensorType", self.ns).text
        self.operational_mode = root.find(".//eop:sensor/eop:Sensor/eop:operationalMode", self.ns).text
        self.swath_id = root.find(".//eop:sensor/eop:Sensor/eop:swathIdentifier", self.ns).text

        # acquisitionParameters
        acq_path = ".//eop:acquisitionParameters/bio:Acquisition"
        self.orbit_number = int(root.find(f"{acq_path}/eop:orbitNumber", self.ns).text)
        self.last_orbit_number = int(root.find(f"{acq_path}/eop:lastOrbitNumber", self.ns).text)
        self.orbit_direction = root.find(f"{acq_path}/eop:orbitDirection", self.ns).text
        self.ascending_node_date = root.find(f"{acq_path}/eop:ascendingNodeDate", self.ns).text
        self.polarisation_mode = root.find(f"{acq_path}/sar:polarisationMode", self.ns).text
        self.polarisation_channels = root.find(f"{acq_path}/sar:polarisationChannels", self.ns).text.split(", ")
        self.antenna_look_direction = root.find(f"{acq_path}/sar:antennaLookDirection", self.ns).text
        self.mission_phase = root.find(f"{acq_path}/bio:missionPhase", self.ns).text
        self.data_take_id = root.find(f"{acq_path}/bio:dataTakeID", self.ns).text
        self.orbit_drift_flag = root.find(f"{acq_path}/bio:orbitDriftFlag", self.ns).text == "true"
        
        #etc 
        

class BiomassProductSCS:
    
    """
    Class representing a BIOMASS L1A SCS product.

    This class parses and organizes the product directory structure, including measurement, annotation,
    preview, and schema folders. It automatically identifies key files such as .tiff, .xml, and .kmz.
    """
    def __init__(self, path):
        print('__init__')
        self.path = Path(path)
        
        self.mph = next(self.path.glob("bio*.xml"), None)
        
        #search directory
        self.measurement_dir = self.path / "measurement" 
        self.annotation_dir = self.path / "annotation"
        self.annotation_navigation_dir = self.path / "annotation/navigation"      
        self.preview_dir = self.path / "preview"          
        self.schema_dir = self.path / "schema"

        #search file in  measurement
        self.measurement_abs_file = next(self.measurement_dir.glob("*abs*.tiff"), None)
        self.measurement_phase_file = next(self.measurement_dir.glob("*phase*.tiff"), None)
        self.measurement_vrt_file = next(self.measurement_dir.glob("*.vrt"), None)
        #search file in  annotation coregistrated
        self.annotation_xml_file = next(self.annotation_dir.glob("*.xml"), None)
        self.annotation_lut_file = next(self.annotation_dir.glob("*.nc"), None)
        self.annotation_att_file = next(self.annotation_navigation_dir.glob("*att*.xml"), None)
        self.annotation_orb_file = next(self.annotation_navigation_dir.glob("*orb*.xml"), None)
        

        
        #search file in  preview
        self.preview_ql_file = next(self.preview_dir.glob("*.png"), None)
        self.preview_kmz_file = next(self.preview_dir.glob("*.kmz"), None)
        
        #search file in  schema
        self.schema_aux_attitude = next(self.schema_dir.glob("*attitude.xds"), None)
        self.schema_aux_orbit = next(self.schema_dir.glob("*orbit.xds"), None)
        self.schema_common_types = next(self.schema_dir.glob("*types.xds"), None)
        self.schema_l1ab_main_annotation = next(self.schema_dir.glob("*-l1ab-main-annotation.xsd"), None)
        self.schema_l1_annotations = next(self.schema_dir.glob("*l1-annotations.xds"), None)
        self.schema_l1_overlay = next(self.schema_dir.glob("*l1-overlay.xds"), None)
        self.schema_l1_overlay_support = next(self.schema_dir.glob("*l1-overlay-support.xds"), None)
        self.schema_l1_vrt = next(self.schema_dir.glob("*l1-vrt.xds"), None)
        
        
        
                
        self.load_lut_variables()
        self.extract_footprint()
        self.load_mph()

    def load_mph(self):
       """
       Parse the MPH XML (root of the product) and extract:
         - Acquisition parameters (eop:acquisitionParameters/bio:Acquisition)
         - Processing information (eop:processing/bio:ProcessingInformation)
    
       The extracted values are stored as class attributes (self.*).
       """
       
    
       # Locate the main XML file if not already stored
       if  self.mph is None:          
           self.mph = next(Path(self.path).glob("*.xml"), None)
    
       if self.mph is None:
           raise FileNotFoundError("MPH XML not found in product root.")
    
       # XML namespaces
       ns = {
           "bio": "http://earth.esa.int/biomass/1.0",
           "eop": "http://www.opengis.net/eop/2.1",
           "gml": "http://www.opengis.net/gml/3.2",
           "om":  "http://www.opengis.net/om/2.0",
           "ows": "http://www.opengis.net/ows/2.0",
           "sar": "http://www.opengis.net/sar/2.1",
           "xlink": "http://www.w3.org/1999/xlink",
           "xsi": "http://www.w3.org/2001/XMLSchema-instance",
       }
    
       # Helper to safely extract text and cast values
       def _get_text(root, xpath, cast=str):
           el = root.find(xpath, ns)
           if el is None or el.text is None:
               return None
           txt = el.text.strip()
           if cast is bool:
               return txt.lower() == "true"
           if cast is int:
               try: return int(txt)
               except ValueError: return None
           if cast is float:
               try: return float(txt)
               except ValueError: return None
           return txt
    
       # Parse the XML tree
       tree = ET.parse(self.mph)
       root = tree.getroot()
    
       # =========================
       # Acquisition parameters
       # =========================
       base = ".//eop:acquisitionParameters/bio:Acquisition"
    
       self.orbitNumber       = _get_text(root, f"{base}/eop:orbitNumber", int)
       self.lastOrbitNumber   = _get_text(root, f"{base}/eop:lastOrbitNumber", int)
       self.orbitDirection    = _get_text(root, f"{base}/eop:orbitDirection", str)
    
       self.wrsLongitudeGrid  = _get_text(root, f"{base}/eop:wrsLongitudeGrid", int)
       self.wrsLatitudeGrid   = _get_text(root, f"{base}/eop:wrsLatitudeGrid", int)
    
       self.ascendingNodeDate = _get_text(root, f"{base}/eop:ascendingNodeDate", str)
       self.startTimeFromAscendingNode      = _get_text(root, f"{base}/eop:startTimeFromAscendingNode", int)
       self.completionTimeFromAscendingNode = _get_text(root, f"{base}/eop:completionTimeFromAscendingNode", int)
    
       self.polarisationMode      = _get_text(root, f"{base}/sar:polarisationMode", str)
       pol_channels               = _get_text(root, f"{base}/sar:polarisationChannels", str)
       self.polarisationChannels  = [p.strip() for p in pol_channels.split(",")] if pol_channels else None
       self.antennaLookDirection  = _get_text(root, f"{base}/sar:antennaLookDirection", str)
    
       self.missionPhase     = _get_text(root, f"{base}/bio:missionPhase", str)
       self.instrumentConfID = _get_text(root, f"{base}/bio:instrumentConfID", int)
       self.dataTakeID       = _get_text(root, f"{base}/bio:dataTakeID", int)
       self.orbitDriftFlag   = _get_text(root, f"{base}/bio:orbitDriftFlag", bool)
       self.globalCoverageID = _get_text(root, f"{base}/bio:globalCoverageID", str)
       self.majorCycleID     = _get_text(root, f"{base}/bio:majorCycleID", str)
       self.repeatCycleID    = _get_text(root, f"{base}/bio:repeatCycleID", str)
    
       # =========================
       # Processing information
       # =========================
       pbase = ".//eop:processing/bio:ProcessingInformation"
    
       self.processingCenter   = _get_text(root, f"{pbase}/eop:processingCenter", str)
       self.processingDate     = _get_text(root, f"{pbase}/eop:processingDate", str)  # ISO 8601 string
       self.processorName      = _get_text(root, f"{pbase}/eop:processorName", str)
       self.processorVersion   = _get_text(root, f"{pbase}/eop:processorVersion", str)
       self.processingLevel    = _get_text(root, f"{pbase}/eop:processingLevel", str)
       self.processingMode     = _get_text(root, f"{pbase}/eop:processingMode", str)
    
       # Repeated fields: collect as lists
       self.auxiliaryDataSetFiles = [
           e.text.strip() for e in root.findall(f"{pbase}/eop:auxiliaryDataSetFileName", ns)
           if e is not None and e.text
       ]
       self.sourceProducts = [
           e.text.strip() for e in root.findall(f"{pbase}/bio:sourceProduct", ns)
           if e is not None and e.text
       ]
            

    def extract_footprint(self):
        """
        Extracts the footprint polygon from the annotation XML.
        Stores it as a Shapely Polygon in self.footprint_polygon.
        """
        if self.annotation_xml_file is None:
            print("[WARNING] No annotation XML found.")
            self.footprint_polygon = None
            return
    
        try:
            tree = ET.parse(self.annotation_xml_file)
            root = tree.getroot()
    
            # Cerca il nodo footprint
            footprint_elem = root.find(".//sarImage/footprint")
            if footprint_elem is None or not footprint_elem.text:
                print("[WARNING] No <footprint> tag found in sarImage.")
                self.footprint_polygon = None
                return
    
            coords = list(map(float, footprint_elem.text.strip().split()))
            if len(coords) % 2 != 0:
                print("[ERROR] Invalid number of footprint coordinates.")
                self.footprint_polygon = None
                return
    
            # Converte in lista di tuple (lon, lat)
            points = [(coords[i+1], coords[i]) for i in range(0, len(coords), 2)]
    
            # Chiude il poligono se non è già chiuso
            if points[0] != points[-1]:
                points.append(points[0])
    
            self.footprint_polygon = Polygon(points)
            print(f"[INFO] Extracted footprint with {len(points)-1} points.")
        except Exception as e:
            print(f"[ERROR] Failed to extract footprint: {e}")
            self.footprint_polygon = None



    def load_lut_variables(self):
        """
        Load all variables and global attributes from the LUT NetCDF file into self.
        Includes root-level variables, group variables, and global metadata (NC_GLOBAL).
        """

    
        if not self.annotation_lut_file:
            raise FileNotFoundError("LUT NetCDF file not found.")
    
        # === 1. Carica variabili root
        try:
            ds_root = xr.open_dataset(self.annotation_lut_file)
            for var in ds_root.variables:
                setattr(self, var, ds_root[var])
            # Carica anche gli attributi globali
            for attr in ds_root.attrs:
                attr_name = f"global_{attr}"
                setattr(self, attr_name, ds_root.attrs[attr])
        except Exception as e:
            print(f"Could not load root-level variables or global attributes: {e}")
    
        # === 2. Carica gruppi strutturati
        groups = ["ionosphereCorrection","denoising", "geometry", "radiometry"]
    
        for group in groups:
            try:
                ds_group = xr.open_dataset(self.annotation_lut_file, group=group)
                for var in ds_group.variables:
                    attr_name = f"{group}_{var}"
                    print(attr_name)
                    setattr(self, attr_name, ds_group[var])
            except Exception as e:
                print(f"Could not load variables from group '{group}': {e}")
    
        print("All LUT variables and global attributes loaded as attributes.")   
        
        
        
    
    def plot_lut_variable(self, variable_name, save_geotiff=False):
        """
        Plot a variable from the LUT file and optionally save it as a GeoTIFF.
        Searches across all known groups.
        """
        if not self.annotation_lut_file:
            raise FileNotFoundError("LUT NetCDF file not found.")
    
        groups = ["ionosphereCorrection","denoising", "geometry", "radiometry"]
    
        var = None
        found_group = None
    
        for group in groups:
            try:
                ds_group = xr.open_dataset(self.annotation_lut_file, group=group)
                if variable_name in ds_group:
                    var = ds_group[variable_name]
                    found_group = group
                    break
            except Exception:
                continue
    
        if var is None:
            raise KeyError(f"Variable '{variable_name}' not found in any known group.")
    
        data = var.values
        # Maschera i valori nodata
        data = np.ma.masked_equal(data, -9999)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(data, cmap="RdYlBu")
        plt.colorbar(label=f"{found_group}/{variable_name}")
        plt.title(f"{variable_name} from group '{found_group}'")
        plt.tight_layout()
        plt.show()
    
        if save_geotiff:
            try:
                lat_ds = xr.open_dataset(self.annotation_lut_file, group="geometry")
                lats = lat_ds["latitude"].values
                lons = lat_ds["longitude"].values
    
                transform = from_origin(
                    np.min(lons),
                    np.max(lats),
                    abs(lons[0, 1] - lons[0, 0]),
                    abs(lats[1, 0] - lats[0, 0])
                )
    
                with rasterio.open(
                    f"{variable_name.replace('/', '_')}.tif",
                    'w',
                    driver='GTiff',
                    height=data.shape[0],
                    width=data.shape[1],
                    count=1,
                    dtype=data.dtype,
                    crs='EPSG:4326',
                    transform=transform
                ) as dst:
                    dst.write(data, 1)
    
                print(f"✅ Saved {variable_name} as GeoTIFF.")
            except Exception as e:
                print(f"⚠️ Failed to save GeoTIFF: {e}")
                
                
                
                
        
    def export_ionosphere_to_geotiff(self, output_dir):
        """
        Export all variables in the ionosphereCorrection group to individual GeoTIFF files.
        Uses geometry/latitude and geometry/longitude for georeferencing.
        """
        if not hasattr(self, "geometry_latitude") or not hasattr(self, "geometry_longitude"):
            print("[ERROR] Latitude and longitude not loaded from geometry group.")
            return
    
        lat = self.geometry_latitude.values
        lon = self.geometry_longitude.values
    
        if lat.shape != lon.shape:
            print("[ERROR] Latitude and longitude shapes do not match.")
            return
    
        # Calcola risoluzione e origine
        lat_res = abs(lat[1, 0] - lat[0, 0])
        lon_res = abs(lon[0, 1] - lon[0, 0])
        top_left_lat = lat[0, 0]
        top_left_lon = lon[0, 0]
        transform = from_origin(top_left_lon, top_left_lat, lon_res, lat_res)
    
        
        iono_vars = [attr for attr in self.__dict__ if attr.startswith("ionosphereCorrection_")]
    
        os.makedirs(output_dir, exist_ok=True)
    
        for var_name in iono_vars:
            data = getattr(self, var_name).values.astype("float32")
            tif_name = var_name.replace("ionosphereCorrection_", "") + ".tif"
            tif_path = os.path.join(output_dir, tif_name)
    
            with rasterio.open(
                tif_path,
                'w',
                driver='GTiff',
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype='float32',
                crs='EPSG:4326',
                transform=transform
            ) as dst:
                dst.write(data, 1)
    
            print(f"[INFO] Exported {tif_name} to {tif_path}")
        
    def load_complex_polarizations(self):
        """
        Carica i dati complessi polarimetrici (HH, HV, VH, VV) da file abs e phase.
        Salva:
        - self.S: dati complessi
        - self.A: ampiezze
        - self.P: fasi
        - self.profile, self.crs, self.transform, self.gcps, self.gcp_crs
          per permettere un corretto salvataggio geolocalizzato (GeoTIFF con GCP).
        """
        import rasterio
        import numpy as np
        from rasterio.control import GroundControlPoint
    
        polarizations = ['HH', 'HV', 'VH', 'VV']
        self.S = {}
        self.A = {}
        self.P = {}
    
        if not self.measurement_abs_file or not self.measurement_phase_file:
            raise FileNotFoundError("Missing abs or phase file in measurement directory.")
    
        # === Leggi il file delle ampiezze ===
        with rasterio.open(self.measurement_abs_file) as abs_src:
            abs_data = abs_src.read()  # shape (4, rows, cols)
            self.profile = abs_src.profile
            self.crs = abs_src.crs
            self.transform = abs_src.transform
            self.driver = abs_src.driver
    
            # Leggi e correggi i GCP
            raw_gcps, gcp_crs = abs_src.gcps
            self.gcps = [
                GroundControlPoint(row=g.row, col=g.col, x=g.x, y=g.y, z=g.z)
                for g in raw_gcps
            ]
            self.gcp_crs = gcp_crs
    
        # === Leggi il file delle fasi ===
        with rasterio.open(self.measurement_phase_file) as phase_src:
            phase_data = phase_src.read()  # shape (4, rows, cols)
    
        # === Ricostruisci dati complessi ===
        for i, pol in enumerate(polarizations):
            amp = abs_data[i, :, :]
            phase = phase_data[i, :, :]
            self.A[pol] = amp
            self.P[pol] = phase
            self.S[pol] = amp * np.exp(1j * phase)
    
        print("✔️ Complex polarizations loaded: HH, HV, VH, VV.")
        print(f"🗺  CRS: {self.crs}, GCPs: {len(self.gcps)} trovati.")
        
    def save_raster(self, array, out_path, dtype="float32"):
        import rasterio
    
        profile = self.profile.copy()
        profile.update({
            "count": 1,
            "dtype": dtype,
            "crs": self.crs,
            "transform": self.transform or rasterio.Affine.identity(),
            "driver": "GTiff",
            "compress": "DEFLATE"  # ✅ SNAP-friendly
        })
    
        # 1. Scrivi immagine base
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(array.astype(dtype), 1)
    
        # 2. Aggiungi i GCP
        if hasattr(self, "gcps") and hasattr(self, "gcp_crs"):
            with rasterio.open(out_path, "r+") as dst:
                dst.gcps = (self.gcps, self.gcp_crs)
    
        print(f"✔️ GeoTIFF salvato con GCP: {out_path}")     
        
    def generate_pauli_rgb(self, normalize=True):
        """
        Crea un'immagine RGB da decomposizione di Pauli.
        Restituisce un array shape (3, rows, cols).
        """

    
        HH = self.S["HH"]
        VV = self.S["VV"]
        HV = self.S["HV"]
        VH = self.S["VH"]
    
        R = np.abs(HH - VV) / np.sqrt(2)
        G = np.abs(HV + VH) / np.sqrt(2)
        B = np.abs(HH + VV) / np.sqrt(2)
    
        rgb = np.stack([R, G, B])
    
        if normalize:
            rgb_max = np.percentile(rgb, 99)  # taglio valori outlier
            rgb = np.clip(rgb / rgb_max, 0, 1)
    
        return rgb
    

    

class BiomassProductDGM:
    def __init__(self, path):
        print('__init__')
        self.path = Path(path)
        self.mph = next(self.path.glob("bio*dgm*.xml"), None)
        
        #search directory
        self.measurement_dir = self.path / "measurement" 
        self.annotation_dir = self.path / "annotation"
        self.annotation_navigation_dir = self.path / "annotation/navigation"      
        self.preview_dir = self.path / "preview"          
        self.schema_dir = self.path / "schema"

        #search file in  measurement
        self.measurement_abs_file = next(self.measurement_dir.glob("*abs*.tiff"), None)

        #search file in  annotation coregistrated
        self.annotation_xml_file = next(self.annotation_dir.glob("*.xml"), None)
        self.annotation_lut_file = next(self.annotation_dir.glob("*.nc"), None)
        self.annotation_att_file = next(self.annotation_navigation_dir.glob("*att*.xml"), None)
        self.annotation_orb_file = next(self.annotation_navigation_dir.glob("*orb*.xml"), None)
        

        
        #search file in  preview
        self.preview_ql_file = next(self.preview_dir.glob("*.png"), None)
        self.preview_kmz_file = next(self.preview_dir.glob("*.kmz"), None)
        
        #search file in  schema
        self.schema_aux_attitude = next(self.schema_dir.glob("*attitude.xds"), None)
        self.schema_aux_orbit = next(self.schema_dir.glob("*orbit.xds"), None)
        self.schema_common_types = next(self.schema_dir.glob("*types.xds"), None)
        self.schema_l1ab_main_annotation = next(self.schema_dir.glob("*-l1ab-main-annotation.xsd"), None)
        self.schema_l1_annotations = next(self.schema_dir.glob("*l1-annotations.xds"), None)
        self.schema_l1_overlay = next(self.schema_dir.glob("*l1-overlay.xds"), None)
        self.schema_l1_overlay_support = next(self.schema_dir.glob("*l1-overlay-support.xds"), None)

                
        self.load_lut_variables()
        self.load_mph()
        
    
    def load_mph(self):
        """
        Parse the MPH XML (root of the product) and extract:
          - Acquisition parameters (eop:acquisitionParameters/bio:Acquisition)
          - Processing information (eop:processing/bio:ProcessingInformation)
     
        The extracted values are stored as class attributes (self.*).
        """
        
     
        # Locate the main XML file if not already stored
        if  self.mph is None:          
            self.mph = next(Path(self.path).glob("*.xml"), None)
     
        if self.mph is None:
            raise FileNotFoundError("MPH XML not found in product root.")
     
        # XML namespaces
        ns = {
            "bio": "http://earth.esa.int/biomass/1.0",
            "eop": "http://www.opengis.net/eop/2.1",
            "gml": "http://www.opengis.net/gml/3.2",
            "om":  "http://www.opengis.net/om/2.0",
            "ows": "http://www.opengis.net/ows/2.0",
            "sar": "http://www.opengis.net/sar/2.1",
            "xlink": "http://www.w3.org/1999/xlink",
            "xsi": "http://www.w3.org/2001/XMLSchema-instance",
        }
     
        # Helper to safely extract text and cast values
        def _get_text(root, xpath, cast=str):
            el = root.find(xpath, ns)
            if el is None or el.text is None:
                return None
            txt = el.text.strip()
            if cast is bool:
                return txt.lower() == "true"
            if cast is int:
                try: return int(txt)
                except ValueError: return None
            if cast is float:
                try: return float(txt)
                except ValueError: return None
            return txt
     
        # Parse the XML tree
        tree = ET.parse(self.mph)
        root = tree.getroot()
     
        # =========================
        # Acquisition parameters
        # =========================
        base = ".//eop:acquisitionParameters/bio:Acquisition"
     
        self.orbitNumber       = _get_text(root, f"{base}/eop:orbitNumber", int)
        self.lastOrbitNumber   = _get_text(root, f"{base}/eop:lastOrbitNumber", int)
        self.orbitDirection    = _get_text(root, f"{base}/eop:orbitDirection", str)
     
        self.wrsLongitudeGrid  = _get_text(root, f"{base}/eop:wrsLongitudeGrid", int)
        self.wrsLatitudeGrid   = _get_text(root, f"{base}/eop:wrsLatitudeGrid", int)
     
        self.ascendingNodeDate = _get_text(root, f"{base}/eop:ascendingNodeDate", str)
        self.startTimeFromAscendingNode      = _get_text(root, f"{base}/eop:startTimeFromAscendingNode", int)
        self.completionTimeFromAscendingNode = _get_text(root, f"{base}/eop:completionTimeFromAscendingNode", int)
     
        self.polarisationMode      = _get_text(root, f"{base}/sar:polarisationMode", str)
        pol_channels               = _get_text(root, f"{base}/sar:polarisationChannels", str)
        self.polarisationChannels  = [p.strip() for p in pol_channels.split(",")] if pol_channels else None
        self.antennaLookDirection  = _get_text(root, f"{base}/sar:antennaLookDirection", str)
     
        self.missionPhase     = _get_text(root, f"{base}/bio:missionPhase", str)
        self.instrumentConfID = _get_text(root, f"{base}/bio:instrumentConfID", int)
        self.dataTakeID       = _get_text(root, f"{base}/bio:dataTakeID", int)
        self.orbitDriftFlag   = _get_text(root, f"{base}/bio:orbitDriftFlag", bool)
        self.globalCoverageID = _get_text(root, f"{base}/bio:globalCoverageID", str)
        self.majorCycleID     = _get_text(root, f"{base}/bio:majorCycleID", str)
        self.repeatCycleID    = _get_text(root, f"{base}/bio:repeatCycleID", str)
     
        # =========================
        # Processing information
        # =========================
        pbase = ".//eop:processing/bio:ProcessingInformation"
     
        self.processingCenter   = _get_text(root, f"{pbase}/eop:processingCenter", str)
        self.processingDate     = _get_text(root, f"{pbase}/eop:processingDate", str)  # ISO 8601 string
        self.processorName      = _get_text(root, f"{pbase}/eop:processorName", str)
        self.processorVersion   = _get_text(root, f"{pbase}/eop:processorVersion", str)
        self.processingLevel    = _get_text(root, f"{pbase}/eop:processingLevel", str)
        self.processingMode     = _get_text(root, f"{pbase}/eop:processingMode", str)
     
        # Repeated fields: collect as lists
        self.auxiliaryDataSetFiles = [
            e.text.strip() for e in root.findall(f"{pbase}/eop:auxiliaryDataSetFileName", ns)
            if e is not None and e.text
        ]
        self.sourceProducts = [
            e.text.strip() for e in root.findall(f"{pbase}/bio:sourceProduct", ns)
            if e is not None and e.text
        ]

    def load_lut_variables(self):
        """
        Load all variables and global attributes from the LUT NetCDF file into self.
        Includes root-level variables, group variables, and global metadata (NC_GLOBAL).
        """

    
        if not self.annotation_lut_file:
            raise FileNotFoundError("LUT NetCDF file not found.")
    
        # === 1. Carica variabili root
        try:
            ds_root = xr.open_dataset(self.annotation_lut_file)
            for var in ds_root.variables:
                setattr(self, var, ds_root[var])
            # Carica anche gli attributi globali
            for attr in ds_root.attrs:
                attr_name = f"global_{attr}"
                setattr(self, attr_name, ds_root.attrs[attr])
        except Exception as e:
            print(f"Could not load root-level variables or global attributes: {e}")
    
        # === 2. Carica gruppi strutturati
        groups = ["denoising", "geometry", "radiometry"]
    
        for group in groups:
            try:
                ds_group = xr.open_dataset(self.annotation_lut_file, group=group)
                for var in ds_group.variables:
                    attr_name = f"{group}_{var}"
                    setattr(self, attr_name, ds_group[var])
            except Exception as e:
                print(f"Could not load variables from group '{group}': {e}")
    
        print("All LUT variables and global attributes loaded as attributes.")  
        
        
        
    def plot_lut_variable(self, variable_name, save_geotiff=False):
            """
            Plot a variable from the LUT file and optionally save it as a GeoTIFF.
            Searches across all known groups.
            """
            if not self.annotation_lut_file:
                raise FileNotFoundError("LUT NetCDF file not found.")
        
            groups = ["ionosphereCorrection","denoising", "geometry", "radiometry"]
        
            var = None
            found_group = None
        
            for group in groups:
                try:
                    ds_group = xr.open_dataset(self.annotation_lut_file, group=group)
                    if variable_name in ds_group:
                        var = ds_group[variable_name]
                        found_group = group
                        break
                except Exception:
                    continue
        
            if var is None:
                raise KeyError(f"Variable '{variable_name}' not found in any known group.")
        
            data = var.values
            # Maschera i valori nodata
            data = np.ma.masked_equal(data, -9999)
            
            plt.figure(figsize=(8, 6))
            plt.imshow(data, cmap="RdYlBu")
            plt.colorbar(label=f"{found_group}/{variable_name}")
            plt.title(f"{variable_name} from group '{found_group}'")
            plt.tight_layout()
            plt.show()
        
            if save_geotiff:
                try:
                    lat_ds = xr.open_dataset(self.annotation_lut_file, group="geometry")
                    lats = lat_ds["latitude"].values
                    lons = lat_ds["longitude"].values
        
                    transform = from_origin(
                        np.min(lons),
                        np.max(lats),
                        abs(lons[0, 1] - lons[0, 0]),
                        abs(lats[1, 0] - lats[0, 0])
                    )
        
                    with rasterio.open(
                        f"{variable_name.replace('/', '_')}.tif",
                        'w',
                        driver='GTiff',
                        height=data.shape[0],
                        width=data.shape[1],
                        count=1,
                        dtype=data.dtype,
                        crs='EPSG:4326',
                        transform=transform
                    ) as dst:
                        dst.write(data, 1)
        
                    print(f"✅ Saved {variable_name} as GeoTIFF.")
                except Exception as e:
                    print(f"⚠️ Failed to save GeoTIFF: {e}")    
    
    
    
    

    def export_abs_bands_separately(self, output_dir):
        """
        Esporta ogni banda del file _abs.tiff in file singoli, compressi con DEFLATE.
        I file verranno salvati come 'band_1.tif', 'band_2.tif', ecc.
        """
        if not self.measurement_abs_file:
            raise FileNotFoundError("File _abs.tiff non trovato.")

        os.makedirs(output_dir, exist_ok=True)

        with rasterio.open(self.measurement_abs_file) as src:
            profile = src.profile.copy()
            gcps, gcp_crs = src.gcps
            count = src.count

            for i in range(1, count + 1):
                band_data = src.read(i)
                band_name = f"band_{i}.tif"
                band_path = os.path.join(output_dir, band_name)

                band_profile = profile.copy()
                band_profile.update({
                    "count": 1,
                    "compress": "DEFLATE",
                    "predictor": 2,
                    "tiled": True
                })

                with rasterio.open(band_path, "w", **band_profile) as dst:
                    dst.write(band_data, 1)

                if gcps:
                    with rasterio.open(band_path, "r+") as dst:
                        dst.gcps = (gcps, gcp_crs)

        print(f"✅ {count} bande esportate singolarmente in: {output_dir}")

    def recombine_abs_bands(self, input_dir, output_path):
        """
        Ricombina i file 'band_1.tif', ..., 'band_N.tif' in un GeoTIFF multibanda,
        mantenendo compressione DEFLATE e i GCP.
        """
        # Trova i band_*.tif in ordine
        band_paths = sorted([
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.startswith("band_") and f.endswith(".tif")
        ])

        if not band_paths:
            raise FileNotFoundError("Nessun file banda trovato nella cartella.")

        with rasterio.open(band_paths[0]) as ref:
            profile = ref.profile.copy()
            gcps, gcp_crs = ref.gcps
            dtype = ref.dtypes[0]

        profile.update({
            "count": len(band_paths),
            "compress": "DEFLATE",
            "predictor": 2,
            "tiled": True,
            "dtype": dtype
        })

        with rasterio.open(output_path, "w", **profile) as dst:
            for i, band_path in enumerate(band_paths):
                with rasterio.open(band_path) as src:
                    dst.write(src.read(1), i + 1)

        if gcps:
            with rasterio.open(output_path, "r+") as dst:
                dst.gcps = (gcps, gcp_crs)

        print(f"✅ File multibanda finale creato: {output_path}")
        


    def load_polarizations(self):
        """
        Carica i dati complessi polarimetrici (HH, HV, VH, VV) da file abs e phase.
        Salva:
        - self.S: dati complessi
        - self.A: ampiezze
        - self.P: fasi
        - self.profile, self.crs, self.transform, self.gcps, self.gcp_crs
          per permettere un corretto salvataggio geolocalizzato (GeoTIFF con GCP).
        """
        import rasterio
        import numpy as np
        from rasterio.control import GroundControlPoint
    
        polarizations = ['HH', 'HV', 'VH', 'VV']
        self.A = {}

    
        if not self.measurement_abs_file :
            raise FileNotFoundError("Missing abs or phase file in measurement directory.")
    
        # === Leggi il file delle ampiezze ===
        with rasterio.open(self.measurement_abs_file) as abs_src:
            abs_data = abs_src.read()  # shape (4, rows, cols)
            self.profile = abs_src.profile
            self.crs = abs_src.crs
            self.transform = abs_src.transform
            self.driver = abs_src.driver
    
            # Leggi e correggi i GCP
            raw_gcps, gcp_crs = abs_src.gcps
            self.gcps = [
                GroundControlPoint(row=g.row, col=g.col, x=g.x, y=g.y, z=g.z)
                for g in raw_gcps
            ]
            self.gcp_crs = gcp_crs
    
    
        # === Ricostruisci dati complessi ===
        for i, pol in enumerate(polarizations):
            amp = abs_data[i, :, :]

            self.A[pol] = amp

    
        print("✔️ Complex polarizations loaded: HH, HV, VH, VV.")
        print(f"🗺  CRS: {self.crs}, GCPs: {len(self.gcps)} trovati.")



class STAProductGroupLoader:
    def __init__(self, folder_TDS):
        """
        Initialize the loader with a folder containing STA products.
        Automatically identifies primary and secondary products.
        """
        self.folder_TDS = Path(folder_TDS)
        self.products = []  # Will be list of dicts: {'name': ..., 'path': ..., 'instance': ...}
        self._load_products()

    def _load_products(self):
        """
        Search and instantiate STA products, identify primary and secondary ones.
        """
        sta_dirs = [d for d in self.folder_TDS.iterdir() if d.is_dir() and "STA__1S" in d.name]
        temp_list = []

        # Create initial list of (path, instance)
        for path in sta_dirs:
            product = BiomassProductSTA(str(path))
            temp_list.append((path, product))

        # Identify primary product
        primary_pair = next(((p, obj) for (p, obj) in temp_list if getattr(obj, "is_primary", False)), None)
        if primary_pair is None:
            raise ValueError("❌ No primary product found (is_primary=True).")

        primary_path, primary_instance = primary_pair
        self.products.append({
            "name": "product_primary",
            "path": primary_path,
            "instance": primary_instance
        })

        # Identify and add secondary products
        secondary_counter = 1
        for path, instance in temp_list:
            if instance is not primary_instance:
                self.products.append({
                    "name": f"product_secondary{secondary_counter}",
                    "path": path,
                    "instance": instance
                })
                secondary_counter += 1

    def print_summary(self):
        """
        Print a summary of the loaded product instances.
        """
        print("📦 Summary of STA products:")
        for prod in self.products:
            print(f"  - {prod['name']}: {prod['path'].name}")

    def get_instance_by_name(self, name):
        """
        Retrieve a product instance by its assigned variable name.
        """
        for prod in self.products:
            if prod["name"] == name:
                return prod["instance"]
        return None


class BiomassProductSTA:
    
    """
    Class representing a BIOMASS STA product.

    This class parses and organizes the product directory structure, including measurement, annotation,
    preview, and schema folders. It automatically identifies key files such as .tiff, .xml, and .kmz.
    """
    def __init__(self, path):
        self.path = Path(path)
        print (self.path)
        self.mph = next(self.path.glob("bio*sta*.xml"), None)
        #search directory
        self.measurement_dir = self.path / "measurement" 
        self.annotation_coregistered_dir = self.path / "annotation_coregistered"
        self.annotation_coregistrated_navigation_dir = self.path / "annotation_coregistered/navigation"        
        self.annotation_primary_dir = self.path / "annotation_primary"
        self.annotation_primary_navigation_dir = self.path / "annotation_primary/navigation"
        self.preview_dir = self.path / "preview"          
        self.schema_dir = self.path / "schema"

        #search file in  measurement
        self.measurement_abs_file = next(self.measurement_dir.glob("*abs*.tiff"), None)
        self.measurement_phase_file = next(self.measurement_dir.glob("*phase*.tiff"), None)
        self.measurement_vrt_file = next(self.measurement_dir.glob("*.vrt"), None)
        #search file in  annotation coregistrated
        self.annotation_coregistered_xml_file = next(self.annotation_coregistered_dir.glob("*.xml"), None)
        self.annotation_coregistered_lut_file = next(self.annotation_coregistered_dir.glob("*.nc"), None)
        self.annotation_cor_att_file = next(self.annotation_coregistrated_navigation_dir.glob("*att*.xml"), None)
        self.annotation_cor_orb_file = next(self.annotation_coregistrated_navigation_dir.glob("*orb*.xml"), None)
        
        #search file in  annotation primary
        self.annotation_primary_xml_file = next(self.annotation_primary_dir.glob("*.xml"), None)
        self.annotation_pri_att_file = next(self.annotation_primary_navigation_dir.glob("*att*.xml"), None)
        self.annotation_pri_orb_file = next(self.annotation_primary_navigation_dir.glob("*orb*.xml"), None)
        
        #search file in  preview
        self.preview_ql_file = next(self.preview_dir.glob("*.png"), None)
        self.preview_kmz_file = next(self.preview_dir.glob("*.kmz"), None)
        
        self.load_lut_variables()        
        self.is_primary = self.check_if_primary()
        self.noDataValue=self.get_nodata_value()
        
        self.load_mph()
        
    
    def load_mph(self):
        """
        Parse the MPH XML (root of the product) and extract:
          - Acquisition parameters (eop:acquisitionParameters/bio:Acquisition)
          - Processing information (eop:processing/bio:ProcessingInformation)
     
        The extracted values are stored as class attributes (self.*).
        """
        
     
        # Locate the main XML file if not already stored
        if  self.mph is None:          
            self.mph = next(Path(self.path).glob("*.xml"), None)
     
        if self.mph is None:
            raise FileNotFoundError("MPH XML not found in product root.")
     
        # XML namespaces
        ns = {
            "bio": "http://earth.esa.int/biomass/1.0",
            "eop": "http://www.opengis.net/eop/2.1",
            "gml": "http://www.opengis.net/gml/3.2",
            "om":  "http://www.opengis.net/om/2.0",
            "ows": "http://www.opengis.net/ows/2.0",
            "sar": "http://www.opengis.net/sar/2.1",
            "xlink": "http://www.w3.org/1999/xlink",
            "xsi": "http://www.w3.org/2001/XMLSchema-instance",
        }
     
        # Helper to safely extract text and cast values
        def _get_text(root, xpath, cast=str):
            el = root.find(xpath, ns)
            if el is None or el.text is None:
                return None
            txt = el.text.strip()
            if cast is bool:
                return txt.lower() == "true"
            if cast is int:
                try: return int(txt)
                except ValueError: return None
            if cast is float:
                try: return float(txt)
                except ValueError: return None
            return txt
     
        # Parse the XML tree
        tree = ET.parse(self.mph)
        root = tree.getroot()
     
        # =========================
        # Acquisition parameters
        # =========================
        base = ".//eop:acquisitionParameters/bio:Acquisition"
     
        self.orbitNumber       = _get_text(root, f"{base}/eop:orbitNumber", int)
        self.lastOrbitNumber   = _get_text(root, f"{base}/eop:lastOrbitNumber", int)
        self.orbitDirection    = _get_text(root, f"{base}/eop:orbitDirection", str)
     
        self.wrsLongitudeGrid  = _get_text(root, f"{base}/eop:wrsLongitudeGrid", int)
        self.wrsLatitudeGrid   = _get_text(root, f"{base}/eop:wrsLatitudeGrid", int)
     
        self.ascendingNodeDate = _get_text(root, f"{base}/eop:ascendingNodeDate", str)
        self.startTimeFromAscendingNode      = _get_text(root, f"{base}/eop:startTimeFromAscendingNode", int)
        self.completionTimeFromAscendingNode = _get_text(root, f"{base}/eop:completionTimeFromAscendingNode", int)
     
        self.polarisationMode      = _get_text(root, f"{base}/sar:polarisationMode", str)
        pol_channels               = _get_text(root, f"{base}/sar:polarisationChannels", str)
        self.polarisationChannels  = [p.strip() for p in pol_channels.split(",")] if pol_channels else None
        self.antennaLookDirection  = _get_text(root, f"{base}/sar:antennaLookDirection", str)
     
        self.missionPhase     = _get_text(root, f"{base}/bio:missionPhase", str)
        self.instrumentConfID = _get_text(root, f"{base}/bio:instrumentConfID", int)
        self.dataTakeID       = _get_text(root, f"{base}/bio:dataTakeID", int)
        self.orbitDriftFlag   = _get_text(root, f"{base}/bio:orbitDriftFlag", bool)
        self.globalCoverageID = _get_text(root, f"{base}/bio:globalCoverageID", str)
        self.majorCycleID     = _get_text(root, f"{base}/bio:majorCycleID", str)
        self.repeatCycleID    = _get_text(root, f"{base}/bio:repeatCycleID", str)
     
        # =========================
        # Processing information
        # =========================
        pbase = ".//eop:processing/bio:ProcessingInformation"
     
        self.processingCenter   = _get_text(root, f"{pbase}/eop:processingCenter", str)
        self.processingDate     = _get_text(root, f"{pbase}/eop:processingDate", str)  # ISO 8601 string
        self.processorName      = _get_text(root, f"{pbase}/eop:processorName", str)
        self.processorVersion   = _get_text(root, f"{pbase}/eop:processorVersion", str)
        self.processingLevel    = _get_text(root, f"{pbase}/eop:processingLevel", str)
        self.processingMode     = _get_text(root, f"{pbase}/eop:processingMode", str)
     
        # Repeated fields: collect as lists
        self.auxiliaryDataSetFiles = [
            e.text.strip() for e in root.findall(f"{pbase}/eop:auxiliaryDataSetFileName", ns)
            if e is not None and e.text
        ]
        self.sourceProducts = [
            e.text.strip() for e in root.findall(f"{pbase}/bio:sourceProduct", ns)
            if e is not None and e.text
        ]        
        
    
    def check_if_primary(self):

        """
        Determine whether the current STA product is the primary one in a coregistered pair.
    
        This is done by:
        1. Extracting the 'primaryImage' name from the annotation_coregistered_xml_file.
        2. Extracting the 'startTime' of the current product from the same XML.
        3. Comparing the dates (YYYYMMDD) from both sources.
        
        If the date of 'startTime' matches the one in 'primaryImage', this product is the primary.
        """
        try:
            tree = ET.parse(self.annotation_coregistered_xml_file)
            root = tree.getroot()
    
            # Extract <primaryImage> and get the date part (YYYYMMDD)
            primary_image_tag = root.find(".//primaryImage")
            
            primary_image_text = primary_image_tag.text
            date_from_primary = primary_image_text.split("_")[5][:8]  # '20170228'
            
            # Extract <startTime> and convert to YYYYMMDD
            start_time_tag = root.find(".//acquisitionInformation/startTime")
            
            start_time_text = start_time_tag.text  # e.g. '2017-02-28T09:45:37.702537'
            date_from_start = datetime.datetime.strptime(start_time_text, "%Y-%m-%dT%H:%M:%S.%f").strftime("%Y%m%d")
            
            
            return date_from_primary == date_from_start
    
        except Exception as e:
            print(f"Error in check_if_primary: {e}")
            return False
        
        
    def get_nodata_value(self):
        """
        Extract the noDataValue from the annotation_coregistered_xml_file.
    
        Returns:
            float: the noDataValue if found, otherwise None.
        """
        try:
            tree = ET.parse(self.annotation_coregistered_xml_file)
            root = tree.getroot()
    
            no_data_tag = root.find(".//sarImage/noDataValue")
            if no_data_tag is not None:
                return float(no_data_tag.text.strip())
            else:
                print("noDataValue tag not found in XML.")
                return None
        except Exception as e:
            print(f"Error in get_nodata_value: {e}")
            return None
 
    

    def load_lut_variables(self):
        """
        Load all LUT variables from the NetCDF file and store them as both attributes (self.<var>)
        and in a dictionary (self.lut_variables) for easy access and inspection.
        """
        if not self.annotation_coregistered_lut_file:
            raise FileNotFoundError("LUT NetCDF file path not set.")
    
        self.lut_variables = {}  # Initialize dictionary
    
        # Load root-level variables
        try:
            root_ds = xr.open_dataset(self.annotation_coregistered_lut_file)
            for var in root_ds.variables:
                self.lut_variables[var] = root_ds[var]
                setattr(self, var, root_ds[var])
                #print(f"✅ Loaded (root): self.{var}")
        except Exception as e:
            print(f"❌ Could not load root-level variables: {e}")
    
        # Expected groups based on provided dump
        groups = [
            "radiometry",
            "denoising",
            "geometry",
            "coregistration",
            "skpPhaseCalibration",
            "baselineAndIonosphereCorrection"
        ]
    
        for group in groups:
            try:
                ds = xr.open_dataset(self.annotation_coregistered_lut_file, group=group)
                if not ds.variables:
                    print(f"⚠️ Group '{group}' exists but has no variables.")
                for var in ds.variables:
                    full_var_name = f"{group}_{var}"
                    self.lut_variables[full_var_name] = ds[var]
                    setattr(self, full_var_name, ds[var])
                    #print(f"✅ Loaded: self.{full_var_name}")
            except OSError as e:
                if "group not found" in str(e):
                    print(f"❌ Group '{group}' not found.")
                else:
                    print(f"⚠️ Error loading group '{group}': {e}")
            except Exception as e:
                print(f"⚠️ Unexpected error loading group '{group}': {e}")
        
        print(" Done loading all available LUT variables.")
               
        


    def parse_structure(self):
        print(f"Parsing base structure for product at {self.path}")
        print("- annotation_coregistered directory:", self.annotation_coregistered_dir)
        print("- annotation_coregistrated_navigation_dir directory:", self.annotation_coregistrated_navigation_dir)
        
        print("- annotation_primary directory:", self.annotation_primary_dir)
        print("- annotation_primary_navigation_dir directory:", self.annotation_primary_navigation_dir)
        
        print("- Measurement directory:", self.measurement_dir)
        print("- Preview directory:", self.preview_dir)
        print("- Schema directory:", self.schema_dir)


    def check_structure(self):
        print("\n[ Checking required directories and files ]")
        required_dirs = [self.measurement_dir, self.annotation_coregistered_dir, self.annotation_coregistrated_navigation_dir ,self.annotation_primary_dir,self.annotation_primary_navigation_dir,self.preview_dir,self.schema_dir ]
        for d in required_dirs:
            print(f"{'✔️' if d.exists() else '❌'} Directory exists: {d}")
        print(' ')
        
     

    def check_tiff_files(self):
        print("\n[TIFF Files Check]")
        if not self.measurement_dir.exists():
            print("❌ Measurement directory not found.")
            return
        for tiff_file in self.measurement_dir.glob("*.tiff"):
            try:
                with rasterio.open(tiff_file) as src:
                    src.read(1)
                print(f"✔️ {tiff_file.name} is valid.")
            except Exception as e:
                print(f"❌ {tiff_file.name} failed to read: {e}")
        print(' ')
        
        
    def inspect_vrt(self):
        """
        Parses the VRT file and returns information about the referenced TIFFs.

        Returns:
            dict: VRT metadata including raster size and referenced bands/files.
        """
        print("\nChecking validity of VRT files...")
        
        if self.measurement_vrt_file is None or not self.measurement_vrt_file.exists():
            print("❌ No .vrt file found in the measurement directory.")
            return

        tree = ET.parse(self.measurement_vrt_file)
        root = tree.getroot()

        rasterXSize = root.attrib.get("rasterXSize")
        rasterYSize = root.attrib.get("rasterYSize")
        print(f"✅ VRT Dimensions: {rasterXSize} x {rasterYSize}")

        for band in root.findall(".//VRTRasterBand"):
            band_num = band.attrib.get("band")
            desc = band.findtext("Description") or "N/A"
            func = band.findtext("PixelFunctionType") or "N/A"
            print(f"\n🔹 Band {band_num}")
            print(f"   Description: {desc}")
            print(f"   Pixel Function: {func}")
            sources = [src.text for src in band.findall(".//SimpleSource/SourceFilename")]
            for i, src in enumerate(sources, start=1):
                print(f"   Source {i}: {src}")
        print(' ')

    def check_xml_validity(self):
        print("\n[Checking validity of XML files]")
        for xml_file in [self.annotation_coregistered_xml_file, self.annotation_primary_xml_file]:
            try:
                if xml_file:
                    ET.parse(xml_file)
                    print(f"✔️ {xml_file.name} is well-formed")
            except ET.ParseError as e:
                print(f"❌ Error in {xml_file.name}: {e}")
        print(' ')   
                
    def check_lut_contents(self):
        """
        Check the LUT file and print available variables. Warn if expected variables are missing.
        Uses group-aware inspection.
        """
        if not self.annotation_coregistered_lut_file:
            raise FileNotFoundError("LUT NetCDF file not found.")
    
        expected_groups = {
        "radiometry": [
                        "sigmaNought", "gammaNought"
                        ],
        "denoising": [
                        "denoisingHH", "denoisingXX", "denoisingVV"
                        ],
        "geometry": [
                    "latitude", "longitude", "height", "incidenceAngle", "elevationAngle", "terrainSlope"
                    ],
        "coregistration": [
                    "azimuthCoregistrationShifts", "rangeCoregistrationShifts",
                    "coregistrationShiftsQuality", "flatteningPhaseScreen","waveNumbers"
                    ],
    "skpPhaseCalibration": [
        "skpCalibrationPhaseScreen", "skpCalibrationPhaseScreenQuality"
    ],
    "baselineAndIonosphereCorrection": [
        "baselineErrorPhaseScreen",
        "residualIonospherePhaseScreen",
        "residualIonospherePhaseScreenQuality"
    ]
            }

        print("--- Checking LUT contents ---")
        for group, variables in expected_groups.items():
            try:
                ds = xr.open_dataset(self.annotation_coregistered_lut_file, group=group)
                for var in variables:
                    if var in ds.variables:
                        print(f"✔️ {group}/{var}")
                    else:
                        print(f"❌ MISSING: {group}/{var}")
            except Exception as e:
                for var in variables:
                    print(f"❌ MISSING: {group}/{var} (group error: {e})")
        print(' ')



    def plot_lut_variable(self, variable_name, save_geotiff=False):
        """
        Plot a variable from the LUT file and optionally save it as a GeoTIFF.
        Searches across all known groups.
        """
        if not self.annotation_coregistered_lut_file:
            raise FileNotFoundError("LUT NetCDF file not found.")
    
        groups = [
            "radiometry",
            "denoising",
            "geometry",
            "coregistration",
            "skpPhaseCalibration",
            "baselineAndIonosphereCorrection"
        ]
    
        var = None
        found_group = None
    
        for group in groups:
            try:
                ds_group = xr.open_dataset(self.annotation_coregistered_lut_file, group=group)
                if variable_name in ds_group:
                    var = ds_group[variable_name]
                    found_group = group
                    break
            except Exception:
                continue
    
        if var is None:
            raise KeyError(f"Variable '{variable_name}' not found in any known group.")
    
        data = var.values
        # Maschera i valori nodata
        data = np.ma.masked_equal(data, -9999)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(data, cmap="RdYlBu")
        plt.colorbar(label=f"{found_group}/{variable_name}")
        plt.title(f"{variable_name} from group '{found_group}'")
        plt.tight_layout()
        plt.show()
    
        if save_geotiff:
            try:
                lat_ds = xr.open_dataset(self.annotation_coregistered_lut_file, group="geometry")
                lats = lat_ds["latitude"].values
                lons = lat_ds["longitude"].values
    
                transform = from_origin(
                    np.min(lons),
                    np.max(lats),
                    abs(lons[0, 1] - lons[0, 0]),
                    abs(lats[1, 0] - lats[0, 0])
                )
    
                with rasterio.open(
                    f"{variable_name.replace('/', '_')}.tif",
                    'w',
                    driver='GTiff',
                    height=data.shape[0],
                    width=data.shape[1],
                    count=1,
                    dtype=data.dtype,
                    crs='EPSG:4326',
                    transform=transform
                ) as dst:
                    dst.write(data, 1)
    
                print(f"✅ Saved {variable_name} as GeoTIFF.")
            except Exception as e:
                print(f"⚠️ Failed to save GeoTIFF: {e}")


    def check_cog_integrity(self, tiff_path):
        with rasterio.open(tiff_path) as src:
            overviews = src.overviews(1)
            is_tiled = src.is_tiled
            compression = src.compression.name if src.compression else "None"

            print(f"COG Check for {tiff_path.name}:")
            print(f"  - Size: {src.width} x {src.height}")
            print(f"  - Tiled: {'✔' if is_tiled else '✖'}")
            print(f"  - Overviews: {overviews if overviews else '✖ None'}")
            print(f"  - Compression: {compression}")
            print(f"  - CRS: {src.crs}")
            print(f"  - Nodata: {src.nodata}")


    def visualize_geotiff(self, scale_factor=40):
        if not self.measurement_abs_file or not self.measurement_phase_file:
            print("❌ Required TIFF files (abs or phase) are missing.")
            return

        def prepare_overlay_image(tiff_path, cmap_name='viridis', scale_factor=4):
            with rasterio.open(tiff_path) as src:
                bounds = src.bounds
                data = src.read(
                    1,
                    out_shape=(
                        int(src.height / scale_factor),
                        int(src.width / scale_factor)
                    ),
                    resampling=Resampling.average
                )
                nodata = src.nodata if src.nodata is not None else -9999
                data = np.ma.masked_equal(data, nodata)
                vmin, vmax = np.percentile(data.compressed(), [2, 98])
                norm = Normalize(vmin=vmin, vmax=vmax)
                cmap = cm.get_cmap(cmap_name)
                rgba = cmap(norm(data.filled(vmin)))
                img = (rgba * 255).astype(np.uint8)
                pil_img = Image.fromarray(img)
                buffer = io.BytesIO()
                pil_img.save(buffer, format='PNG')
                encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
                data_url = f"data:image/png;base64,{encoded}"
            return data_url, bounds

        print("\nGenerating map overlay for ABS and PHASE...")

        abs_img_buf, abs_bounds = prepare_overlay_image(self.measurement_abs_file, cmap_name='viridis', scale_factor=scale_factor)
        phase_img_buf, _ = prepare_overlay_image(self.measurement_phase_file, cmap_name='twilight', scale_factor=scale_factor)

        lat_center, lon_center = self.compute_gcps_centroid(self.measurement_abs_file)

        m = folium.Map(
            location=[lat_center, lon_center],
            zoom_start=10
        )

        abs_overlay = ImageOverlay(image=abs_img_buf,
                                   bounds=[[abs_bounds.bottom, abs_bounds.left], [abs_bounds.top, abs_bounds.right]],
                                   opacity=0.6,
                                   name='ABS')

        phase_overlay = ImageOverlay(image=phase_img_buf,
                                     bounds=[[abs_bounds.bottom, abs_bounds.left], [abs_bounds.top, abs_bounds.right]],
                                     opacity=0.6,
                                     name='PHASE')

        abs_overlay.add_to(m)
        phase_overlay.add_to(m)
        folium.Marker([lat_center, lon_center], popup="GCP Center", icon=folium.Icon(color="blue")).add_to(m)
        folium.LayerControl().add_to(m)

        return m

    def compute_gcps_centroid(self, tiff_path):
        with rasterio.open(tiff_path) as src:
            if not src.gcps[0]:
                print("No GCPs found in the file.")
                return (0, 0)
            gcps = src.gcps[0]
            xs = [gcp.x for gcp in gcps]
            ys = [gcp.y for gcp in gcps]
            lon_centroid = np.mean(xs)
            lat_centroid = np.mean(ys)
            print(f"GCP centroid: Latitude {lat_centroid:.6f}, Longitude {lon_centroid:.6f}")
            return lat_centroid, lon_centroid
        
    def check_preview_png(self):
        """
        Checks integrity of the preview PNG file and displays it inline.
        """
        print("\n[ Preview PNG Check ]")
        if not self.preview_ql_file or not self.preview_ql_file.exists():
            print("❌ Preview PNG file not found.")
            return
    
        try:
            img = Image.open(self.preview_ql_file)
            img.verify()  # Check for corruption
            img = Image.open(self.preview_ql_file)  # Reopen for display
            print(f"✅ {self.preview_ql_file.name} is a valid PNG.")
            display(img)
        except Exception as e:
            print(f"❌ Failed to open {self.preview_ql_file.name}: {e}")

    def check_and_show_kml_overlay(self):
        """
        Check if KML is valid and show it on a Folium map.
        Supports only standard .kml (not .kmz).
        """

    
        file = self.preview_kmz_file
        if not file or not file.exists():
            print("❌ No KMz file found in preview.")
            return
    
        if file.suffix != ".kmz":
            print("❌ The file is not a valid .kml. Only plain KML is supported.")
            return
    
        try:
            # Try parsing to confirm it's well-formed XML
            ET.parse(file)
            print(f"✅ Valid KMz file: {file.name}")
    
            # Display in Folium
            m = Map(location=[0, 0], zoom_start=2)
            folium.Kml(str(file)).add_to(m)
            LayerControl().add_to(m)
            return m
    
        except ET.ParseError as e:
            print(f"❌ Error parsing KMz file: {e}")
        



class L2AProductPaths:
    def __init__(self, base_folder):
        self.base_folder = Path(base_folder)

    def get_path(self, prefix):
        """Trova il path del prodotto L2A in base al prefisso (FH, FD, GN)."""
        matches = list(self.base_folder.glob(f"BIO_FP_{prefix}__L2A_*"))
        if not matches:
            raise FileNotFoundError(f"No L2A product found for {prefix} in {self.base_folder}")
        return matches[0]  # se ce n'è uno solo, restituisci quello
        


    @property
    def fh(self):
        
        return self.get_path("FH")

    @property
    def fd(self):
        
        return self.get_path("FD")

    @property
    def gn(self):
        
        return self.get_path("GN")
    
    

class L2BProductPaths:
    def __init__(self, base_folder):
        self.base_folder = Path(base_folder)

    def get_path(self, prefix):
        """Trova il path del prodotto L2A in base al prefisso (FH, FD, GN)."""
        matches = list(self.base_folder.glob(f"BIO_FP_{prefix}_L2B_*"))
        if not matches:
            raise FileNotFoundError(f"No L2B product found for {prefix} in {self.base_folder}")
        return matches[0]  # se ce n'è uno solo, restituisci quello

    @property
    def fh(self):
        return self.get_path("FH_")

    @property
    def fd(self):
        return self.get_path("FD_")

    @property
    def agb(self):
        return self.get_path("AGB")



class BiomassProductL2:
    def __init__(self, path):
        self.path = Path(path)
        self.mph = next(self.path.glob("bio*.xml"), None)
        self.measurement_dir = self.path / "measurement"
        self.annotation_dir = self.path / "annotation"
        self.preview_dir = self.path / "preview"
        self.schema_dir = self.path / "schema" 
        
        # Default value for float NoData (will be filled after XML parsing)
        self.floatNoDataValue = None

        # Search for the main annotation XML file and parse it if available
        xml_file = next(self.annotation_dir.glob("*.xml"), None)
        if xml_file and xml_file.exists():
            self._parse_annotation_xml(xml_file)
        
        self.load_mph()
        
    
    def load_mph(self):
        """
        Parse the MPH XML (root of the product) and extract:
          - Acquisition parameters (eop:acquisitionParameters/bio:Acquisition)
          - Processing information (eop:processing/bio:ProcessingInformation)
     
        The extracted values are stored as class attributes (self.*).
        """
        
     
        # Locate the main XML file if not already stored
        if  self.mph is None:          
            self.mph = next(Path(self.path).glob("*.xml"), None)
     
        if self.mph is None:
            raise FileNotFoundError("MPH XML not found in product root.")
     
        # XML namespaces
        ns = {
            "bio": "http://earth.esa.int/biomass/1.0",
            "eop": "http://www.opengis.net/eop/2.1",
            "gml": "http://www.opengis.net/gml/3.2",
            "om":  "http://www.opengis.net/om/2.0",
            "ows": "http://www.opengis.net/ows/2.0",
            "sar": "http://www.opengis.net/sar/2.1",
            "xlink": "http://www.w3.org/1999/xlink",
            "xsi": "http://www.w3.org/2001/XMLSchema-instance",
        }
     
        # Helper to safely extract text and cast values
        def _get_text(root, xpath, cast=str):
            el = root.find(xpath, ns)
            if el is None or el.text is None:
                return None
            txt = el.text.strip()
            if cast is bool:
                return txt.lower() == "true"
            if cast is int:
                try: return int(txt)
                except ValueError: return None
            if cast is float:
                try: return float(txt)
                except ValueError: return None
            return txt
     
        # Parse the XML tree
        tree = ET.parse(self.mph)
        root = tree.getroot()
     
        # =========================
        # Acquisition parameters
        # =========================
        base = ".//eop:acquisitionParameters/bio:Acquisition"
     
        self.orbitNumber       = _get_text(root, f"{base}/eop:orbitNumber", int)
        self.lastOrbitNumber   = _get_text(root, f"{base}/eop:lastOrbitNumber", int)
        self.orbitDirection    = _get_text(root, f"{base}/eop:orbitDirection", str)
     
        self.wrsLongitudeGrid  = _get_text(root, f"{base}/eop:wrsLongitudeGrid", int)
        self.wrsLatitudeGrid   = _get_text(root, f"{base}/eop:wrsLatitudeGrid", int)
     
        self.ascendingNodeDate = _get_text(root, f"{base}/eop:ascendingNodeDate", str)
        self.startTimeFromAscendingNode      = _get_text(root, f"{base}/eop:startTimeFromAscendingNode", int)
        self.completionTimeFromAscendingNode = _get_text(root, f"{base}/eop:completionTimeFromAscendingNode", int)
     
        self.polarisationMode      = _get_text(root, f"{base}/sar:polarisationMode", str)
        pol_channels               = _get_text(root, f"{base}/sar:polarisationChannels", str)
        self.polarisationChannels  = [p.strip() for p in pol_channels.split(",")] if pol_channels else None
        self.antennaLookDirection  = _get_text(root, f"{base}/sar:antennaLookDirection", str)
     
        self.missionPhase     = _get_text(root, f"{base}/bio:missionPhase", str)
        self.instrumentConfID = _get_text(root, f"{base}/bio:instrumentConfID", int)
        self.dataTakeID       = _get_text(root, f"{base}/bio:dataTakeID", int)
        self.orbitDriftFlag   = _get_text(root, f"{base}/bio:orbitDriftFlag", bool)
        self.globalCoverageID = _get_text(root, f"{base}/bio:globalCoverageID", str)
        self.majorCycleID     = _get_text(root, f"{base}/bio:majorCycleID", str)
        self.repeatCycleID    = _get_text(root, f"{base}/bio:repeatCycleID", str)
     
        # =========================
        # Processing information
        # =========================
        pbase = ".//eop:processing/bio:ProcessingInformation"
     
        self.processingCenter   = _get_text(root, f"{pbase}/eop:processingCenter", str)
        self.processingDate     = _get_text(root, f"{pbase}/eop:processingDate", str)  # ISO 8601 string
        self.processorName      = _get_text(root, f"{pbase}/eop:processorName", str)
        self.processorVersion   = _get_text(root, f"{pbase}/eop:processorVersion", str)
        self.processingLevel    = _get_text(root, f"{pbase}/eop:processingLevel", str)
        self.processingMode     = _get_text(root, f"{pbase}/eop:processingMode", str)
     
        # Repeated fields: collect as lists
        self.auxiliaryDataSetFiles = [
            e.text.strip() for e in root.findall(f"{pbase}/eop:auxiliaryDataSetFileName", ns)
            if e is not None and e.text
        ]
        self.sourceProducts = [
            e.text.strip() for e in root.findall(f"{pbase}/bio:sourceProduct", ns)
            if e is not None and e.text
        ]
    
    def _parse_annotation_xml(self, xml_file):
        """
        Parse the main annotation XML file to extract relevant metadata.
        
        Parameters
        ----------
        xml_file : Path
            Path to the annotation XML file.
        """
        try:
            # Load and parse the XML file
            tree = ET.parse(xml_file)
            root = tree.getroot()
    
            # Find the <floatNoDataValue> tag in the XML and store its value
            float_no_data_elem = root.find(".//floatNoDataValue")
            if float_no_data_elem is not None:
                self.floatNoDataValue = float(float_no_data_elem.text.strip())
    
        except Exception as e:
            print(f"⚠️ Error while parsing XML {xml_file}: {e}")
    
    
    def parse_structure(self):
        print(f"Parsing base structure for product at {self.path}")
        print("- Measurement directory:", self.measurement_dir)
        print("- Annotation directory:", self.annotation_dir)
        print("- Preview directory:", self.preview_dir)
        print("- Schema directory:", self.schema_dir)

class l2a_fh(BiomassProductL2):
    def __init__(self, path):
        super().__init__(path)
        
        # Cerca i file di measurement
        self.measurement_file = None
        self.measurement_quality_file = None
        for file in self.measurement_dir.glob("*.tiff"):
            if "quality" in file.name.lower():
                self.measurement_quality_file = file
            else:
                self.measurement_file = file
        
        #search file in  annotation coregistrated
        self.annotation_xml_file = next(self.annotation_dir.glob("*.xml"), None)
        self.annotation_lut_file = next(self.annotation_dir.glob("*.nc"), None)     
        
        
        # Load LUT variables as self attributes
        self.lut_variables = {}

        if self.annotation_lut_file and self.annotation_lut_file.exists():
            try:
                group_names = [None]
                try:
                    import netCDF4
                    root = netCDF4.Dataset(self.annotation_lut_file, mode='r')
                    group_names.extend(list(root.groups.keys()))
                    root.close()
                except Exception:
                    pass

                for group in group_names:
                    ds = xr.open_dataset(self.annotation_lut_file, group=group) if group else xr.open_dataset(self.annotation_lut_file)
                    for var in ds.variables:
                        attr_name = f"{group}_{var}" if group else var
                        setattr(self, attr_name, ds[var])
                        self.lut_variables[attr_name] = ds[var]
            except Exception as e:
                print(f"⚠️ Failed to preload LUT variables: {e}")
        
                      
        #search file in  preview
        self.preview_ql_file = next(self.preview_dir.glob("*fh_ql.png"), None)
        self.preview_quality_file = next(self.preview_dir.glob("*fhquality_ql.png"), None)
        
        

    def parse_l2a_specific(self):
        print("Parsing L2A FH specific content...")
        print("- Main Measurement File:", self.measurement_file)
        print("- Quality Measurement File:", self.measurement_quality_file)
        print("- Annotation XML File:", self.annotation_xml_file)
        print("- Annotation LUT File:", self.annotation_lut_file)
        
        
 
    def check_structure(self):
        print("\n[Checking required directories]")
        for d in [self.measurement_dir, self.annotation_dir, self.preview_dir, self.schema_dir]:
            print(f"{'✔️' if d.exists() else '❌'} {d}")

    def check_tiff_files(self):
        print("\n[Checking TIFF files]")
        for tiff_file in [self.measurement_file, self.measurement_quality_file]:
            if tiff_file and tiff_file.exists():
                try:
                    with rasterio.open(tiff_file) as src:
                        src.read(1)
                    print(f"✔️ {tiff_file.name} is valid.")
                except Exception as e:
                    print(f"❌ {tiff_file.name} could not be read: {e}")
            else:
                print(f"❌ TIFF file missing: {tiff_file}")

    def check_xml_validity(self):
        print("\n[Checking XML file]")
        if self.annotation_xml_file and self.annotation_xml_file.exists():
            try:
                ET.parse(self.annotation_xml_file)
                print(f"✔️ {self.annotation_xml_file.name} is well-formed.")
            except ET.ParseError as e:
                print(f"❌ XML Parse Error: {e}")
        else:
            print("❌ XML file not found.")

    def check_lut_contents(self):
            print("[Listing available LUT contents]")
            if not self.annotation_lut_file or not self.annotation_lut_file.exists():
                print("❌ LUT NetCDF file not found.")
                return
    
            try:
                group_names = [None]
                try:
                    # Try opening with netcdf4 engine to access groups
                    root = netCDF4.Dataset(self.annotation_lut_file, mode='r')
                    group_names.extend(list(root.groups.keys()))
                    root.close()
                except Exception:
                    pass
    
                for group in group_names:
                    print(f" --- Group: {group or 'root'} ---")
                    ds = xr.open_dataset(self.annotation_lut_file, group=group) if group else xr.open_dataset(self.annotation_lut_file)
    
                    for var in ds.variables:
                        attr_name = f"{group}_{var}" if group else var
                        print(f" - {attr_name}	 shape: {ds[var].shape}")
    
                    if ds.attrs:
                        print("Global attributes:")
                        for key, val in ds.attrs.items():
                            print(f" - {key}: {val}")
            except Exception as e:
                print(f"❌ Failed to list LUT contents: {e}")


            
    

    def show_lut_fnf_on_map(self):
        if not self.annotation_lut_file or not self.annotation_lut_file.exists():
            print("❌ LUT NetCDF file not found.")
            return
        try:
            ds = xr.open_dataset(self.annotation_lut_file, group="FNF")
            fnf = ds["FNF"].values
            lat = ds["Latitude"].values
            lon = ds["Longitude"].values

            # Check for dimensionality
            if lat.ndim == 1 and lon.ndim == 1:
                lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
            elif lat.ndim == 2 and lon.ndim == 2:
                lat2d, lon2d = lat, lon
            else:
                print("❌ Unexpected dimensions for lat/lon arrays.")
                return

            if fnf.shape != lat2d.shape:
                print("🔁 Transposing FNF to match coordinate shape.")
                fnf = fnf.T

            norm = Normalize(vmin=0, vmax=1)
            cmap = cm.get_cmap("Greens")
            rgba = cmap(norm(fnf))
            img = (rgba * 255).astype(np.uint8)

            pil_img = Image.fromarray(img)
            buffer = io.BytesIO()
            pil_img.save(buffer, format='PNG')
            encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
            data_url = f"data:image/png;base64,{encoded}"

            top = float(lat2d.max())
            bottom = float(lat2d.min())
            left = float(lon2d.min())
            right = float(lon2d.max())

            m = folium.Map(location=[(top + bottom) / 2, (left + right) / 2], zoom_start=8)
            ImageOverlay(
                image=data_url,
                bounds=[[bottom, left], [top, right]],
                opacity=0.6,
                name="FNF"
            ).add_to(m)
            folium.LayerControl().add_to(m)
            return m

        except Exception as e:
            print(f"❌ Could not display FNF on map: {e}")


    def check_and_show_previews(self):
        """
        Check and display both quicklook PNGs: main and quality.
        """
        import matplotlib.pyplot as plt
    
        paths = {
            "Main Preview": self.preview_ql_file,
            "Quality Preview": self.preview_quality_file
        }
    
        valid_files = {}
    
        # Check existence and loadability
        for label, path in paths.items():
            if not path or not path.exists():
                print(f"❌ {label} not found.")
                continue
            try:
                img = Image.open(path)
                img.verify()  # check format, does not load
                img = Image.open(path)  # reopen after verify
                valid_files[label] = img
                print(f"✔️ {label} loaded: {path.name}, size: {img.size}")
            except Exception as e:
                print(f"❌ Error loading {label}: {e}")
    
        # Show if at least one is valid
        if valid_files:
            fig, axs = plt.subplots(1, len(valid_files), figsize=(10, 5))
            if len(valid_files) == 1:
                axs = [axs]
            for ax, (label, img) in zip(axs, valid_files.items()):
                ax.imshow(img)
                ax.set_title(label)
                ax.axis('off')
            plt.tight_layout()
            plt.show()

    def check_cog_integrity(self):
        print("\n[COG Integrity Check]")
        files = {
            "Measurement": self.measurement_file,
            "Quality": self.measurement_quality_file
        }
        for label, path in files.items():
            if not path or not path.exists():
                print(f"❌ {label} TIFF not found.")
                continue
            try:
                with rasterio.open(path) as src:
                    overviews = src.overviews(1)
                    is_tiled = src.is_tiled
                    compression = src.compression.name if src.compression else "None"
                    print(f"✅ {label} TIFF: {path.name}")
                    print(f"   - Size: {src.width} x {src.height}")
                    print(f"   - Tiled: {'✔' if is_tiled else '✖'}")
                    print(f"   - Overviews: {overviews if overviews else '✖ None'}")
                    print(f"   - Compression: {compression}")
                    print(f"   - CRS: {src.crs}")
                    print(f"   - Nodata: {src.nodata}")
            except Exception as e:
                print(f"❌ Error with {label}: {e}")
    
    def show_tiffs_on_map(self, scale_factor=1):
    
    
        files = {
            "Measurement": self.measurement_file,
            "Quality": self.measurement_quality_file
        }
    
        images = {}
        bounds = None
        center = None
    
        for label, path in files.items():
            if not path or not path.exists():
                continue
            try:
                with rasterio.open(path) as src:
                    data = src.read(
                        1,
                        out_shape=(
                            int(src.height / scale_factor),
                            int(src.width / scale_factor)
                        ),
                        resampling=Resampling.average
                    )
                    nodata = src.nodata if src.nodata is not None else -9999
                    data = np.ma.masked_equal(data, nodata)
                    vmin, vmax = np.percentile(data.compressed(), [2, 98])
                    norm = Normalize(vmin=vmin, vmax=vmax)
                    cmap = cm.get_cmap('viridis' if label == "Measurement" else 'magma')
                    rgba = cmap(norm(data.filled(vmin)))
                    img = (rgba * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img)
                    buffer = io.BytesIO()
                    pil_img.save(buffer, format='PNG')
                    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    images[label] = f"data:image/png;base64,{encoded}"
                    if not bounds:
                        bounds = src.bounds
                        center = [(bounds.top + bounds.bottom) / 2, (bounds.left + bounds.right) / 2]
            except Exception as e:
                print(f"❌ Error reading {label} TIFF: {e}")
    
        if images and bounds:
            dm = DualMap(location=center, zoom_start=10)
    
            # Left map: Measurement
            if "Measurement" in images:
                ImageOverlay(
                    image=images["Measurement"],
                    bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                    opacity=0.7,
                    name="Measurement"
                ).add_to(dm.m1)
    
            # Right map: Quality
            if "Quality" in images:
                ImageOverlay(
                    image=images["Quality"],
                    bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                    opacity=0.7,
                    name="Quality"
                ).add_to(dm.m2)
    
            folium.LayerControl().add_to(dm.m1)
            folium.LayerControl().add_to(dm.m2)
    
            return dm
        else:
            print("⚠️ No valid TIFFs to display on map.")
            




class l2a_fd(BiomassProductL2):
    def __init__(self, path):
        super().__init__(path)
        
        # Cerca i file di measurement
        self.measurement_cfm_file = None
        self.measurement_file = None
        self.measurement_probability_file = None
        
        
        for file in self.measurement_dir.glob("*.tiff"):
            if "cfm" in file.name.lower():
                self.measurement_cfm_file = file
                
            if "fd" in file.name.lower():
                    self.measurement_file = file

        
            if "probability" in file.name.lower():
                    self.measurement_probability_file = file
        
        # Cerca i file di annotation
        self.annotation_xml_file = None
        self.annotation_lut_file = None
        for file in self.annotation_dir.glob("*"):
            if file.suffix == ".xml":
                self.annotation_xml_file = file
            elif file.suffix == ".nc":
                self.annotation_lut_file = file
                                          
         
         
        # Load LUT variables as self attributes
        self.lut_variables = {}

        if self.annotation_lut_file and self.annotation_lut_file.exists():
             try:
                 group_names = [None]
                 try:
                     import netCDF4
                     root = netCDF4.Dataset(self.annotation_lut_file, mode='r')
                     group_names.extend(list(root.groups.keys()))
                     root.close()
                 except Exception:
                     pass

                 for group in group_names:
                     ds = xr.open_dataset(self.annotation_lut_file, group=group) if group else xr.open_dataset(self.annotation_lut_file)
                     for var in ds.variables:
                         attr_name = f"{group}_{var}" if group else var
                         setattr(self, attr_name, ds[var])
                         self.lut_variables[attr_name] = ds[var]
             except Exception as e:
                 print(f"⚠️ Failed to preload LUT variables: {e}")
         
                       
        #search file in  preview
        self.preview_ql_file = next(self.preview_dir.glob("*fd_ql.png"), None)
        self.preview_quality_file = next(self.preview_dir.glob("*probability_ql.png"), None)               
        self.preview_quality_file = next(self.preview_dir.glob("*cfm_ql.png"), None)                 
                

    def parse_l2a_specific(self):
        print("Parsing L2A FD specific content...")
        print("- Main Measurement File:", self.measurement_file)
        print("- cfm Measurement File:", self.measurement_cfm_file)
        print("- Probability Measurement File:", self.measurement_probability_file)
        
        print("- Annotation XML File:", self.annotation_xml_file)
        print("- Annotation LUT File:", self.annotation_lut_file)    
        
        
    def check_structure(self):
        print("\n[Checking required directories]")
        for d in [self.measurement_dir, self.annotation_dir, self.preview_dir, self.schema_dir]:
            print(f"{'✔️' if d.exists() else '❌'} {d}")

    def check_tiff_files(self):
        print("\n[Checking TIFF files]")
        for tiff_file in [self.measurement_file, self.measurement_cfm_file, self.measurement_probability_file]:
            if tiff_file and tiff_file.exists():
                try:
                    with rasterio.open(tiff_file) as src:
                        src.read(1)
                    print(f"✔️ {tiff_file.name} is valid.")
                except Exception as e:
                    print(f"❌ {tiff_file.name} could not be read: {e}")
            else:
                print(f"❌ TIFF file missing: {tiff_file}")

    def check_xml_validity(self):
        print("\n[Checking XML file]")
        if self.annotation_xml_file and self.annotation_xml_file.exists():
            try:
                ET.parse(self.annotation_xml_file)
                print(f"✔️ {self.annotation_xml_file.name} is well-formed.")
            except ET.ParseError as e:
                print(f"❌ XML Parse Error: {e}")
        else:
            print("❌ XML file not found.")

    def check_lut_contents(self):
            print("[Listing available LUT contents]")
            if not self.annotation_lut_file or not self.annotation_lut_file.exists():
                print("❌ LUT NetCDF file not found.")
                return
    
            try:
                group_names = [None]
                try:
                    # Try opening with netcdf4 engine to access groups
                    root = netCDF4.Dataset(self.annotation_lut_file, mode='r')
                    group_names.extend(list(root.groups.keys()))
                    root.close()
                except Exception:
                    pass
    
                for group in group_names:
                    print(f" --- Group: {group or 'root'} ---")
                    ds = xr.open_dataset(self.annotation_lut_file, group=group) if group else xr.open_dataset(self.annotation_lut_file)
    
                    for var in ds.variables:
                        attr_name = f"{group}_{var}" if group else var
                        print(f" - {attr_name}	 shape: {ds[var].shape}")
    
                    if ds.attrs:
                        print("Global attributes:")
                        for key, val in ds.attrs.items():
                            print(f" - {key}: {val}")
            except Exception as e:
                print(f"❌ Failed to list LUT contents: {e}")


            
    

    def plot_lut_variable(self, variable_name, save_geotiff=False):
            """
            Plot a variable from the LUT file and optionally save it as a GeoTIFF.
            Searches across all known groups.
            """
            if not self.annotation_lut_file:
                raise FileNotFoundError("LUT NetCDF file not found.")
        
            groups = ["FNF","ACM", "numberOfAverages"]
        
            var = None
            found_group = None
        
            for group in groups:
                try:
                    ds_group = xr.open_dataset(self.annotation_lut_file, group=group)
                    if variable_name in ds_group:
                        var = ds_group[variable_name]
                        found_group = group
                        break
                except Exception:
                    continue
        
            if var is None:
                raise KeyError(f"Variable '{variable_name}' not found in any known group.")
        
            data = var.values
            # Maschera i valori nodata
            data = np.ma.masked_equal(data, -9999)
            
            plt.figure(figsize=(8, 6))
            plt.imshow(data, cmap="RdYlBu")
            plt.colorbar(label=f"{found_group}/{variable_name}")
            plt.title(f"{variable_name} from group '{found_group}'")
            plt.tight_layout()
            plt.show()
        
            if save_geotiff:
                try:
                    lat_ds = xr.open_dataset(self.annotation_lut_file, group="geometry")
                    lats = lat_ds["latitude"].values
                    lons = lat_ds["longitude"].values
        
                    transform = from_origin(
                        np.min(lons),
                        np.max(lats),
                        abs(lons[0, 1] - lons[0, 0]),
                        abs(lats[1, 0] - lats[0, 0])
                    )
        
                    with rasterio.open(
                        f"{variable_name.replace('/', '_')}.tif",
                        'w',
                        driver='GTiff',
                        height=data.shape[0],
                        width=data.shape[1],
                        count=1,
                        dtype=data.dtype,
                        crs='EPSG:4326',
                        transform=transform
                    ) as dst:
                        dst.write(data, 1)
        
                    print(f"✅ Saved {variable_name} as GeoTIFF.")
                except Exception as e:
                    print(f"⚠️ Failed to save GeoTIFF: {e}")  


    def check_and_show_previews(self):
        """
        Check and display both quicklook PNGs: main and quality.
        """
        import matplotlib.pyplot as plt
    
        paths = {
            "Main Preview": self.preview_ql_file,
            "Quality Preview": self.preview_quality_file
        }
    
        valid_files = {}
    
        # Check existence and loadability
        for label, path in paths.items():
            if not path or not path.exists():
                print(f"❌ {label} not found.")
                continue
            try:
                img = Image.open(path)
                img.verify()  # check format, does not load
                img = Image.open(path)  # reopen after verify
                valid_files[label] = img
                print(f"✔️ {label} loaded: {path.name}, size: {img.size}")
            except Exception as e:
                print(f"❌ Error loading {label}: {e}")
    
        # Show if at least one is valid
        if valid_files:
            fig, axs = plt.subplots(1, len(valid_files), figsize=(10, 5))
            if len(valid_files) == 1:
                axs = [axs]
            for ax, (label, img) in zip(axs, valid_files.items()):
                ax.imshow(img)
                ax.set_title(label)
                ax.axis('off')
            plt.tight_layout()
            plt.show()

    def check_cog_integrity(self):
        print("\n[COG Integrity Check]")
        files = {
            "Measurement": self.measurement_file,
            "Quality": self.measurement_quality_file
        }
        for label, path in files.items():
            if not path or not path.exists():
                print(f"❌ {label} TIFF not found.")
                continue
            try:
                with rasterio.open(path) as src:
                    overviews = src.overviews(1)
                    is_tiled = src.is_tiled
                    compression = src.compression.name if src.compression else "None"
                    print(f"✅ {label} TIFF: {path.name}")
                    print(f"   - Size: {src.width} x {src.height}")
                    print(f"   - Tiled: {'✔' if is_tiled else '✖'}")
                    print(f"   - Overviews: {overviews if overviews else '✖ None'}")
                    print(f"   - Compression: {compression}")
                    print(f"   - CRS: {src.crs}")
                    print(f"   - Nodata: {src.nodata}")
            except Exception as e:
                print(f"❌ Error with {label}: {e}")
    
    
        
        
        
        
        

class l2a_gn(BiomassProductL2):
    def __init__(self, path):
        super().__init__(path)
        
        # Cerca i file di measurement
        self.measurement_file = None
                
        for file in self.measurement_dir.glob("*.tiff"):
            if "gn" in file.name.lower():

                self.measurement_file = file
        
        # Cerca i file di annotation
        self.annotation_xml_file = None
        self.annotation_lut_file = None
        for file in self.annotation_dir.glob("*"):
            if file.suffix == ".xml":
                self.annotation_xml_file = file
            elif file.suffix == ".nc":
                self.annotation_lut_file = file
                
        # Load LUT variables as self attributes
        self.lut_variables = {}

        if self.annotation_lut_file and self.annotation_lut_file.exists():
             try:
                 group_names = [None]
                 try:
                     import netCDF4
                     root = netCDF4.Dataset(self.annotation_lut_file, mode='r')
                     group_names.extend(list(root.groups.keys()))
                     root.close()
                 except Exception:
                     pass

                 for group in group_names:
                     ds = xr.open_dataset(self.annotation_lut_file, group=group) if group else xr.open_dataset(self.annotation_lut_file)
                     for var in ds.variables:
                         attr_name = f"{group}_{var}" if group else var
                         setattr(self, attr_name, ds[var])
                         self.lut_variables[attr_name] = ds[var]
             except Exception as e:
                 print(f"⚠️ Failed to preload LUT variables: {e}")
         
                       
        #search file in  preview
        self.preview_ql_file = next(self.preview_dir.glob("*gn_ql.png"), None)
               
                
        

    def parse_l2a_specific(self):
        print("Parsing L2A GN specific content...")
        print("- Main Measurement File:", self.measurement_file)

        
        print("- Annotation XML File:", self.annotation_xml_file)
        print("- Annotation LUT File:", self.annotation_lut_file)
        
        
        

class l2b_fh(BiomassProductL2):
    def __init__(self, path):
        super().__init__(path)
        
        # Cerca i file di measurement
        self.measurement_file = None
        self.measurement_fhquality_file = None
        
        for file in self.measurement_dir.glob("*.tiff"):
            if "fhquality" in file.name.lower():
                self.measurement_fhquality_file = file
            else:
                self.measurement_file = file
        
        # Cerca i file di annotation
        
        # Cerca i file di annotation
        self.annotation_xml_file = None
        self.annotation_acquisition_file = None
        self.annotation_heatmap_file = None
        self.annotation_bps_fnf_file = None
        
        for file in self.annotation_dir.glob("*"):
            if file.suffix == ".xml":
                self.annotation_xml_file = file
                
            if "acquisition_id_image" in file.name.lower():
                self.annotation_acquisition_file = file
                
            if "heatmap" in file.name.lower():
                    self.annotation_heatmap_file = file
                
            if "bps" in file.name.lower():
                    self.annotation_bps_fnf_file = file                
                

    def parse_l2b_specific(self):
        print("Parsing L2B FH specific content...")
        print("- Main Measurement File:", self.measurement_file)
        print("- Quality Measurement File:", self.measurement_fhquality_file)
        
        
        
        print("- Annotation XML File:", self.annotation_xml_file)
        print("- Annotation Heatmap File:", self.annotation_heatmap_file)
        print("- Annotation Acquisition id  File:", self.annotation_acquisition_file)
        print("- Annotation bps File:", self.annotation_bps_fnf_file)        

class l2b_fd(BiomassProductL2):
    def __init__(self, path):
        super().__init__(path)
        
        # Cerca i file di measurement
        self.measurement_cfm_file = None
        self.measurement_file = None
        self.measurement_probability_file = None
        
        
        for file in self.measurement_dir.glob("*.tiff"):
            if "cfm" in file.name.lower():
                self.measurement_cfm_file = file
                
            if "probability" in file.name.lower():
                    self.measurement_probability_file = file
            else:
                self.measurement_file = file
        
        # Cerca i file di annotation
        self.annotation_xml_file = None
        self.annotation_acquisition_file = None
        self.annotation_heatmap_file = None
        
        for file in self.annotation_dir.glob("*"):
            if file.suffix == ".xml":
                self.annotation_xml_file = file
                
            if "acquisition_id_image" in file.name.lower():
                self.annotation_acquisition_file = file
                
            if "heatmap" in file.name.lower():
                    self.annotation_heatmap_file = file

    def parse_l2b_specific(self):
        print("Parsing L2B FD specific content...")
        print("- Main Measurement File:", self.measurement_file)
        print("- cfm Measurement File:", self.measurement_cfm_file)
        print("- Probability Measurement File:", self.measurement_probability_file)
        
        print("- Annotation XML File:", self.annotation_xml_file)
        print("- Annotation Heatmap File:", self.annotation_heatmap_file)
        print("- Annotation Acquisition id  File:", self.annotation_acquisition_file)

class l2b_agb(BiomassProductL2):
    def __init__(self, path):
        super().__init__(path)
        
        # Cerca i file di measurement
        self.measurement_file = None
        self.measurement_std_file = None
        
        for file in self.measurement_dir.glob("*.tiff"):
            if "std" in file.name.lower():
                self.measurement_std_file = file
            else:
                self.measurement_file = file
        
        # Cerca i file di annotation
        
        # Cerca i file di annotation
        self.annotation_xml_file = None
        self.annotation_acquisition_file = None
        self.annotation_heatmap_file = None
        self.annotation_bps_fnf_file = None
        
        for file in self.annotation_dir.glob("*"):
            if file.suffix == ".xml":
                self.annotation_xml_file = file
                
            if "acquisition_id_image" in file.name.lower():
                self.annotation_acquisition_file = file
                
            if "heatmap" in file.name.lower():
                    self.annotation_heatmap_file = file
                
            if "bps" in file.name.lower():
                    self.annotation_bps_fnf_file = file                
                

    def parse_l2b_specific(self):
        print("Parsing L2B AGB specific content...")
        print("- Main Measurement File:", self.measurement_file)
        print("- Quality Measurement File:", self.measurement_std_file)
        
        
        
        print("- Annotation XML File:", self.annotation_xml_file)
        print("- Annotation Heatmap File:", self.annotation_heatmap_file)
        print("- Annotation Acquisition id  File:", self.annotation_acquisition_file)
        print("- Annotation bps File:", self.annotation_bps_fnf_file)  


# test
if __name__ == "__main__":
    '''
    product_STA_path = r"E:\BIOMASS\03_SCRIPTS\TDS-BPS-L2A-010b\l1c\BIO_S2_STA__1S_20170125T163833_20170125T163900_I_G03_M03_C03_T000_F001_01_D4PPSH" 
    product_STA= BiomassProductSTA(product_STA_path)
    
    print(product_STA.annotation_coregistered_lut_file)
    product_STA.parse_structure()
    product_STA.check_structure()
    product_STA.check_tiff_files()
    product_STA.check_xml_validity()
    product_STA.inspect_vrt()
    product_STA.check_lut_contents()
    product_STA.plot_lut_variable("skpCalibrationPhasesScreen", save_geotiff=True)
    product_STA.plot_lut_variable("flatteningPhasesScreen")
    product_STA.visualize_geotiff()
    '''
    
    
   

    '''
    product_FH__L2A_path = r"E:\BIOMASS\03_SCRIPTS\example-notebooks\data\BIO_FP_FH__L2A_20170125T163833_20170206T163906_T_G03_M03_C___T000_F001_02_D6IOT7"
    product_FD__L2A_path = r"E:\BIOMASS\03_SCRIPTS\example-notebooks\data\BIO_FP_FD__L2A_20170125T163833_20170206T163906_T_G03_M03_C___T000_F001_02_D6IOR3"
    product_GN__L2A_path = r"E:\BIOMASS\03_SCRIPTS\example-notebooks\data\BIO_FP_GN__L2A_20170125T163833_20170206T163906_T_G03_M03_C___T000_F001_02_D6IOWK"
    
    
    product_FH__L2B_path = r"E:\BIOMASS\03_SCRIPTS\example-notebooks\data\BIO_FP_FH__L2B_I_G01_TS35W056_B200_02_D5MSK0"
    product_FD__L2B_path = r"E:\BIOMASS\03_SCRIPTS\example-notebooks\data\BIO_FP_FD__L2B_I_G01_TS35W056_B200_02_D5MSKG"
    product_AGB__L2B_path = r"E:\BIOMASS\03_SCRIPTS\example-notebooks\data\BIO_FP_AGB_L2B_C_G___TS34W055_B001_02_D74PU1"    
    
    
    
    product_fh_l2a = l2a_fh(product_FH__L2A_path)
    product_fh_l2a.check_structure()
    product_fh_l2a.check_lut_contents()
    product_fh_l2a.check_tiff_files()
    product_fh_l2a.check_xml_validity()
    product_fh_l2a.check_and_show_quicklook()
    product_fh_l2a.check_and_show_kml_overlay()
    
    
    product_fd_l2a = l2a_fd(product_FD__L2A_path)    
    product_gn_l2a = l2a_gn(product_GN__L2A_path)    
    
    product_fh_l2b = l2b_fh(product_FH__L2B_path)
    product_fd_l2b = l2b_fd(product_FD__L2B_path)    
    product_gn_l2b = l2b_agb(product_AGB__L2B_path)     
    
    
    product_fh_l2a.parse_l2a_specific()
    product_fd_l2a.parse_l2a_specific()
    product_fh_l2a.parse_l2a_specific()
    
    product_fh_l2b.parse_l2b_specific()
    product_fd_l2b.parse_l2b_specific()
    product_gn_l2b.parse_l2b_specific()
    '''
    
    product_SCS_path = r"E:\BIOMASS\02_DATA\20250522\BIO_S1_SCS__1S_20250525T095726_20250525T095919_C_G___M___C___T____F001_01_D9AIP8" 
    product_SCS= BiomassProductSCS(product_SCS_path)
    print(product_SCS.denoising_denoisingHH)
    print(product_SCS.denoising_denoisingHV)
    print(product_SCS.denoising_denoisingVH)
    print(product_SCS.denoising_denoisingVV)

    print(product_SCS.geometry_latitude)
    print(product_SCS.geometry_longitude)
    print(product_SCS.geometry_height)
    print(product_SCS.geometry_incidenceAngle)
    print(product_SCS.geometry_elevationAngle)
    print(product_SCS.geometry_terrainSlope)
    
    print(product_SCS.radiometry_sigmaNought)
    print(product_SCS.radiometry_gammaNought)
    
    
    print(product_SCS.geometry_incidenceAngle)  
    print(product_SCS.denoising_denoisingVV)   
    print(product_SCS.radiometry_sigmaNought)  
    print(product_SCS.global_polarisationList)  
    print(product_SCS.global_orbitPass)       

    '''
    product_SCS.load_complex_polarizations()

    hh_complex = product_SCS.S['HH']

    vv_amp = product_SCS.A['VV']

    hv_phase = product_SCS.P['HV']
    
    print (hv_phase)
    print (vv_amp)
    print(hh_complex)
    '''    
    #ratio = vv_amp / (product_SCS.A['HH'] + 1e-6)
    #product_SCS.save_raster(vv_amp, "E:\BIOMASS\VV_.tif")
    #rgb = product_SCS.generate_pauli_rgb()

    # Salva come GeoTIFF 3 bande
    #product_SCS.save_rgb(rgb, "E:/BIOMASS/pauli_rgb.tif")
    print(product_SCS.footprint_polygon)
   
    '''
    product_DGM_path = r"E:\BIOMASS\02_DATA\V330\BIO_S1_DGM__1S_20170101T060309_20170101T060330_I_G03_M03_C03_T010_F001_01_D87CQE" 
    product_DGM= BiomassProductDGM(product_DGM_path)
    print(product_DGM.denoising_denoisingHH)
    print(product_DGM.denoising_denoisingHV)
    print(product_DGM.denoising_denoisingVH)
    print(product_DGM.denoising_denoisingVV)
    
    print(product_DGM.geometry_latitude)
    print(product_DGM.geometry_longitude)
    print(product_DGM.geometry_height)
    print(product_DGM.geometry_incidenceAngle)
    print(product_DGM.geometry_elevationAngle)
    print(product_DGM.geometry_terrainSlope)
    
    print(product_DGM.radiometry_sigmaNought)
    print(product_DGM.radiometry_gammaNought)
    
    
    print(product_DGM.geometry_incidenceAngle)  
    print(product_DGM.denoising_denoisingVV)   
    print(product_DGM.radiometry_sigmaNought)  
    print(product_DGM.global_polarisationList)  
    print(product_DGM.global_orbitPass) 
    

    
    # 1. Esporta le bande singole
    product_DGM.export_abs_bands_separately("E:/BIOMASS/temp_bands")

    # 2. Ricombina in un file GeoTIFF multibanda
    product_DGM.recombine_abs_bands("E:/BIOMASS/temp_bands", "E:/BIOMASS/abs_final_snap_ready.tif")

    
    product_RAW0S_path=r'E:\BIOMASS\03_SCRIPTS\STACK_SELECTOR\WD_01\L0_inputs\S3_RAW__0S\BIO_S3_RAW__0S_20170403T220008_20170403T220205_I_G01_M03_C01_T040_F001_01_D6LL6U'
    product_RAW0S=BiomassProductRAWS(product_RAW0S_path)
    print(product_RAW0S.measurement_rxv_file)
    print(product_RAW0S.valid_start_time)
    print(product_RAW0S.valid_end_time)
    
    
    product_L1VFRA_path=r'E:\BIOMASS\03_SCRIPTS\STACK_SELECTOR\WD_01\L0_inputs\S3_RAW__0S\framer_BIO_S3_RAW__0S_20170403T220008_20170403T220205_I_G01_M03_C01_T040_F001_01_D6LL6U/BIO_OPER_CPF_L1VFRA_20170403T220015_20170403T220040_01_D8Q3HZ.EOF'
    product_L1VFRA=BiomassProductL1VFRA(product_L1VFRA_path)
    print(product_L1VFRA.frame_start_time)
    print(product_L1VFRA.frame_stop_time)
    print(product_L1VFRA.frame_id)
    '''   