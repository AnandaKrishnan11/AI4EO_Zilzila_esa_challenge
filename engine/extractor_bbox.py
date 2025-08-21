import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
import os
from tqdm import tqdm
from shapely.geometry import mapping
import matplotlib.pyplot as plt
from PIL import Image
import json
import pandas as pd
import argparse


class BitemporalImageCropper:
    def __init__(self, pre_image_path, post_image_path, shapefile_path, output_base_dir):
        """
        Initialize the bitemporal image cropper
        
        Args:
            pre_image_path (str): Path to pre-event image
            post_image_path (str): Path to post-event image
            shapefile_path (str): Path to shapefile with bounding boxes
            output_base_dir (str): Base directory for output
        """
        self.pre_image_path = pre_image_path
        self.post_image_path = post_image_path
        self.shapefile_path = shapefile_path
        self.output_base_dir = output_base_dir
        
        # Create output directories
        self.pre_output_dir = os.path.join(output_base_dir, 'pre_event_crops')
        self.post_output_dir = os.path.join(output_base_dir, 'post_event_crops')
        self.label_dir = os.path.join(output_base_dir, 'labels')

        os.makedirs(self.pre_output_dir, exist_ok=True)
        os.makedirs(self.post_output_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)
        
        # Load shapefile
        self.gdf = gpd.read_file(shapefile_path)
        print(f"Loaded {len(self.gdf)} features from shapefile")
        
    def validate_geometries(self):
        """Validate and clean geometries in the shapefile"""
        valid_features = []
        invalid_features = []
        
        for idx, row in self.gdf.iterrows():
            geometry = row.geometry
            if geometry.is_valid and not geometry.is_empty:
                valid_features.append(idx)
            else:
                invalid_features.append(idx)
                print(f"Invalid geometry at index {idx}")
        
        # Keep only valid geometries
        if invalid_features:
            self.gdf = self.gdf.iloc[valid_features].copy()
            print(f"Removed {len(invalid_features)} invalid geometries")
    
    def check_crs_compatibility(self):
        """Check and handle CRS compatibility between shapefile and images"""
        with rasterio.open(self.pre_image_path) as src:
            raster_crs = src.crs
        
        if self.gdf.crs != raster_crs:
            print(f"CRS mismatch: Shapefile CRS {self.gdf.crs}, Raster CRS {raster_crs}")
            print("Reprojecting shapefile to match raster CRS...")
            self.gdf = self.gdf.to_crs(raster_crs)
    
    def crop_single(self, image_path, geometry, output_path, feature_id, all_touched=True):
        """
        Crop a single image using a geometry
        
        Args:
            image_path (str): Path to input image
            geometry: Shapely geometry object
            output_path (str): Path to save cropped image
            feature_id: ID of the feature for metadata
            all_touched (bool): Include all touched pixels
        
        Returns:
            dict: Metadata about the cropped image
        """
        try:
            with rasterio.open(image_path) as src:
                
                geojson_geom = [mapping(geometry)]
                
                cropped_image, cropped_transform = mask(
                    src, 
                    geojson_geom, 
                    crop=True, 
                    all_touched=all_touched
                )
                
                # Update metadata
                out_meta = src.meta.copy()
                out_meta.update({
                    "height": cropped_image.shape[1],
                    "width": cropped_image.shape[2],
                    "transform": cropped_transform
                })
                
                # Save the cropped image
                with rasterio.open(output_path, "w", **out_meta) as dest:
                    dest.write(cropped_image)
                
                
        except Exception as e:
            print(f"Error cropping image for feature {feature_id}: {e}")
            return None
    
    
    def crop_all_features(self, feature_id_field='id', all_touched=True):
        """
        Crop both pre and post images for all features
        
        Args:
            feature_id_field (str): Field name for feature identification
            save_png (bool): Whether to save PNG versions for visualization
            all_touched (bool): Include all touched pixels
        
        Returns:
            list: List of metadata for all cropped images
        """
        # Validate geometries and CRS
        self.validate_geometries()
        self.check_crs_compatibility()

        
        
        for idx, row in tqdm(self.gdf.iterrows(), total=len(self.gdf), desc="Cropping features"):
            
            if feature_id_field in row:
                feature_id = row[feature_id_field]
            else:
                feature_id = f"feature_{idx}"
            
            geometry = row.geometry
            dem_cls = row.damaged
            
           
            pre_tif_path = os.path.join(self.pre_output_dir, f"{feature_id}.tif")
            post_tif_path = os.path.join(self.post_output_dir, f"{feature_id}.tif")
            label_path = os.path.join(self.label_dir, f"{feature_id}.txt")
            
            
            self.crop_single(
                self.pre_image_path, geometry, pre_tif_path, feature_id, all_touched
            )
            
            
            self.crop_single(
                self.post_image_path, geometry, post_tif_path, feature_id, all_touched
            )

            with open(label_path, 'w') as txt:
                txt.write(f'{dem_cls}\n')  # check this line 


def parser():
    pars_vals = argparse.ArgumentParser(description='parsing arguments')
    pars_vals.add_argument('--pre_image_path', type=str, help='the pre image path')
    pars_vals.add_argument('--post_image_path', type=str, help='post image path')
    pars_vals.add_argument('--shapefile_path', type=str, help='shape file path')
    pars_vals.add_argument('--output_base_dir', type=str, help='output directory to save pre, post and labels')
    return pars_vals.parse_args()


def main():

    args = parser()
    
    cropper = BitemporalImageCropper(
        pre_image_path = args.pre_image_path,
        post_image_path= args.post_image_path,
        shapefile_path=args.shapefile_path,
        output_base_dir=args.output_base_dir
    )
    
    cropper.crop_all_features(
        feature_id_field='id',
        all_touched=True
    )
    

    print(f"Pre-event crops saved in: {cropper.pre_output_dir}")
    print(f"Post-event crops saved in: {cropper.post_output_dir}")
    print(f"Post-event crops saved in: {cropper.post_output_dir}")
    

if __name__ == "__main__":
    main()