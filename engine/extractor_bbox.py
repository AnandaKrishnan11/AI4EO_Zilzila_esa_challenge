import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
import os
from utils import PathCollector
from tqdm import tqdm
from shapely.geometry import mapping
import matplotlib.pyplot as plt
from PIL import Image
import json
import pandas as pd
import argparse


class BitemporalImageCropper:
    def __init__(self,img_path, excel_path,output_base_dir):
        """
        Initialize the bitemporal image cropper
        
        Args:
            pre_image_path (str): Path to pre-event image
            post_image_path (str): Path to post-event image
            shapefile_path (str): Path to shapefile with bounding boxes
            output_base_dir (str): Base directory for output
        """

        collector = PathCollector(img_path,excel_path)
        self.image_collection = collector.collect()

        self.output_base_dir = output_base_dir
        
        # Create output directories
        self.pre_output_dir = os.path.join(output_base_dir, 'pre_event_crops')
        self.post_output_dir = os.path.join(output_base_dir, 'post_event_crops')
        self.label_dir = os.path.join(output_base_dir, 'labels')

        os.makedirs(self.pre_output_dir, exist_ok=True)
        os.makedirs(self.post_output_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)
        
        # Empty loader
        self.gdf = gpd.GeoDataFrame()

   
        
    def validate_geometries(self,shapefile_path):
        """Validate and clean geometries in the shapefile"""
        valid_features = []
        invalid_features = []
        
        self.gdf = gpd.read_file(shapefile_path)
        print(f"Loaded {len(self.gdf)} features from shapefile")


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
    
    def check_crs_compatibility(self, image_path):
        """Check and handle CRS compatibility between shapefile and images"""
        with rasterio.open(image_path) as src:
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
                
                return cropped_image, out_meta
                
                
        except Exception as e:
            print(f"Error cropping image for feature {feature_id}: {e}")
            
    
    
    def crop_all_features(self,feature_id_field='id', all_touched=True):
        """
        Crop both pre and post images for all features
        
        Args:
            feature_id_field (str): Field name for feature identification
            save_png (bool): Whether to save PNG versions for visualization
            all_touched (bool): Include all touched pixels
        
        Returns:
            list: List of metadata for all cropped images
        """

        counter = 0
    
        for i, row in self.image_collection.iterrows():
                pre_img = row["pre"]
                post_img = row["post"]
                shp = row["shp"]

                # Validate geometries and CRS
                self.validate_geometries(shp)
                self.check_crs_compatibility(post_img)

                
                
                for idx, row in tqdm(self.gdf.iterrows(), total=len(self.gdf), desc="Cropping features"):
                    
                    if feature_id_field in row:
                        feature_id = row[feature_id_field]
                    else:
                        feature_id = f"feature_{idx}"
                    
                    geometry = row.geometry
                    dem_cls = row.damaged
                    
                
                    pre_tif_path = os.path.join(self.pre_output_dir, f"{counter}.tif")
                    post_tif_path = os.path.join(self.post_output_dir, f"{counter}.tif")
                    label_path = os.path.join(self.label_dir, f"{counter}.txt")
                    
                    
                    
                    
                    pre_result = self.crop_single(pre_img, geometry, pre_tif_path, feature_id, all_touched)
                
                
                    post_result = self.crop_single(post_img, geometry, post_tif_path, feature_id, all_touched)

                    if pre_result and post_result:
                        pre,out_meta_pre = pre_result
                        post,out_meta_post = post_result

                        if pre is not None and post is not None:
                            # Save the cropped image
                            with rasterio.open(pre_tif_path, "w", **out_meta_pre) as dest:
                                dest.write(pre)

                            with rasterio.open(post_tif_path, "w", **out_meta_post) as dest:
                                dest.write(post)


                            with open(label_path, 'w') as txt:
                                txt.write(f'{dem_cls}\n')  # check this line 
                            counter+=1
                        else:
                            continue


def parser():
    pars_vals = argparse.ArgumentParser(description='parsing arguments')
    pars_vals.add_argument('--image_path', type=str, help='the path to the directory')
    pars_vals.add_argument('--excel_path', type=str, help='excel sheet path')
    pars_vals.add_argument('--output_base_dir', type=str, help='output directory to save pre, post and labels')
    return pars_vals.parse_args()


def main():



    args = parser()
    
    cropper = BitemporalImageCropper(
        img_path = args.image_path,
        excel_path= args.excel_path,
        output_base_dir=args.output_base_dir
    )
    
    cropper.crop_all_features(
        feature_id_field='id',
        all_touched=True
    )

    

if __name__ == "__main__":
    main()