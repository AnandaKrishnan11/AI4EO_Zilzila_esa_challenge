import os
import glob
import pandas as pd
import geopandas as gpd

COUNTRIES_NEED = ['Call_970_Afghanistan','Call_1075_China']

class PathCollector:

    def __init__(self, file_path, excel_path):
        self.file_path = file_path
        self.excel_path = excel_path
        self.gdf = gpd.GeoDataFrame(columns=["pre", "post", "shp"])  

    def collect(self):
        pre, post, shp = [], [], []

        if os.path.isfile(self.excel_path):
            df = pd.read_excel(self.excel_path)
            pre_list = df.loc[df["Pre or Post"]=="Pre-event"]["Dataset ID"].to_list()
            post_list = df.loc[df["Pre or Post"]=="Post-event"]["Dataset ID"].to_list()
        else:
            raise FileNotFoundError("Excel file not found!")

        if os.path.isdir(self.file_path):
            countries = os.listdir(self.file_path)
            for sing_count in countries:
                if sing_count in COUNTRIES_NEED:
                    country_path = os.path.join(self.file_path, sing_count)
                    country_level = os.listdir(country_path)

                    for items in pre_list:
                        if items in country_level:
                            pre_path = os.path.join(country_path, items, "Optical_Calibration", "overview-trc_PROJ.tif")
                            pre.append(pre_path)

                    for items in post_list:
                        if items in country_level:
                            post_path = os.path.join(country_path, items, "Optical_Calibration", "overview-trc_PROJ.tif")
                            post.append(post_path)

                    shp_country_path = sorted(glob.glob(os.path.join(self.file_path, "Annotations", sing_count, "*_training.gpkg")))
                    shp.extend(shp_country_path)

        self.gdf = gpd.GeoDataFrame(
            [{"pre": p, "post": d, "shp": s} for p, d, s in zip(pre, post, shp)]
        )

        return self.gdf
