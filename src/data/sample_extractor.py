import os
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Point

def extract_features(
    points_csv="data/vector/train_points.csv",
    raster_path="data/processed/features_Sentinel2_Original_2026.tif",
    output_csv="data/processed/train_features.csv"
):
    df_points = pd.read_csv(points_csv)
    geometry = [Point(xy) for xy in zip(df_points.longitude, df_points.latitude)]
    gdf = gpd.GeoDataFrame(df_points, geometry=geometry, crs="EPSG:4326")

    with rasterio.open(raster_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        coords = [(geom.x, geom.y) for geom in gdf.geometry]
        sampled_values = list(src.sample(coords))
        
        band_names = [f"Band_{i+1}" if not desc else desc for i, desc in enumerate(src.descriptions)]
        
        features_df = pd.DataFrame(sampled_values, columns=band_names)
        
        # Drop redundant categorical/mask bands from feature set
        drop_cols = [c for c in ['SCL', 'QA60', 'MSK_CLDPRB'] if c in features_df.columns]
        features_df = features_df.drop(columns=drop_cols)

        final_df = pd.concat([gdf[['class_id', 'class_name', 'longitude', 'latitude']], features_df], axis=1)
        final_df.to_csv(output_csv, index=False)
        print(f"Extracted feature matrix: {output_csv} (Shape: {final_df.shape})")

if __name__ == "__main__":
    extract_features()