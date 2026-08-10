import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from src.visualization.plot_unet_overlay import generate_clean_overlay


def run_lgbm_overlays():
    for year in ['2017', '2026']:
        raw_path = f"data/processed/features_Sentinel2_Original_{year}.tif"
        lgbm_path = f"data/classified_lgbm/lulc_map_lgbm_{year}.tif"
        out_path = f"docs/figures/lgbm_overlay_clean_{year}.png"
        
        if os.path.exists(raw_path) and os.path.exists(lgbm_path):
            generate_clean_overlay(raw_path, lgbm_path, out_path, title_prefix="LightGBM", year_label=year)
        else:
            print(f"Missing input files for LightGBM year {year}")


if __name__ == "__main__":
    run_lgbm_overlays()