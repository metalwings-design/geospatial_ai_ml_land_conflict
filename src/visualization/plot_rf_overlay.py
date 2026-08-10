import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from src.visualization.plot_unet_overlay import generate_clean_overlay


def run_rf_overlays():
    for year in ['2017', '2026']:
        raw_path = f"data/processed/features_Sentinel2_Original_{year}.tif"
        rf_path = f"data/classified/lulc_map_{year}.tif"
        out_path = f"docs/figures/rf_overlay_clean_{year}.png"
        
        if os.path.exists(raw_path) and os.path.exists(rf_path):
            generate_clean_overlay(raw_path, rf_path, out_path, title_prefix="Random Forest", year_label=year)
        else:
            print(f"Missing input files for RF year {year}")


if __name__ == "__main__":
    run_rf_overlays()