import os
import sys

# Resolve project root path dynamically
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def normalize_rgb(rgb_array, valid_mask):
    rgb = np.nan_to_num(rgb_array)
    valid_pixels = rgb[valid_mask]
    
    if len(valid_pixels) > 0:
        p2, p98 = np.percentile(valid_pixels, (2, 98))
    else:
        p2, p98 = 0.0, 1.0
        
    rgb_clipped = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
    rgb_clipped[~valid_mask] = 0.0
    return rgb_clipped


def generate_clean_overlay(
    raw_raster_path,
    classified_raster_path,
    output_png_path,
    title_prefix="U-Net",
    year_label="2026"
):
    with rasterio.open(raw_raster_path) as src_raw:
        descriptions = [d if d else f"Band_{i+1}" for i, d in enumerate(src_raw.descriptions)]
        
        r_idx = descriptions.index('B4') + 1 if 'B4' in descriptions else 4
        g_idx = descriptions.index('B3') + 1 if 'B3' in descriptions else 3
        b_idx = descriptions.index('B2') + 1 if 'B2' in descriptions else 2
        
        red = src_raw.read(r_idx).astype(np.float32)
        green = src_raw.read(g_idx).astype(np.float32)
        blue = src_raw.read(b_idx).astype(np.float32)
        
        valid_mask = (red > 0) & (green > 0) & (blue > 0) & (~np.isnan(red))
        
        rgb = np.stack([red, green, blue], axis=-1)
        rgb_norm = normalize_rgb(rgb, valid_mask)

    with rasterio.open(classified_raster_path) as src_cls:
        class_map = src_cls.read(1)

    # Isolate Class 1 (Solar/Built) strictly within valid predicted non-zero pixels
    solar_detected = (class_map == 1) & valid_mask
    solar_overlay = np.ma.masked_where(~solar_detected, class_map)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Panel 1: True-Color RGB Image
    axes[0].imshow(rgb_norm)
    axes[0].set_title(f"Sentinel-2 True Color ({year_label})", fontsize=12)
    axes[0].axis('off')

    # Panel 2: True-Color RGB + Cleaned Solar Overlay
    axes[1].imshow(rgb_norm)
    cmap_red = ListedColormap(['#ff0000'])
    axes[1].imshow(solar_overlay, cmap=cmap_red, alpha=0.65)
    axes[1].set_title(f"{title_prefix} Solar PV Detection ({year_label})", fontsize=12)
    axes[1].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cleaned overlay image to: {output_png_path}")


def run_unet_overlays():
    for year in ['2017', '2026']:
        raw_path = f"data/processed/features_Sentinel2_Original_{year}.tif"
        unet_path = f"data/classified_unet/lulc_map_unet_{year}.tif"
        out_path = f"docs/figures/unet_overlay_clean_{year}.png"
        
        if os.path.exists(raw_path) and os.path.exists(unet_path):
            generate_clean_overlay(raw_path, unet_path, out_path, title_prefix="U-Net", year_label=year)


if __name__ == "__main__":
    run_unet_overlays()