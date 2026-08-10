import os
import sys
import glob
import torch
import numpy as np
import pandas as pd
import rasterio

# Resolve module import paths from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.models.predict_unet import UNet


def predict_tiled_raster(model, image, patch_size=128, stride=64, device="cpu"):
    """
    Executes sliding-window tiled inference across full raster dimensions, 
    ensuring complete edge coverage and explicit NoData masking.
    """
    c, h, w = image.shape

    # Construct coordinate step lists ensuring exact coverage to raster edges
    y_steps = list(range(0, h - patch_size + 1, stride))
    if len(y_steps) == 0 or y_steps[-1] != h - patch_size:
        y_steps.append(max(0, h - patch_size))

    x_steps = list(range(0, w - patch_size + 1, stride))
    if len(x_steps) == 0 or x_steps[-1] != w - patch_size:
        x_steps.append(max(0, w - patch_size))

    output_probs = np.zeros((4, h, w), dtype=np.float32)
    count_matrix = np.zeros((h, w), dtype=np.float32)

    model.eval()

    for y in y_steps:
        for x in x_steps:
            patch = image[:, y:y + patch_size, x:x + patch_size].copy()

            # Sanitize input patch array
            patch = np.nan_to_num(patch, nan=0.0, posinf=1.0, neginf=-1.0)

            # Skip pure NoData background patches
            if np.all(patch == 0):
                continue

            mean = np.mean(patch, axis=(1, 2), keepdims=True)
            std = np.std(patch, axis=(1, 2), keepdims=True)
            std[std == 0] = 1e-6
            norm_patch = (patch - mean) / std

            tensor_patch = torch.tensor(norm_patch, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor_patch)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            output_probs[:, y:y + patch_size, x:x + patch_size] += probs
            count_matrix[y:y + patch_size, x:x + patch_size] += 1.0

    # Mask unvisited pixels
    visited_mask = count_matrix > 0
    count_matrix[~visited_mask] = 1.0

    avg_probs = output_probs / count_matrix
    pred_map = np.argmax(avg_probs, axis=0) + 1

    # Force unvisited or pure NoData pixels to 0
    raw_valid = np.any(image != 0, axis=0)
    pred_map[~visited_mask | ~raw_valid] = 0

    return pred_map.astype(np.int16)


def evaluate_pretrained_unet(
    model_path="models/unet_lulc_model.pth",
    processed_dir="data/processed",
    output_dir="data/classified_unet"
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pre-trained weights file not found at: {model_path}")

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_files = sorted(glob.glob(os.path.join(processed_dir, "features_*.tif")))
    if not feature_files:
        raise FileNotFoundError(f"No feature rasters found in: {processed_dir}")

    with rasterio.open(feature_files[0]) as src:
        in_channels = src.count

    print("=======================================================")
    print("   U-NET INFERENCE ENGINE (EDGE COVERAGE FIXED)        ")
    print("=======================================================")
    print(f"Loading Model Weights: {model_path}")
    print(f"Input Feature Channels: {in_channels}")
    print(f"Compute Device:         {device}\n")

    model = UNet(
        in_channels=in_channels, 
        num_classes=4, 
        features=[16, 32, 64, 128]
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))

    classified_rasters = {}

    for file_path in feature_files:
        year = os.path.basename(file_path).split("_")[-1].replace(".tif", "")
        out_raster_path = os.path.join(output_dir, f"lulc_map_unet_{year}.tif")
        
        with rasterio.open(file_path) as src:
            profile = src.profile.copy()
            image = src.read().astype(np.float32)

            pred_map = predict_tiled_raster(model, image, patch_size=128, stride=64, device=device)

            profile.update(
                count=1,
                dtype=rasterio.int16,
                nodata=0
            )

            with rasterio.open(out_raster_path, "w", **profile) as dst:
                dst.write(pred_map.reshape(1, pred_map.shape[0], pred_map.shape[1]))
                dst.set_band_description(1, f"UNet_Prediction_{year}")

            classified_rasters[year] = out_raster_path
            print(f"Generated U-Net LULC map for year {year}: {out_raster_path}")

    if '2017' in classified_rasters and '2026' in classified_rasters:
        compute_unet_metrics(classified_rasters['2017'], classified_rasters['2026'])


def compute_unet_metrics(baseline_path, target_path):
    with rasterio.open(baseline_path) as src_base, rasterio.open(target_path) as src_targ:
        base_arr = src_base.read(1)
        targ_arr = src_targ.read(1)
        
        pixel_area_ha = 0.01  # 10m x 10m Sentinel-2 pixel = 0.01 ha
        classes = {1: "Solar/Built", 2: "Vegetation", 3: "Bare Soil/Scrub", 4: "Water"}
        
        print("\n=======================================================")
        print("   U-NET PIXEL COUNT DISTRIBUTION (2017 vs 2026)       ")
        print("=======================================================")
        for cid, cname in classes.items():
            cnt_2017 = np.sum(base_arr == cid)
            cnt_2026 = np.sum(targ_arr == cid)
            print(f"Class {cid} ({cname}): 2017 = {cnt_2017} px ({round(cnt_2017*pixel_area_ha, 2)} ha) | 2026 = {cnt_2026} px ({round(cnt_2026*pixel_area_ha, 2)} ha)")

        print("\n=======================================================")
        print("   U-NET LAND COVER TRANSITION MATRIX (2017 -> 2026) HA ")
        print("=======================================================")
        
        matrix = {}
        for c1, name1 in classes.items():
            matrix[name1] = {}
            for c2, name2 in classes.items():
                pixel_count = np.sum((base_arr == c1) & (targ_arr == c2))
                area_ha = round(pixel_count * pixel_area_ha, 2)
                matrix[name1][name2] = area_ha
                
        df_matrix = pd.DataFrame(matrix).T
        print(df_matrix)
        
        output_csv = "data/processed/unet_transition_matrix_2017_2026.csv"
        df_matrix.to_csv(output_csv)
        print(f"\nU-Net transition matrix exported to: {output_csv}")


if __name__ == "__main__":
    evaluate_pretrained_unet()