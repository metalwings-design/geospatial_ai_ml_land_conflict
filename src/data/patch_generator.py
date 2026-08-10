import os
import numpy as np
import rasterio

def generate_spatial_patches(
    feature_raster_path="data/processed/features_Sentinel2_Original_2026.tif",
    mask_raster_path="data/classified/lulc_map_2026.tif",
    output_dir="data/patches",
    patch_size=150,
    stride=32
):
    """
    Slices 2D multi-spectral rasters and corresponding LULC masks 
    into overlapping square patches for deep learning training.
    """
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    with rasterio.open(feature_raster_path) as src_img, rasterio.open(mask_raster_path) as src_mask:
        image = src_img.read().astype(np.float32)  # Shape: (C, H, W)
        mask = src_mask.read(1).astype(np.int64)    # Shape: (H, W)

    c, h, w = image.shape
    patch_count = 0

    # Extract overlapping patches via sliding window
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            img_patch = image[:, y:y + patch_size, x:x + patch_size]
            mask_patch = mask[y:y + patch_size, x:x + patch_size]

            # Exclude patches containing invalid or zero values
            if np.sum(img_patch == 0) > (0.2 * img_patch.size):
                continue

            img_out_path = os.path.join(output_dir, "images", f"patch_{patch_count}.npy")
            mask_out_path = os.path.join(output_dir, "masks", f"patch_{patch_count}.npy")

            np.save(img_out_path, img_patch)
            np.save(mask_out_path, mask_patch)
            patch_count += 1

    print(f"Generated {patch_count} spatial patches ({patch_size}x{patch_size}) saved to: {output_dir}")

if __name__ == "__main__":
    generate_spatial_patches()