import os
import glob
import numpy as np
import rasterio

def compute_spectral_indices(src_raster_path, output_raster_path):
    """
    Reads a multi-band Sentinel-2 GeoTIFF, computes spectral indices 
    (NDVI, NDBI, NDWI, BSI), and writes an augmented GeoTIFF.
    
    Band Indexing Assumption (1-based index in rasterio):
    Band 2 = Blue (B2)
    Band 3 = Green (B3)
    Band 4 = Red (B4)
    Band 8 = NIR (B8)
    Band 11 = SWIR1 (B11)
    """
    with rasterio.open(src_raster_path) as src:
        profile = src.profile.copy()
        
        # Read reflectance bands and convert to float32 to prevent division by zero
        blue = src.read(2).astype(np.float32)
        green = src.read(3).astype(np.float32)
        red = src.read(4).astype(np.float32)
        nir = src.read(8).astype(np.float32)
        swir1 = src.read(11).astype(np.float32)

        # Numerical stability epsilon
        eps = 1e-6

        # Compute Normalized Difference Indices
        ndvi = (nir - red) / (nir + red + eps)
        ndbi = (swir1 - nir) / (swir1 + nir + eps)
        ndwi = (green - nir) / (green + nir + eps)
        
        # Bare Soil Index (BSI)
        bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + eps)

        # Stack original bands with computed spectral indices
        # Reading original bands array
        original_bands = src.read()
        
        # Expand dimensions for stacking
        computed_indices = np.stack([ndvi, ndbi, ndwi, bsi], axis=0)
        
        # Combine original bands + 4 indices
        augmented_data = np.concatenate([original_bands, computed_indices], axis=0)

        # Update metadata profile for exported raster
        profile.update(
            count=augmented_data.shape[0],
            dtype=rasterio.float32
        )

        # Write output raster
        with rasterio.open(output_raster_path, 'w', **profile) as dst:
            dst.write(augmented_data.astype(np.float32))
            
            # Label band names in metadata
            band_names = list(src.descriptions) if src.descriptions[0] is not None else [f"Band_{i}" for i in range(1, src.count + 1)]
            band_names.extend(['NDVI', 'NDBI', 'NDWI', 'BSI'])
            for idx, name in enumerate(band_names, start=1):
                dst.set_band_description(idx, name)

    print(f"Processed feature engineering: {output_raster_path}")

def process_all_years(raw_dir="data/raw", processed_dir="data/processed"):
    os.makedirs(processed_dir, exist_ok=True)
    raw_files = glob.glob(os.path.join(raw_dir, "*.tif"))
    
    for file_path in raw_files:
        filename = os.path.basename(file_path)
        out_path = os.path.join(processed_dir, f"features_{filename}")
        compute_spectral_indices(file_path, out_path)

if __name__ == "__main__":
    process_all_years()


    