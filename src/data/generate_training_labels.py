import os
import numpy as np
import pandas as pd
import rasterio

def generate_labels_from_scl(
    raster_path="data/processed/features_Sentinel2_Original_2026.tif",
    output_csv="data/vector/train_points.csv",
    samples_per_class=300
):
    """
    Generates ground-truth training coordinates using SCL band and spectral thresholding.
    
    Target Classes:
    1: Solar PV / Infrastructure
    2: Vegetation / Agriculture
    3: Bare Soil / Scrubland
    4: Water
    """
    with rasterio.open(raster_path) as src:
        # Read SCL (Band 15 in raw output), NDVI (Band 20), NDBI (Band 21)
        # Note: Band indexing depends on total counts. Reading by description if available.
        descriptions = src.descriptions
        
        scl_idx = descriptions.index('SCL') + 1 if 'SCL' in descriptions else 15
        ndvi_idx = descriptions.index('NDVI') + 1 if 'NDVI' in descriptions else 20
        ndbi_idx = descriptions.index('NDBI') + 1 if 'NDBI' in descriptions else 21

        scl = src.read(scl_idx)
        ndvi = src.read(ndvi_idx)
        ndbi = src.read(ndbi_idx)
        
        transform = src.transform

        # Define class masks
        veg_mask = (scl == 4) & (ndvi > 0.4)           # Class 2: Vegetation
        bare_mask = (scl == 5) & (ndvi < 0.25)          # Class 3: Bare Soil/Scrub
        water_mask = (scl == 6)                         # Class 4: Water
        
        # Solar PV Mask: High NDBI, Very Low NDVI, non-water
        solar_mask = (ndbi > 0.15) & (ndvi < 0.15) & (scl != 6)  # Class 1: Solar/Built-up

        masks = {
            1: ("Solar_Infrastructure", solar_mask),
            2: ("Vegetation", veg_mask),
            3: ("Bare_Soil_Scrub", bare_mask),
            4: ("Water", water_mask)
        }

        np.random.seed(42)
        records = []

        for class_id, (class_name, mask) in masks.items():
            rows, cols = np.where(mask)
            if len(rows) == 0:
                continue
            
            # Subsample coordinates randomly
            n_samples = min(samples_per_class, len(rows))
            chosen_indices = np.random.choice(len(rows), size=n_samples, replace=False)
            
            for idx in chosen_indices:
                r, c = rows[idx], cols[idx]
                x, y = rasterio.transform.xy(transform, r, c)
                records.append({
                    'longitude': x,
                    'latitude': y,
                    'class_id': class_id,
                    'class_name': class_name
                })

        df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Generated {len(df)} training points saved to: {output_csv}")

if __name__ == "__main__":
    generate_labels_from_scl()