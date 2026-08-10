import os
import glob
import pickle
import numpy as np
import pandas as pd
import rasterio

def run_lgbm_inference(
    processed_dir="data/processed",
    output_dir="data/classified_lgbm",
    model_path="models/lgbm_lulc_model.pkl"
):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
        model = artifact['model']
        feature_cols = artifact['feature_cols']

    feature_files = sorted(glob.glob(os.path.join(processed_dir, "features_*.tif")))
    classified_rasters = {}

    for file_path in feature_files:
        year = os.path.basename(file_path).split("_")[-1].replace(".tif", "")
        out_raster_path = os.path.join(output_dir, f"lulc_map_lgbm_{year}.tif")
        
        with rasterio.open(file_path) as src:
            profile = src.profile.copy()
            descriptions = [d if d else f"Band_{i+1}" for i, d in enumerate(src.descriptions)]
            
            data = src.read()
            c, h, w = data.shape
            flattened_data = data.reshape(c, -1).T
            
            df_features = pd.DataFrame(flattened_data, columns=descriptions)
            for col in feature_cols:
                if col not in df_features.columns:
                    df_features[col] = 0.0
                    
            X_infer = df_features[feature_cols].fillna(0).values
            preds = model.predict(X_infer)
            
            prediction_map = preds.reshape(1, h, w).astype(np.int16)
            profile.update(count=1, dtype=rasterio.int16, nodata=0)
            
            with rasterio.open(out_raster_path, "w", **profile) as dst:
                dst.write(prediction_map)
                dst.set_band_description(1, f"LGBM_Prediction_{year}")
                
            classified_rasters[year] = out_raster_path
            print(f"Generated LightGBM classified map for {year}: {out_raster_path}")

    if '2017' in classified_rasters and '2026' in classified_rasters:
        compute_lgbm_transition_matrix(classified_rasters['2017'], classified_rasters['2026'])

def compute_lgbm_transition_matrix(baseline_path, target_path):
    with rasterio.open(baseline_path) as src_base, rasterio.open(target_path) as src_targ:
        base_arr = src_base.read(1)
        targ_arr = src_targ.read(1)
        pixel_area_ha = 0.01

        classes = {1: "Solar/Built", 2: "Vegetation", 3: "Bare Soil/Scrub", 4: "Water"}
        
        print("\n=======================================================")
        print("   LIGHTGBM PIXEL COUNT DISTRIBUTION (2017 vs 2026)     ")
        print("=======================================================")
        for cid, cname in classes.items():
            cnt_2017 = np.sum(base_arr == cid)
            cnt_2026 = np.sum(targ_arr == cid)
            print(f"Class {cid} ({cname}): 2017 = {cnt_2017} px ({round(cnt_2017*pixel_area_ha, 2)} ha) | 2026 = {cnt_2026} px ({round(cnt_2026*pixel_area_ha, 2)} ha)")

        print("\n=======================================================")
        print("   LIGHTGBM TRANSITION MATRIX (2017 -> 2026) IN HA     ")
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
        
        output_csv = "data/processed/lgbm_transition_matrix_2017_2026.csv"
        df_matrix.to_csv(output_csv)
        print(f"\nLightGBM transition matrix exported to: {output_csv}")

if __name__ == "__main__":
    run_lgbm_inference()