import pandas as pd
import os
from pathlib import Path

# Define base paths for WSL
base_path = Path("/home/ralbe/Data/CirrMRI600+")

# Read demographic data
healthy_df = pd.read_csv(base_path / "Healthy_demographics.csv")
cirrhotic_df = pd.read_csv(base_path / "CirrMRI600+_CompleteData_age_gender_evaluation.csv")

# Add image and mask paths for healthy subjects
def get_healthy_paths(row):
    id_str = str(row['ID'])  # No padding with zeros
    t1_img_path = base_path / "Healthy_subjects/T1_W_Healthy/T1_images" / f"{id_str}.nii.gz"
    t1_mask_path = base_path / "Healthy_subjects/T1_W_Healthy/T1_masks" / f"{id_str}.nii.gz"
    t2_img_path = base_path / "Healthy_subjects/T2_W_Healthy/T2_images" / f"{id_str}.nii.gz"
    t2_mask_path = base_path / "Healthy_subjects/T2_W_Healthy/T2_masks" / f"{id_str}.nii.gz"
    return pd.Series({
        'T1_img': str(t1_img_path) if t1_img_path.exists() else None,
        'T1_mask': str(t1_mask_path) if t1_mask_path.exists() else None,
        'T2_img': str(t2_img_path) if t2_img_path.exists() else None,
        'T2_mask': str(t2_mask_path) if t2_mask_path.exists() else None,
        'group': None  # Empty group for healthy subjects
    })

# Add image and mask paths for cirrhotic subjects
def get_cirrhotic_paths(row):
    id_str = str(row['Patient ID'])  # No padding with zeros
    
    # Check all possible locations for images and masks
    paths = {
        'T1_train_img': base_path / "Cirrhosis_T1_3D/train_images" / f"{id_str}.nii.gz",
        'T1_train_mask': base_path / "Cirrhosis_T1_3D/train_masks" / f"{id_str}.nii.gz",
        'T1_valid_img': base_path / "Cirrhosis_T1_3D/valid_images" / f"{id_str}.nii.gz",
        'T1_valid_mask': base_path / "Cirrhosis_T1_3D/valid_masks" / f"{id_str}.nii.gz",
        'T1_test_img': base_path / "Cirrhosis_T1_3D/test_images" / f"{id_str}.nii.gz",
        'T1_test_mask': base_path / "Cirrhosis_T1_3D/test_masks" / f"{id_str}.nii.gz",
        'T2_train_img': base_path / "Cirrhosis_T2_3D/train_images" / f"{id_str}.nii.gz",
        'T2_train_mask': base_path / "Cirrhosis_T2_3D/train_masks" / f"{id_str}.nii.gz",
        'T2_valid_img': base_path / "Cirrhosis_T2_3D/valid_images" / f"{id_str}.nii.gz",
        'T2_valid_mask': base_path / "Cirrhosis_T2_3D/valid_masks" / f"{id_str}.nii.gz",
        'T2_test_img': base_path / "Cirrhosis_T2_3D/test_images" / f"{id_str}.nii.gz",
        'T2_test_mask': base_path / "Cirrhosis_T2_3D/test_masks" / f"{id_str}.nii.gz",
    }
    
    # Convert paths to strings and check existence
    path_dict = {k: str(v) if v.exists() else None for k, v in paths.items()}
    
    # Determine which group (train/valid/test) the subject belongs to
    group = None
    for split in ['train', 'valid', 'test']:
        if (path_dict.get(f'T1_{split}_img') is not None or 
            path_dict.get(f'T2_{split}_img') is not None):
            group = split
            break
    
    # Create the final series with the desired structure
    return pd.Series({
        'T1_img': path_dict.get('T1_train_img') or path_dict.get('T1_valid_img') or path_dict.get('T1_test_img'),
        'T1_mask': path_dict.get('T1_train_mask') or path_dict.get('T1_valid_mask') or path_dict.get('T1_test_mask'),
        'T2_img': path_dict.get('T2_train_img') or path_dict.get('T2_valid_img') or path_dict.get('T2_test_img'),
        'T2_mask': path_dict.get('T2_train_mask') or path_dict.get('T2_valid_mask') or path_dict.get('T2_test_mask'),
        'group': group
    })

# Add paths to dataframes
healthy_df = pd.concat([healthy_df, healthy_df.apply(get_healthy_paths, axis=1)], axis=1)
cirrhotic_df = pd.concat([cirrhotic_df, cirrhotic_df.apply(get_cirrhotic_paths, axis=1)], axis=1)

# Map gender codes to labels
gender_map = {1: 'Female', 2: 'Male'}
healthy_df['Gender'] = healthy_df['Gender'].map(gender_map)
cirrhotic_df['Gender'] = cirrhotic_df['Gender'].map(gender_map)

# Map radiological evaluation codes to labels
eval_map = {1: 'Mild', 2: 'Moderate', 3: 'Severe'}
cirrhotic_df['Radiological Evaluation'] = cirrhotic_df['Radiological Evaluation'].map(eval_map)

# Reorder columns for both dataframes
column_order = ['ID', 'Age', 'Gender', 'T1_img', 'T1_mask', 'T2_img', 'T2_mask', 'group']
healthy_df = healthy_df[column_order]
cirrhotic_df = cirrhotic_df[['Patient ID', 'Age', 'Gender', 'T1_img', 'T1_mask', 'T2_img', 'T2_mask', 'group', 'Radiological Evaluation']]
cirrhotic_df = cirrhotic_df.rename(columns={'Patient ID': 'ID'})

# Save the dataframes
healthy_df.to_csv('healthy_subjects_data.csv', index=False)
cirrhotic_df.to_csv('cirrhotic_subjects_data.csv', index=False)

print("Dataframes created and saved!")
print("\nHealthy subjects dataframe shape:", healthy_df.shape)
print("Cirrhotic subjects dataframe shape:", cirrhotic_df.shape)

# Display sample of each dataframe
print("\nSample of healthy subjects dataframe:")
print(healthy_df.head())
print("\nSample of cirrhotic subjects dataframe:")
print(cirrhotic_df.head()) 