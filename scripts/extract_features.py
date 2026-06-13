"""
Extract the exact 31 feature names used to train the best model
and save them to best_model_meta.json so the ML service can align features correctly.
"""
import pandas as pd
import numpy as np
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ml_pipeline.preprocessing import load_and_prepare_all_data
from ml_pipeline.feature_engineering import engineer_features

# Load the data and run through the full preprocessing + feature engineering pipeline
df, preprocessor = load_and_prepare_all_data()
df = engineer_features(df)

# Get feature columns (all except target)
feature_cols = [c for c in df.columns if c != 'attrition']
print(f"Total features: {len(feature_cols)}")
for i, col in enumerate(feature_cols):
    print(f"  {i}: {col}")

# Update metadata
meta_path = os.path.join('ml_pipeline', 'models', 'best_model_meta.json')
with open(meta_path, 'r') as f:
    meta = json.load(f)

meta['feature_names'] = feature_cols
meta['n_features'] = len(feature_cols)

with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved {len(feature_cols)} feature names to {meta_path}")
