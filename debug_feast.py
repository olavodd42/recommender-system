import pandas as pd
from feast import FeatureStore
from pathlib import Path
import os

# Setup
REPO_PATH = "./feature_repo"
store = FeatureStore(repo_path=REPO_PATH)

# Create dummy entity df
entity_df = pd.DataFrame({
    "customer_id": ["000058a12d5b43e67d225668fa1f8d618c13dc232df0cad8ffe7ad4a1091e318"],
    "article_id": ["0663713001"],
    "event_timestamp": [pd.Timestamp("2021-01-01 00:00:00")]
})

print("Entity DF:")
print(entity_df)

print("\nRetrieving features (User Stats ONLY)...")
try:
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "user_stats:avg_spend",
            "user_stats:purchase_count"
        ]
    ).to_df()
    print("\nResult:")
    print(training_df)
except Exception as e:
    print(f"\nError: {e}")
