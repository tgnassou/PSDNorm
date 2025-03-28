# %%
import pandas as pd
from pathlib import Path
# %%

path = Path(f"metadata/metadata_per_dataset")

results = []
for file in path.iterdir():
    print(file)
    metadata = pd.read_parquet(file)
    results.append(metadata)
# %%
metadata = pd.concat(results).drop(columns=["Unnamed: 0"])
metadata["session"] = metadata["session"].astype(str)
metadata["sub+session"] = metadata.apply(lambda x: f"{x['subject_id']}_{x['session']}", axis=1)

# %%
metadata.to_parquet("metadata/metadata_sleep.parquet", index=False)

# %%
