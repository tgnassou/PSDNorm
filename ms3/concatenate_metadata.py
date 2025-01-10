# %%
import pandas as pd
from pathlib import Path

# %%

path = Path("metadata/")
# iter on all subdirs
metadata = pd.DataFrame()
for dataset in path.iterdir():
    if dataset.is_dir():
        for file in dataset.iterdir():
            if file.is_file():
                metadata = pd.concat([metadata, pd.read_csv(file)])
metadata.to_csv("metadata/metadata_sleep.csv", index=False)

# %%
