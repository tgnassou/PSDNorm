# %%
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed

# %%

def read_csv(file):
    return pd.read_csv(file)


path = Path("metadata/")
metadata = pd.DataFrame()

def process_file(file):
    try:
        return read_csv(file)
    except:
        print(f"Error reading {file}")

files = [file for dataset in path.iterdir() if dataset.is_dir() for file in dataset.iterdir() if file.is_file()]

results = Parallel(n_jobs=-1)(delayed(process_file)(file) for file in files)

metadata = pd.concat(results)
metadata.to_csv("metadata/metadata_sleep.parquet", index=False)

# %%
