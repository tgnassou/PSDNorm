# %%
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
import argparse
# %%
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
args = parser.parse_args()
dataset = args.dataset


def read_csv(file):
    return pd.read_csv(file)


path = Path(f"metadata/{dataset}")
metadata = pd.DataFrame()


def process_file(file):
    try:
        return read_csv(file)
    except:
        print(f"Error reading {file}")


files = [file for file in path.iterdir() if file.is_file()]

results = Parallel(n_jobs=-1)(delayed(process_file)(file) for file in files)

metadata = pd.concat(results)
metadata.to_parquet(f"metadata/metadata_sleep_{dataset}.parquet", index=False)

# %%
