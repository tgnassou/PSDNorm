# %%
from ms3.utils import create_metadata
import argparse
# %%

parser = argparse.ArgumentParser(description="Create metadata for dataset")
parser.add_argument("--dataset", type=str, default="ABC")
parser.add_argument("--n_jobs", type=int, default=10)

args = parser.parse_args()

create_metadata(n_subjects=-1, dataset_name=args.dataset, n_jobs=args.n_jobs)
