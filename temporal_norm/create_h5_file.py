# %%
import h5py
import numpy as np
import os
import argparse
# %%
# Create an HDF5 file
# make parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="Path to the data folder")
args = parser.parse_args()

DATA_PATH = f"/raid/derivatives/{args.dataset}/npy"

# Create an HDF5 file
with h5py.File(f"/raid/derivatives/h5_dataset/{args.dataset}.h5", "w") as f:
    # get all subfolfer of DATA_PATH
    for subject in os.listdir(DATA_PATH):
        print(f"Processing {subject}")
        for session in os.listdir(f"{DATA_PATH}/{subject}"):
            for file in os.listdir(f"{DATA_PATH}/{subject}/{session}"):
                if file.endswith(".npy"):
                    name = file.split(".")[0][2:]
                    data = np.load(f"{DATA_PATH}/{subject}/{session}/{file}")
                    f.create_dataset(f"{subject}/{session}/{name}", data=data, compression="gzip")

print("Data successfully saved in hierarchical HDF5 format!")
# %%