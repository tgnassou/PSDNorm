import h5py
import numpy as np
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="Dataset name")
args = parser.parse_args()

DATA_PATH = f"/raid/derivatives/{args.dataset}/npy"
OUT_PATH = f"/raid/derivatives/h5_dataset_V2/{args.dataset}.h5"

with h5py.File(OUT_PATH, "w") as f:
    for subject in tqdm(sorted(os.listdir(DATA_PATH))):
        tqdm.write(f"Processing {subject}")
        for session in sorted(os.listdir(f"{DATA_PATH}/{subject}")):
            session_path = f"{DATA_PATH}/{subject}/{session}"
            samples = sorted([
                file for file in os.listdir(session_path) if file.endswith(".npy")
            ], key=lambda x: int(x.split(".")[0][2:]))

            # Load all session samples into memory
            data = np.stack(
                [np.load(f"{session_path}/{file}") for file in samples],
                axis=0
            )

            # Store as a single large dataset per session
            f.create_dataset(
                f"{subject}/{session}",
                data=data,
                compression="gzip",
                chunks=(128, data.shape[1], data.shape[2])
            )
