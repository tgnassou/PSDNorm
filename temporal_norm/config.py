from pathlib import Path


RESULTS_PATH = Path(__file__).parent / "." / "results"
# MODEL_PATH = Path(__file__).parent / "." / "model"

# DATA_PATH = Path("/storage/local/derivatives/")
DATA_PATH = Path("/storage/store3/derivatives/")
DATA_LOCAL_PATH = Path("/raid/derivatives/")
if not DATA_PATH.exists():
    DATA_PATH = Path(__file__).parent / "." / "data_bids/"

DATA_H5_PATH = Path("/raid/derivatives/h5_dataset/")
if not DATA_H5_PATH.exists():
    DATA_H5_PATH = "/lustre/fswork/projects/rech/chr/ujq48hj/datasets/"               
# MODEL_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
