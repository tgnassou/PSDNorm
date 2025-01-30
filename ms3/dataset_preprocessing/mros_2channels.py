import os
import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids, write_raw_bids

bids_root = "/storage/store3/derivatives/MROS/100Hz/"
preproc_bids_root = "/storage/store3/derivatives/MROS/2channels/"
datatype = "eeg"

all_sub = (
    pd.read_csv(
        bids_root + "participants.tsv", delimiter="\t", engine="python"
    )[
        "participant_id"
    ]
    .transform(lambda x: x[4:])
    .tolist()
)


def preprocess_and_save(bids_path, preproc_bids_path):
    raw = read_raw_bids(bids_path=bids_path)
    try:
        raw.pick_channels(
            ["C3", "C4", "A1", "A2"],
            ordered=True,
        )
        sfreq = raw.info["sfreq"]
        linefreq = raw.info["line_freq"]

        data, times = raw[:]
        C3_A2 = data[0] - data[3]
        C4_A1 = data[1] - data[2]

        info_C3_A2 = mne.create_info(["C3-A2"], sfreq=sfreq, ch_types=datatype)
        info_C3_A2["line_freq"] = linefreq
        raw_C3_A2 = mne.io.RawArray(C3_A2[np.newaxis, :], info_C3_A2)

        info_C4_A1 = mne.create_info(["C4-A1"], sfreq=sfreq, ch_types=datatype)
        info_C4_A1["line_freq"] = linefreq
        raw_C4_A1 = mne.io.RawArray(C4_A1[np.newaxis, :], info_C4_A1)

        raw_final = raw_C3_A2.copy().add_channels(
            [raw_C4_A1]
        )
        raw_final.set_meas_date(raw.info["meas_date"])
        raw_final.set_annotations(raw.annotations)

        # Write new BIDS
        # Set output path

        write_raw_bids(
            raw_final,
            preproc_bids_path,
            overwrite=True,
            allow_preload=True,
            format="BrainVision",
        )
    except ValueError:
        print(f"Subject {bids_path.subject} has no A1/A2 channels")


for subject_id in all_sub:
    bids_path = BIDSPath(
        subject=subject_id, root=bids_root, datatype=datatype, task="sleep"
    )
    sessions = os.listdir(f"{bids_path.root}/sub-{bids_path.subject}")
    for session in sessions:
        if not session.startswith("ses-"):
            continue
        bids_path.update(session=session[4:])
        preproc_bids_path = bids_path.copy().update(root=preproc_bids_root)
        preprocess_and_save(
            bids_path, preproc_bids_path,
        )
