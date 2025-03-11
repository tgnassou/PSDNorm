import numpy as np
import mne
import os
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids, write_raw_bids


bids_root = "/storage/store3/data/abc-bids/"
preproc_bids_root = "/storage/store3/derivatives/ABC/100Hz/"
datatype = "eeg"

all_sub = (
    pd.read_csv(
        bids_root + "participants.tsv",
        delimiter="\t",
        skiprows=1,
        names=["participant_id", "age", "sex", "hand", "weight", "height"],
        engine="python",
    )["participant_id"]
    .transform(lambda x: x[4:])
    .tolist()
)


def preprocess_and_save(
    bids_path,
    preproc_bids_path,
    l_freq,
    h_freq,
    sfreq,
    to_microvolt=False,
    channels_to_keep=None,
    remove_ch_ref=False,
    load_eeg_only=False,
    crop_wake_mins=30,
):
    raw = read_raw_bids(bids_path=bids_path)
    raw.load_data()

    # Preprocessing
    if to_microvolt:
        raw.apply_function(
            lambda x: x * 1e6, channel_wise=False, verbose=False
        )
    if channels_to_keep is not None:
        raw.pick_channels(channels_to_keep)
    if sfreq != raw.info["sfreq"]:
        raw.resample(sfreq=sfreq, npad="auto", verbose=False)
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    if remove_ch_ref:
        mapping = {name: name.split("-")[0] for name in raw.info["ch_names"]}
        mne.rename_channels(raw.info, mapping)
    if crop_wake_mins > 0:
        annots = raw.annotations
        # Find first and last sleep stages
        mask = [x[-1] in ["1", "2", "3", "4", "R"] for x in annots.description]
        sleep_event_inds = np.where(mask)[0]

        # Crop raw
        tmin = annots[int(sleep_event_inds[0])]["onset"] - crop_wake_mins * 60
        tmin = max(raw.times[0], tmin)
        tmax = annots[int(sleep_event_inds[-1])]["onset"] + crop_wake_mins * 60
        tmax = min(tmax, raw.times[-1])
        raw.crop(tmin=tmin, tmax=tmax)
    # Write new BIDS

    write_raw_bids(
        raw,
        preproc_bids_path,
        overwrite=True,
        verbose=False,
        allow_preload=True,
        format="BrainVision",
    )


l_freq, h_freq = None, 30
sfreq = 100
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
            bids_path, preproc_bids_path, l_freq, h_freq, sfreq
        )
