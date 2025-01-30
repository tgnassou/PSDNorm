import os
import glob
from lxml import etree
from pathlib import Path

import mne
from mne_bids import BIDSPath, write_raw_bids


def annot_from_xml(path):
    tree = etree.parse(path)
    onset = list()
    for user in tree.xpath("/PSGAnnotation/ScoredEvents/ScoredEvent/Start"):
        onset.append(user.text)

    duration = list()
    for user in tree.xpath("/PSGAnnotation/ScoredEvents/ScoredEvent/Duration"):
        duration.append(user.text)

    description = list()
    for user in tree.xpath(
        "/PSGAnnotation/ScoredEvents/ScoredEvent/EventConcept"
    ):
        if user.text == "Wake|0":
            user.text = "Sleep stage W"
        if user.text == "Stage 1 sleep|1":
            user.text = "Sleep stage 1"
        if user.text == "Stage 2 sleep|2":
            user.text = "Sleep stage 2"
        if user.text == "Stage 3 sleep|3":
            user.text = "Sleep stage 3"
        if user.text == "Stage 4 sleep|4":
            user.text = "Sleep stage 4"
        if user.text == "REM sleep|5":
            user.text = "Sleep stage R"

        description.append(user.text)

    annotation = mne.Annotations(onset, duration, description)
    return annotation


def save_to_bids(
    raw_path,
    annot_path,
    bids_root,
    rec,
):
    raw_files = glob.glob(raw_path + "/*.edf")
    annot_files = glob.glob(annot_path + "/*.xml")
    raw_names = [Path(raw_file).stem for raw_file in raw_files]
    annot_names = [Path(annot_file).stem[:-5] for annot_file in annot_files]
    common = list(set(raw_names) & set(annot_names))
    common.sort()

    for fileref in common:
        raw_filepath = raw_path + "/" + fileref + ".edf"
        annot_filepath = annot_path + "/" + fileref + "-nsrr.xml"
        subject = fileref[-6:]
        raw = mne.io.read_raw_edf(raw_filepath)
        raw.load_data()

        annots = annot_from_xml(annot_filepath)
        raw.set_annotations(annots, emit_warning=False)
        channels_picked = [
            'EEG(sec)',
            'ECG',
            'EMG',
            'EOG(L)',
            'EOG(R)',
            'EEG',
        ]
        raw.pick_channels(channels_picked)
        mappings = [
            {"EOG(L)": "eog"},
            {"EOG(R)": "eog"},
            {"ECG": "ecg"},
            {"EMG": "emg"},
        ]
        for mapping in mappings:
            try:
                raw.set_channel_types(mapping)
            except ValueError:
                continue
        # write BIDS
        raw.info["line_freq"] = 50

        bids_path = BIDSPath(
            subject=subject,
            session=rec,
            root=bids_root,
            task="sleep",
        )

        write_raw_bids(
            raw,
            bids_path,
            overwrite=True,
            verbose=False,
            allow_preload=True,
            format="BrainVision",
        )


bids_root = "/storage/store3/data/shhs-bids/"
raw_path = "/storage/store2/data/shhs/polysomnography/edfs/"
annot_path = (
    "/storage/store2/data/shhs/polysomnography/annotations-events-nsrr/"
)

for i, session in enumerate(os.listdir(raw_path)):
    session_path = os.path.join(raw_path, session)
    annot_session_path = os.path.join(annot_path, session)
    save_to_bids(session_path, annot_session_path, bids_root, f"{i+1}")
