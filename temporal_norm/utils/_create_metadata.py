from mne_bids import BIDSPath
import mne

import pandas as pd
import numpy as np
from temporal_norm.config import DATA_PATH, DATA_LOCAL_PATH

from pathlib import Path

from temporal_norm.utils import read_raw_bids_with_preprocessing
from joblib import Parallel, delayed

mne.set_log_level("warning")


def _create_data_per_subject(dataset_name, subj_id, all_sub, data_path, eog=False, emg=False, scaler="sample"):
    datatype = "eeg"
    suffix = "eeg"
    subject = all_sub[subj_id]
    print(f"Processing subject {subject}")
    try:
        bids_path = BIDSPath(
            datatype=datatype,
            root=data_path,
            suffix=suffix,
            task="sleep",
            subject=subject,
        )
        list_sessions = []
        for path_ in bids_path.match():
            ses = path_.session
            if ses not in list_sessions:
                list_sessions.append(ses)

        for ses in list_sessions:
            bids_path = BIDSPath(
                datatype=datatype,
                root=data_path,
                suffix=suffix,
                task="sleep",
                session=ses,
                subject=subject,
            )
            data, events = read_raw_bids_with_preprocessing(
                bids_path, scaler, eog, emg
            )

            Path(f"metadata/{dataset_name}").mkdir(parents=True, exist_ok=True)

            metadata_path = f"metadata/{dataset_name}/metadata_{dataset_name}_{subj_id}_{ses}.csv"

            Path(f"{DATA_LOCAL_PATH}/{dataset_name}/npy/subject_{subj_id}").mkdir(
                parents=True, exist_ok=True
            )
            Path(f"{DATA_LOCAL_PATH}/{dataset_name}/npy/subject_{subj_id}/session_{ses}").mkdir(
                parents=True, exist_ok=True
            )
            for sample in range(len(data)):
                path = (
                    f"{DATA_LOCAL_PATH}/{dataset_name}/npy/subject_{subj_id}/session_{ses}/X_{sample}.npy"
                )
                np.save(path, data[sample])
                metadata = [
                    {
                        "dataset_name": dataset_name,
                        "subject_id": subj_id,
                        "session": ses,
                        "y": events[sample],
                        "sample": sample,
                        "path": path,
                    }
                ]
                try:
                    df_metadata = pd.read_csv(metadata_path).drop("Unnamed: 0", axis=1)
                except FileNotFoundError:
                    df_metadata = pd.DataFrame()
                df_metadata = pd.concat((df_metadata, pd.DataFrame(metadata)))
                df_metadata.to_csv(metadata_path)

    # except ValueError:
    #     print("{} subject no valid".format(subject))

    except PermissionError:
        print("subject no valid")

    except TypeError:
        print("subject no valid")

    except FileNotFoundError:
        print("File not found")


def create_data(
    dataset_name,
    n_subjects,
    data_path,
    eog=False,
    emg=False,
    scaler="sample",
    n_jobs=1,
):
    """XXX docstring"""

    all_sub = (
        pd.read_csv(
            data_path / "participants.tsv",
            delimiter="\t",
        )["participant_id"]
        .transform(lambda x: x[4:])
        .tolist()
    )
    if n_subjects == -1:
        n_subjects = len(all_sub)
    print(n_subjects)

    # use joblib to parallelize the process
    Parallel(n_jobs=n_jobs)(
        delayed(_create_data_per_subject)(dataset_name, subj_id, all_sub, data_path, eog, emg, scaler)
        for subj_id in range(n_subjects)
    )


def create_metadata(
    n_subjects,
    dataset_name,
    eog=False,
    emg=False,
    data_path=None,
    scaler=True,
    n_jobs=1,
):
    if dataset_name == "MASS":
        if data_path is None:
            data_path = DATA_PATH
        data_path = data_path / "MASS" / "SS3" / "4channels-eeg_eog_emg"
        return create_data(dataset_name, n_subjects, data_path, eog, emg, scaler, n_jobs)

    if dataset_name == "ABC":
        if data_path is None:
            data_path = DATA_PATH
        data_path = data_path / "ABC" / "2channels"
        return create_data(dataset_name, n_subjects, data_path, eog, emg, scaler, n_jobs)

    if dataset_name == "CHAT":
        if data_path is None:
            data_path = DATA_PATH
        data_path = data_path / "CHAT" / "2channels"
        return create_data(dataset_name, n_subjects, data_path, eog, emg, scaler, n_jobs)

    if dataset_name == "CFS":
        if data_path is None:
            data_path = DATA_PATH
        data_path = data_path / "CFS" / "2channels"
        return create_data(dataset_name, n_subjects, data_path, eog, emg, scaler, n_jobs)

    if dataset_name == "HOMEPAP":
        if data_path is None:
            data_path = DATA_PATH
        data_path = data_path / "HOMEPAP" / "2channels"
        return create_data(dataset_name, n_subjects, data_path, eog, emg, scaler, n_jobs)

    if dataset_name == "CCSHS":
        if data_path is None:
            data_path = DATA_PATH
        data_path = data_path / "CCSHS" / "2channels"
        return create_data(dataset_name, n_subjects, data_path, eog, emg, scaler, n_jobs)

    if dataset_name == "SOF":
        if data_path is None:
            data_path = DATA_PATH
        data_path = data_path / "SOF" / "2channels"
        return create_data(dataset_name, n_subjects, data_path, eog, emg, scaler, n_jobs)

    if dataset_name == "MROS":
        if data_path is None:
            data_path = DATA_PATH
        data_path = data_path / "MROS" / "2channels"
        return create_data(dataset_name, n_subjects, data_path, eog, emg, scaler, n_jobs)

    if dataset_name == "PhysioNet":
        if data_path is None:
            data_path = DATA_PATH
        data_path = data_path / "Physionet" / "4channels-eeg_eog_emg"
        return create_data(dataset_name, n_subjects, data_path, eog, emg, scaler, n_jobs)

    if dataset_name == "SHHS":
        if data_path is None:
            data_path = DATA_PATH
        data_path = data_path / "SHHS" / "2channels"
        return create_data(dataset_name, n_subjects, data_path, eog, emg, scaler, n_jobs)
