# PSDNorm: Test Time Temporal Normalization for Deep Learning on EEG Signals

This repository contains the code for the paper "Test Time Temporal Normalization for Deep Learning on EEG Signals" by Anonymised.

## Downloading the datasets

To get access to the dataset used in the paper you have to download it from the following link: [NSRR](https://sleepdata.org/)

## Preprocessing the data

To preprocess the data you have to run the following commands. This is an example for ABC dataset.
First, you transform the data from the NSRR format to the BIDS format::

```bash
python temporal_norm/dataset_preprocessing/abc_to_bids.py
```

Then you can preprocess the data by running the following command::

```bash
python temporal_norm/dataset_preprocessing/abc_preprocessing.py
```

Finally, you can select the channels you want to use by running the following command::

```bash
python temporal_norm/dataset_preprocessing/abc_2channels.py
```

## Create metadata for dataloader

To avoid to reach the disk capactiry we need to load a batch of data, we create a metadata file that contains the path to the data. To create this file you have to run the following command::

```bash
python temporal_norm/create_metadata_per_dataset.py --dataset ABC
```

It will create a metadata file and save data per windows of 30 seconds in npy format.

Then you can concatenate the metadata files by running the following command::

```bash
python temporal_norm/concatenate_metadata.py
```

## Training the model

After downloading the dataset and preprocessing the data, you can train the model by running the following command::

```bash
python temporal_norm/run_all_subjects.py
```

If you want to train over all subjects.

or 

```bash
python temporal_norm/run_percentage_subject.py --percentage 0.15
```

If you want to train over a percentage of the subjects.

## Plot the results

To get the results of the model you can run the following command::

```bash
python temporal_norm/plot_table.py
```

for the result table of the paper.

```bash
python temporal_norm/plot_scatter.py
```

for the scatter plot of the paper.

```bash
python temporal_norm/plot_percentage.py
```

for the lineplot of the paper.