# PSDNorm: Test Time Temporal Normalization for Deep Learning on EEG Signals

This repository contains the code for the paper "PSDNorm: Test Time Temporal Normalization for Deep Learning on Sleep Staging" by Anonymised.

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

To avoid to reach the disk capacity we need to load a batch of data, we create a metadata file that contains the path to the data. To create this file you have to run the following command::

```bash
python temporal_norm/create_metadata_per_subjects.py --dataset ABC
```

It will create a metadata file and save data per windows of 30 seconds in npy format.

Then you can concatenate the metadata files per dataset by running the following command::

```bash
python temporal_norm/concatenate_metadata_per_dataset.py --dataset ABC
```

And then you can concatenate all the metadata by running::

```bash
python temporal_norm/concatenate_metadata.py
```

To make the code faster when loading the data we create h5 files per dataset with::

```bash
python temporal_norm/create_h5_file.py --dataset ABC
```


## Training the model

After downloading the dataset and preprocessing the data, you can train the model by running the following command::

```bash
python temporal_norm/run_LODO.py
```

This will run the model with the Leave One Dataset Out (LODO) cross-validation. 

If you want to fasten the training you can run the model with compile option::

```bash
python temporal_norm/run_LODO.py --compile
```
This will compile the model with torch.compile and run it.

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
python temporal_norm/plot_sensitivity.py
```

for the lineplot of the paper.