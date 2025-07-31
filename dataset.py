import os
import numpy as np
import pandas as pd
import scipy
import variables as v


def load_dataset(data_type="ica_filtered", test_type="Arithmetic"):
    ''''
    Load data from the SAM 40 dataset
    Args:
        data_type (string): The data type to load. Defaults to "ica_filtered".
        test_type (string): The test type to load. Defaults to "Arithmetic".
    Returns:
        ndarray: The specified dataset.
    '''
    assert (test_type in v.TEST_TYPES)
    assert (data_type in v.DATA_TYPES)

    if data_type == "ica_filtered" and test_type != "Arithmatic":
        print("Data of type", data_type, "doesn't have test type", test_type)
        return 0

    if data_type == "raw":
        dir = v.DIR_RAW
        data_key = 'Data'
    elif data_type == "wt_filtered":
        dir = v.DIR_FILTERED
        data_key = 'Clean_data'
    else:
        dir = v.DIR_ICA_FILTERED
        data_key = 'Clean_data'

    dataset = np.empty((120, 32, 3200))

    counter = 0
    for filename in os.listdir(dir):
        if test_type not in filename:
            continue

        f = os.path.join(dir, filename)
        data = scipy.io.loadmat(f)[data_key]
        dataset[counter] = data
        counter += 1
    return dataset


def load_labels():
    '''
    Load labels from the dataset and transforms the label value to binary values
    Returns:
        ndarray: The labels
    '''
    labels = pd.read_excel(v.LABELS_PATH)
    labels = labels.rename(columns=v.COLUMNS_TO_RENAME)
    labels = labels[1:]
    labels = labels.reset_index(drop=True)
    labels = labels.astype("int")
    labels = labels > 5
    return labels


def format_labels(labels, test_type="Arithmatic", epochs=1):
    '''
    Filter the labels and repeat for the specified amount of epochs.
    Args:
        labels (ndarray): The labels
        test_type (sttring): the test_type to filter by Defaults to "Arithmertic"
        epochs (int): the amount of epochs. Default to 1
    Returns: 
        ndarray: The formatted labels
    '''
    assert (test_type in v.TEST_TYPES)

    formatted_labels = []
    for trial in v.TEST_TYPE_COLUMNS[test_type]:
        formatted_labels.append(labels[trial])

    formatted_labels = pd.concat(formatted_labels).to_numpy()
    formatted_labels = formatted_labels.repeat(epochs)
    return formatted_labels


def load_channels():
    """
    Memuat nama channel dari file .locs dan mengembalikannya sebagai list.
    """
    # Membaca file .locs, menggunakan spasi sebagai pemisah, dan tanpa header
    df = pd.read_csv(v.CHANNELS_PATH, sep='\s+', header=None)

    # Nama channel ada di kolom ke-4 (indeks 3)
    # Kita ambil kolom tersebut dan ubah menjadi list
    channel_names = df[3].tolist()

    return channel_names


def convert_to_epochs(data, n_genes, sfreq):
    '''
    Convert raw data to epochs
    Args:
        data (ndarray): The raw data in shape [trial, channel, time]
        n_genes (int): The number of genes to extract
        sfreq (int): The sampling frequency
    Returns:
        ndarray: The data in shape [epoch, channel, time_per_epoch]
    '''
    n_trials = data.shape[0]
    n_channels = data.shape[1]
    n_samples = data.shape[2]

    # Calculate the number of samples per epoch
    samples_per_epoch = int(sfreq * 2)  # 2 seconds per epoch

    # Calculate the number of epochs
    n_epochs = n_samples // samples_per_epoch

    # Reshape the data to epochs
    epochs = np.empty((n_trials * n_epochs, n_channels, samples_per_epoch))

    for trial in range(n_trials):
        for epoch in range(n_epochs):
            start = epoch * samples_per_epoch
            end = start + samples_per_epoch
            epochs[trial * n_epochs + epoch] = data[trial, :, start:end]

    return epochs
