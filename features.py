import numpy as np
import mne
import mne_features as mne_f
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def hjorth_features(data):
    """
    Menghitung parameter Hjorth (Activity, Mobility, Complexity) secara manual
    menggunakan NumPy. Ini lebih stabil daripada bergantung pada mne-features.

    Bentuk input data: (n_epochs, n_channels, n_times)
    """
    # Menghitung turunan pertama dan kedua dari sinyal di sepanjang sumbu waktu (axis=2)
    d_data = np.diff(data, axis=2)
    dd_data = np.diff(d_data, axis=2)

    # 1. Activity (adalah varians dari sinyal)
    activity = np.var(data, axis=2)

    # 2. Mobility (std dev dari turunan pertama / std dev dari sinyal asli)
    mobility = np.std(d_data, axis=2) / np.std(data, axis=2)

    # 3. Complexity (Mobility dari turunan pertama / Mobility dari sinyal asli)
    complexity_num = np.std(dd_data, axis=2) / np.std(d_data, axis=2)
    complexity = complexity_num / mobility

    # Gabungkan semua fitur menjadi satu array per epoch
    # Hasilnya akan menjadi (n_epochs, n_channels * 3)
    features = np.hstack([activity, mobility, complexity])

    return features


def decompose_to_bands(mne_objects_dict):
    """
    Memfilter dan memecah setiap objek MNE Raw dalam dictionary ke dalam 4 pita frekuensi.

    Args:
        mne_objects_dict (dict): Dictionary berisi objek MNE Raw. 
                                 Kunci adalah nama file/trial, value adalah objek MNE Raw.

    Returns:
        dict: Dictionary bersarang. Kunci luar adalah nama file/trial, 
              kunci dalam adalah nama band ('Theta', 'Alpha', dll.), 
              dan value adalah objek MNE Raw yang sudah terfilter.
    """
    decomposed_dict = {}
    freq_bands = {'Theta': [4, 8], 'Alpha': [
        8, 13], 'Beta': [13, 30], 'Gamma': [30, 45]}

    for name, raw_object in mne_objects_dict.items():
        bands = {}
        for band_name, (l_freq, h_freq) in freq_bands.items():
            # Salin objek agar tidak mengubah data asli
            raw_copy = raw_object.copy()
            # Terapkan band-pass filter
            raw_copy.filter(l_freq=l_freq, h_freq=h_freq,
                            fir_design='firwin', verbose=False)
            bands[band_name] = raw_copy
        decomposed_dict[name] = bands

    return decomposed_dict


def _calculate_de_windowed(decomposed_data, window_size_sec=1):
    """
    Fungsi pembantu untuk menghitung Mean DE dari objek MNE.
    Tanda underscore (_) di awal nama menandakan ini adalah fungsi internal.
    """
    fs = int(decomposed_data.info['sfreq'])
    data = decomposed_data.get_data()
    n_channels, n_samples = data.shape
    window_samples = window_size_sec * fs
    n_windows = n_samples // window_samples

    # Jika sinyal lebih pendek dari 1 jendela, hitung DE untuk keseluruhan sinyal
    if n_windows == 0:
        variances = np.var(data, axis=1)
        return 0.5 * np.log(2 * np.pi * np.e * variances)

    # Jika tidak, hitung DE per jendela dan ambil rata-ratanya
    data_reshaped = data[:, :n_windows *
                         window_samples].reshape(n_channels, n_windows, window_samples)
    variances = np.var(data_reshaped, axis=2)
    epsilon = 1e-10  # Menghindari log(0)
    de_windows = 0.5 * np.log(2 * np.pi * np.e * (variances + epsilon))
    mean_de = np.mean(de_windows, axis=1)

    return mean_de


def differential_entropy_features(decomposed_data, band_order=['Theta', 'Alpha', 'Beta', 'Gamma']):
    """
    Ekstraksi fitur Differential Entropy (DE) dengan pengurangan baseline.

    Args:
        decomposed_data (dict): Dictionary berisi objek MNE yang sudah di-bandpass filter.
                                Kunci dictionary adalah nama file, valuenya adalah dict lain
                                dengan kunci nama band frekuensi.
        band_order (list): Urutan band frekuensi yang akan digabungkan fiturnya.

    Returns:
        dict: Dictionary berisi vektor fitur final untuk setiap file eksperimen.
    """
    eeg_features_baselined = {}

    for filename_key, bands_dict in decomposed_data.items():
        # Lewati file baseline (Relax) karena hanya digunakan sebagai referensi
        if "Relax" in filename_key:
            continue

        # Cari file baseline yang sesuai
        baseline_key = "Relax"

        if baseline_key not in decomposed_data:
            # print(f"Peringatan: Baseline '{baseline_key}' tidak ditemukan untuk '{filename_key}'. File ini dilewati.")
            continue

        features_for_this_file = []
        for band_name in band_order:
            # Mengambil objek MNE
            baseline_band_obj = decomposed_data[baseline_key][band_name]
            experiment_band_obj = decomposed_data[filename_key][band_name]

            # Menghitung mean DE untuk baseline
            base_mean_de = _calculate_de_windowed(baseline_band_obj)

            # Menghitung mean DE untuk eksperimen
            exper_de = _calculate_de_windowed(experiment_band_obj)

            # Menghitung final DE (baseline reduction)
            final_de = exper_de - base_mean_de
            features_for_this_file.append(final_de)

        # Menggabungkan semua fitur dari semua band menjadi satu vektor tunggal
        final_feature_vector = np.concatenate(features_for_this_file)

        # Menyimpan vektor fitur final ke dalam dictionary
        eeg_features_baselined[filename_key] = final_feature_vector
        # print(f"✅ Selesai memproses {filename_key}. Ukuran vektor fitur: {final_feature_vector.shape}")

    return eeg_features_baselined


def create_3d_feature_maps(feature_vectors_dict, montage_path, selected_channels_list, grid_resolution=9, band_order=['Theta', 'Alpha', 'Beta', 'Gamma']):
    montage = mne.channels.read_custom_montage(montage_path)
    full_ch_positions = montage.get_positions()['ch_pos']

    # Mengambil posisi HANYA untuk channel yang terpilih, dengan urutan yang sama seperti di selected_channels_list
    points = np.array([full_ch_positions[ch][:2]
                      for ch in selected_channels_list if ch in full_ch_positions])

    # Buat grid interpolasi
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    grid_x, grid_y = np.mgrid[min(x_coords):max(x_coords):complex(grid_resolution),
                              min(y_coords):max(y_coords):complex(grid_resolution)]

    n_channels = len(points)
    n_bands = len(band_order)
    expected_feature_size = n_bands * n_channels

    feature_maps_3d = {}
    for key, vector in feature_vectors_dict.items():
        if vector.size != expected_feature_size:
            print(
                f"Warning: Ukuran vektor fitur untuk '{key}' ({vector.size}) tidak cocok dengan yang diharapkan ({expected_feature_size}). File dilewati.")
            continue

        features_by_band = vector.reshape(n_bands, n_channels)

        interpolated_bands = []
        for i in range(n_bands):
            values = features_by_band[i]
            grid_z = griddata(points, values, (grid_x, grid_y),
                              method='cubic', fill_value=np.mean(values))  # Menggunakan mean sebagai fill_value
            interpolated_bands.append(grid_z.T)

        feature_maps_3d[key] = np.stack(interpolated_bands, axis=-1)

    return feature_maps_3d


def visualize_3d_feature_map(feature_map_3d, montage_path, title="Feature Map", band_order=['Theta', 'Alpha', 'Beta', 'Gamma']):
    """
    Memvisualisasikan semua band dari satu peta fitur 3D dalam subplot 2x2.
    """
    montage = mne.channels.read_custom_montage(montage_path)
    ch_names = montage.ch_names
    ch_positions = montage.get_positions()['ch_pos']

    x_coords = [pos[0] for pos in ch_positions.values()]
    y_coords = [pos[1] for pos in ch_positions.values()]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    grid_h, grid_w, n_bands = feature_map_3d.shape

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()  # Ubah dari 2x2 grid menjadi array 1D agar mudah di-loop

    fig.suptitle(title, fontsize=16)

    for i, band_name in enumerate(band_order):
        ax = axes[i]
        feature_map_2d = feature_map_3d[:, :, i]
        im = ax.imshow(feature_map_2d, cmap='viridis',
                       origin='upper', aspect='auto')

        # Tambahkan label nama kanal
        for ch_name in ch_names:
            if ch_name in ch_positions:
                x, y = ch_positions[ch_name][:2]
                epsilon = 1e-9
                col = ((x - x_min) / (x_max - x_min + epsilon)) * (grid_w - 1)
                row = ((y_max - y) / (y_max - y_min + epsilon)) * (grid_h - 1)
                ax.text(col, row, ch_name, ha='center', va='center', color='white', fontsize=8,
                        bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f'Band {band_name}')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
