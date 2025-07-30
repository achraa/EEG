import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report


def plot_conf_matrix_and_stats(conf_matrix, class_names=['Non-Stressed', 'Stressed']):
    """
    Memvisualisasikan confusion matrix menggunakan seaborn
    dan menampilkan statistik performa (Akurasi, Sensitivitas, Spesifisitas).
    """
    # Ekstrak nilai True Positive, True Negative, False Positive, False Negative
    TN, FP, FN, TP = conf_matrix.ravel()

    # Hitung metrik
    accuracy = (TP + TN) / (TP + TN + FP +
                            FN) if (TP + TN + FP + FN) > 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0  # Recall
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    # Siapkan teks statistik
    stats_text = (f"Accuracy: {accuracy:.2%}\n"
                  f"Sensitivity (Recall): {sensitivity:.2%}\n"
                  f"Specificity: {specificity:.2%}")

    # Visualisasi Confusion Matrix
    plt.figure(figsize=(7, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix\n\n{stats_text}')
    plt.show()

    print("\nLaporan Klasifikasi Rinci:")
    # Buat y_true dan y_pred dummy untuk menghasilkan laporan lengkap
    y_true = np.concatenate([np.zeros(TN + FP), np.ones(TP + FN)])
    y_pred = np.concatenate(
        [np.zeros(TN), np.ones(FP), np.zeros(FN), np.ones(TP)])
    print(classification_report(y_true, y_pred,
          target_names=class_names, zero_division=0))
