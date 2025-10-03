import numpy as np
import random
import pandas as pd

import variables as v
from dataset import convert_to_epochs
from feature import hjorth_features
from classifiers import knn


def cal_pop_fitness(equation_inputs, pop_performance):
    """Menghitung skor fitness dari array performa [akurasi, sensitivitas, spesifisitas]."""
    # equation_inputs adalah bobot, contoh: [1.5, 1, 1]
    fitness = np.sum(pop_performance * equation_inputs, axis=1)
    return fitness


def select_mating_pool(population, fitness, num_parents):
    """Memilih individu terbaik untuk menjadi orang tua."""
    parents = np.empty((num_parents, population.shape[1]), dtype='<U5')
    for parent_num in range(num_parents):
        max_fitness_idx = np.argmax(fitness)
        parents[parent_num, :] = population[max_fitness_idx, :]
        # Pastikan individu ini tidak terpilih lagi
        fitness[max_fitness_idx] = -99999999999
    return parents


def crossover(parents, offspring_size):
    """Menciptakan generasi baru (offspring) dari para orang tua."""
    offspring = np.empty(offspring_size, dtype='<U5')
    # Titik pindah silang gen (misal, 4 gen pertama)
    crossover_point = offspring_size[1] // 2

    for i in range(offspring_size[0]):
        # Pilih dua orang tua secara acak untuk setiap anak
        parent1_idx = i % parents.shape[0]
        parent2_idx = (i + 1) % parents.shape[0]

        # Ambil gen dari orang tua pertama
        offspring[i, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]

        # Ambil gen dari orang tua kedua, pastikan tidak ada duplikat
        parent2_genes = list(parents[parent2_idx, :])
        genes_from_parent2 = []
        for gene in parent2_genes:
            if gene not in offspring[i, 0:crossover_point]:
                genes_from_parent2.append(gene)

        offspring[i, crossover_point:] = genes_from_parent2[:offspring_size[1] - crossover_point]

    return offspring


def get_subset(data, all_genes, subset_genes):
    """Mengambil data hanya untuk subset channel yang dipilih."""
    # Cari indeks dari channel yang dipilih di daftar semua channel
    indices = [all_genes.index(gene) for gene in subset_genes]
    return data[:, indices, :]


def convert_pop_to_fitness(all_data, all_channels, current_pop, label, n_genes):
    """Mengevaluasi setiap individu (kombinasi channel) dalam populasi."""
    new_pop_fitness = np.empty((len(current_pop), 3))
    for i, individual_channels in enumerate(current_pop):
        # Ambil data hanya untuk channel-channel di individu ini
        subset_data = get_subset(
            all_data, all_channels, individual_channels.tolist())

        # Ubah ke format epoch
        # epoched_data = convert_to_epochs(subset_data, n_genes, v.SFREQ)

        # Ekstrak fitur
        features = hjorth_features(subset_data)

        # Hitung performa dengan classifier cepat
        performance = knn(features, label)
        new_pop_fitness[i] = performance
    return new_pop_fitness


def make_init_pop(all_genes, num_genes_in_person, num_people):
    """Membuat populasi awal secara acak tanpa duplikasi channel."""
    init_pop = np.empty([num_people, num_genes_in_person], dtype='<U5')
    for i in range(num_people):
        # Ambil sampel acak dari semua gen tanpa penggantian
        init_pop[i] = np.random.choice(
            all_genes, size=num_genes_in_person, replace=False)
    return init_pop


def run_channel_selection(full_dataset, all_channels_list, labels_for_ga, ga_params):
    """
    Menjalankan keseluruhan proses Algoritma Genetika untuk seleksi channel.
    """
    # Unpack parameter dari dictionary
    num_generations = ga_params['num_generations']
    num_parents_mating = ga_params['num_parents_mating']
    num_people_in_pop = ga_params['num_people_in_pop']
    num_channels_to_select = ga_params['num_channels_to_select']

    # Buat populasi awal
    curr_pop = make_init_pop(
        all_channels_list, num_channels_to_select, num_people_in_pop)
    pop_size = curr_pop.shape[0]

    best_fitness_overall = -1
    best_channels_overall = None

    for generation in range(num_generations):
        print(f"  -> Generasi {generation + 1}/{num_generations}...")

        curr_pop_fitness_scores = convert_pop_to_fitness(
            full_dataset, all_channels_list, curr_pop, labels_for_ga, num_channels_to_select)

        # Bobot [akurasi, sensitivitas, spesifisitas]
        equation_inputs = [1.5, 1, 1]
        fitness = cal_pop_fitness(equation_inputs, curr_pop_fitness_scores)

        best_fitness_in_gen = np.max(fitness)
        if best_fitness_in_gen > best_fitness_overall:
            best_fitness_overall = best_fitness_in_gen
            best_idx = np.argmax(fitness)
            best_channels_overall = curr_pop[best_idx]
            print(
                f"-> Ditemukan solusi terbaik baru dengan fitness {best_fitness_overall:.2f}")
            print(f"-> Channel: {best_channels_overall}")

        parents = select_mating_pool(curr_pop, fitness, num_parents_mating)

        offspring_crossover = crossover(parents, offspring_size=(
            pop_size - num_parents_mating, num_channels_to_select))

        curr_pop[0:num_parents_mating, :] = parents
        curr_pop[num_parents_mating:, :] = offspring_crossover

    print("\n--- Seleksi Channel Selesai ---")
    final_best_channels = best_channels_overall.tolist()
    print(f"Channel terbaik yang ditemukan: {final_best_channels}")

    return final_best_channels
