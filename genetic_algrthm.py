import numpy as np
import random
import pandas as pd

# Impor yang dibutuhkan dari file-file Anda
import variables as v
from dataset import convert_to_epochs
from features import hjorth_features
from classifiers import knn

# =============================================================================
# FUNGSI-FUNGSI INTI UNTUK ALGORITMA GENETIKA
# (Ini adalah versi rapi dari kode Anda sebelumnya)
# =============================================================================


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
        epoched_data = convert_to_epochs(subset_data, n_genes, v.SFREQ)

        # Ekstrak fitur
        features = hjorth_features(epoched_data)

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

# =============================================================================
# FUNGSI UTAMA YANG AKAN DIPANGGIL DARI NOTEBOOK
# =============================================================================


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
                f"     -> Ditemukan solusi terbaik baru dengan fitness {best_fitness_overall:.2f}")
            print(f"     -> Channel: {best_channels_overall}")

        parents = select_mating_pool(curr_pop, fitness, num_parents_mating)

        offspring_crossover = crossover(parents, offspring_size=(
            pop_size - num_parents_mating, num_channels_to_select))

        curr_pop[0:num_parents_mating, :] = parents
        curr_pop[num_parents_mating:, :] = offspring_crossover

    print("\n--- Seleksi Channel Selesai ---")
    final_best_channels = best_channels_overall.tolist()
    print(f"Channel terbaik yang ditemukan: {final_best_channels}")

    return final_best_channels


# import numpy as np
# import random
# import pandas as pd

# from dataset import load_dataset, load_labels, convert_to_epochs, load_channels
# from features import hjorth_features
# from classifiers import knn  # , svm, NN
# import variables as v

# # This project is extended and a library called PyGAD is released to build the
# # genetical gorithm.
# # PyGAD documentation: https://pygad.readthedocs.io
# # Install PyGAD: pip install pygad
# # PyGAD source code at GitHub: https://github.com/ahmedfgad/GeneticAlgorithmPython


# def cal_pop_fitness(equation_inputs, pop):
#     # Calculating the fitness value of each solution in the current population.
#     # The fitness function calculates the sum of products between each input and
#     # its corresponding weight.
#     # In our case this is 4*accuracy + 1*sensitivity + 1*specificity
#     fitness = np.sum(pop * equation_inputs, axis=1)
#     return fitness


# def select_mating_pool(pop, fitness, num_parents):
#     # Selecting the best individuals in the current generation as parents for
#     # producing the offspring of the next generation.
#     parents = np.empty((num_parents, pop.shape[1]), dtype='<U5')
#     print(parents)
#     for parent_num in range(num_parents):
#         max_fitness_idx = np.where(fitness == np.max(fitness))
#         max_fitness_idx = max_fitness_idx[0][0]
#         parents[parent_num, :] = pop[max_fitness_idx, :]
#         fitness[max_fitness_idx] = -99999999999
#     return parents


# def crossover(parents, offspring_size):
#     # Produces offspring with a random combination of the twop arents’ genes
#     n_offspring = offspring_size[0]
#     n_genes_in_person = int(offspring_size[1])
#     offspring = np.empty(offspring_size, dtype='<U5')
#     # The point at which crossover takes place between two parents.
#     # Usually it is at the center.
#     crossover_point = round(n_genes_in_person/2)

#     offspring_indx = 0
#     for i in range(parents.shape[0]):
#         for j in range(i+1, parents.shape[0]):
#             gene_indx = 0
#             while gene_indx < n_genes_in_person:
#                 if gene_indx < crossover_point:
#                     rand_int = random.randint(0, 7)
#                     if parents[i][rand_int] not in offspring[offspring_indx]:
#                         offspring[offspring_indx][gene_indx] = parents[i][rand_int]
#                         gene_indx += 1
#                     else:
#                         rand_int = random.randint(0, 7)
#                         if parents[j][rand_int] not in offspring[offspring_indx]:
#                             offspring[offspring_indx][gene_indx] = parents[j][rand_int]
#                             gene_indx += 1
#             offspring_indx += 1
#     return offspring


# def make_init_pop(all_data, all_genes, num_genes_in_person, num_people):
#     # Makes a random first population
#     # Initialize empty population
#     init_pop = np.empty([num_people, num_genes_in_person], dtype='<U5')
#     person_index = 0

#     while person_index != num_people:
#         # Initialize new person
#         person = np.empty(num_genes_in_person, dtype='<U5')
#         gene_index = 0
#         while gene_index != num_genes_in_person:
#             # Gives a random index
#             index = random.randint(0, len(all_genes)-1)
#             # Checks if the gene is not already in the gene pool of the person
#             if all_genes[index] not in person:
#                 person[gene_index] = all_genes[index]
#                 gene_index += 1
#     init_pop[person_index] = person
#     person_index += 1

#     # Create labels to match the dataset
#     # Creating labels
#     subset_data = get_subset(all_data, all_genes, init_pop[0])
#     dataset = convert_to_epochs(subset_data, num_genes_in_person, v.SFREQ)
#     label = create_labels(dataset)
#     return init_pop, label


# def get_subset(data, all_genes, subset_genes):
#     # Retrieves the data that belongs to the subset of genes
#     subset_data = np.empty((120, 8, 3200))
#     n_genes = 8

#     j = 0
#     for i in range(len(all_genes)):
#         if j < (n_genes + 1) and all_genes[i] in subset_genes:
#             subset_data[:, j, :] = data[:, i, :]
#             j += 1
#     return subset_data


# def check_nan(array):
#     # Checks if there is any NaN values in array
#     # Used for debugging
#     x = np.isnan(array)
#     if True in x:
#         print('NAN in array')
#         return 0
#     print('No NAN found')


# def create_labels(dataset):
#     # Loads labels in to the correct shape
#     labels = load_labels()
#     label = pd.concat([labels['t1_math'], labels['t2_math'],
#                       labels['t3_math']]).to_numpy()
#     label = label.repeat(dataset.shape[1])
#     return label


# def convert_pop_to_fitness(all_data, all_channels, current_pop, label, n_genes):
#     # Calculates population fitness (accuracy, sensitivity, specificity)
#     data = np.empty((3000, 16))
#     new_pop_fitness = np.empty((len(current_pop), 3))
#     for i in range(len(current_pop)):
#         subset_data = get_subset(all_data, all_channels, current_pop[i])
#         dataset = convert_to_epochs(subset_data, n_genes, v.SFREQ)
#         # hjorth features perform th ebest
#         features = hjorth_features(dataset, n_genes, v.SFREQ)
#         data = features

#         # Use KNN or SVM
#         new_pop_fitness[i] = knn(data, label)
#     return new_pop_fitness


# def convert_parents_to_fitness(all_data, all_genes, parents, label, num_parents_mating, n_genes):
#     # Calculates parents’ fitness(accuracy, sensitivity, specificity)
#     new_pop_fitness = np.empty((num_parents_mating, 3))
#     data = np.empty((3000, 16))
#     for i in range(num_parents_mating):
#         subset_data = get_subset(all_data, all_genes, parents[i])
#         dataset = convert_to_epochs(subset_data, n_genes, v.SFREQ)
#         features = hjorth_features(dataset, n_genes, v.SFREQ)
#         data = features
#         print(f'Parent genes: {parents[i]} ')
#         results = knn(data, label)
#         print(results)
#         new_pop_fitness[i] = results
#     return new_pop_fitness
