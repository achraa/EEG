# import genetic_algrthm as ga
# import variables as v
# from dataset import load_dataset, load_labels, convert_to_epochs, load_channels
# from features import time_series_features, hjorth_features
# from classifiers import KNN, SVM, NN

# # importing ICA filtered two times
# dataset_ica_2 = load_dataset(data_type= "ica2", test_type = "Arithmetic")
# channels = load_channels()
# labels = ga.create_labels(convert_to_epochs(dataset_ica_2_,32,v.SFREQ))

# num_generations = 10
# num_genes_in_person = 8
# num_parents_mating = 5
# num_people_in_pop = 15

# equation_inputs = [1.5,1,1] # weight for accuracy, sensitivity and specificity
# init_pop,label = ga.make_init_pop(dataset_ica_2_, channels, num_genes_in_person, num_people_in_pop)
# print(init_pop)

# pop_size = init_pop.shape[0]
# curr_pop = init_pop

# for generation in range(num_generations):
#     print(f'Generation␣number:␣{generation}')
#     # Measuring the fitness of each chromosome in the population.
#     curr_pop_fitness = ga.convert_pop_to_fitness(dataset_ica_2_, channels, curr_pop,label, num_genes_in_person)
#     fitness = ga.cal_pop_fitness(equation_inputs, curr_pop_fitness)
#     print(fitness)

#     # Selecting the best parents in the population formating.
#     parents = ga.select_mating_pool(curr_pop, fitness, num_parents_mating)

#     # Generating next generation using the crossover.
#     offspring_crossover = ga.crossover(parents, offspring_size = (pop_size - num_parents_mating, num_genes_in_person))

#     # Creating the new population based on the parents and offspring.
#     curr_pop[0:num_parents_mating,:] = parents
#     curr_pop[num_parents_mating:,:] = offspring_crossover
