import json

import numpy as np

import utils
import setup

RUN_SETUP = False


def load_word_data():
    with open(setup.LIST_OF_ANIMALS_FILE, "r") as file:
        animals = json.load(file)

    similarities = np.load(setup.ANIMAL_SIMILARITIES_FILE)

    return animals, similarities


def main():
    if RUN_SETUP:
        with utils.announce("Running the setup, this might take a while"):
            setup.setup()

    list_of_animals, animal_similarity_matrix = load_word_data()
    print(list_of_animals)
    print(animal_similarity_matrix)

    # individual_size = 5
    # sampling, mutation, crossover, selection = task.make_operators(similarity_matrix, individual_size)
    # winning_individual, winning_fitness = run_metaheuristic(
    #     objective = task.make_maxmin_distance_objective(similarity_matrix),
    #     mutation_operator = mutation,
    #     crossover = crossover,
    #     population_size = 100,
    #     budget = 10000
    # )
    #
    # print(f"The ideal set of animals has fitness: {winning_fitness:.3f}")
    # for animal in winning_individual:
    #     print(f"\t{animal}")



main()



