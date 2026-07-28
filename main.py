import json

import numpy as np

import utils
import setup
from Metaheuristic.GeneticAlgorithm import run_GA
from Metaheuristic.objective import make_maxmin_distance_objective
from Metaheuristic.operators import make_operators

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

    individual_size = 5
    operators = make_operators(animal_similarity_matrix, individual_size)
    sampling, mutation, crossover, make_tournament_selection, truncation_selection = operators

    winning_individual, winning_fitness = run_GA(
        sampling_operator = sampling,
        objective = make_maxmin_distance_objective(animal_similarity_matrix),
        mutation_operator = mutation,
        crossover = crossover,
        make_tournament_selection = make_tournament_selection,
        tournament_size = 3,
        truncation_selection = truncation_selection,
        population_size = 100,
        budget = 10000,
        verbose=True
    )

    print(f"The ideal set of animals has fitness: {winning_fitness:.3f}")
    for animal in winning_individual:
        print(f"\t{animal}")



main()



