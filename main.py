import json
import os

import numpy as np

import utils
import setup
from Metaheuristic.GeneticAlgorithm import run_genetic_algorithm
from Metaheuristic.objective import make_maxmin_distance_objective, make_clique_objective
from Metaheuristic.operators import make_operators

RUN_SETUP = False
RUN_GA_FOR_EVERY_N = False


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

    def get_optimal(individual_size: int):
        operators = make_operators(animal_similarity_matrix, individual_size)
        sampling, mutation, crossover, make_tournament_selection, truncation_selection = operators

        winning_individual, winning_fitness = run_genetic_algorithm(
            sampling_operator=sampling,
            objective=make_maxmin_distance_objective(animal_similarity_matrix),
            mutation_operator=mutation,
            crossover=crossover,
            make_tournament_selection=make_tournament_selection,
            tournament_size=30,
            truncation_selection=truncation_selection,
            population_size=1000,
            budget=100000,
            verbose=True
        )

        print(f"The ideal set of animals has fitness: {winning_fitness:.6f}")
        for index in sorted(winning_individual):
            print(f"\t{list_of_animals[index]}")

        return winning_individual, winning_fitness

    #get_optimal(individual_size=3)

    if RUN_GA_FOR_EVERY_N:
        for N in range(10, 11):
            winning_individual, winning_fitness = get_optimal(N)
            file = os.path.join("results", f"N_is{N}.json")
            with open(file, "w+") as file:
                json.dump({"N": N,
                           "winning_individual": [list_of_animals[index] for index in winning_individual],
                           "winning_fitness": winning_fitness}, file)


main()
