import random
from typing import Callable

import numpy as np

import utils

Individual = frozenset[int]
Population = dict[Individual, float]


def make_transition_matrix_from_similarity_matrix(similarity_matrix, mutation_rate):
    # requirements for a matrix M to be usable as a transition matrix:
    #  all cells in [0, 1]
    #  sum of each row is 1
    #  diagonals are probability of re-selecting itself (1- mutation rate)

    transition_matrix = np.array(similarity_matrix)
    np.fill_diagonal(transition_matrix, 0)  # set diagonals to 0 for now, we only care about the off-diagonal
    # sum_of_row = diagonal + rest_of_row = (1-mutationRate) + rest_of_row = 1
    # therefore, rest_of_row = 1 - 1 + mutationRate = mutationRate

    # first, make sure we get rid of crazy values (negatives and above 1)
    transition_matrix = np.array([utils.remap_0_1(row) for row in transition_matrix])
    # then set the sum
    transition_matrix = np.array([row * (mutation_rate / np.sum(row)) for row in transition_matrix])
    # then set the diagonal
    np.fill_diagonal(transition_matrix, 1 - mutation_rate)

    return transition_matrix

def make_operators(similarity_matrix: np.ndarray, individual_size: int) -> (
Callable, Callable, Callable, Callable, Callable):
    quantity_values = int(similarity_matrix.shape[0])
    all_values = list(range(quantity_values))

    def sample() -> Individual:
        return Individual(random.sample(all_values, individual_size))

    def add_random_elements_until_correct_size(individual: set) -> Individual:
        mutated = set(individual)
        while len(mutated) < individual_size:
            mutated.add(random.choice(all_values))
        return Individual(mutated)

    mutation_rate = 1 / individual_size

    def mutate_naive(individual: Individual) -> Individual:
        mutated = set()  # not frozen set because we need to mutate it

        # Each item has a probability of not being added.
        # This would not scale well if individual_size was big.
        for item in individual:
            if random.random() > mutation_rate:  # note that it's flipped from normal
                mutated.add(item)

        # re-add elements until we get to the right size
        return add_random_elements_until_correct_size(mutated)

    # precalculated for mutate_markov
    transition_matrix = make_transition_matrix_from_similarity_matrix(similarity_matrix, mutation_rate)

    def mutate_markov(individual: Individual) -> Individual:
        individual_array = np.fromiter(individual, dtype=int)
        # sparce multiplication
        sum_probabilities_per_item = np.sum(transition_matrix[individual_array, :], axis=0)
        # the sum of the probabilities is scaled so that the average quantity quantised is individual_size
        final_per_item = sum_probabilities_per_item * (individual_size / np.sum(sum_probabilities_per_item))

        # then quantise
        result = set()
        for index, prob in enumerate(final_per_item):
            if random.random() < prob:
                result.add(index)

        # if there are not enough
        if len(result) < individual_size:
            return add_random_elements_until_correct_size(result)
        elif len(result) > individual_size:
            return Individual(random.sample(list(result), k=individual_size))
        else:
            return Individual(result)


    def crossover_naive(parent_a: Individual, parent_b: Individual) -> (Individual, Individual):
        probability_of_crossover_per_gene = 0.5
        # children are guaranteed the intersection
        child_1 = set(parent_a.intersection(parent_b))
        child_2 = set(child_1)  # copy

        exclusive_to_parent_a = parent_a.difference(child_1)
        exclusive_to_parent_b = parent_b.difference(child_1)

        for from_a, from_b in zip(exclusive_to_parent_a, exclusive_to_parent_b):
            if random.random() < probability_of_crossover_per_gene:
                child_1.add(from_b)
                child_2.add(from_a)
            else:
                child_1.add(from_a)
                child_2.add(from_b)
        return Individual(child_1), Individual(child_2)

    def make_tournament_select(pool: Population, tournament_size: int) -> Callable:
        # we have to re-make the operator for every pool, because there is pre-computed information
        # this stems from the fact that you can't do random.choice(pool)
        list_of_pairs = list(pool.items())

        def tournament_select():
            items = random.sample(list_of_pairs, k=tournament_size)
            winner_key, winner_value = min(items, key=utils.second)  # MINIMISATION TASK
            return winner_key

        return tournament_select

    def truncation_select(pool: Population, how_many: int) -> Population:
        sorted_pairs: list = sorted(pool.items(), key=utils.second)
        kept_pairs = sorted_pairs[:how_many]  # MINIMISATION TASK

        return dict(kept_pairs)

    return sample, mutate_markov, crossover_naive, make_tournament_select, truncation_select
