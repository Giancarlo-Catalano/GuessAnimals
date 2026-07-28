import random
from typing import Callable

import numpy as np

import utils

Individual = set[int]
Population = dict[Individual, float]



def make_operators(similarity_matrix: np.ndarray, individual_size: int) -> (Callable, Callable, Callable, Callable, Callable):
    quantity_values = int(similarity_matrix.shape[0])
    all_values = list(range(quantity_values))
    def sample() -> Individual:
        return set(random.sample(all_values, individual_size))


    mutation_rate = 1/individual_size
    def mutate_naive(individual: Individual) -> Individual:
        mutated = set()

        # Each item has a probability of not being added.
        # This would not scale well if individual_size was big.
        for item in individual:
            if random.random() > mutation_rate:  # note that it's flipped from normal
                mutated.add(item)

        # re-add elements until we get to the right size
        while len(mutated) < individual_size:
            mutated.add(random.choice(all_values))
        return set(mutated)

    def mutate_markov(individual: Individual) -> Individual:
        raise NotImplementedError("Use the Naive operator instead, for now")

    def crossover_naive(parent_a: Individual, parent_b: Individual) -> (Individual, Individual):
        probability_of_crossover_per_gene = 0.5
        # children are guaranteed the intersection
        child_1 = parent_a.intersection(parent_b)
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
        return child_1, child_2

    def make_tournament_select(pool: Population, tournament_size: int) -> Callable:
        # we have to re-make the operator for every pool, because there is pre-computed information
        # this stems from the fact that you can't do random.choice(pool)
        list_of_pairs = list(pool.items())

        def tournament_select():
            items = random.sample(list_of_pairs, k=tournament_size)
            winner_key, winner_value = max(items, key=utils.second)
            return winner_key
        return tournament_select

    def truncation_select(pool: Population, how_many: int) -> Population:
        sorted_pairs: list = sorted(pool.items(), key=utils.second)
        kept_pairs = sorted_pairs[:how_many] # MINIMISATION TASK

        return dict(kept_pairs)

    return sample, mutate_naive, crossover_naive, make_tournament_select, truncation_select


