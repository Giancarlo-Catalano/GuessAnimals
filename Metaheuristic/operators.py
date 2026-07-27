from typing import Callable

import numpy as np


Individual = set[int]





def make_operators(similarity_matrix: np.ndarray) -> (Callable, Callable, Callable):
    def sample() -> Individual:
        pass


    def mutate(individual: Individual) -> Individual:
        pass

    def crossover(first: Individual, second: Individual) -> (Individual, Individual):
        pass

    def tournament_select(pool: dict[Individual, float], how_many: int) -> list[Individual]:
        pass

    def truncation_select(pool: dict[Individual, float], how_many: int) -> list[Individual]:
        pass


