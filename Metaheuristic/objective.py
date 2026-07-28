import numpy as np

from Metaheuristic import operators


def make_maxmin_distance_objective(similarity_matrix: np.ndarray):

    total_quantity = similarity_matrix.shape[0]
    all_items_set = set(range(total_quantity))
    def objective(individual: operators.Individual) -> float:
        array_of_present = np.fromiter(individual, dtype=int)
        array_of_missing = np.fromiter(all_items_set.difference(individual), dtype=int)

        best_distance_for_every_present = np.max(similarity_matrix[array_of_present, :][:, array_of_missing])
        return -float(np.min(best_distance_for_every_present)) # we invert it because my GA minimises

    return objective


def make_clique_objective(similarity_matrix: np.ndarray):
    def objective(individual: operators.Individual) -> float:
        array_of_present = np.fromiter(individual, dtype=int)
        return -np.average(similarity_matrix[array_of_present, :][:, array_of_present])

    return objective