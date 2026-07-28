import numpy as np

from Metaheuristic import operators


def make_maxmin_distance_objective(similarity_matrix: np.ndarray):

    total_quantity = similarity_matrix.shape[0]
    all_items_set = set(range(total_quantity))
    def objective(individual: operators.Individual) -> float:
        array_of_present = np.array(individual, dtype=int)
        array_of_missing = np.array(all_items_set.difference(individual), dtype=float)

        smallest_distance_for_every_present = np.min(similarity_matrix[array_of_present, array_of_missing], axis=1)
        return float(np.max(smallest_distance_for_every_present))

    return objective