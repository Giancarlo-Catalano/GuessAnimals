import utils
import setup

RUN_SETUP = True


def main():
    if RUN_SETUP:
        with utils.announce("Running the setup, this might take a while"):
            setup.setup()

    # list_of_animals, animal_similarity_matrix = load_word_data()

    mutation, crossover, selection = task.make_operators(similarity_matrix)
    N = 5
    winning_individual, winning_fitness = run_metaheuristic(
        objective = task.make_maxmin_distance_objective(similarity_matrix),
        mutation_operator = mutation,
        crossover = crossover,
        population_size = 100,
        budget = 10000
    )

    print(f"The ideal set of animals has fitness: {winning_fitness:.3f}")
    for animal in winning_individual:
        print(f"\t{animal}")









