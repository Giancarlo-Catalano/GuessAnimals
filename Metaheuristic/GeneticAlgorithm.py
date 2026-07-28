import random
import sys
from typing import Callable

import utils
from Metaheuristic import operators

Ind = operators.Individual

def run_GA(
        objective: Callable[[Ind], float],
        sampling_operator: Callable[[], Ind],
        mutation_operator: Callable[[Ind], Ind],
        crossover: Callable[[Ind, Ind], tuple[Ind, Ind]],
        make_tournament_selection: Callable[[operators.Population, int], Callable],
        tournament_size: int,
        truncation_selection: Callable[[operators.Population, int], operators.Population],
        population_size: int,
        budget: int,
        verbose: bool = True):

    def log(msg):
        if verbose:
            print(msg, file=sys.stderr)

    log("Starting GA")

    fitness_evaluations = 0
    population = dict() # so that we avoid duplicates, and we cache the fitness values


    consecutive_failed_generations = 0
    sampling_fails_count = 0
    while len(population) < population_size:
        if sampling_fails_count > population_size:
            log("It appears to be impossible to generate an initial population, "
                            "perhaps the search space is too small, "
                            "or the sampling operator is not implemented correctly.")
            break # Yes, the program continues with a small population
        new_ind = sampling_operator()
        if new_ind in population:
            sampling_fails_count+=1
        else:
            population[new_ind] = objective(new_ind)

    fitness_evaluations += len(population)
    log("Initial population has been generated")

    def create_new_children(tournament_selector) -> (Ind, Ind):
        probability_of_sexual_reproduction = 0.5
        parent_1, parent_2 = (tournament_selector(), tournament_selector())

        if random.random() < probability_of_sexual_reproduction:
            parent_1, parent_2 = crossover(tournament_selector(), tournament_selector())

        return mutation_operator(parent_1), mutation_operator(parent_2)

    while (fitness_evaluations < budget):
        if consecutive_failed_generations > 10:
            log("Ending generational loop early, too many failed generations.")
            break

        log(f"Creating new generation, {fitness_evaluations = }")
        # create the offspring
        offspring = dict()
        tournament_selector = make_tournament_selection(population, tournament_size)
        rejected_children_count = 0
        while len(offspring) < population_size:
            if rejected_children_count > 2*population_size: # arbitrary
                log("Could not produce enough children, continuing")
                break
            children = create_new_children(tournament_selector)
            for child in children:
                # we reject duplicate entries
                if (child in population) or (child in offspring):
                    rejected_children_count += 1
                else:
                    offspring[child] = objective(child)

        if rejected_children_count > 2*population_size:
            log(f"Rejected children count = {rejected_children_count}")
            consecutive_failed_generations += 1
        else:
            consecutive_failed_generations = 0


        fitness_evaluations += len(offspring)

        # make the offspring compete with the parents
        population = truncation_selection(population | offspring, population_size)

    return max(population.items(), key=utils.second)


