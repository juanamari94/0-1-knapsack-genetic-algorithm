"""
Author: Juan Amari
Class for the Bounded Knapsack Genetic Algorithm.
"""

import copy
import logging
import random
import sys
from typing import List, Callable

import numpy as np

from genetic_algorithm.Chromosome import Chromosome
from genetic_algorithm.Gene import Gene

MAX_WEIGHT = 15

logger = logging.getLogger("bounded-knapsack-ga-logger")


class BoundedKnapsackGA:
    """
    Main wrapper class for the Bounded Knapsack Genetic Algorithm.
    """
    # Thanks Python
    MAX_WEIGHT = MAX_WEIGHT
    DEFAULT_POPULATION_LIMIT = 200
    MAX_GENERATIONS = 50
    MUTATION_PROBABILITY = 0.1
    ELITE_RATIO = 0.2

    def __init__(self, max_weight=MAX_WEIGHT,
                 pop_limit=DEFAULT_POPULATION_LIMIT,
                 max_generations=MAX_GENERATIONS,
                 mutation_probability=MUTATION_PROBABILITY,
                 elite_ratio=ELITE_RATIO):
        self.max_weight = max_weight
        self.pop_limit = pop_limit
        self.max_generations = max_generations
        self.mutation_probability = mutation_probability
        self.elite_ratio = elite_ratio

    def _initialize_first_population(self, gene_pool: List[Gene]):
        """
        Helper method that creates a new population with randomly generated gene states.
        :param gene_pool: The pool of genes that we can use for each chromosome.
        :return: A population of chromosomes composed of genes.
        """
        from genetic_algorithm.Population import Population
        population = list()
        population.append(Chromosome(gene_pool))
        for _ in range(self.pop_limit):
            new_chromosome = []
            for gene in gene_pool:
                toss = bool(random.getrandbits(1))
                new_chromosome.append(Gene(gene.value, gene.weight, toss))
            population.append(Chromosome(new_chromosome))
        return Population(population, population_limit=self.pop_limit)

    def _choose_best_chromosome_across_generations(self, candidate, champion):
        """
        Helper function that is used to determine the best chromosome seen across generations.
        :param candidate: Canditate that may become the new champion gene.
        :param champion: Current champion gene.
        :return: The new champion gene, if there's one.
        """
        candidate_fitness = candidate.fitness_score(
            BoundedKnapsackGA.fitness_func(self.max_weight))
        if champion:
            champion_fitness = champion.fitness_score(
                BoundedKnapsackGA.fitness_func(self.max_weight))
            if champion_fitness < candidate_fitness:
                champion = copy.deepcopy(candidate)
                champion_fitness = candidate_fitness
        else:
            champion = candidate
            champion_fitness = candidate_fitness

        return champion, champion_fitness

    def run(self, filepath):
        """
        Runs the genetic algorithm using chromosome elitism and non-elitist crossover. The best chromosomes are preserved
        across generations unless they mutate or are displaced by better chromosomes.
        :param filepath: The path of the file containing the gene pool.
        :return: The population of the last generation along with the best chromosone seen across all iterations.
        """
        from genetic_algorithm.Population import Population
        gene_pool = BoundedKnapsackGA.load_from_file(filepath)
        logger.info("### START ###")
        logger.info("Our items are:")
        for i, gene in enumerate(gene_pool):
            logger.info("Item {} - Value: {} | Weight: {}".format(i, gene.value, gene.weight))
        current_pop = self._initialize_first_population(gene_pool)
        champion = None
        champion_fitness = None
        for i in range(self.max_generations):
            logger.info("Generation {}".format(i + 1))
            selected_pop = current_pop.selection(self.pop_limit, BoundedKnapsackGA.fitness_func)

            logger.info("Selected population: {}".format(selected_pop))
            elite_length = int(len(selected_pop.chromosomes) * self.elite_ratio)
            elite_chromosomes = selected_pop.chromosomes[:elite_length]

            generation_best_chromosome = elite_chromosomes[0]
            champion, champion_fitness = self._choose_best_chromosome_across_generations(generation_best_chromosome,
                                                                                         champion)

            logger.info("Best chromosomes in the generation: {}".format(str(elite_chromosomes)))
            logger.debug("Best chromosome seen so far is: {} with fitness: {}".format(str(champion), champion_fitness))

            non_elite_chromosomes = selected_pop.chromosomes[elite_length:]
            crossover_candidates_pop = Population(non_elite_chromosomes, self.pop_limit - elite_length, self.max_weight)
            crossover_pop = crossover_candidates_pop.crossover()

            new_chromosomes = list()
            new_chromosomes.extend(elite_chromosomes)
            new_chromosomes.extend(crossover_pop.chromosomes)
            full_pop = Population(new_chromosomes, self.pop_limit, self.max_weight)

            if i == self.max_generations - 1:
                break
            logger.info("Mutating population...")
            full_pop.mutate(self.mutation_probability)
            current_pop = full_pop
        return current_pop, champion, champion_fitness

    @staticmethod
    def load_from_file(filepath: str) -> List[Gene]:
        """
        Attemps to load the gene data from a file.
        :param filepath: The path of the file.
        :return: A list of genes generated from the file specificall for the knapsack problem.
        """
        try:
            data = np.loadtxt(filepath)
            genes = []
            for row in data:
                genes.append(Gene(row[0], row[1], True))
            return genes
        except Exception as err:
            logger.error(err)
            sys.exit(0)

    @staticmethod
    def fitness_func(max_weight: int) -> Callable[[Chromosome], int]:
        """
        Curried fitness function for the Bounded Knapsack Problem.
        :param max_weight: The maximum weight constraint used in this specific fitness function.
        :return: A function that will calculate the fitness of the chromosome with the given max_weight.
        """
        # https://www.python-course.eu/currying_in_python.php

        if not max_weight:
            max_weight = MAX_WEIGHT

        def parameterized_fitness_func(chromosome: Chromosome) -> int:
            """
            Fitness function that calculates the fitness for a given chromosome.
            :param chromosome: The given chromosome.
            :return: An integer denoting the fitness of the chromosome.
            """
            profits, weights = chromosome.calculate_active_values_and_weights()
            return profits - weights if weights > MAX_WEIGHT else profits

        return parameterized_fitness_func
