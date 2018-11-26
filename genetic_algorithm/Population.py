"""
Author: Juan Amari

Main file for the Population class.
"""

import logging
import random
from typing import List, Callable

from genetic_algorithm.BoundedKnapsackGA import BoundedKnapsackGA
from genetic_algorithm.Chromosome import Chromosome

logger = logging.getLogger("bounded-knapsack-ga-logger")


class Population:
    DEFAULT_MAXIMUM_SELECTION = 10
    DEFAULT_POPULATION_LIMIT = 10
    DEFAULT_MUTATION_PROBABILITY = 0.2

    def __init__(self, chromosomes: List[Chromosome],
                 population_limit: int = BoundedKnapsackGA.DEFAULT_POPULATION_LIMIT,
                 max_weight: int = BoundedKnapsackGA.MAX_WEIGHT):

        self.chromosomes = chromosomes
        self.max_weight = max_weight
        self.population_limit = population_limit

    def selection(self, maximum_selection: int = DEFAULT_MAXIMUM_SELECTION,
                  fitness_func: Callable[[int], Callable[[Chromosome], int]] = BoundedKnapsackGA.fitness_func):
        """
        Sorts the chromosomes by fitness score and returns a selection up to the given amount of the ones with the highest
        fitness score.
        :param maximum_selection: Threshold by up to which to limit the selected population.
        :param fitness_func: Function that is used to calculate the fitness of a chromosome.
        :return: A new population containing the selected chromosomes.
        """

        if maximum_selection <= 0:
            raise ValueError("Maximum Selection can't be less than 1.")

        chromosomes, fitness_scores = self.calculate_fitness_scores(fitness_func)
        logger.info("Best fitness: {} for {}".format(fitness_scores[0], chromosomes[0]))

        return Population(chromosomes[:maximum_selection],
                          population_limit=self.population_limit,
                          max_weight=self.max_weight)

    def calculate_fitness_scores(self, fitness_func: Callable[
        [int], Callable[[Chromosome], int]] = BoundedKnapsackGA.fitness_func):
        """
        Wrapper function that calculates the fitness score for each chromosome in the population and returns them
        in sorted order.
        :param fitness_func: The fitness function to use.
        :return: A tuple with the chromosomes and their fitness score sorted in descending order.
        """

        fitness_score = [chromosome.fitness_score(fitness_func(self.max_weight)) for chromosome in self.chromosomes]
        zipped_chromosomes = zip(self.chromosomes, fitness_score)
        chromosomes, fitness_scores = zip(*sorted(zipped_chromosomes, key=lambda x: x[1], reverse=True))
        return chromosomes, fitness_scores

    def crossover(self):
        """
        Performs a crossover between the chromosomes of the given population.
        :return: A new population with the newly generated chromosomes from the crossover.
        """
        new_population = []
        for i in range(self.population_limit):
            new_chromosome = self._perform_crossover()
            new_population.append(new_chromosome)

        return Population(new_population, self.population_limit, self.max_weight)

    def mutate(self, mutation_probability: float = DEFAULT_MUTATION_PROBABILITY):
        """
        Mutates genes of each chromosome given a parameter probability.
        :param mutation_probability: Probability that a gene will mutate.
        """
        for chromosome in self.chromosomes:
            chromosome.mutate(mutation_probability)

    def _perform_crossover(self, random_seed=None):
        """
        Helper function to perform crossover. Chooses randomly two chromosomes and mixes half and half of each.
        :param random_seed: An optional random seed to use.
        :return: A new chromosome resulting from the crossover of two parent chromosomes.
        """
        if not random_seed:
            rng = random.Random()
        else:
            rng = random.Random(random_seed)

        # Choose two parents randomly
        chain = []
        parents = rng.sample(self.chromosomes, 2)
        parent1 = parents[0]
        parent2 = parents[1]
        chromosome_length = len(parent1.genes) // 2
        chain.extend(parent1.genes[:chromosome_length])
        chain.extend(parent2.genes[chromosome_length:])

        # Return unique genes. This is a 0-1 knapsack after all.
        return Chromosome(list(chain))

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Population) or len(self.chromosomes) != len(o.chromosomes):
            return False

        this_sorted = sorted(self.chromosomes,
                             key=lambda chromosome: chromosome.fitness_score(
                                 BoundedKnapsackGA.fitness_func(self.max_weight)))
        other_sorted = sorted(o.chromosomes,
                              key=lambda chromosome: chromosome.fitness_score(
                                  BoundedKnapsackGA.fitness_func(self.max_weight)))

        return this_sorted == other_sorted

    def __repr__(self):
        return str(self.chromosomes)
