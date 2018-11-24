"""
Author: Juan Amari

Main file for the Population class.
"""

from typing import List, Callable

from genetic_algorithm.BoundedKnapsackGA import BoundedKnapsackGA
from genetic_algorithm.Chromosome import Chromosome
from genetic_algorithm.Gene import Gene


class Population:
    DEFAULT_MAXIMUM_SELECTION = 10
    DEFAULT_POPULATION_LIMIT = 10

    def __init__(self, chromosomes: List[Chromosome],
                 population_limit: int = BoundedKnapsackGA.DEFAULT_POPULATION_LIMIT,
                 max_weight: int = BoundedKnapsackGA.MAX_WEIGHT):

        self.chromosomes = chromosomes
        self.max_weight = max_weight
        self.population_limit = population_limit

    def selection(self, maximum_selection: int = DEFAULT_MAXIMUM_SELECTION,
                  fitness_func: Callable[[int], Callable[[List[Gene]], int]] = BoundedKnapsackGA.fitness_func):

        if maximum_selection <= 0:
            raise ValueError("Maximum Selection can't be less than 1.")

        selected_chromosomes = sorted(self.chromosomes,
                                      key=lambda chromosome: chromosome.fitness_score(
                                          fitness_func(self.max_weight)),
                                      reverse=True)

        return Population(selected_chromosomes[:maximum_selection])

    def crossover(self, population_limit=None):
        if not population_limit:
            population_limit = self.population_limit
        pass

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
