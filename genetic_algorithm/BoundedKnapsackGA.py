import random
from typing import List, Callable

import numpy as np

from genetic_algorithm.Chromosome import Chromosome
from genetic_algorithm.Gene import Gene

MAX_WEIGHT = 15


class BoundedKnapsackGA:
    # Thanks Python
    MAX_WEIGHT = MAX_WEIGHT
    DEFAULT_POPULATION_LIMIT = 10
    MAX_GENERATIONS = 50
    MUTATION_PROBABILITY = 0.2

    def __init__(self, max_weight=MAX_WEIGHT,
                 pop_limit=DEFAULT_POPULATION_LIMIT,
                 max_generations=MAX_GENERATIONS,
                 mutation_probability=MUTATION_PROBABILITY):
        self.max_weight = max_weight
        self.pop_limit = pop_limit
        self.max_generations = max_generations
        self.mutation_probability = mutation_probability

    def _initialize_first_population(self, gene_pool: List[Gene]):
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

    def run(self, filepath):
        gene_pool = BoundedKnapsackGA.load_from_file(filepath)
        current_pop = self._initialize_first_population(gene_pool)
        for i in range(self.max_generations):
            print("Generation {}".format(i + 1))
            selected_pop = current_pop.selection(self.pop_limit, BoundedKnapsackGA.fitness_func)
            print("Selected population: {}".format(selected_pop))
            crossover_pop = selected_pop.crossover()
            crossover_pop.mutate(self.mutation_probability)
            current_pop = crossover_pop
        return current_pop

    @staticmethod
    def load_from_file(filepath: str) -> List[Gene]:
        data = np.loadtxt(filepath)
        genes = []
        for row in data:
            genes.append(Gene(row[0], row[1], True))
        return genes

    @staticmethod
    def fitness_func(max_weight: int) -> Callable[[Chromosome], int]:
        # https://www.python-course.eu/currying_in_python.php

        if not max_weight:
            max_weight = MAX_WEIGHT

        def parameterized_fitness_func(chromosome: Chromosome) -> int:
            # https://www.dataminingapps.com/2017/03/solving-the-knapsack-problem-with-a-simple-genetic-algorithm/
            profits, weights = chromosome.calculate_active_values_and_weights()
            if weights >= max_weight:
                return 0
            return profits

        return parameterized_fitness_func
