from typing import List, Callable

import numpy as np

from genetic_algorithm.Chromosome import Chromosome
from genetic_algorithm.Gene import Gene
from helpers.combinatorics import powerset

import copy

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

    def _initialize_first_population(self, gene_powerset: list):
        from genetic_algorithm.Population import Population
        chromosomes = []
        for genes in gene_powerset:
            if genes:
                chromosomes.append(Chromosome(genes))
        return Population(chromosomes, self.max_weight)

    def run(self, filepath):
        gene_pool = BoundedKnapsackGA.load_from_file(filepath)
        current_pop = self._initialize_first_population(list(powerset(copy.deepcopy(gene_pool))))
        for i in range(self.max_generations):
            print("Generation {}".format(i + 1))
            selected_pop = current_pop.selection(self.pop_limit, BoundedKnapsackGA.fitness_func)
            crossover_pop = selected_pop.crossover()
            crossover_pop.mutate(gene_pool, self.mutation_probability)
            current_pop = crossover_pop
            print("Current population:")
            for chromosome in current_pop.chromosomes:
                print(str(chromosome))
        return current_pop

    @staticmethod
    def load_from_file(filepath: str) -> List[Gene]:
        data = np.loadtxt(filepath)
        genes = []
        for row in data:
            genes.append(Gene(row[0], row[1]))
        return genes

    @staticmethod
    def fitness_func(max_weight: int) -> Callable[[List[Gene]], int]:
        # https://www.python-course.eu/currying_in_python.php

        if not max_weight:
            max_weight = MAX_WEIGHT

        def parameterized_fitness_func(genes: List[Gene]) -> int:
            # https://www.dataminingapps.com/2017/03/solving-the-knapsack-problem-with-a-simple-genetic-algorithm/
            profits = sum([gene.value for gene in genes])
            weights = sum([gene.weight for gene in genes])
            item_count = len(genes)
            fitness = profits * item_count
            # penalty = weights * abs((item_count * weights) - max_weight)
            penalty = abs((item_count * weights) - max_weight)
            return fitness - penalty

        return parameterized_fitness_func
