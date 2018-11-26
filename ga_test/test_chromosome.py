import unittest

from genetic_algorithm.BoundedKnapsackGA import BoundedKnapsackGA
from genetic_algorithm.Chromosome import Chromosome
from genetic_algorithm.Gene import Gene


class ChromosomeTest(unittest.TestCase):

    def test_equality_between_chromosomes(self):
        genes = [Gene(1, 2, True), Gene(3, 4, True)]
        first_chromosome = Chromosome(genes)
        second_chromosome = Chromosome(genes)
        self.assertTrue(first_chromosome == second_chromosome)

    def test_inequality_between_chromosomes(self):
        first_genes = [Gene(1, 2, True), Gene(3, 4, True)]
        second_genes = [Gene(3, 4, True), Gene(1, 2, True)]
        first_chromosome = Chromosome(first_genes)
        second_chromosome = Chromosome(second_genes)
        self.assertFalse(first_chromosome == second_chromosome)

    def test_fitness_func_with_reward(self):
        genes = [Gene(2, 1, True), Gene(3, 2, True), Gene(4, 3, True)]
        chromosome = Chromosome(genes)
        score = chromosome.fitness_score(BoundedKnapsackGA.fitness_func(BoundedKnapsackGA.MAX_WEIGHT))
        self.assertTrue(score > 0)

    def test_fitness_func_negative_fitness(self):
        genes = [Gene(1, 5, True), Gene(2, 8, True), Gene(3, 2, True)]
        chromosome = Chromosome(genes)
        score = chromosome.fitness_score(BoundedKnapsackGA.fitness_func(BoundedKnapsackGA.MAX_WEIGHT))
        self.assertTrue(score < 0)

    def test_fitness_positive_fitness_high_values(self):
        genes = [Gene(6, 1, True), Gene(8, 2, True), Gene(4, 3, True)]
        chromosome = Chromosome(genes)
        score = chromosome.fitness_score(BoundedKnapsackGA.fitness_func(BoundedKnapsackGA.MAX_WEIGHT))
        other_genes = [Gene(2, 1, True), Gene(3, 2, True), Gene(4, 3, True)]
        other_chromosome = Chromosome(other_genes)
        other_score = other_chromosome.fitness_score(BoundedKnapsackGA.fitness_func(BoundedKnapsackGA.MAX_WEIGHT))
        self.assertTrue(score > 0 and score > other_score)

    def test_mutation_alters_chromosome_length(self):
        genes = [Gene(6, 1, True), Gene(8, 2, True), Gene(4, 3, True)]
        expected_length = 3
        chromosome = Chromosome(genes)
        chromosome.mutate(1)
        self.assertTrue(len(chromosome) == expected_length)
