import unittest

from genetic_algorithm.Chromosome import Chromosome
from genetic_algorithm.Gene import Gene
from genetic_algorithm.Population import Population


class PopulationTest(unittest.TestCase):

    def test_selection(self):
        chromosome2 = Chromosome([Gene(2, 1), Gene(3, 1), Gene(4, 1)])
        chromosome1 = Chromosome([Gene(1, 2), Gene(3, 4), Gene(5, 6)])
        chromosome3 = Chromosome([Gene(5, 1), Gene(8, 2), Gene(3, 1)])
        pop = Population([chromosome1, chromosome2, chromosome3])
        self.assertEqual(pop, pop.selection())

    def test_selection_with_less_chromosomes(self):
        chromosome1 = Chromosome([Gene(2, 1), Gene(3, 1), Gene(4, 1)])
        chromosome2 = Chromosome([Gene(1, 2), Gene(3, 4), Gene(5, 6)])
        chromosome3 = Chromosome([Gene(5, 1), Gene(8, 2), Gene(3, 1)])
        pop = Population([chromosome1, chromosome2, chromosome3])
        expected_pop = Population([chromosome3, chromosome1])
        self.assertEqual(expected_pop, pop.selection(maximum_selection=2))

    def test_selection_with_different_fitness_func(self):
        chromosome1 = Chromosome([Gene(2, 1), Gene(3, 1), Gene(4, 1)])
        chromosome2 = Chromosome([Gene(1, 2), Gene(3, 4), Gene(5, 6)])
        chromosome3 = Chromosome([Gene(5, 1), Gene(8, 2), Gene(3, 1)])
        pop = Population([chromosome1, chromosome2, chromosome3], max_weight=9)
        expected_pop = Population([chromosome3, chromosome1], max_weight=9)
        self.assertEqual(expected_pop, pop.selection(maximum_selection=2))

    def test_perform_crossover(self):
        chromosome1 = Chromosome([Gene(2, 1), Gene(3, 1), Gene(4, 1)])
        chromosome2 = Chromosome([Gene(1, 2), Gene(3, 4), Gene(5, 6)])
        pop = Population([chromosome1, chromosome2])
        crossover_pop = pop._perform_crossover()
        expected_len = 4
        self.assertTrue(len(crossover_pop) == expected_len)

    def test_perform_crossover_with_seed(self):
        chromosome1 = Chromosome([Gene(2, 1), Gene(3, 1), Gene(4, 1)])
        chromosome2 = Chromosome([Gene(1, 2), Gene(3, 4), Gene(5, 6)])
        pop = Population([chromosome1, chromosome2])
        crossover_chromosome = pop._perform_crossover(1)
        expected_crossover_chromosome = Chromosome([Gene(5, 6), Gene(3, 1), Gene(3, 4), Gene(2, 1)])
        self.assertEqual(crossover_chromosome, expected_crossover_chromosome)

    def test_crossover_returns_same_population_size(self):
        population_limit = 10
        chromosome1 = Chromosome([Gene(2, 1), Gene(3, 1), Gene(4, 1)])
        chromosome2 = Chromosome([Gene(1, 2), Gene(3, 4), Gene(5, 6)])
        chromosome3 = Chromosome([Gene(5, 1), Gene(8, 2), Gene(3, 1)])
        pop = Population([chromosome1, chromosome2, chromosome3], population_limit=population_limit)
        new_pop = pop.crossover()
        expected_pop_size = population_limit
        for chrom in new_pop.chromosomes:
            self.assertTrue(len(chrom.genes) > 0)
        self.assertTrue(len(new_pop.chromosomes) == expected_pop_size)

    def test_crossover_does_not_return_empty_chromosomes(self):
        population_limit = 10
        chromosome1 = Chromosome([Gene(2, 1), Gene(3, 1), Gene(4, 1)])
        chromosome2 = Chromosome([Gene(1, 2), Gene(3, 4), Gene(5, 6)])
        chromosome3 = Chromosome([Gene(5, 1), Gene(8, 2), Gene(3, 1)])
        pop = Population([chromosome1, chromosome2, chromosome3], population_limit=population_limit)
        new_pop = pop.crossover()
        for chrom in new_pop.chromosomes:
            self.assertTrue(len(chrom.genes) > 0)

    def test_mutation_does_not_alter_chromosome_length(self):
        population_limit = 10
        chromosome1 = Chromosome([Gene(2, 1), Gene(3, 1), Gene(4, 1)])
        chromosome2 = Chromosome([Gene(1, 2), Gene(3, 4), Gene(5, 6)])
        chromosome3 = Chromosome([Gene(5, 1), Gene(8, 2), Gene(3, 1)])
        expected_chromosome_length = 3
        gene_pool = list(set(chromosome1.genes).union(set(chromosome2.genes)).union(set(chromosome3.genes)))
        pop = Population([chromosome1, chromosome2, chromosome3], population_limit=population_limit)
        pop.mutate(gene_pool, 0.2)
        for chromosome in pop.chromosomes:
            self.assertTrue(len(chromosome.genes), expected_chromosome_length)
