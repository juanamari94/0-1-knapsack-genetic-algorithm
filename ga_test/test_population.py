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
