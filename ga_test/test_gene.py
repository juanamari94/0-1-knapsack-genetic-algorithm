import unittest

from genetic_algorithm.Gene import Gene


class GeneTest(unittest.TestCase):

    def test_gene_mutation(self):
        gene = Gene(1, 2, False)
        gene.mutate(mutation_chance=1.0)
        self.assertTrue(gene.is_active)

    def test_gene_mutation_multiple_choice_pool(self):
        gene = Gene(1, 2, True)
        gene.mutate(mutation_chance=0.0)
        self.assertTrue(gene.is_active)

    def test_gene_mutation_with_chance(self):
        gene = Gene(1, 2, True)
        gene.mutate(mutation_chance=0.2, mutation_seed=1)
        self.assertFalse(gene.is_active)

    def test_gene_non_mutation(self):
        gene = Gene(1, 2, True)
        gene.mutate(mutation_chance=0.2, mutation_seed=2)
        self.assertTrue(gene.is_active)
