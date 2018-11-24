import unittest

from genetic_algorithm.Gene import Gene


class GeneTest(unittest.TestCase):

    def test_gene_mutation(self):
        gene = Gene(1, 2)
        choice_pool = [Gene(2, 3)]
        gene.mutate(choice_pool, mutation_chance=1.0)
        self.assertEqual(gene, choice_pool[0])

    def test_gene_mutation_multiple_choice_pool(self):
        gene = Gene(1, 2)
        choice_pool = [Gene(2, 3), Gene(4, 5)]
        gene.mutate(choice_pool, mutation_chance=1.0)
        self.assertTrue(gene in choice_pool)

    def test_gene_mutation_with_chance(self):
        gene = Gene(1, 2)
        choice_pool = [Gene(2, 3), Gene(4, 5)]
        gene.mutate(choice_pool, mutation_chance=0.2, mutation_seed=1)
        self.assertTrue(gene in choice_pool)

    def test_gene_non_mutation(self):
        gene = Gene(1, 2)
        expected_gene_after_mutation = Gene(1, 2)
        choice_pool = [Gene(2, 3), Gene(4, 5)]
        gene.mutate(choice_pool, mutation_chance=0.2, mutation_seed=2)
        self.assertEqual(gene, expected_gene_after_mutation)