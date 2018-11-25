"""
Author: Juan Amari

File for the Chromosome class
"""

from typing import List, Callable

from genetic_algorithm.Gene import Gene


class Chromosome:

    def __init__(self, genes: List[Gene]):
        self.genes = genes

    def fitness_score(self, fitness_func: Callable) -> int:
        return fitness_func(self.genes)

    def mutate(self, gene_pool: List[Gene], mutation_probability: float):
        for gene in self.genes:
            gene.mutate(gene_pool, mutation_probability)

    def __eq__(self, other):
        return isinstance(other, Chromosome) and len(other) == len(self) and all(
            [this == other for this, other in zip(self.genes, other.genes)])

    def __len__(self):
        return len(self.genes)

    def __repr__(self):
        return "; ".join(str(gene) for gene in self.genes)
