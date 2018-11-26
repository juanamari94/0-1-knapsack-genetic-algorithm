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
        return fitness_func(self)

    def mutate(self, mutation_probability: float):
        for gene in self.genes:
            gene.mutate(mutation_probability)

    def calculate_active_values_and_weights(self):
        weights = 0
        values = 0
        for gene in self.genes:
            if gene.is_active:
                weights += gene.weight
                values += gene.value

        return values, weights

    def calculate_active_length(self):
        return sum([gene.is_active for gene in self.genes])

    def __eq__(self, other):
        return isinstance(other, Chromosome) and len(other) == len(self) and all(
            [this == other for this, other in zip(self.genes, other.genes)])

    def __len__(self):
        return len(self.genes)

    def __repr__(self):
        return "".join(str(int(gene.is_active)) for gene in self.genes)
