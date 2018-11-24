"""
Author: Juan Amari

File for the Chromosome class
"""

from typing import List, Callable

from genetic_algorithm.Gene import Gene


class Chromosome:

    def fitness_score(self, fitness_func: Callable) -> int:
        return fitness_func(self.genes)

    def __init__(self, genes: List[Gene]):
        self.genes = genes

    def __eq__(self, other):
        return isinstance(other, Chromosome) and len(other) == len(self) and all(
            [this == other for this, other in zip(self.genes, other.genes)])

    def __len__(self):
        return len(self.genes)
