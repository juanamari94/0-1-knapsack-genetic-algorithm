"""
Author: Juan Amari

File for the Chromosome class
"""


class Chromosome:

    def __init__(self, genes):
        self.genes = genes

    def fitness_score(self, fitness_func):
        return fitness_func(self.genes)

    def __eq__(self, other):
        return isinstance(other, Chromosome) and len(other) == len(self) and all(
            [this == other for this, other in zip(self.genes, other.genes)])

    def __len__(self):
        return len(self.genes)
