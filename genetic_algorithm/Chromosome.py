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
        """
        Calculates the fitness score of the chromosome given a fitness function.
        :param fitness_func: The fitness function to use.
        :return: An integer representing the calculated fitness.
        """
        return fitness_func(self)

    def mutate(self, mutation_probability: float):
        """
        Attempts to mutate each gene given the probability of mutation parameter.
        :param mutation_probability: Parameter that represents how likely is a gene to mutate.
        """
        for gene in self.genes:
            gene.mutate(mutation_probability)

    def calculate_active_values_and_weights(self):
        """
        Since each chromosome is represented by activated and non-activated genes, this method returns the sum of values
        and weights of all activates genes.
        :return: The sum of values and weights of all activates genes.
        """
        weights = 0
        values = 0
        for gene in self.genes:
            if gene.is_active:
                weights += gene.weight
                values += gene.value

        return values, weights

    def calculate_active_length(self):
        """
        :return: Returns the amount of active genes in this chromosome.
        """
        return sum([gene.is_active for gene in self.genes])

    def __eq__(self, other):
        return isinstance(other, Chromosome) and len(other) == len(self) and all(
            [this == other for this, other in zip(self.genes, other.genes)])

    def __len__(self):
        return len(self.genes)

    def __repr__(self):
        return "".join(str(int(gene.is_active)) for gene in self.genes)
