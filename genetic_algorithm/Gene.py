"""
Author: Juan Amari

File for the Gene class.
"""

import random


class Gene:

    def __init__(self, value: int, weight: int, is_active: bool):
        self.value = value
        self.weight = weight
        self.is_active = is_active

    def mutate(self, mutation_chance, mutation_seed=None):
        """
        Mutates the gene in-place if a randomly generated number is less than the mutation probability.
        :param mutation_chance: Probability to mutate.
        :param mutation_seed: A seed to pre-determine the random generated number.
        """
        rng = random.Random()
        if mutation_seed:
            rng.seed(mutation_seed)

        mutation_trial = rng.uniform(0, 1)

        if mutation_trial <= mutation_chance:
            self.is_active = not self.is_active

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Gene) \
               and self.value == o.value \
               and self.weight == o.weight \
               and self.is_active == o.is_active

    def __hash__(self):
        return hash((self.value, self.weight, self.is_active))

    def __repr__(self):
        return "Value: {} - Weight: {} - Is Active: {}".format(self.value, self.weight, self.is_active)

    def copy(self):
        return Gene(self.value, self.weight, self.is_active)
