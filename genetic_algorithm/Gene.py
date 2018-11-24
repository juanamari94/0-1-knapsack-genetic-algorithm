"""
Author: Juan Amari

File for the Gene class.
"""

import random

from typing import List

DEFAULT_MUTATION_CHANCE = 0.2


class Gene:

    def __init__(self, value: int, weight: int):
        self.value = value
        self.weight = weight

    def mutate(self, choice_pool: List, mutation_seed=None, mutation_chance=DEFAULT_MUTATION_CHANCE):
        rng = random.Random()
        if mutation_seed:
            rng.seed(mutation_seed)

        mutation_trial = rng.uniform(0, 1)

        if mutation_trial <= mutation_chance:
            new_gene = rng.choice(choice_pool)
            self.value = new_gene.value
            self.weight = new_gene.weight

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Gene) and self.value == o.value and self.weight == o.weight
