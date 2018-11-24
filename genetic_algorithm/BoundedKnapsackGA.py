from typing import List, Callable

from genetic_algorithm.Gene import Gene

MAX_WEIGHT = 15


class BoundedKnapsackGA:
    # Thanks Python
    MAX_WEIGHT = MAX_WEIGHT

    @staticmethod
    def fitness_func(max_weight: int) -> Callable[[List[Gene]], int]:
        # https://www.python-course.eu/currying_in_python.php

        if not max_weight:
            max_weight = MAX_WEIGHT

        def parameterized_fitness_func(genes: List[Gene]) -> int:
            # https://www.dataminingapps.com/2017/03/solving-the-knapsack-problem-with-a-simple-genetic-algorithm/
            profits = sum([gene.value for gene in genes])
            weights = sum([gene.weight for gene in genes])
            item_count = len(genes)
            fitness = profits * item_count
            # penalty = weights * abs((item_count * weights) - max_weight)
            penalty = abs((item_count * weights) - max_weight)
            return fitness - penalty

        return parameterized_fitness_func
