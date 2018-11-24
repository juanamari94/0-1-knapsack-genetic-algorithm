class BoundedKnapsackGA:
    MAX_WEIGHT = 15

    @staticmethod
    def fitness_func(genes, max_weight=MAX_WEIGHT):
        # https://www.dataminingapps.com/2017/03/solving-the-knapsack-problem-with-a-simple-genetic-algorithm/
        profits = sum([gene.value for gene in genes])
        weights = sum([gene.weight for gene in genes])
        item_count = len(genes)
        fitness = profits * item_count
        # penalty = weights * abs((item_count * weights) - max_weight)
        penalty = abs((item_count * weights) - max_weight)
        return fitness - penalty
