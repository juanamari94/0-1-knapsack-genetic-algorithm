import logging
import sys

import click

from genetic_algorithm.BoundedKnapsackGA import BoundedKnapsackGA

logging.basicConfig(filename='log.txt', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("bounded-knapsack-ga-logger")
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)


@click.command()
@click.option('--items_filename', required=True, type=str)
@click.option('--max_weight', default=BoundedKnapsackGA.MAX_WEIGHT, help="Weight constraint for the knapsack problem.",
              type=float)
@click.option('--population_limit', default=BoundedKnapsackGA.DEFAULT_POPULATION_LIMIT,
              help="The maximum population limit.", type=int)
@click.option('--max_generations', default=BoundedKnapsackGA.MAX_GENERATIONS,
              help="The maximum amount of generations to run for.", type=int)
@click.option('--mutation_probability', default=BoundedKnapsackGA.MUTATION_PROBABILITY,
              help="The probability of a gene mutating.", type=float)
@click.option('--elite_ratio', default=BoundedKnapsackGA.ELITE_RATIO,
              help="How many best chromosomes to pick from each generation.", type=float)
def run(items_filename, max_weight, population_limit, max_generations, mutation_probability, elite_ratio):
    if max_weight <= 0 or population_limit <= 0 or max_generations <= 0 or mutation_probability <= 0 or elite_ratio <= 0:
        logger.error("No hyperparameter must be <= 0.")
        sys.exit(0)
    bk_ga = BoundedKnapsackGA(max_weight, population_limit, max_generations, mutation_probability, elite_ratio)
    pop, champion, champion_fitness = bk_ga.run(items_filename)
    logger.info("### END ###")
    logger.info("The last population was: {}".format(str(pop.chromosomes)))
    logger.debug(
        "The champion chromosome seen accross generations was: {} with a fitness score of {}.".format(str(champion),
                                                                                                      champion_fitness))


def main():
    run()


if __name__ == '__main__':
    main()
