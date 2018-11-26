import logging
import sys

from genetic_algorithm.BoundedKnapsackGA import BoundedKnapsackGA

logging.basicConfig(filename='log.txt', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("bounded-knapsack-ga-logger")

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)


def main():
    bk_ga = BoundedKnapsackGA()
    logger.info("### START ###")
    pop, champion, champion_fitness = bk_ga.run(
        "/Users/juanamari/Google-Drive/ITC/Curso/Assignments/Optimization/Knapsack/items.txt")

    logger.info("### END ###")
    logger.info("The last population was: {}".format(str(pop.chromosomes)))
    logger.info(
        "The champion chromosome seen accross generations was: {} with a fitness score of {}.".format(str(champion),
                                                                                                      champion_fitness))


if __name__ == '__main__':
    main()
