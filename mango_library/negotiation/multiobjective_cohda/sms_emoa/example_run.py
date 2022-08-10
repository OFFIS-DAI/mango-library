import random

from algorithm import run
from individual import Individual


# objective function gibt Tuple von Werten zurück (für jedes Ziel ein Wert)
def obj_function(phenome):
    return sum(x ** 2 for x in phenome), sum((x - 2) ** 2 for x in phenome)


dim = 10
population_size = 2
population = []

for _ in range(population_size):
    ind = [random.random() * 5 for _ in range(dim)]
    obj = obj_function(ind)
    population.append(Individual(genome=ind, objective_values=obj))

reduced_population_size = len(population) - 1
population = run(population=population,
                 reduced_population_size=reduced_population_size,
                 objective_function=obj_function)
