import math
import random
from optproblems import Problem
from evoalgos.algo import SMSEMOA
from evoalgos.individual import ESIndividual


def obj_function(phenome):
    return sum(x ** 2 for x in phenome), sum((x - 2) ** 2 for x in phenome)


# num objectives: when not given, guessed from number of obj functions
# objective_values = self.objective_function(phenome)
# assert objective_values == num_objectives
# BundledObjectives: calls functions iteratively, returns values for each
# function

problem = Problem(obj_function, num_objectives=2, max_evaluations=1000,
                  name="Example")
dim = 10
popsize = 10
population = []
init_step_sizes = [0.25]

# genome wird mutiert
# population besteht aus verschiedenen Individuuen

# 1 step:
# create offspring: nimm als parent 1 objekt der population, rekombiniere mit
# anderen parents aus der Population, genome wird random aus genomen der
# parents erstellt.
# offspring wird evaluiert (phenome erstellt),
# survivor_selection: remove according to bad fitness (with parents also if
# too old),
# SMS-EMOA: HyperVolumeContributionSelection
# sort according to HV, remove numbers to remove
# num_parents: wie viele Individuals sind in einer rekombination (andere)
# strategy params: strategy parameters (mutation strengths)
#         By default, the mutation strengths are bounded to the smallest and
#         largest representable positive values of the system
# evaluate is called on class Problem (uses objective functions)


for _ in range(popsize):
    population.append(
        ESIndividual(genome=[random.random() * 5 for _ in range(dim)],
                     learning_param1=1.0 / math.sqrt(dim),
                     learning_param2=0.0,
                     strategy_params=init_step_sizes,
                     recombination_type="none",
                     num_parents=1))

ea = SMSEMOA(problem, population, popsize, num_offspring=40)
ea.run()
for individual in ea.population:
    print(individual)
