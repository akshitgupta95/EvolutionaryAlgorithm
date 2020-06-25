#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import random
import numpy
import sys


from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from jmetal.core.quality_indicator import GenerationalDistance,InvertedGenerationalDistance,EpsilonIndicator,HyperVolume

MAX_ITEM = 50
MAX_WEIGHT = 520 



NGEN = 50
MU = 20
LAMBDA = 5
CXPB = 0.7
MUTPB = 0.2

filename = 'knapsack_instances/%s/%d.txt' % ("dense", 10)
data = open(filename, 'r').readlines()
# Create random items and store them in the items' dictionary.
items = {}
weights=[]

for weight in data[1:]:  # skip first line defining the graph structure
    weight = weight.strip().split(' ')
    weight = [int(e) for e in weight]
    weights.append(weight)
    print(weights)
    # weights.sort(key=lambda x: x[0], reverse=False)
    # print(weights)

for i in range(len(weights)):
    items[i] = (weights[i][0], weights[i][1],weights[i][2])
print("items")
print(items)

NBR_ITEMS = len(weights)
IND_INIT_SIZE = NBR_ITEMS

def evalKnapsack(individual):    
    weight = 0.0
    objective1 = 0.0
    objective2= 0.0

    for item in individual:
        weight += items[item][0]
    while(weight > MAX_WEIGHT):
        return 0,0
        # removed=individual.pop()
        # weight=weight-items[removed][0]
        # if(weight==0):
        #     print("error")

    # weight = 0.0
    for item in individual:
        # weight += items[item][0]
        objective1 +=items[item][1]
        objective2 +=items[item][2]
    # Ensure overweighted bags are dominated
    # return weight, objective1, objective2
    return objective1, objective2

# def evalKnapsackBalanced(individual):
#     """
#     Variant of the original weight-value knapsack problem with added third object being minimizing weight difference between items.
#     """
#     weight, value = evalKnapsack(individual)
#     balance = 0.0
#     for a,b in zip(individual, list(individual)[1:]):
#         balance += abs(items[a][0]-items[b][0])
#     if len(individual) > MAX_ITEM or weight > MAX_WEIGHT:
#         return weight, value, 1e30 # Ensure overweighted bags are dominated
#     return weight, value, balance

def cxSet(ind1, ind2):
    """Apply a crossover operation on input sets. The first child is the
    intersection of the two sets, the second child is the difference of the
    two sets.
    """
    temp = set(ind1)                # Used in order to keep type
    ind1 &= ind2                    # Intersection (inplace)
    ind2 ^= temp                    # Symmetric Difference (inplace)
    return ind1, ind2
    
def mutSet(individual):
    """Mutation that pops or add an element."""
    if random.random() < 0.5:
        if len(individual) > 0:     # We cannot pop from an empty set
            individual.remove(random.choice(sorted(tuple(individual))))
    else:
        individual.add(random.randrange(NBR_ITEMS))
    return individual,

def main(objectives=2, seed=64):
    random.seed(seed)

    # Create the item dictionary: item name is an integer, and value is 
    # a (weight, value) 2-uple.
    if objectives == 2:
        # creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0,1.0))
        creator.create("Fitness", base.Fitness, weights=( 1.0,1.0))
    
    else:
        print ("No evaluation function available for", objectives, "objectives.")
        sys.exit(-1)

        
    creator.create("Individual", set, fitness=creator.Fitness)

    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("attr_item", random.randrange, NBR_ITEMS)

    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                     toolbox.attr_item, IND_INIT_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    if objectives == 2:
        toolbox.register("evaluate", evalKnapsack)
    # elif objectives == 3:
    #     toolbox.register("evaluate", evalKnapsackBalanced)
    else:
        print ("No evaluation function available for", objectives, "objectives.")
        sys.exit(-1)
        

    toolbox.register("mate", cxSet)
    toolbox.register("mutate", mutSet)
    toolbox.register("select", tools.selSPEA2)

    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()
    
    HV = HyperVolume([1.0,1.0])
    stats = tools.Statistics(lambda ind: ind.fitness.values)
   
     HV.compute(list(zip(objective1,objective2))
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)
    
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                               halloffame=hof)
    # algorithms.eaSimple(pop, toolbox,  CXPB, MUTPB, NGEN, stats,halloffame=hof)                          

    

    # stats = {}
    # def lambda_factory(idx):
    #     return lambda ind: ind.fitness.values[idx]                

    # fitness_tags = ["Weight", "Value"]
    # for tag in fitness_tags:
    #     s = tools.Statistics( key=lambda_factory(
    #                 fitness_tags.index(tag)
    #             ))
    #     stats[tag] = s

    # mstats = tools.MultiStatistics(**stats)
    # mstats.register("avg", numpy.mean, axis=0)
    # mstats.register("std", numpy.std, axis=0)
    # mstats.register("min", numpy.min, axis=0)
    # mstats.register("max", numpy.max, axis=0)

    # ea = MOEAD(pop, toolbox, MU, CXPB, MUTPB, ngen=NGEN, stats=mstats, halloffame=hof, nr=LAMBDA)
    # pop = ea.execute()
    
    return pop, stats, hof
                 
if __name__ == "__main__":
    objectives = 2
    seed = 64
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    if len(sys.argv) > 2:
        objectives = int(sys.argv[2])

    pop,stats,hof = main(objectives)

    pop = [str(p) +" "+ str(p.fitness.values) for p in pop]
    hof = [str(h) +" "+ str(h.fitness.values) for h in hof]
    for item in hof:
        print(item)

    print ("POP:")
    print ("\n".join(pop))
    
    print ("PF:")
    print ("\n".join(hof))