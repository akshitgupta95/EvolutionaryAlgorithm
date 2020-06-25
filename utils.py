import numpy as np


def comparator(solution1, solution2):
    '''
    Comparator based on solutions rank and crowding distance
    '''

    if solution1.rank < solution2.rank:
        return 1
    if solution1.rank == solution2.rank and solution1.crowdingDistance > solution2.crowdingDistance:
        return 1

    return -1

def calculateHypervolume(solutions):
    '''
    Function for calculating hypervolume
    Input: list of Individuals assuming that these solutions do not dominate each other
    Output: single real value - the hypervolume formed by solutions with (1,1) as the reference point
    '''

    result = 0.0   
    sorted_solutions_fitnesses = solutions   
    bottom = 0
    left = 0

    #sort by first objective is descending order (from right to left)
    sorted_solutions_fitnesses = sorted(sorted_solutions_fitnesses, key = lambda x: x[0], reverse=True)
    for i, fitness in enumerate(sorted_solutions_fitnesses):
        current_hypervolume = (fitness[0] - left) * (fitness[1] - bottom)
        bottom = fitness[1]
        result += current_hypervolume

    return result


def calculateIGD(solutions, fitnessFunction):
    result = 0.0
    sorted_solutions_fitnesses = solutions


    sorted_solutions_fitnesses = sorted(sorted_solutions_fitnesses, key = lambda x: x[0], reverse=True)
    IGD_matrix = []

    for fitness_solution in sorted_solutions_fitnesses:
        lowest_dist = 200000 #eucl. distance cannot be higher than 2
        for fitness_estfront in fitnessFunction:
            dist = np.linalg.norm(np.array(fitness_estfront)-np.array(fitness_solution))#euclidean distance

            if dist < lowest_dist:
                lowest_dist = dist
        IGD_matrix.append(lowest_dist)

    return sum(IGD_matrix)/len(IGD_matrix)

