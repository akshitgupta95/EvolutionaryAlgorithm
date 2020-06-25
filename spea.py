from typing import TypeVar, List
from jmetal.algorithm.multiobjective import SPEA2
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.comparator import MultiComparator
from jmetal.util.density_estimator import CrowdingDistance
from jmetal.util.ranking import StrengthRanking
from jmetal.util.density_estimator import KNearestNeighborDensityEstimator
from jmetal.util.replacement import RankingAndDensityEstimatorReplacement, RemovalPolicyType
from jmetal.core.operator import Mutation, Crossover
from jmetal.core.problem import Problem
from jmetal.util.termination_criterion import TerminationCriterion
from jmetal.util.solution import get_non_dominated_solutions
import numpy as np
from utils import calculateHypervolume,calculateIGD
from jmetal.core.quality_indicator import InvertedGenerationalDistance

S = TypeVar('S')
R = TypeVar('R')

class MyAlgorithm(SPEA2):
    
    def __init__(self, problem:Problem, population_size:int,mutation:Mutation, crossover:Crossover,termination:TerminationCriterion,utopian: tuple, referenceFront:np.ndarray):
        super().__init__(problem = problem, 
                         population_size = population_size, 
                         offspring_population_size = population_size,
                         mutation = mutation,
                         crossover = crossover, 
                         termination_criterion = termination
                        )
        self.hypervolumeByGeneration=[]
        self.IGDbyGeneration=[]
        self.Uobjecive1, self.Uobjecive2= utopian
        self.reference_front=referenceFront
        
     # how to generate the next population
    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[S]]:
     
        ranking = StrengthRanking(self.dominance_comparator)
        density_estimator = KNearestNeighborDensityEstimator()

        r = RankingAndDensityEstimatorReplacement(ranking, density_estimator, RemovalPolicyType.SEQUENTIAL)
        solutions = r.replace(population, offspring_population)
        front = get_non_dominated_solutions(solutions)
        objective1=-1/self.Uobjecive1*np.array([solution.objectives[0] for solution in front])
        objective2=-1/self.Uobjecive2*np.array([solution.objectives[1] for solution in front])
        HV=calculateHypervolume(list(zip(objective1,objective2)))
        print('Hypervolume;',HV)
        self.hypervolumeByGeneration.append(HV)
        # DO IGD calculation here
        IGD = InvertedGenerationalDistance(self.reference_front)
        igd=IGD.compute(list(zip(-np.array([solution.objectives[0] for solution in front]),-np.array([solution.objectives[1] for solution in front]))))
        print('IGD1:',igd)
        # obj1front=1/self.Uobjecive1*self.reference_front[:, 0]
        # obj2front=1/self.Uobjecive2*self.reference_front[:,1]
        # igd2 = calculateIGD(list(zip(objective1,objective2)), list(zip(obj1front,obj2front)))
        # print('IGD2:',igd2)
        self.IGDbyGeneration.append(igd)
        return solutions

    def getHyperVolumeByGeneration(self):
        return np.array(self.hypervolumeByGeneration) 

    def getIGDByGeneration(self):
        return np.array(self.IGDbyGeneration)
      