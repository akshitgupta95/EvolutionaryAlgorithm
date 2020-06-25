import random

import numpy as np

from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution

class MOKnapsack(BinaryProblem):

    def __init__(self, number_of_items: int = 50, capacity: int = 1000, weights: list = None,
                 objective1: list = None, objective2: list = None,from_file: bool = False, filename: str = None):
        super(MOKnapsack, self).__init__()

        if from_file:
            self.__read_from_file(filename)
        else:
            self.capacity = capacity
            self.weights = weights
            self.objective1 = objective1
            self.objective2 = objective2
            self.number_of_bits = number_of_items

        self.number_of_objectives = 2
        self.number_of_variables = 1
        self.obj_directions = [self.MAXIMIZE,self.MAXIMIZE]
        self.obj_labels = ['objective1', 'objective2']
        self.number_of_constraints = 0

    def __read_from_file(self, filename: str):
        """
        This function reads a Knapsack Problem instance from a file.
        It expects the following format:

            num_of_items (dimension)
            capacity of the knapsack
            num_of_items-tuples of weight-profit

        :param filename: File which describes the instance.
        :type filename: str.
        """

        if filename is None:
            raise FileNotFoundError('Error Filename can not be None')

        with open(filename) as file:
            lines = file.readlines()
            data = [line.split() for line in lines if len(line.split()) >= 1]
        
            self.number_of_bits = int(data[0][0])
            self.capacity = int(data[0][1])
        
            weights_and_profits = np.asfarray(data[1:], dtype=np.float64).astype(np.int64)
      
            self.weights = weights_and_profits[:, 0]
            self.objective1 = weights_and_profits[:, 1]
            self.objective2 = weights_and_profits[:, 2]

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        total_objective1 = 0.0
        total_objective2=0
        total_weigths = 0.0

        for index, bits in enumerate(solution.variables[0]):
            if bits:
                total_objective1 += self.objective1[index]
                total_objective2+=self.objective2[index]
                total_weigths += self.weights[index]

        if total_weigths > self.capacity:
            total_objective1 = 0.0
            total_objective2= 0.0
            

        solution.objectives[0] = -1.0 * total_objective1
        solution.objectives[1] = -1.0 * total_objective2
        return solution

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=self.number_of_variables,
                                      number_of_objectives=self.number_of_objectives)

        new_solution.variables[0] = \
            [True if random.randint(0, 1) == 0 else False for _ in range(
                self.number_of_bits)]

        return new_solution

    def get_name(self):
        return 'MOKnapsack'          