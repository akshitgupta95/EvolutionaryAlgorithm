
from jmetal.algorithm.multiobjective.spea2 import SPEA2
from jmetal.operator import SPXCrossover, BitFlipMutation
from jmetal.problem import ZDT1
from MOKnapsack import MOKnapsack
from spea import MyAlgorithm
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.lab.visualization.plotting import Plot
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.operator.selection import RouletteWheelSelection, BinaryTournamentSelection, BestSolutionSelection
from jmetal.core.quality_indicator import GenerationalDistance,InvertedGenerationalDistance,EpsilonIndicator,HyperVolume
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.ticker as mtick

def oneRun(problemType,numberOfitems:int, population_size:int,maxEvaluations:int):
    filename = 'knapsack_instances/%s/%d.txt' % (problemType, numberOfitems)
    problem = MOKnapsack(from_file=True,filename=filename)
    filename= 'PF/%s/%d.txt' % (problemType, numberOfitems)
    with open(filename) as file:
            lines = file.readlines()
            data = [line.split() for line in lines if len(line.split()) >= 1]  
            calibratedFront = np.asfarray(data[0:], dtype=np.int32)

    max_evaluations = maxEvaluations
    utopianPoint=(np.amax(calibratedFront[:, 0]),np.amax(calibratedFront[:,1]))
    algorithm = MyAlgorithm(
        problem=problem,
        population_size=population_size,
        mutation=BitFlipMutation(probability=1.0 / problem.number_of_bits),
        crossover=SPXCrossover(probability=0.9),
        termination=StoppingByEvaluations(max_evaluations),utopian=utopianPoint,
        referenceFront= calibratedFront
    )

    algorithm.run()
    solutions = algorithm.get_result()
    HVbyGen=algorithm.getHyperVolumeByGeneration()
    IGDbyGen=algorithm.getIGDByGeneration()
    front = get_non_dominated_solutions(solutions)
    # for solution in front:
    #     print(solution.variables)

    #     print(-solution.objectives[0],-solution.objectives[1])
    objective1=-1/utopianPoint[0]*np.array([solution.objectives[0] for solution in front])
    objective2=-1/utopianPoint[1]*np.array([solution.objectives[1] for solution in front])
    # print(calculateHypervolume(list(zip(objective1,objective2))))
    fig=plt.figure(numberOfitems+3)
    plt.scatter(objective1, objective2)
    plt.title('%s_%d'%(problemType,numberOfitems))
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.savefig('%s_%d'%(problemType,numberOfitems))
    return HVbyGen,IGDbyGen
# plot_front = Plot(title='Pareto front approximation', axis_labels=['x', 'y'])
# plot_front.plot(front, label='SPEA2-ZDT1')
def main():
    
    numberOfRuns=2
    problemType="dense"
    numberOfItemsArray=np.array([20,80,320])
    max_evaluations=50000
    populationSizes=np.array([4,16,64,128,256])
    # fig, ax = plt.subplots()
    figureNumber=0
    for numberOfItems in numberOfItemsArray:
        figureNumber=numberOfItems
        for popSize in populationSizes:
            
            print("populationSize: ",popSize)
            numberOfEvaluations=np.arange(popSize, max_evaluations, popSize)
            AvgHV=[]
            AvgIGD=[] 
            for i in range(numberOfRuns):
                print("run: ",i)
                HVSigleRun,IGDRun=oneRun(problemType=problemType,numberOfitems=numberOfItems,population_size=popSize,maxEvaluations=max_evaluations)
                AvgHV.append(HVSigleRun)
                AvgIGD.append(IGDRun)
            HVresult=np.mean(AvgHV, axis=0)
            HVresult=1-HVresult
            IGDResult=np.mean(AvgIGD, axis=0)
            if(popSize==4):
                plt.figure(figureNumber)
                plt.loglog(numberOfEvaluations,HVresult, basex=2, basey=2,label='Population Size=4',c='r')
                plt.figure(figureNumber+1)
                plt.loglog(numberOfEvaluations,IGDResult, basex=2, basey=2,label='Population Size=4',c='r')
            if(popSize==16):
                plt.figure(figureNumber)
                plt.loglog(numberOfEvaluations,HVresult, basex=2, basey=2,label='Population Size=16',c='b')
                plt.figure(figureNumber+1)
                plt.loglog(numberOfEvaluations,IGDResult, basex=2, basey=2,label='Population Size=16',c='b')
            if(popSize==64):
                plt.figure(figureNumber)
                plt.loglog(numberOfEvaluations,HVresult, basex=2, basey=2,label='Population Size=64',c='g')
                plt.figure(figureNumber+1)
                plt.loglog(numberOfEvaluations,IGDResult, basex=2, basey=2,label='Population Size=64',c='g')
            if(popSize==128):
                plt.figure(figureNumber)
                plt.loglog(numberOfEvaluations,HVresult, basex=2, basey=2,label='Population Size=128',c='y')
                plt.figure(figureNumber+1)
                plt.loglog(numberOfEvaluations,IGDResult, basex=2, basey=2,label='Population Size=128',c='y')
            if(popSize==256):
                plt.figure(figureNumber)
                plt.loglog(numberOfEvaluations,HVresult, basex=2, basey=2,label='Population Size=256',c='m')
                plt.figure(figureNumber+1)
                plt.loglog(numberOfEvaluations,IGDResult, basex=2, basey=2,label='Population Size=256',c='m')
        # plt.get_yaxis().set_major_formatter(mtick.FormatStrFormatter('%.3f'))
        # plt.xticks(np.arange(popSize, max_evaluations, popSize))
        fig=plt.figure(figureNumber)
        ax=fig.axes
        ax[0].xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
        ax[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
        plt.xlabel('Number of Evaluations')
        plt.ylabel('1 minus hypervolume')
        ax[0].legend()
        plt.title('%s_%d'%(problemType,numberOfItems))
        plt.savefig("comparison_%d_HV"%(numberOfItems))

        fig=plt.figure(figureNumber+1)
        ax=fig.axes
        ax[0].xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
        ax[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        plt.xlabel('Number of Evaluations')
        plt.ylabel('IGD')
        ax[0].legend()
        plt.title('%s_%d'%(problemType,numberOfItems))
        plt.savefig("comparison_%d_IGD"%(numberOfItems))
        

if __name__=="__main__": 
    main()     
