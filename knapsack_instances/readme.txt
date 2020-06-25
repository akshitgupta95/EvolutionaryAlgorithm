Folders "dense" and "sparse" contain two different sets of MO-Knapsack instances.
Filename [N]/.txt contains an instance with N variables.

The difference between these two types of instances is that in "dense" instances the area (in the objective space) close to the pareto front is dense and contains many solutions, while in "sparse" one part of it is dense and one is sparse. So, you can think about it as "it's easy to find solutions with good objective one, but much more difficult to find solutions with good objective two".
Such test cases are interesting for MO algorithms benchmarking in general as different algorithms can behave differently depending on solutions distribution in the search space.

It is important to test your algorithms on both types and write down some differences in performance (if there are any).

Files format:

--------------
N W
w_1 o1_1 o1_1
w_2 o1_2 o2_2
...
w_N o1_N o2_N
--------------

N is the number of items
W is the knapsack capacity
w_i is the weight of the i-th item
o1_i is the objective1 of the i-th item
o2_i is the objective2 of the i-th item

All numbers are integers, though in "dense" instances objectives values are positive and in "sparse" one objective is negative and one is positive, in both cases objectives should be MAXIMIZED.

