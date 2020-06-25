import numpy as np

filename = 'PF/%s/%d.txt' % ("dense",20)
with open(filename) as file:
        lines = file.readlines()
        data = [line.split() for line in lines if len(line.split()) >= 1]
        
        # number_of_bits = int(data[0][0])
        # capacity = int(data[0][1])
        
        weights_and_profits = np.asfarray(data[0:], dtype=np.int32)
      
        # weights = weights_and_profits[:, 0]
        # objective1 = weights_and_profits[:, 0]
        # objective2 = weights_and_profits[:, 1]
