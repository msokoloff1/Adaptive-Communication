import numpy as np
import sys
import re
from multiprocessing import Process
import time
class Tree():
        def __init__(self,k,n,l):
                self.generateTree(k,n,l)
                #Min of l or 15 because we only have 4 bit unsigned representations (5 bits with the signed version)
                self.NUM_BITS = 5
                self.l = min(l, ((self.NUM_BITS-1)**2)-1 )


        def generateTree(self, k,n,l):
                self.weights = np.random.randint(-l,l+1, [k, n])

        def getActivations(self, inputs, nonActivated = False):
                hidden = []
                for weightGroup, inputGroup in zip(self.weights, inputs):
                        hidden.append(np.dot(weightGroup, inputGroup))

                if(nonActivated):
                        return np.abs(np.array(hidden))

                hidden = np.sign(np.array(hidden))
                output = np.prod(hidden)
                return [hidden, output]

        def updateWeights(self,inputs, hidden, outputSelf, outputOther):
            for index, input in enumerate(inputs):
                update = input*float(hidden[index]==outputSelf)*float(outputSelf==outputOther==1)
                self.weights = np.clip((self.weights+update), -self.l, self.l)



        def getKey(self, messageLength, batchSize,iteration):
            np.random.seed(iteration)
            integerVersion = np.random.choice(self.weights.reshape(-1),(batchSize,(messageLength//self.NUM_BITS)*2))
            formatted = [list("".join(["{0:05b}".format(int(val)).replace("-","1") for val in row])) for row in integerVersion]
            return np.array(formatted).reshape(-1, messageLength*2)





