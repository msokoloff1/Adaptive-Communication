import numpy as np
import sys
import re

class Tree():
        def __init__(self,k,n,l):
                self.generateTree(k,n,l)
                #Min of l or 15 because we only have 4 bit unsigned representations (5 bits with the signed version)
                self.l = min(l, 15)

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


        def getLine(self,messageLength, startIndex):
            #Each weight is a decimal number that we map to an 8 bit binary value.
            binary = []
            count = 0
            reshaped = self.weights.reshape(-1)
            newIndex = 0
            for index in range(messageLength):
                newIndex = (index + startIndex) % reshaped.shape[0]
                weight = reshaped[newIndex]
                sign = str(int(np.clip(np.sign(weight), 0, 1)))
                binaryVersion = sign + re.findall(r'\d+', bin(int(weight)))[-1].zfill(4)
                for bit in binaryVersion:
                    binary.append(int(bit))

                if (len(binary) == messageLength * 2):
                    return np.array(binary).reshape(-1)


        def getKey(self, messageLength, indices):
            results = []
            for startIndex in indices:
                results.append(self.getLine(messageLength,startIndex))

            return np.array(results).reshape(-1, messageLength*2)




