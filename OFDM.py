import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.signal import hilbert

np.seed = 1
random.seed(1)
def randsrc(dim=1, qnt=10, src=np.array([-3, -1, 1, 3])):
    aux = np.array([])
    outVar = np.empty(shape=(0,qnt),dtype=int)

    for i in range(dim):
        outVar = np.vstack((outVar, [random.choice(src) for _ in range(qnt)]))
    return outVar

def slicer(arrayMoura):
    aux = []
    for i in range(len(arrayMoura)):
        if(arrayMoura[i] < 0):
            aux.append(-1)
        else:
            aux.append(1)
    return aux
# Canal h
alfa = 0.1
h = np.array([1, alfa])
mu = len(h) - 1

a = randsrc(1, 8, [-1, 1])

s = np.fft.ifft(a[0],8)

s = s + np.random.normal(0, 0.5, np.shape(s))

r = np.fft.fft(s,8)

# posIndexes = np.nonzero(r>=0)
# negIndexes = np.nonzero(r<0)

aChapeu = slicer(r)

#ProbErroTeor = Seria aquela função Q(sqrt(2*Eb/N0)) mas quem é N0? N0 é 2*sigma^2 porque N0 é complexo
probErroPrat = len(np.nonzero((a - aChapeu) != 0))/8

print(probErroPrat, a - aChapeu)



# print(a, r, r)