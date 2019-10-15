import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.stats import norm


np.random.seed(1)
random.seed(1)

def ErrProb(c,sigma,nBlocks, Ns):
    return (1 - norm.cdf(c / np.sqrt(((sigma ** 2) * nBlocks * Ns))))

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
# Variaveis
alfa = 0.1
Ns = 2**15
nBlocks = 1
c = 1
sigma = 0.5

# Canal h
h = np.array([1, alfa])
mu = len(h) - 1

a = randsrc(nBlocks, Ns, [-c, c])
aChapeu = np.zeros(np.shape(a))

for i in range(nBlocks):
    s = np.fft.ifft(a[i],Ns)
    s = np.append(s[Ns-1], s)
    s = np.convolve(s, h)
    s = s + np.random.normal(0, sigma, np.shape(s))

    s = s[1:]
    r = np.fft.fft(s,Ns)

    # posIndexes = np.nonzero(r>=0)
    # negIndexes = np.nonzero(r<0)

    aChapeu[i] = slicer(r)

#ProbErroTeor = Seria aquela função Q(sqrt(2*Eb/N0)) mas quem é N0? N0 é 2*sigma^2 porque N0 é complexo
probErroPrat = np.shape((np.nonzero((a - aChapeu) != 0)))[1]/(Ns*nBlocks)
errosTeor = ErrProb(c,sigma,nBlocks, Ns)

print(probErroPrat, errosTeor)



# print(a, r, r)