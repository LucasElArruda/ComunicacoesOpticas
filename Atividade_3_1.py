# Calcular probabilidade de erro de simbolo teorica e simulada do 4QAM

import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm

def randsrc(dim=1, qnt=10, src=np.array([-3, -1, 1, 3])):
    aux = np.array([])
    outVar = np.empty(shape=(0,qnt),dtype=int)

    for i in range(dim):
        outVar = np.vstack((outVar, [random.choice(src) for _ in range(qnt)]))
    return outVar

tam = int(1e5)
sigma = 0.5
Ea = 1
xest = np.zeros(tam, dtype=complex)
n = np.random.normal(0, sigma, tam) + np.random.normal(0, sigma, tam)*1j

# plt.figure()
# plt.scatter(np.real(z), np.imag(z), s=8, marker='*')

dB = np.arange(1,20)

for c in 10**(dB/10):
    x = c*(randsrc(qnt=tam, src=[complex(1,1), complex(-1,1), complex(-1,-1), complex(1,-1)]))

    z = x[0]+n
    # Estimador
    xest[np.where((np.real(z) > 0) & (z.imag > 0))[0]] = c*np.complex(1,1)
    xest[np.where((np.real(z) > 0) & (z.imag < 0))[0]] = c*np.complex(1, -1)
    xest[np.where((np.real(z) < 0) & (z.imag > 0))[0]] = c*np.complex(-1, 1)
    xest[np.where((np.real(z) < 0) & (z.imag < 0))[0]] = c*np.complex(-1, -1)

    erros=np.sum(x!=xest)/tam
    errosTeor = 2*norm.sf(c/(2*sigma))

plt.figure()
plt.plot(errosTeor)
plt.plot(erros, '*')

plt.show()