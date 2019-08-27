import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.signal import hilbert

def randsrc(dim=1, qnt=10, src=np.array([-3, -1, 1, 3])):
    aux = np.array([])
    outVar = np.empty(shape=(0,qnt),dtype=int)

    for i in range(dim):
        outVar = np.vstack((outVar, [random.choice(src) for _ in range(qnt)]))
    return outVar

def intersperse(lst, item, interval):
    result = [item] * ((len(lst)-1) * interval+1)
    result[0::interval] = lst
    return result

## Começa a atividade aqui
## Primeiro fez com pulso quadrado
T = 100
fc = 100
# t = np.arange(0,2*1/fc, 0.001)


g = np.ones((1,T))
g = np.concatenate((g, 0), axis=None)

g = g/(np.sqrt(np.sum(g**2)))

am = randsrc(qnt=10, src=[complex(1,1), complex(-1,1), complex(-1,-1), complex(1,-1)])
print(am)
am = intersperse(am[0], 0, T)
# plt.stem(am, use_line_collection=True)

s = np.convolve(am, g)

t = np.linspace(0,20/fc, len(s))
carrier = np.sqrt(2)*np.exp(2*np.pi*fc*t*1j)
plt.figure()
plt.plot(np.real(s))

tx = s*carrier
tx = np.real(tx)

# tx = tx + np.random.normal(0, 0.1, len(s))
plt.figure()
plt.plot(tx)

rx = hilbert(tx)
plt.figure()
plt.plot(rx)
print(t)

carrier_neg = np.sqrt(2)*np.exp(-2*np.pi*fc*t*1j)
s_chapeu = rx*carrier_neg
plt.figure()
plt.plot(s_chapeu)

s_casado = np.convolve(s_chapeu, g)
plt.figure()
plt.scatter(np.real(s_casado[::T]), np.imag(s_casado[::T]), s=8, marker='*')

plt.show()