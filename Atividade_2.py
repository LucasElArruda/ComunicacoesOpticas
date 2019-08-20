import numpy as np
import random
import matplotlib.pyplot as plt

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

## Começa a atividade aqui.
## Primeiro fez com pulso quadrado
T = 100

g = np.ones((1,T))
g = np.concatenate((g, 0), axis=None)

g = g/(np.sqrt(np.sum(g**2)))

am = randsrc(qnt=10)
am = intersperse(am[0], 0, T)
# plt.stem(am, use_line_collection=True)

s = np.convolve(am, g)
# plt.figure()
# plt.plot(s)

s = s + np.random.normal(0, 0.1, len(s))
# plt.figure()
# plt.plot(s)

yRx = np.convolve(s, g)
# plt.figure()
# plt.stem(yRx[::T])

## Aeee deu bom com pulso quadrado! Bora fazer com sinc/coseno levanto
t2 = np.arange(-500, 500, 1)
alfa = 0


g2 = (np.sin(2*np.pi*t2/T))/(2*np.pi*t2/T)*np.cos(alfa*np.pi*t2/T)/(1-(2*alfa*t2/T)**2)
g2[500] = 1
g2 = g2/(np.sqrt(np.sum(g2**2)))
# g2 = np.concatenate(np.flip(g2))

s2 = np.convolve(am, g2)
print(len(s2))

plt.figure()
plt.stem(am)

plt.figure()
plt.plot(s2)

plt.figure()
plt.plot(g2)
s2 = s2 + np.random.normal(0, 0.1, len(s2))

plt.figure()
plt.plot(s2)

yRx2 = np.convolve(s2, g2)

plt.figure()
plt.stem(yRx2[500::T])





plt.show()


print(g)