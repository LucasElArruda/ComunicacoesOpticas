import numpy as np
import random
import matplotlib.pyplot as plt

print("ae")

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

g = np.ones((1,100))
g = np.concatenate((g, 0), axis=None)
#l = [1, 2, 3, 4, 5]
#intersperse(l, 0, 100)
print(g)

am = randsrc(qnt=5)
print(am)
am = intersperse(am[0], 0, 100)

s = np.convolve(am, g)

#plt.stem(s, use_line_collection=True)
plt.plot(s)
plt.show()
#plt.plot(np.sinc(np.arange(10,step=0.1)))
