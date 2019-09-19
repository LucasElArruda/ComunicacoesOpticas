import numpy as np
import matplotlib.pyplot as plt

# Fazer taxa de amostragem com Fs = 8192 para tocar o som do PC)
# Ele botou tambem fc = 440

def intersperse(lst, item, interval):
    result = [item] * ((len(lst)-1) * interval+1)
    result[0::interval] = lst
    return result

def str2ascii(s):
    return [ord(c) for c in s]

def dec2bin(integer, nbits=8):
    return bin(integer)[2:].zfill(nbits)

def composeSignal(arrayMoura):
    aux = ''
    for i in range(len(arrayMoura)):
        aux = str(aux) +dec2bin(arrayMoura[i])
    return aux

def decomposeSignal(arrayMoura, nbits=8):
    aux = []
    for i in range(int(len(arrayMoura)/nbits)):
        aux.append(int(str(arrayMoura[i*nbits:i*nbits+nbits]), 2))
    return aux

def dec2ascii(arrayMoura):
    return ''.join([chr(c) for c in arrayMoura])

def bpskMap(stringMoura, scale=1):
    aux = []
    for i in range(len(stringMoura)):
        if(stringMoura[i]=='1'):
            aux.append(scale)
        else:
            aux.append(-1*scale)
    return aux

def bpskDemap(arrayMoura):
    aux = []
    for i in range(len(arrayMoura)):
        if(arrayMoura[i] == -1):
            aux.append('0')
        else:
            aux.append('1')
    return ''.join(aux)

s = 'Oi tudo bem'

sBits = composeSignal(str2ascii(s))

print("Bits tx: {}".format(sBits))

sMapped = bpskMap(sBits)


deltaT = 100

am = intersperse(sMapped, 0, T)
g = np.ones((1,T))
g = np.concatenate((g, 0), axis=None)
g = g/(np.sqrt(np.sum(g**2)))

plt.figure()
plt.stem(am, use_line_collection=True)

s = np.convolve(am,g)
plt.figure()
plt.plot(s)

fc = 440
t = np.linspace(0,2/T, len(s))
carrier = np.sqrt(2)*np.exp(2*np.pi*fc*t*1j)
plt.figure()
plt.plot(carrier)

print("Mapped signal: {}".format(sMapped))

sDemapped = bpskDemap(sMapped)

print("Demapd signal: {}".format(sDemapped))
# print(decomposeSignal(sDemapped))
print(dec2ascii(decomposeSignal(sDemapped)))
plt.show()
