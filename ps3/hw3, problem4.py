import numpy as np
import matplotlib.pyplot as plt

tau = 3.053*60    #half-life in secs
N = 1_000         #initial condition

def f(u):
    return -tau*np.log2(1-u)

t = np.sort(f(np.random.random(size = N)))
atoms = N - np.arange(1, N+1)

plt.plot(t, atoms)
plt.xlabel('time, s')
plt.ylabel('Undecayed atoms left')
plt.grid()
plt.show()