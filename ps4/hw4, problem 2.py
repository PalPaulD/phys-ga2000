import numpy as np
import matplotlib.pyplot as plt
import scipy

a_min = 0
a_max = 2
da = 0.01

def f(x):
    return 1/np.sqrt(1+x*x)

def integrator(N = 20):
    a = -1
    b = 1
    x, w = scipy.special.roots_chebyt(N) # chebyshev polynomial (1st kind) roots
    xp = 0.5*(b-a)*x + 0.5*(b+a)                     # sample points, rescaled to bounds 0,1
    wp = 0.5*(b-a)*w                            # rescale weights to bounds 0,1
    return np.sqrt(8)*sum(f(xp)*wp)/2      # add them up!

amplitudes = np.arange(a_min, a_max + da, da, dtype = float)
print('The a-dependence is {}/a'.format(integrator()))

plt.plot(amplitudes, integrator()/amplitudes)
plt.legend(['$T(a)$ dependence'])
plt.grid()
plt.show()