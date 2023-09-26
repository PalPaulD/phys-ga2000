import numpy as np
import matplotlib.pyplot as plt

def f(x=None):
    return x*(x-1)

def derivative(x0 = None, dx = None):
    return (f(x0+dx) - f(x0))/dx

delta1 = np.power(10, (-np.arange(2, 19, 0.1, dtype = np.float32)))
delta2 = np.power(10, (-np.arange(2, 19, 0.1, dtype = np.float64)))

result1 = derivative(1, delta1)
result2 = derivative(1, delta2)
print('The value of step at which the error is minimal at 32-bit precision: {}'.format(delta1[np.argmin(np.abs(result1-1))]))
print('The vakue of step at which the error is minimal at 64-bit precision: {}'.format(delta2[np.argmin(np.abs(result2-1))]))


plt.plot(delta1, result1, delta2, result2)
plt.xlabel('$h = \delta$')
plt.semilogx()
plt.xlim(10**-19, 10**-2)
plt.ylabel('$\dfrac{df}{dx}(x=1)$')
#plt.title('Values of derivative for different $h$')
plt.legend(['32-bit precision','64-bit precision', 'exact difference formula'])
plt.grid()
plt.show()