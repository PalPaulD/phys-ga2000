import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy import random


def f(x, a):       #the function
    return np.power(x, a-1.)*np.exp(-x)

def fz(z, a):       #the integrand after the change of variables with c = a-1 
    return np.power(a-1., a)*np.power(z, a-1)*np.exp(-(a-1)*z/(1-z))/np.power(1-z, a+1)

def fz1(z, a):      #the integrand after the change of variables with c = a
    return np.power(a, a)*np.power(z, a-1)*np.exp(-a*z/(1-z))/np.power(1-z, a+1)

def integrand(x, a):    #the integrand for G-L method
    return np.power(x, a-1.)

def integrate(a):
    if type(a) == int:
        N = np.ceil((a+1)//2)
    else:
        N = np.ceil(a)*100 
    x, w = scipy.special.roots_laguerre(N)
    return sum(integrand(x, a)*w)


x = np.arange(0, 5, 0.001, dtype = float)

plt.plot(x,f(x, 2), x, f(x, 3), x, f(x, 4))
plt.xlim(0, 5)
plt.xlabel('$x$')
plt.ylabel('$x^{a} e^{-x}$')
plt.legend(['$x e^{-x}$', '$x^2 e^{-x}$', '$x^3 e^{-x}$'])
plt.grid()
plt.show()

print('The values of gamma functions are: \u0393(3/2) = {}, \u0393(3) = {}, \u0393(6) = {}, \u0393(10) = {}'.format(integrate(3/2), integrate(3), integrate(6), integrate(10)))

##This code plots integrands with rescaling c = a-1
#x = np.arange(0, 1, 0.001, dtype = float)
#plt.plot(x, fz(x, 2), x, fz(x, 3), x, fz(x, 4))
#plt.legend(['Integrand with $a = 2$', 'Integrand with $a = 3$', 'Integrand with $a = 4$'])
#plt.ylabel('f(x)')
#plt.xlim(0, 1)
#plt.grid()
#plt.show()

##This code plots integrands with rescaling c = a
#x = np.arange(0, 1, 0.001, dtype = float)
#plt.plot(x, fz1(x, 2), x, fz1(x, 3), x, fz1(x, 4))
#plt.legend(['Integrand with $a = 2$', 'Integrand with $a = 3$', 'Integrand with $a = 4$'])
#plt.ylabel('f(x)')
#plt.xlim(0, 1)
#plt.grid()