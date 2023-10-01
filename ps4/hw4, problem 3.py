import numpy as np
import matplotlib.pyplot as plt
import math
import scipy

def H(n, x):
    if (n==0 or n==1):
        return (2*x)**n
    else:
        return 2*x*H(n-1, x) - 2*(n-1)*H(n-2, x)

def gaussxw(N):

    # Initial approximation to roots of the Legendre polynomial
    a = np.linspace(3,4*N-1,N)/(4*N+2)
    x = np.cos(np.pi*a+1/(8*N*N*np.tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = np.ones(N,float)
        p1 = np.copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)
    return x,w

def integrator_gauss_quad(n = 100):                                                                                                     #change n here to change the number of gaussian quad points
    x, w = gaussxw(n)
    f_x = np.power(wavefunction(5, np.tan(np.pi*x/2)),2)*np.power(np.tan(np.pi*x/2), 2)*(np.power(np.tan(np.pi*x/2), 2) + 1)*np.pi/2    #change the first argument of wavefunction to change the number of the wavefunction
    return sum(w*f_x)                

def wavefunction(n, x):
    return H(n,x)*np.exp(-x*x/2)/np.sqrt(np.power(2, n)*math.factorial(n)*np.sqrt(np.pi))

def integrand_factor_hermite(x, n):
    return x*x*H(n, x)*H(n,x)/(np.power(2, n)*math.factorial(n)*np.sqrt(np.pi))

def gauss_hermite_integrator(n = 6):                      #change n here to change the number of gauss-hermite quadrature points
    x, w = scipy.special.roots_hermite(n)
    return sum(integrand_factor_hermite(x, 5)*w)        #change the second argument of integrand_factor to change the number of the wavefunction

x = np.arange(-4, 4, 0.1, dtype = float)
plt.plot(x, H(0, x), x, H(1, x), x, H(2, x), x, H(3,x))
plt.xlabel('x')
plt.ylabel('$H_{n}(x)$')
plt.legend(['$H_{0}(x)$', '$H_{1}(x)$', '$H_{2}(x)$', '$H_{3}(x)$'])
plt.xlim(-4, 4)
plt.ylim(-20, 20)
plt.grid()
plt.show()

x = np.arange(-10, 10, 0.1, dtype = float)
plt.plot(x, wavefunction(30, x))
plt.xlabel('x')
plt.ylabel(r'$\psi_{30}(x)$')
plt.xlim(-10, 10)
plt.grid()
plt.show()

print('The mean of x**2 is: {} (Gaussian quadrature)'.format(integrator_gauss_quad()))
print('The mean of x**2 is {} (Gauss-Hermite quadrature)'.format(gauss_hermite_integrator()))

print('Sqrt(<x**2>) is: {} (Gaussian quadrature)'.format(np.sqrt(integrator_gauss_quad())))
print('Sqrt(<x**2>) is {} (Gauss-Hermite quadrature)'.format(np.sqrt(gauss_hermite_integrator())))