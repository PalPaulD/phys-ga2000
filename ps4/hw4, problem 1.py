import numpy as np
import matplotlib.pyplot as plt

Debye_temp = 428

def f(x):
    return (x**4)*np.exp(x)/((np.exp(x)-1)**2)

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

def gaussxwab(N, a, b):
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a), 0.5*(b-a)*w

#the integrator
def cv(T, N = 50):
    xp, wp = gaussxwab(N, 0, Debye_temp/T)  #take the weights and the point
    integral = sum(f(xp)*wp)                #an sum their products
    return ((T/Debye_temp)**3)*integral     #all constants are omitted in the answer    

temperatures = np.arange(5, 500, 1, dtype = np.float64)
results = np.zeros(len(temperatures), dtype = np.float64) 

for i in range(0, len(temperatures)):
    results[i] = cv(temperatures[i])

plt.plot(temperatures/Debye_temp, results)
plt.xlabel('$T/\Theta_D$')
plt.ylabel('$C_V / (9 V 'r'\rho k_B)$')
plt.legend(['Dimensionless heat capacity'])
plt.grid()
plt.show()

Ns = np.arange(10, 80, dtype = int)
results = np.zeros(len(Ns), dtype = float)

for i in range(0, len(results)):
    results[i] = cv(100*Debye_temp, Ns[i])

plt.plot(Ns, results)
plt.xlabel('N')
plt.ylabel('$C_V / (9 V 'r'\rho k_B)$')
plt.legend(['Dimensionless heat capacity against N'])
plt.grid()
plt.show()