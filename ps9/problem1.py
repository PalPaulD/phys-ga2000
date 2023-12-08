import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm

class solver():

    def __init__(self, x0, sigma, mass, k, L, T, N, M):
        self.x0=x0
        self.sigma=sigma
        self.m=mass
        self.k=k
        self.dx=L/N
        self.dt=T/M
        self.N=N
        self.M=M

        #declare an array of values of function on the grid

        self.u=np.zeros([N+1,M+1], dtype=np.complex64)

        #initialize the initial conditions

        x=self.dx*np.arange(0, N+1, 1, dtype=np.float32)

        self.u[:,0]=np.exp(-(x-self.x0)**2/(2*self.sigma**2))*np.exp(1j*self.k*x)


    def sweep_forward(self, d1, sigma1, d_Np1, sigma_Np1, A, B, C, j):
        d=np.zeros(self.N+1, dtype=np.complex64)
        sigma=np.zeros(self.N+1, dtype=np.complex64)
        d[0]=d1
        sigma[0]=sigma1

        for i in range(0, self.N-1):
            F=-self.u[i,j-1]-(self.u[i+1,j-1] + self.u[i-1,j-1] - 2*self.u[i,j-1])*self.dt*1j/(4*self.m*self.dx**2)

            d[i+1]=C/(B-A*d[i])
            sigma[i+1]=(F-A*sigma[i])/(A*d[i]-B)

        d[self.N]=d_Np1
        sigma[self.N]=sigma_Np1    
        return d, sigma
        
    def sweep_back(self, d, sigma, j):
        for i in range(self.N, 1, -1):
            self.u[i-1,j]=d[i]*self.u[i,j]+sigma[i]

    def solve(self):

        A = self.dt*1j/(4*self.m*self.dx**2)
        B = 1 + self.dt*1j/(2*self.m*self.dx**2)
        C = A

        d1=0.
        sigma1=0.
        sigma_Np1=0.
        d_Np1=0.

        for j in tqdm(range(1, self.M+1)):
            d, sigma = self.sweep_forward(d1, sigma1, sigma_Np1, d_Np1, A, B, C, j)
            self.sweep_back(d, sigma, j)

    def plot(self):
        
        x=self.dx*np.arange(0, self.N+1, 1, dtype=np.float32)
        y=np.abs(self.u[:,0])**2

        fig, ax = plt.subplots()
        fig.suptitle('$|\psi(x)|^2$')
        plt.grid()
        line, = ax.plot(x, y)

        def update(i, x, line):
            line.set_data(x, np.abs(self.u[:,i])**2)
            return line,
    
        ani=animation.FuncAnimation(fig, update, len(x), interval=100, fargs=[x,line], blit=True)

        ani.save('animation.gif', fps=60)

        plt.close()        

solve=solver(x0=4, sigma=1, mass=1, k=4, L=8, T=64, N=512, M=2048)
solve.solve()
solve.plot()