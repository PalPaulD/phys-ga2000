import scipy
import numpy as np
import matplotlib.pyplot as plt

def RHS(t, r):
    sigma_=10
    r_=28
    b_=8/3
    x=r[0]
    y=r[1]
    z=r[2]
    fx=sigma_*(y-x)
    fy=r_*x - y - x*z
    fz=x*y - b_*z
    return [fx, fy, fz]

def numerical_traj_ex(t_span, y0, t):
    solution=scipy.integrate.solve_ivp(RHS, t_span, y0, t_eval=t)
    return solution.y[0,:], solution.y[2,:]

exp_fps=1_000
t_span=[0, 100]
t=np.arange(*t_span, 1/exp_fps)
y0=[0,1,0]
x, z=numerical_traj_ex(t_span, y0, t)

plt.plot(x, z, label='attractor')
plt.grid()
plt.xlabel('x')
plt.ylabel('z')
plt.legend()
plt.show()