import numpy as np
import matplotlib.pyplot as plt
import time

N_start = 1
N_finish = 101 
dN = 1

def self_made_dot(n):
    A = np.zeros(n*n, dtype = float).reshape(n,n)
    B = np.zeros(n*n, dtype = float).reshape(n,n)
    C = np.zeros(n*n, dtype = float).reshape(n,n)
    start_time = time.perf_counter_ns()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i,j] += A[i,k]*B[k,j]
    return (time.perf_counter_ns() - start_time)/(10**9) 

def npdot(n):
    A = np.zeros(n*n, dtype = float).reshape(n,n)
    B = np.zeros(n*n, dtype = float).reshape(n,n)
    start_time = time.perf_counter_ns()
    C = np.dot(A,B)
    return (time.perf_counter_ns() - start_time)/(10**9) 


size_grid = np.arange(N_start, N_finish, dN)
self_made_dot_times = np.zeros(len(size_grid))
npdot_times = np.zeros(len(size_grid))


for i in range(0, len(size_grid)):
    npdot_times[i] = npdot(size_grid[i])
    self_made_dot_times[i] = self_made_dot(size_grid[i])

plt.plot(size_grid, self_made_dot_times, size_grid, npdot_times)
plt.xlabel('Size of matricies')
plt.ylabel('Computational time, seconds')
plt.legend(['Self made multiplication times', 'Dot made multiplication times'])
plt.grid()
plt.show()