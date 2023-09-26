import numpy as np
import matplotlib.pyplot as plt
import scipy
import time

#sizes of matrices
N_start_selfmade = 1
N_finish_selfmade = 50
dN_selfmade = 1

N_start_npdot = 100
N_finish_npdot = 500
dN_npdot = 1

#multiplication functions
def self_made_dot(n):
    A = np.zeros(n*n, dtype = float).reshape(n,n)
    B = np.zeros(n*n, dtype = float).reshape(n,n)
    C = np.zeros(n*n, dtype = float).reshape(n,n)
    start_time = time.perf_counter_ns()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i,j] += A[i,k]*B[k,j]
    del(A, B, C)
    return (time.perf_counter_ns() - start_time)/(10**9) 

def npdot(n):
    A = np.zeros(n*n, dtype = float).reshape(n,n)
    B = np.zeros(n*n, dtype = float).reshape(n,n)
    C = np.zeros(n*n, dtype = float).reshape(n,n)
    start_time = time.perf_counter_ns()
    C = np.dot(A,B)
    del(A, B, C)
    return (time.perf_counter_ns() - start_time)/(10**9) 

#fitted function
def fit_func(x, a, b):
    return a*(x**b)

#create and fill arrays with computational time for different multipication funcions
def fitting():
    size_grid_selfmade = np.arange(N_start_selfmade, N_finish_selfmade, dN_selfmade)
    size_grid_npdot = np.arange(N_start_npdot, N_finish_npdot, dN_npdot)

    npdot_times = np.zeros(len(size_grid_npdot))
    for i in range(1, len(size_grid_npdot)):
        npdot_times[i] = npdot(size_grid_npdot[i])

    selfmade_times = np.zeros(len(size_grid_selfmade))
    for i in range(1, len(size_grid_selfmade)):
        selfmade_times[i] = self_made_dot(size_grid_selfmade[i])

    #fitting the curves
    fit_selfmade_times, _ = scipy.optimize.curve_fit(fit_func, size_grid_selfmade, selfmade_times, p0 = (0, 0), maxfev = 10**4)
    fit_npdot_times, _ = scipy.optimize.curve_fit(fit_func, size_grid_npdot, npdot_times, p0 = (0, 0), maxfev = 10**4)

    print('For a self-made multiplication a*x^n, where a = {}, and n = {}'.format(fit_selfmade_times[0], fit_selfmade_times[1]))
    print('For a np.dot() multiplication a*x^n, where a = {}, and n = {}'.format(fit_npdot_times[0], fit_npdot_times[1]))
    return (selfmade_times, npdot_times, fit_selfmade_times, fit_npdot_times)

#plot everything from the data (including the fits)
def plot_everything(selfmade_times, npdot_times, fit_selfmade_times, fit_npdot_times):
    size_grid_selfmade = np.arange(N_start_selfmade, N_finish_selfmade, dN_selfmade)
    size_grid_npdot = np.arange(N_start_npdot, N_finish_npdot, dN_npdot)

    plt.plot(size_grid_selfmade, selfmade_times, size_grid_selfmade, fit_func(size_grid_selfmade, fit_selfmade_times[0], fit_selfmade_times[1]))
    plt.xlabel('Size of matricies')
    plt.title('Selfmade dot multiplication times and its fit')
    plt.ylabel('Computational time, seconds')
    plt.legend(['Self made multiplication times', 'A fit'])
    plt.grid()
    plt.show()

    plt.plot(size_grid_npdot, npdot_times, size_grid_npdot, fit_func(size_grid_npdot, fit_npdot_times[0], fit_npdot_times[1]))
    plt.xlabel('Size of matricies')
    plt.title('Np.dot multiplication times and its fit')
    plt.ylabel('Computational time, seconds')
    plt.legend(['Np.dot() multiplication times', 'A fit'])
    plt.grid()
    plt.show()


#calling fitting() = calculations + fit, calling plot_everything() = plotting the results
selfmade_times, npdot_times, fit_selfmade_times, fit_npdot_times = fitting()
plot_everything(selfmade_times, npdot_times, fit_selfmade_times, fit_npdot_times)