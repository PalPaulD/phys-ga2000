import numpy as np
import matplotlib.pyplot as plt
import sys 
import time

start_time = time.time()
N=1000
M=500

def no_loops_calculation(c, z = 0, n = M):
    ''' 
    ----------
    The function applies the Mandelbrot sequence with c to z n times
    and returns the absolute value of the last calculated element.

    Note that there are no for-loops!
    -----------
    '''
    if n==0:
        return np.absolute(z)
    else:
        return no_loops_calculation(c, z**2 + c, n-1)
    
def one_loop_calculation(c):
    '''
    -------
    This function works in a same way as no_loops_calculation,
    but it uses a loop to do the same thing
    -------
    '''
    z=0
    for i in range(0, M+1):
        z = z**2 + c
    return np.absolute(z)


step = 4/N
sys.setrecursionlimit(N*M)               #had to use this, so I could do a lot of recursion. On my machine there is a limit of 
points = np.arange(-2., 2. + step, step)
cx, cy = np.meshgrid(points, points)
result = no_loops_calculation(cx + cy*1j)<2
#result = one_loop_calculation(cx+cy*1j)<2

print('Execution time: {} seconds'.format(round(time.time()-start_time, 2)))
plt.imshow(result, aspect = 'auto', cmap = 'bone_r', extent = [-2, 2, -2,2])
plt.grid()
plt.xlabel('$Re(c)$')
plt.ylabel('$Im(c)$')
plt.show()