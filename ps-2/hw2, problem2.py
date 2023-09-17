import numpy as np
import time

start_time = time.time()
L = 200

points = np.arange(-L, L+1, dtype = np.float32)
Is, Js, Ks = np.meshgrid(points, points, points)
V = (-1)**(Is + Js + Ks)/np.sqrt(Is*Is + Js*Js + Ks*Ks)
V[L][L][L] = 0                 #turn inf into 0, so the sum will stay finite
print('Madelung constant: {}, execution time: {} seconds'.format(V.sum(), round(time.time() - start_time, 2)))