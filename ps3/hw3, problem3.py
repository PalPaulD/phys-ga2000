import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

T_final = 20_000     #observation time
N = 4                #10**N atoms of 213-Bi is considered

tau_Bi_213 = 46.*60       #decay rates in seconds
tau_Tl_209 = 3.053*60
tau_Pb_209 = 3.3*60
prob = 0.9791              #probability of 213-Bi->209-Pb channel

Bi_213 = np.zeros(T_final+1, dtype = int)      #These are functins of time!!! So Bi_213 = Bi_213[t]
Tl_209 = np.zeros(T_final+1, dtype = int)
Pb_209 = np.zeros(T_final+1, dtype = int)
Bi_209 = np.zeros(T_final+1, dtype = int)
N_total = np.zeros(T_final+1, dtype = int)      #Total number is conserved

def random_decay(dt, tau):
    f_T = np.log(2.)*np.power(2, -dt/tau)/tau
    return np.random.binomial(1, f_T)


Bi_213[0] = 10**N                   #initial condtions
Tl_209[0] = 0
Pb_209[0] = 0
Bi_209[0] = 0
N_total[0] = Pb_209[0] + Bi_209[0] + Tl_209[0] + Bi_213[0] 

for t in range(0, T_final):

    decayed_Pb_209 = random_decay(np.ones(Pb_209[t], dtype = int), tau_Pb_209).sum()          #209-Pb -> 209-Bi
    Pb_209[t+1] = Pb_209[t] - decayed_Pb_209
    Bi_209[t+1] = Bi_209[t] + decayed_Pb_209

    decayed_Ti_209 = random_decay(np.ones(Tl_209[t], dtype = int), tau_Tl_209).sum()          #209-Ti -> 209-Pb 
    Tl_209[t+1] = Tl_209[t] - decayed_Ti_209
    Pb_209[t+1] = Pb_209[t+1] + decayed_Ti_209

    total_decayed = random_decay(np.ones(Bi_213[t], dtype = int), tau_Bi_213).sum()           #213-Bi -> 209-Pb or 209-Ti
    to_Pb_209 = np.random.binomial(np.ones(total_decayed, dtype = int), prob).sum()
    to_Tl_209 = total_decayed - to_Pb_209
    Bi_213[t+1] = Bi_213[t] - to_Pb_209 - to_Tl_209
    Pb_209[t+1] = Pb_209[t+1] + to_Pb_209
    Tl_209[t+1] = Tl_209[t+1] + to_Tl_209

    N_total[t+1] = Bi_209[t+1] + Pb_209[t+1] + Tl_209[t+1] + Bi_213[t+1]


time_grid = np.arange(0, len(Pb_209), dtype = int)

plt.plot(time_grid, Pb_209, time_grid, Bi_209, time_grid, Tl_209, time_grid, Bi_213 ,time_grid, N_total) 
plt.legend(['Pb_209', 'Bi_209', 'Ti_209', 'Bi_213', '$N_{total}$'])
plt.xlabel('time $t$, seconds')
plt.ylabel('The amount of nuclei of each type')
plt.grid()
plt.show()