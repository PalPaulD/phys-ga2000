import numpy as np
from tqdm import tqdm
import hw2_proper_problem2 as calc

M = 100

fit_selfmade_a = np.zeros(M, dtype = float)
fit_selfmade_n = np.zeros(M, dtype = float)
fit_npdot_a = np.zeros(M, dtype = float)
fit_npdot_n = np.zeros(M, dtype = float)

for i in tqdm(range(0, len(fit_npdot_a))):
    calc_results = calc.fitting()
    fit_selfmade_a[i] = calc_results[2][0]
    fit_selfmade_n[i] = calc_results[2][1]
    fit_npdot_a[i] = calc_results[3][0]
    fit_npdot_n[i] = calc_results[3][1]

print('Average estimate of a for hand-made dot product: {}'.format(fit_selfmade_a.sum()/len(fit_selfmade_a)))
print('Average estimate of n for hand-made dot product: {}'.format(fit_selfmade_n.sum()/len(fit_selfmade_n)))
print('Average estimate of a for np.dot: {}'.format(fit_npdot_a.sum()/len(fit_npdot_a)))
print('Average estimate of n for np.dot: {}'.format(fit_npdot_n.sum()/len(fit_npdot_n)))