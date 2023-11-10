import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy import optimize

def f(x, p):
    return p[0]*x + 0.4*np.sin(p[1]*x)

def Covariance(hess_inv, resVariance):
    return hess_inv * resVariance

def error(hess_inv, resVariance):
    covariance = Covariance(hess_inv, resVariance)
    return np.sqrt( np.diag( covariance ))

def p(x, beta_0, beta_1):
    return 1/(1+np.exp(-(beta_0+beta_1*x)))

def log_likelihood(beta_0, beta_1, xs, ys):
    #l_list = [ys*np.log(p(x, beta_0, beta_1)/(1-p(x, beta_0, beta_1))) + np.log(1-p(x, beta_0, beta_1)) for x in xs]
    l_list = []
    for i in range(len(xs)):
        u = ys[i]*np.log(p(xs[i], beta_0, beta_1)/(1-p(xs[i], beta_0, beta_1))) + np.log(1-p(xs[i], beta_0, beta_1))
        l_list.append(u)
    ll = -np.sum(np.array(l_list))
    return ll # return log likelihood

    

data = pandas.read_csv('survey.csv')
xs = data['age'].to_numpy()
ys = data['recognized_it'].to_numpy()
x_sort = np.argsort(xs)
xs = xs[x_sort]
ys = ys[x_sort]

beta = np.array([-0.01, 0.01])

result = optimize.minimize(lambda beta, xs, ys: log_likelihood(beta[0], beta[1], xs, ys), beta,  args=(xs, ys))
hess_inv = result.hess_inv # inverse of hessian matrix
var = result.fun/(len(ys)-len(beta)) 
dFit = error( hess_inv,  var)
print('Optimal parameters and error:\n\tp: ' , result.x, '\n\tdp: ', dFit)
print('Covariance matrix of optimal parameters:\n\tC: ' , Covariance( hess_inv,  var))

grid = np.arange(0, 100, 1, dtype = float)
plt.plot(grid, p(grid, result.x[0], result.x[1]), 'o')
plt.grid()
plt.title(r'$\beta_{0} = -5.62,  \beta_{1} = 0.11$')
#plt.show()
plt.savefig('plot.png')
