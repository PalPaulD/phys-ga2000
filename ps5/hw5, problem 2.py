import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq 
import scipy.signal as signal

def is_float(string):                                          #technical function for uploading data
    """ True if given string is float else False"""
    try:
        return float(string)
    except ValueError:
        return False

def plot_results(fit, name, save = False, plot_fit = False, print_period = False):    #plot signals + a fit (if required) and save in a .png fiel with a given name
    plt.plot(fit.time, fit.signal, '.', label = 'data', color = 'blue')
    if plot_fit==True:
        plt.plot(fit.time, fit.ym, '.', label = 'model', color = 'red')
    if print_period==True:
        plt.title('The periodicity of the fit is T/' + str(fit.period))
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('signal')
    plt.legend()
    if save==True:
        plt.savefig(name)
    else:
        plt.show()
    plt.close()

def plot_residuals(fit, description, name, save = False, cond_number = True):
    plt.plot(fit.time, fit.signal - fit.ym, '.', label = description, color = 'black')
    plt.grid()
    if cond_number==True:
        plt.title('The condition number is ' + str(fit.condition_number))
    plt.xlabel('time')
    plt.ylabel('residuals')
    plt.legend()
    if save==True:
        plt.savefig(name)
    else:
        plt.show()
    plt.close()

def invert_w(w):                            #1./w does not always work, this function does the proper "inversion" of w in SVD (needed only for harmonic fit)
    winv = np.zeros(len(w), dtype = float)
    for i in range(0, len(w)):
        if np.abs(w[i])>0:
            winv[i] = 1/w[i]
    return winv
    

class data_fitter():                        #the class takes care of working with data: uploading, plotting, fitting the curve with several methods
    
    def __init__(self):                     #upload data and pack it in time and signal attributes
        data = []
        with open('signal.dat', 'r') as f:
            d = f.readlines()
            for i in d:
                k = i.rstrip().split('|')
                for i in k:
                    if is_float(i):
                        data.append(float(i)) 
        data = np.array(data, dtype='float')
        self.time = data[::2]
        self.signal = data[1::2]

    def rescale(self, back = False):          #normalize time and signal to 1 for a more efficient fit
        if back==False:
            self.max_time = max(self.time)
            self.max_signal = max(self.signal)
            self.time = self.time/self.max_time
            self.signal = self.signal/self.max_signal
        else:
            self.time = self.time*self.max_time
            self.signal = self.signal*self.max_signal
            self.ym = self.ym*self.max_signal


    def SVD_3rd_order(self):                   #do a 3rd order SVD fit
        A = np.zeros((len(self.time), 4))
        A[:, 0] = 1.
        A[:, 1] = self.time
        A[:, 2] = self.time**2
        A[:, 3] = self.time**3
        (u, w, vt) = np.linalg.svd(A, full_matrices=False)
        ainv = vt.transpose().dot(np.diag(1./w)).dot(u.transpose())
        c = ainv.dot(self.signal) 
        self.ym = A.dot(c)


    def SVD_Nth_order(self, N):                #do an Nth order SVD fit (N is defined in the code below)
        A = np.zeros((len(self.time), N+1))
        for i in range(0, N+1):
            A[:, i] = np.power(self.time, i)
        (u, w, vt) = np.linalg.svd(A, full_matrices=False)
        ainv = vt.transpose().dot(np.diag(1./w)).dot(u.transpose())
        c = ainv.dot(self.signal) 
        self.ym = A.dot(c)
        self.condition_number = np.max(w)/np.min(w)       #calculate the conditional number (can be inf, numpy is smart enough not to throw an error)

    def harmonic_fit(self, N, print_period = False):             #do a fit with a function f(x) = a0 + sum_i=1^{N} a_i sin(i pi T/N x) + b_i cos(i pi T/N x) 
        T = (np.max(self.time) - np.min(self.time))                 #time length of the experiment
        A = np.zeros((len(self.time), 2*N+1))
        A[:, 0] = 1.                                                #a0 term 
        for i in range(1, N+1):
            A[:, 2*i - 1] = np.cos(2*np.pi*self.time/(T/i))       #b_i
            A[:, 2*i] = np.sin(2*np.pi*self.time/(T/i))           #a_i
        (u, w, vt) = np.linalg.svd(A, full_matrices=False)
        ainv = vt.transpose().dot(np.diag(invert_w(w))).dot(u.transpose())
        c = ainv.dot(self.signal)
        self.ym = A.dot(c)
        self.condition_number = np.max(w)/np.min(w)
        self.period = np.argmax(np.abs(c))//2


results = data_fitter()              #initialize fitter class and upload data
plot_results(results,                                   #plot raw data
             name = 'hw5, problem 2, raw_data.png', 
             save = True, plot_fit = False)            

#fitting the 3rd order polynomial with SVD
results.rescale()                   #Rescale data to [0,1]^2 box
results.SVD_3rd_order()             #SVD with a 3rd order polynomial  
results.rescale(back = True)        #rescale data back from the [0,1]^2 box
plot_results(results, 
             name = 'hw5, problem 2, SVD__order_fit.png', 
             save = True, plot_fit = True)
plot_residuals(results, name = 'residuals 3rd order poly fit.png', 
               description = 'residuals for the 3rd order polynomial fit', 
               save = True, cond_number = False)

#fitting the Nth order polynomial with SVD
N = 20                              #set the order of fit
results.rescale()                   #Rescale data to [0,1]^2 box
results.SVD_Nth_order(N)            #SVD with a Nth order polynomial  
results.rescale(back = True)        #rescale data back from the [0,1]^2 box
plot_results(results, 
             name = 'hw5, problem 2, SVD_' + str(N) + '_order_fit.png', 
             save = True, plot_fit = True)
plot_residuals(results, 
               name = 'residuals ' + str(N) + 'rd ord poly fit.png', 
               description = 'residuals for the ' + str(N) + 'th order polynomial fit', 
               save = True, cond_number = True)

#fitting the harmonic function
N = 15                                                      #set the order of fit
results.rescale()                                           #Rescale data to [0,1]^2 box
results.harmonic_fit(N, print_period = True)                #fit a harmonic function
results.rescale(back = True)                                #rescale back from the [0,1]^2 box
plot_results(results, 
             name = 'hw5, problem 2, harmonic_fit_' + str(N) + 'order.png', 
             save = True, plot_fit = True, print_period = True)
plot_residuals(results, name = 'residuals ' + str(N) + 'th order harmonic fit.png', 
               description = 'residuals for the ' + str(N) + 'th order harmonic fit', 
               save = True, cond_number = True)