from pylab import *
from numpy.fft import rfft, irfft
import numpy as np

def plot_data(times, data, instrument=''):
    plot(times, data, label='original sound signal of '+instrument)
    plt.grid()
    plt.xlabel('$t$, s')
    plt.ylabel(instrument + ' signal')
    plt.show()

def plot_transform(frequencies, FT, instrument=''):
    plot(frequencies, FT, label='Fourier transform of a ' + instrument +' signal')
    plt.grid()
    plt.xlabel('$f$, Hz')
    plt.ylabel('Absolute values of Fourier coefficients for ' + instrument)
    plt.show()

def check_lengths(data_piano, data_trumpet):
    if len(data_trumpet)!=len(data_piano):
        print('Data sets have different length! The code might not work properly!')
        exit()



#set a sample rate of a recording

sample_rate=44_100

#upload data from two files

data_piano=np.loadtxt('piano.txt')
data_trumpet=np.loadtxt('trumpet.txt')
check_lengths(data_piano, data_trumpet)               #run a check that both datasets have the same length

data_points=len(data_piano)


#convert a sample grid into a time grid and plot data

times=np.arange(0, len(data_piano)/44_100, 1/44_100, dtype=np.float64)

plot_data(times, data_piano, instrument='piano')
plot_data(times, data_trumpet, instrument='trumpet')

#Take FFT of two data sets

FT_piano=rfft(data_piano)
FT_trumpet=rfft(data_trumpet)

#convert a sample grid in fourier space into a frequency grid and plot data

f=np.arange(0, 10_000, 44_100/len(data_piano))[:10_000]                 #remember to cut-off everything after 10_000 frequencies

plot_transform(f, np.abs(FT_piano[:10_000]), instrument='piano')
plot_transform(f, np.abs(FT_trumpet[:10_000]), instrument='trumpet')

#Calculate the frequency with the maximum absolute value of a Fourier coeff

note_piano = 44_100*np.argmax(np.abs(FT_piano))/len(data_piano)
note_trumpet = 44_100*np.argmax(np.abs(FT_trumpet))/len(data_trumpet)

print('Frequency of the loudest tone for piano is {} Hz'.format(note_piano))               #and print them out
print('Frequency of the loudest tone for trumpet is {} Hz'.format(note_trumpet))