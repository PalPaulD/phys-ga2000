import numpy as np
import matplotlib.pyplot as plt

step = 0.01

def gauss(x, m = 0, sigma = 3):
    return 1/np.sqrt(2.*np.pi*sigma**2)*np.exp(-(x-m)**2/(2*sigma**2))

x = np.arange(-10, 10+step, step)
y = gauss(x)

plt.plot(x,y)
plt.xlabel('x')
plt.xlim(-10, 10)
plt.ylabel('y')
plt.ylim(0, 0.14)
plt.title('Normal distribution N(0, 9)')
plt.grid()
plt.legend(['The gaussian'])
plt.savefig("plot_gaussian.png")
plt.clf()

