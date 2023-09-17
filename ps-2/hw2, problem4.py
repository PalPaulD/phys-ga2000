import numpy as np


def quadratic(a,b,c):
    return (2*c/(-b - np.sqrt(b*b - 4*a*c)), (-1000 - np.sqrt(b*b - 4*a*c))/(2*a))


a,b,c = map(np.float32, input('Please, input values of a,b,c separated by spaces \n').split())

x1 = (-1000 + np.sqrt(b*b - 4*a*c))/(2*a)
x2 = (-1000 - np.sqrt(b*b - 4*a*c))/(2*a)
print('Roots using standard formula -b^2+-sqrt(D)/2a: x1 = {}, x2 = {}'.format(x1,x2))

x1 = 2*c/(-b - np.sqrt(b*b - 4*a*c))
x2 = 2*c/(-b + np.sqrt(b*b - 4*a*c))
print('Roots using modified formula 2c/-b-+sqrt(D): x1 = {}, x2 = {}'.format(x1,x2))

x1, x2 = quadratic(a, b, c)
print(type(x1), type(x2))