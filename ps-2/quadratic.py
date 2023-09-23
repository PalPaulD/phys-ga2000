import numpy as np


def quadratic(a,b,c):
    if np.sign(b)>=0:
        return (2*c/(-b - np.sqrt(b*b - 4*a*c)), (-b - np.sqrt(b*b - 4*a*c))/(2*a))
    else:
        return ((-b + np.sqrt(b*b - 4*a*c))/(2*a), 2*c/(-b + np.sqrt(b*b - 4*a*c)))


def solve():
    a,b,c = map(np.float64, input('Please, input values of a,b,c separated by spaces \n').split())
    x1 = (-b + np.sqrt(b*b - 4*a*c))/(2*a)
    x2 = (-b - np.sqrt(b*b - 4*a*c))/(2*a)
    print('Roots using standard formula -b^2+-sqrt(D)/2a: x1 = {}, x2 = {}'.format(x1,x2))

    x1 = 2*c/(-b - np.sqrt(b*b - 4*a*c))
    x2 = 2*c/(-b + np.sqrt(b*b - 4*a*c))
    print('Roots using modified formula 2c/-b-+sqrt(D): x1 = {}, x2 = {}'.format(x1,x2))

#solve()