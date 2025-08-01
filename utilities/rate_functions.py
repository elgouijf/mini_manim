from math import *
import numpy as np

def smooth(t):
    # ease-in-out rate function
    if t > 1 or t < 0:
        return ValueError("interpolation parameter is in [0,1]")
    return 3*(t**2) - 2*(t**3)

def sinusoidal(t):
    # ease-in-out rate function with a oscillating behavior (for smoother curves)
    if t > 1 or t < 0:
        return ValueError("interpolation parameter is in [0,1]")
    return 0.5*(1 - np.cos(pi*t))

def ease_out(t):
    # starts fast and slows down towards the end
    if t > 1 or t < 0:
        return ValueError("interpolation parameter is in [0,1]")
    return 1 - (1 - t)**2

def ease_in(t):
    # starts slow and accelerates towards the end
    if t > 1 or t < 0:
        return ValueError("interpolation parameter is in [0,1]")
    return t**3

def ease_out(t):
    # starts fast and slows down towards the end
    if t > 1 or t < 0:
        return ValueError("interpolation parameter is in [0,1]")
    return 1 - (1 - t)**3

def wiggle(t, w): # w*pi is the frequency of our wiggling shape
    # wiggles along the way, could give some cartoonish and springy shape
    return t + 0.1*np.sin(w*pi*t)

def there_and_back(t):
    # used to creat a bouncing effect
    if t > 1 or t < 0:
        return ValueError("interpolation parameter is in [0,1]")
    elif t < 0.5:
        return 2*t
    else:
        return 1 - 2*t
    

def exponential_decay(t, c):
    # used for smooth transitions as well but it slows down faster
    # the smaller c is the more time it'll take to reach 1
    return 1 - np.exp(- t/c)