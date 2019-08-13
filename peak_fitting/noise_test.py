import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz

X = np.linspace(-10,10,101)



def profile(x,amp,cen,alpha,gamma,shift):
    sigma = alpha / np.sqrt(2 * np.log(2))
    return amp * np.real(wofz(((x-cen) + 1j*gamma)/sigma/np.sqrt(2))) / sigma / np.sqrt(2*np.pi) + shift

def plot_func(func,*params):
    Y = [func(x) for x in X]
    plt.plot(X,Y,*params)



def is_noise(data):
    sd = np.std(data)
    mean = np.mean(data)
    dev = np.where(np.abs(data-mean)/sd > 2.5)[0]
    #more than 2 points are statistically significant
    if len(dev) > 2:
        return False
    return True

N = 50
amp = 30
Noise = np.array([np.random.random() * N - N/2 for x in X])

for amp in range(0,500,5):
    print(amp)
    plt.cla()

    F = lambda x : profile(x,amp,0,1,1,0)
    Y = np.array([F(x) for x in X]) + Noise
    isnoise = is_noise(Y)
    if isnoise:
        plt.plot(X,Y,'b')
    else:
        plt.plot(X,Y,'r')
    plt.draw()
    plt.pause(.1)
