from scipy.optimize import curve_fit, leastsq
from scipy.signal import find_peaks
from scipy.special import wofz
import numpy as np
import matplotlib.pyplot as plt
import math







def voigt_shift(x,amp,cen,alpha,gamma,shift):
    sigma = alpha / np.sqrt(2 * np.log(2))
    return amp * np.real(wofz(((x-cen) + 1j*gamma)/sigma/np.sqrt(2))) / sigma / np.sqrt(2*np.pi) + shift




peakA = lambda x : voigt_shift(x,160,5,1,.1,0)
peakB = lambda x : voigt_shift(x,360,2,1,.1,0)
N = 30
noise = lambda x : np.random.random() * N - N/2

F = lambda x : noise(x) + peakA(x) + peakB(x)

X = np.linspace(-5,10,101)
Y = [F(x) for x in X]
Y_A = [peakA(x) for x in X]
Y_B = [peakB(x) for x in X]


plt.plot(X,Y,color="black",marker=".",label="data")
#plt.plot(X,Y_A)
#plt.plot(X,Y_B)
plt.ylim(-30,170)
plt.legend()
plt.show()

cen1 = X[np.argmax(Y)]
params1,_ = curve_fit(voigt_shift,X,Y,p0=[360,cen1,1,.1,0],bounds=([300,cen1-5,.01,.01,0],[400,cen1+5,2,1,1]))
guess1 = lambda x : voigt_shift(x,*params1)
Y_G1 = [guess1(x) for x in X]

Y_D1 = [Y[i] - Y_G1[i] for i in range(len(Y))]

plt.plot(X,Y,color="black",marker=".",label="data")
plt.plot(X,Y_G1,"b--",label="guessed curve 1")
plt.ylim(-30,170)
plt.legend()
plt.show()

plt.plot(X,Y_D1,color="black",marker=".",label="data")
plt.ylim(-30,170)
plt.legend()
plt.show()

cen2 = X[np.argmax(Y_D1)]
params2,_ = curve_fit(voigt_shift,X,Y_D1,p0=[160,cen2,1,.1,0],bounds=([100,cen2-5,0,0,0],[200,cen2+5,2,1,1]))
guess2 = lambda x : voigt_shift(x,*params2)
Y_G2 = [guess2(x) for x in X]

Y_D2 = [Y[i] - Y_G1[i] for i in range(len(Y))]

plt.plot(X,Y_D1,color="black",marker=".",label="data")
plt.plot(X,Y_G2,"g--",label="guessed curve 2")
plt.ylim(-30,170)
plt.legend()
plt.show()

#optimize both
voigt_double = lambda x,a1,a2,a3,a4,a5,b1,b2,b3,b4,b5: voigt_shift(x,a1,a2,a3,a4,0) + voigt_shift(x,b1,b2,b3,b4,0)
params_both,_ = curve_fit(voigt_double,X,Y,p0=[*params1,*params2])
opt_both = lambda x : voigt_double(x,*params_both)
opt1 = lambda x : voigt_shift(x,*params_both[0:5])
opt2 = lambda x : voigt_shift(x,*params_both[5:])

Y_OB = [opt_both(x) for x in X]
Y_O1 = [opt1(x) for x in X]
Y_O2 = [opt2(x) for x in X]

Y_DB = [Y[i] - Y_OB[i] - 10 for i in range(len(Y))]

plt.plot(X,Y,color="black",marker=".",label="data")
plt.plot(X,Y_O1,"b:",label="optimized curve 1")
plt.plot(X,Y_O2,"g:",label="optimized curve 2")
plt.plot(X,Y_OB,"r",label="optimized fit")
#plt.plot(X,Y_A)
#plt.plot(X,Y_B)
#plt.plot(X,Y_DB,"r",lebel="residual")
#plt.plot(X,Y_OB)
plt.plot(X,Y_DB,"orange",label="residual")
plt.ylim(-30,170)
plt.legend()
plt.show()
