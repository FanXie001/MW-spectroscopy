import numpy as np
from random import seed
from random import random
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from decimal import *
import scipy
import math
from scipy.optimize import curve_fit
import scipy.interpolate
getcontext().prec = 29

hbar=1
m1 = (2.01568/2)*5.500
m_m = 150
m2 = m_m - m1
mu = m1*m2/(m1+m2)
mu_me = 1822.89  #  atomic mass/electron mass
m = mu*mu_me
N = 3000
k_2 = 0.24
f_1 = 1.0
f_2 = 1.0
deta_E =[]
B = []
for i in range(200):
    f_2 = 1.0
    f_2 = f_2*(0.9)+(i*0.001)
    print(f_2)
    ### Potential well in hatree/borh: for testing purpose
    def V():
        V=np.zeros(N)
        for i in range(N):
            if x[i] > 0:
                V[i] = (x[i]-1)**2
            elif x[i] < 0:
                V[i] = (x[i]+1)**2
        return np.array(V)
    ###

    ### pick up PES and interpolate it
    def pes_pickup(filename,N,k_2,f_1,f_2):
        spec = np.loadtxt(filename)
        x = spec[:,0]*f_1
        y = spec[:,1]*f_2
        a = x[(len(x) - 1)]
        zero = y[(len(x) - 1)]
        def V2(a, n, zero,k_2):
            x = np.linspace(-50, 50, n)
            V = np.zeros(n)
            x_1 = []
            y_1 = []
            x_2 = []
            y_2 = []
            for i in range(n):
                if x[i] > a:
                    V[i] = (k_2)/2*(x[i] - a) ** 2 + zero
                    x_1.append(x[i])
                    y_1.append(V[i])
                elif x[i] < -a:
                    V[i] = (k_2/2)*(x[i] + a) ** 2 + zero
                    x_2.append(x[i])
                    y_2.append(V[i])
            return np.array(x_1), np.array(y_1), np.array(x_2), np.array(y_2)
        x_1,y_1,x_2,y_2 = V2(a,3000,zero,k_2)
        x_3 = np.append((np.append(x_2,x)),x_1)
        y_3 = np.append((np.append(y_2, y)), y_1)
        f = scipy.interpolate.interp1d(x_3, y_3)
        y_4 = f(np.linspace(x_3[0], x_3[(len(x_3) - 1)], N))
        x_4 = np.linspace(x_3[0], x_3[(len(x_3) - 1)], N)
        h = x_4[1] - x_4[0]
        return np.array(x_4),np.array(y_4),h
    ###

    ### intended to fit PES with ahharmonic potential well, it needs a huge data set. fked up.
    def ah_harmonic(x,a,b,c,g):
        #y = (1/2)*a*x**2 + (1/math.factorial(3))*b*x**3 + (1/math.factorial(4))*c*x**4
        #+ (1/math.factorial(5))*d*x**5 + (1/math.factorial(6))*e*x**6
        #+ (1/math.factorial(7))*f*x**7
        return (1/2)*a*x**2 + (1/math.factorial(3))*b*x**3 + (1/math.factorial(4))*c*x**4 +g

    def fit():
        x,y = pes_pickup("pes_new.txt",3000)
        a,b = curve_fit(ah_harmonic, x, y)
        plt.plot(x, ah_harmonic(x, *a), 'r-',label='fit')
    ###

    ###
    def basis_set():
        def V1():
            V1 = np.zeros(N)
            for i in range(N):
                V1[i] = 1 / 2 * (x[i]) ** 2
            return np.array(V1)
        Mdd = 1. / (h * h) * (np.diag(np.ones(N - 1), -1) - 2 * np.diag(np.ones(N), 0) + np.diag(np.ones(N - 1), 1))
        H = -(hbar * hbar) / (2.0 * m) * Mdd + np.diag(V1())
        E, psiT = scipy.linalg.eigh(H)  # This computes the eigen values and eigenvectors
        #psi = np.transpose(psiT)
        return psiT
    ###

    ###
    def ortho_normal_check(basis_set):
        basis_set = np.transpose(basis_set)
        print(integrate.simps((basis_set[0]/np.sqrt(h))**2, x, dx=1, axis=-1, even='avg'))
        a = np.matmul(basis_set,np.linalg.inv(basis_set))
        print(np.shape(basis_set))
        print(np.shape(np.linalg.inv(basis_set)))
        print(a)
    ###

    ###
    def solver():
        Mdd = 1. / (h * h) * (np.diag(np.ones(N - 1), -1) - 2 * np.diag(np.ones(N), 0) + np.diag(np.ones(N - 1), 1))
        H = -(hbar * hbar) / (2.0 * m) * Mdd + np.diag(V)
        n_H = np.matmul(np.linalg.inv(basis_set()),H)
        n_H = np.matmul(n_H,basis_set())
        E, psiT = scipy.linalg.eigh(n_H)  # This computes the eigen values and eigenvectors
        #psi = np.transpose(psiT)
        wave = np.matmul(basis_set(),psiT)
        return np.transpose(wave),E
    ###

    ###
    x,V,h = pes_pickup("pes_new.txt",N,k_2,f_1,f_2)
    V = V*1
    wave,E= solver()
    deta_E.append((E[1]-E[0])*6579681360.732768)
    B.append(f_2)
np.savetxt("E_test5.py.txt", deta_E, delimiter=" ")
np.savetxt("B_test5.py.txt", B, delimiter=" ")
'''
    fig = plt.figure(figsize=(10,7))
    plt.xlabel("x")
    plt.plot(x,V,color="Gray",label="V(x)")
    ###
    for i in range(2):
        if True:
            if wave[i][int(N/2+10)] < 0:   # Flip the wavefunctions if it is negative at large x, so plots are more consistent.
                plt.plot(x,((-wave[i]/np.sqrt(h))/100+E[i]),label="$E_{}$={:>8.16f}".format(i,E[i]))
            else:
                plt.plot(x,((wave[i]/np.sqrt(h))/100+E[i]),label="$E_{}$={:>8.16f}".format(i,E[i]))

                plt.title((E[1]-E[0])*6579681360.732768)
    print((E[1]-E[0])*6579681360.732768)

    #plt.ylim((0,8))
    #plt.xlim((-4,4))
    plt.legend(loc="best")
    #plt.savefig("Conformation_WaveFunctions.pdf")
    #np.savetxt("wavefunctions.csv", wave, delimiter=",")
    #plt.show()
'''
