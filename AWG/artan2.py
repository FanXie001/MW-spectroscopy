import numpy as np
from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy.fftpack import fftfreq
import matplotlib.pyplot as plt
import math

srate = 25.0E9
freq_mult = 1.0E6
def read(filename):
    fid = np.loadtxt(filename)
    time = fid[:,0]
    intensity = fid[:,1]
    points = np.shape(fid)[0]
    return time,intensity,points

def ft(input,start_freq,end_freq):
    fd = fft(input)
    ft = np.column_stack((fftfreq(len(input), 1.00 / srate) / (freq_mult), abs(fd),np.real(fd), np.imag(fd)))
    ft = ft[(start_freq <= ft[:, 0]) & (ft[:, 0] <= end_freq)]
    return ft

def ift(input):
    fd = ifft(input)
    return np.array(fd)

def half(x):
    half = len(x) // 2
    return x[:half]

def inverse(x,y):
    r_x = x + 2.10007600e-005
    r_y = y[::-1]
    n_y = np.append(r_y, y)
    n_x = np.append(r_x, x)
    return n_y

def window(x):
    return np.kaiser(len(x), 9.5)*x

def cut_fixed_width(spec,cut_list,width):

    spec = np.copy(spec)
    mask = np.zeros(spec.shape[0])

    res = spec[1,0]- spec[0,0] # Spectrum spacing

    for val in cut_list:

        low_bound = val - width/2
        high_bound = val + width/2

        idx_low = int(np.floor((low_bound-spec[0, 0])/res) + 1)
        idx_high = int(np.floor((high_bound-spec[0, 0])/res) - 1)

        mask[idx_low:idx_high] = 1

    spec[:, 1] *=  mask
    return spec

def delete(input,output):
    with open(input,'r') as r:
        lines=r.readlines()
    with open(output,'w') as w:
        for l in lines:
           if '"' not in l:
              w.write(l)
    ss=np.loadtxt(output)
    return ss

def wave(srate,frequncy,phase,length):
    x = np.linspace(0.0, length, srate*length)
    y = np.sin(frequncy*2.0 * np.pi * x + phase)
    return y

def artan2(re,im):
    phase = []
    for i in range(0,len(re)):
        phase_x = (math.atan2(im[i], re[i]))/20
        phase.append(phase_x)
    return phase

def pl(fid):
    ft_x = ft(fid,100,12000)
    freq = ft_x [:,0]
    intensity = ft_x [:,1]
    re = ft_x [:,2]
    im = ft_x [:,3]
    phase = artan2(re,im)
    plt.plot(freq,intensity)
    #plt.plot(freq,re)
    #plt.plot(freq,im)
    #plt.plot(freq,phase)
'''
fid =wave(srate,5E9,0.0,20E-6)
pl(fid)
fid =wave(srate,5E9,3.14/2,20E-6)
pl(fid)

fid =wave(srate,5E9,3.14,20E-6)
pl(fid)
fid =wave(srate,5E9,3.14*3/2,20E-6)
pl(fid)
'''

fid = "ass_h2o_4m_fid.txt"
fid = delete(fid,fid)
pl(fid[:,1])


'''
fid = "con_1.txt"
fid = delete(fid,fid)
pl(fid[:,1])

fid = "con_2.txt"
fid = delete(fid,fid)
pl(fid[:,1])

fid = "con_3.txt"
fid = delete(fid,fid)
pl(fid[:,1])

fid = "con_4.txt"
fid = delete(fid,fid)
pl(fid[:,1])

fid = "con_5.txt"
fid = delete(fid,fid)
pl(fid[:,1])
'''
'''
fid = "0.05v.txt"
fid = delete(fid,fid)
pl(fid[:,1])

fid = "0.05_T.txt"
fid = delete(fid,fid)
pl(fid[:,1])

fid = "0.1v.txt"
fid = delete(fid,fid)
pl(fid[:,1])

fid = "0.1_T.txt"
fid = delete(fid,fid)
pl(fid[:,1])
'''



#plt.xlim(4115,4125)
#plt.xlim(4863.0,4863.3)
#plt.xlim(4999.8,5000.2)
plt.show()