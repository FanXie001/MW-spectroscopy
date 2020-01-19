from matplotlib import pyplot as plt
import numpy as np
import re

def Theoryspectrum(linelist_t,thred):
    frequency = []
    intensity = []
    with open(linelist_t) as origin_file:
        for line in origin_file:
            matchObj = re.match(r'(.*) Ground (.*?) .*', line, re.M | re.I)
            if matchObj:
                string = matchObj.group()
                # print(string.split()[0])
                result = re.findall(r"\d+\.?\d*", string)
                # print(result)
                frequency.append(float(result[4]))
                intensity.append(float(string.split()[10]))
    f = np.column_stack((np.array(frequency), np.array(intensity)))
    spec = np.array(f)  # Two-column spectrum file
    linelist_t = spec
    peaks = []
    for i in range(0, len(linelist_t) - 1):
        if linelist_t[i,0] > 2000 and linelist_t[i,0] < 6000 and linelist_t[i,1] > thred:
            peaks.append([linelist_t[i, 0], linelist_t[i, 1]])

    peaks = np.array(peaks)
    band = []

    for i in range(0, len(peaks)-1):
        band.append([(peaks[i, 0] - 0.0001), 0])
        band.append([peaks[i, 0], peaks[i, 1]*-1])
        band.append([(peaks[i, 0] + 0.0001), 0])

    band = np.array(band)
    return band


def plot_spectrum (exp, fit):
    x = np.loadtxt(exp)
    spectrum1 = Theoryspectrum(fit,0.00001)
    plt.plot(x[:,0],1.0*x[:,1],linewidth=1.2,color='#000000',label='Exp')
    plt.plot(spectrum1[:,0],spectrum1[:,1]*7,linewidth=1.2,color='#ff0101',label='I')

plot_spectrum('30psi_Ne_667k_FT_cut hfip monomer dimer trimer.txt','hfip with 1,4 dioxide.out')

plt.xlabel('Frequency (MHz)')
plt.ylabel('Intensity (a.u.)')
plt.xlim(2300,6000)
#plt.ylim(-0.003,0.004)
plt.legend(loc='best')
plt.show()








