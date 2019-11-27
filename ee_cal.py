from matplotlib import pyplot as plt
import numpy as np
import re
import cmath
import statistics


def lp(filename, e_threshold):
    spec = np.loadtxt(filename) # Two-column spectrum file
    peaks = []
    for i in range(0, len(spec)-1):
        if spec[i,1] > e_threshold and spec[i,1] > spec[(i-1),1] and spec[i,1]  > spec[(i+1),1] and spec[i,0] > 2000 and spec[i,0] < 6000:
            peaks.append([spec[i,0],spec[i,1]])
    return np.array(peaks)

def assignment(linelist_t,linelist_e, t_threshold):

    frequency=[]
    intensity=[]
    with open(linelist_t) as origin_file:
        for line in origin_file:
            matchObj = re.match(r'(.*) Ground (.*?) .*', line, re.M | re.I)
            if matchObj:
                string = matchObj.group()
                #print(string.split()[0])
                result = re.findall(r"\d+\.?\d*", string)
                #print(result)
                frequency.append(float(result[4]))
                intensity.append(float(string.split()[10]))
    f = np.column_stack((np.array(frequency), np.array(intensity)))
    spec = np.array(f) # Two-column spectrum file
    linelist_t = spec
    e = []
    t = []
    r = []
    for i in range(0, len(linelist_t) - 1):
        for j in range(0, len(linelist_e) - 1):
            if linelist_e[j,0] - 0.025 < linelist_t[i,0] and linelist_t[i,0] < linelist_e[j,0] + 0.025 and linelist_t[i,1] > t_threshold:
                e.append(linelist_e[j])
                t.append(linelist_t[i])
                r.append(linelist_e[j,1]/linelist_t[i,1])
    f = np.column_stack((np.array(e),np.array(t)))
    f1 = np.column_stack((np.array(f),np.array(r)))
    return np.array(f1), np.array(e)

def production (expspectrum,tlist,exp_thred,theory_thred):
    explinelist = lp(expspectrum,exp_thred)
    ratio, exp = assignment(tlist,explinelist,theory_thred)
    print(np.shape(exp))
    return exp

def ee_cal(racemic,enantiomer,max_in,noise_l):
    re = []
    en = []
    for i in range(0, len(racemic) - 1):
        for j in range(0, len(enantiomer) - 1):
            if enantiomer[j,0] - 0.025 <  racemic[i,0] and racemic[i,0] < enantiomer[j,0] + 0.025:
                en.append(enantiomer[j])
                re.append(racemic[i])
    re = np.array(re)
    en = np.array(en)
    print(np.shape(re))
    print(np.shape(en))
    n_factor = re[:,1]/max_in
    R = (noise_l*n_factor)/en[:,1]
    ee = ((1-R)/(1+R))**0.5
    mean = statistics.mean(ee)
    st = statistics.stdev(ee)
    print("Mean is % s" %(round(mean,4)),"Standard Deviation is % s" %(round(st,4)))
    plt.title(("Mean is % s" %(round(mean,4)),"Standard Deviation is % s" %(round(st,4))))
    plt.xlabel('Freq')
    plt.ylabel('ee')
    plt.scatter(re[:,0],ee,label= "number of transitions % s" %(len(ee)))
    plt.legend(loc='best')
    #plt.ylim(0,1)
    plt.show()




a = production('R-S THFANov13_40psiNe_105C_1224k_FT.txt','RRT-3-3.out',0.0005,0.001)
b = production('Nov27_purified_R_1365k_FT.txt','RRT-3-3.out',0.0005,0.0005)
ee_cal(a,b,0.0015,0.0001)

a = production('R-S THFANov13_40psiNe_105C_1224k_FT.txt','RRK-carbony-1-3.out',0.0005,0.001)
b = production('Nov27_purified_R_1365k_FT.txt','RRK-carbony-1-3.out',0.0005,0.0005)
ee_cal(a,b,0.0015,0.0001)











