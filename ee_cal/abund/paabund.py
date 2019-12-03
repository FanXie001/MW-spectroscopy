from matplotlib import pyplot as plt
import numpy as np
import cmath


def lp(filename, e_threshold):
    spec = np.loadtxt(filename) # Two-column spectrum file
    peaks = []
    for i in range(0, len(spec)-1):
        if spec[i,1] > e_threshold and spec[i,1] > spec[(i-1),1] and spec[i,1]  > spec[(i+1),1] and spec[i,0] > 3000 and spec[i,0] < 5500:
            peaks.append([spec[i,0],spec[i,1]])
    return np.array(peaks)

def assignment(linelist_t,linelist_e, t_threshold):
    linelist_t = np.loadtxt(linelist_t)
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
    return np.array(f1)

def production (expspectrum,tlist):
    explinelist = lp(expspectrum,0.0005)
    ratio = assignment(tlist,explinelist,0.001)
    #print(ratio)
    #print(ratio[:,4])
    a = 0
    a1 = 0
    for i in range(0, len(ratio) - 1):
         a = a + ratio[i,4]
    b = a/len(ratio)
    for i in range(0, len(ratio) - 1):
        a1 = a1 + ((b - ratio[i,4])**2)**0.5/b
    b1 = a1 / len(ratio)
    print(b, b1,len(ratio))

    plt.xlim(2000,6000)
    #plt.ylim(0,100)
    plt.scatter(ratio[:,0],ratio[:,4])
    #plt.show()
    return b


production('Nov27_purified_R_1365k_FT cut 3m.txt','RRK-carbony+-1-3.out')
production('Nov27_purified_R_1365k_FT cut 3m.txt','RRK-carbony-1-3.out')
production('Nov27_purified_R_1365k_FT cut 3m.txt','RRK-ether-1+-3.out')
production('Nov27_purified_R_1365k_FT cut 3m.txt','RRT-3-3.out')




production('R-S THFANov13_40psiNe_105C_1224k_FT cut 3m.txt','RRK-carbony+-1-3.out')
production('R-S THFANov13_40psiNe_105C_1224k_FT cut 3m.txt','RRK-carbony-1-3.out')
production('R-S THFANov13_40psiNe_105C_1224k_FT cut 3m.txt','RRK-ether-1+-3.out')
production('R-S THFANov13_40psiNe_105C_1224k_FT cut 3m.txt','RRT-3-3.out')
production('R-S THFANov13_40psiNe_105C_1224k_FT cut 3m.txt','RSK-carbony+-1-3.out')
production('R-S THFANov13_40psiNe_105C_1224k_FT cut 3m.txt','RSK-ether-1-3.out')








