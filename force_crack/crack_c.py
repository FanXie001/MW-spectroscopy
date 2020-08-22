import numpy as np
import itertools

def step_1(monomer_n):
    list = [-1,1]
    zeros = []
    for i,j,k in itertools.product(list,list,list):
        mask = [[i,0,0],
                [0,j,0],
                [0,0,k]
               ]
        c = np.array(monomer_n).dot(np.array(mask))
        zeros.append(c)
    return np.array(zeros)

def mirror(input):
    total = np.zeros((8,3))
    for i in range(0,n_atom):
        c_i = input[i,2:5]
        total = np.append(total,step_1(c_i),axis=0)
    total = total[8:9*n_atom,]
    return total

def distance(p_1,p_2):
    dis = ((p_1[0] - p_2[0]) ** 2 + (p_1[1] - p_2[1]) ** 2 + (p_1[2] - p_2[2]) ** 2) ** 0.5
    return dis

input_xyz = np.loadtxt('c_atom_hetero1.xyz')
n_atom = len(input_xyz)
total = mirror(input_xyz)

n = 0
for i in range(0,len(total)):
    p_i = total[i,]
    for j in range(0,len(total)):
        p_j = total[j,]
        dis_1 = distance(p_i,p_j)
        if  1.43  < dis_1 < 1.47:
            for k in range(0, len(total)):
                p_k = total[k,]
                dis_2 = distance(p_i,p_k)
                dis_3 = distance(p_j,p_k)
                if (1.47 < dis_2 < 1.54 and dis_2 != dis_1 ) and (2.4 < dis_3 < 2.7):
                    n = n + 1
                    summary = np.append([p_i],[p_j],axis=0)
                    summary = np.append(summary, [p_k], axis=0)
                    print(summary)

print(n)





#print(np.zeros(1,3))