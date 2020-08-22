import numpy as np
import itertools


def distance(m1, m2):
    DIS=[]
    for i in range(0,n_atoms):
        p_1 = m1[i,:]
        for j in range(0,n_atoms):
            p_2 = m2[j,:]
            dis = ((p_1[0] - p_2[0]) ** 2 + (p_1[1] - p_2[1]) ** 2 + (p_1[2] - p_2[2]) ** 2) ** 0.5
            DIS.append(dis)
    return DIS

def step_1(monomer_n):
    list = [-1,1]
    n_atoms = len(monomer_n)
    zeros = np.array(np.zeros((n_atoms,3)))
    for i,j,k in itertools.product(list,list,list):
        mask = [[i,0,0],
                [0,j,0],
                [0,0,k]
               ]
        c = np.array(monomer_n).dot(np.array(mask))
        zeros = np.append(zeros,c,axis=0)
    d = np.array(zeros)[n_atoms:9*n_atoms,:]
    return d



input_xyz = np.loadtxt('test.xyz')

unit_1 = input_xyz[0:10,2:5]

unit_2 = input_xyz[10:20,2:5]

unit_3 = input_xyz[20:30,2:5]

print(np.array(unit_1))
print(np.array(unit_2))
print(np.array(unit_3))

monomer_1 = step_1(unit_1)
monomer_2 = step_1(unit_2)
monomer_3 = step_1(unit_3)
n_atoms = len(unit_1)


n = 0
for i in range(0,8):
    m_1 = monomer_1[i*n_atoms:(i*n_atoms+n_atoms)]
    for j in range(0, 8):
        m_2 = monomer_2[j * n_atoms:(j * n_atoms + n_atoms)]
        for k in range(0, 8):
            m_3 = monomer_3[k * n_atoms:(k * n_atoms + n_atoms)]
            sum = np.append(distance(m_1,m_2),distance(m_1,m_3))
            sum = np.append(sum,distance(m_2,m_3))
            #print(min(sum))
            if 0  < min(sum) < 2:
                n = n + 1
                coor = np.append(m_1,m_2,axis=0)
                coor = np.append(coor,m_3,axis=0)
                print('candidate number', n,   'and the min dis is',min(sum))
                print(coor)
                #print(m_1)
#            print(min(sum))
