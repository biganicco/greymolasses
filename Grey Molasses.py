"""
Grey Molasses Simulation

Will Lab, Columbia University Physics Department

Date created: 27 March 2018
Last edited: 2 April 2018
Last edited by: Niccolò 
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
#import sympy 
from sympy import S
from sympy.physics.wigner import clebsch_gordan as cg
from sympy.physics.quantum.cg import Wigner3j
from sympy.physics.quantum.cg import Wigner6j
'''
from sympy.physics.wigner import wigner_3j as w3j
from sympy.physics.wigner import wigner_3j as w6j
'''
'''
Initializations
'''
#Initialize the molasses states
Fdark = 1
Fbright = 2
Fexcited = 2
Jdark = S(1)/2
Jbright = Jdark
Jexcited = S(3)/2
INa = S(3)/2
m_Fdark = 2*Fdark + 1
m_Fbright = 2*Fbright + 1
m_Fexcited = 2*Fexcited + 1
StatesNumber = m_Fdark + m_Fbright + m_Fexcited
#JRedDip = 3.5247 #Value of <J||e\vec{r}||J'> for the D2 line - from Steck units of [e*a_0]
'''
A = Wigner3j(6, 0, 4, 0, 2, 0).doit()
print(float(A)+0.34)
'''
#Initialize an array containing the state labels for states of the form |J,F,I,m_F>
StLab = np.zeros((4,StatesNumber)) 
for i in range(0,m_Fdark):
    StLab[0,i] = Jdark
    StLab[1,i] = Fdark
    StLab[2,i] = -Fdark+i
for i in range(m_Fdark,m_Fdark+m_Fbright):
    StLab[0,i] = Jbright
    StLab[1,i] = Fbright
    StLab[2,i] = -Fbright-m_Fdark+i
for i in range(m_Fdark+m_Fbright,StatesNumber):
    StLab[0,i] = Jexcited
    StLab[1,i] = Fexcited
    StLab[2,i] = -Fexcited-(m_Fdark+m_Fbright)+i
    
#Initialize lasers parameters
#wc = #cooling laser frequency [GHz]
#wr = #repumper laser frequency [GHz]

#Initialize the unperturbed energies [GHz] - zero set at lowest energy
Ed = 0                                                      #energy of the F=1 manifold 
Eb = 6.83468261090429                                       #energy of the F=2 manifold 
Ee = 4.27167663181519 + 384.2304844685*10**3 - 72.9113*10**(-3)    #energy of the F'=2 manifold 

DeltaRC = 2*np.pi*6834.6*10**(-3)
gamma = 2*np.pi*6.065*10**(-3)
omega_R = 5*gamma+Ee
RabiR = 1.2*gamma
RabiC = 4.2*gamma

Omega = np.matrix([[1,0,0,0,0,0,0,0,RabiR,RabiR,RabiR,RabiR,RabiR],
         [0,1,0,0,0,0,0,0,RabiR,RabiR,RabiR,RabiR,RabiR],
         [0,0,1,0,0,0,0,0,RabiR,RabiR,RabiR,RabiR,RabiR],
         [0,0,0,1,0,0,0,0,RabiC,RabiC,RabiC,RabiC,RabiC],
         [0,0,0,0,1,0,0,0,RabiC,RabiC,RabiC,RabiC,RabiC],
         [0,0,0,0,0,1,0,0,RabiC,RabiC,RabiC,RabiC,RabiC],
         [0,0,0,0,0,0,1,0,RabiC,RabiC,RabiC,RabiC,RabiC],
         [0,0,0,0,0,0,0,1,RabiC,RabiC,RabiC,RabiC,RabiC],
         [RabiR,RabiR,RabiR,RabiC,RabiC,RabiC,RabiC,RabiC,1,0,0,0,0],
         [RabiR,RabiR,RabiR,RabiC,RabiC,RabiC,RabiC,RabiC,0,1,0,0,0],
         [RabiR,RabiR,RabiR,RabiC,RabiC,RabiC,RabiC,RabiC,0,0,1,0,0],
         [RabiR,RabiR,RabiR,RabiC,RabiC,RabiC,RabiC,RabiC,0,0,0,1,0],
         [RabiR,RabiR,RabiR,RabiC,RabiC,RabiC,RabiC,RabiC,0,0,0,0,1]])

#Hamiltonians
H0 = np.matrix(np.zeros((StatesNumber,StatesNumber))) #Initialize the free Hamiltonian
for i in range(0,m_Fdark-1):
    H0[i,i]=Ed
for i in range(m_Fdark,m_Fdark+m_Fbright):
    H0[i,i]=Eb
for i in range(m_Fdark+m_Fbright,StatesNumber):
    H0[i,i]=Ee
    
q = 1
Hp1 = np.matrix(np.zeros((StatesNumber,StatesNumber))) #Initialize the interaction Hamiltonian
for i in range(0,StatesNumber-1):
    for j in range(0,StatesNumber-1):
        A = (-1)**(StLab[1,j]-1+StLab[2,i])*(2*StLab[1,i]+1)**0.5*float(Wigner3j(StLab[1,j], 1, StLab[1,i], StLab[2,j], q, -StLab[2,i]).doit())
        B = (-1)**(StLab[1,j]+StLab[0,i]+1+float(INa))*((2*StLab[1,j]+1)*(2*StLab[0,i]))**0.5*float(Wigner6j(StLab[0,i], StLab[0,j], 1, StLab[1,j], StLab[1,i], INa).doit())
        Hp1[i,j]= A*B*Omega[i,j]

q = 0
Hp0 = np.matrix(np.zeros((StatesNumber,StatesNumber))) #Initialize the interaction Hamiltonian
for i in range(0,StatesNumber-1):
    for j in range(0,StatesNumber-1):
        A = (-1)**(StLab[1,j]-1+StLab[2,i])*(2*StLab[1,i]+1)**0.5*float(Wigner3j(StLab[1,j], 1, StLab[1,i], StLab[2,j], q, -StLab[2,i]).doit())
        B = (-1)**(StLab[1,j]+StLab[0,i]+1+float(INa))*((2*StLab[1,j]+1)*(2*StLab[0,i]))**0.5*float(Wigner6j(StLab[0,i], StLab[0,j], 1, StLab[1,j], StLab[1,i], INa).doit())
        Hp0[i,j] = A*B*Omega[i,j]
        
q = -1
Hm1 = np.matrix(np.zeros((StatesNumber,StatesNumber))) #Initialize the interaction Hamiltonian
for i in range(0,StatesNumber-1):
    for j in range(0,StatesNumber-1):
        A = (-1)**(StLab[1,j]-1+StLab[2,i])*(2*StLab[1,i]+1)**0.5*float(Wigner3j(StLab[1,j], 1, StLab[1,i], StLab[2,j], q, -StLab[2,i]).doit())
        B = (-1)**(StLab[1,j]+StLab[0,i]+1+float(INa))*((2*StLab[1,j]+1)*(2*StLab[0,i]))**0.5*float(Wigner6j(StLab[0,i], StLab[0,j], 1, StLab[1,j], StLab[1,i], INa).doit())
        Hm1[i,j]= A*B*Omega[i,j]

H = np.matrix(np.zeros((StatesNumber,StatesNumber)),dtype=np.complex)
DeltaKZ = 0.001
steps = int(np.int(np.pi/(2*DeltaKZ)))
KZ = np.zeros((steps,))
for i in range(0,steps-1):
    KZ[i] = i*DeltaKZ
'''
k= np.pi/4
for i in range(0,StatesNumber-1):
    for j in range(0,StatesNumber-1):
        if i == j:
            H[i,j] = H0[i,j] + Hp1[i,j] + Hm1[i,j] + Hp0[i,j]
        elif i<j:
            H[i,j] = H0[i,j] + 2**0.5*np.cos(k)*Hp1[i,j]+1j*2**0.5*np.sin(k)*Hm1[i,j]+(np.cos(k)*(1-1j)+np.sin(k)*(1+1j))*Hp0[i,j]
        elif j<i:
            H[i,j] = H0[i,j] + 2**0.5*np.cos(k)*Hp1[i,j]-1j*2**0.5*np.sin(k)*Hm1[i,j]+(np.cos(k)*(1+j)+np.sin(k)*(1-1j))*Hp0[i,j]
Energies, holder = LA.eig(H)
'''
Energies = np.array(np.zeros((13,steps)))
LA.eig(np.diag((1, 2, 3)))
for t in range(0,steps-1):
    k = np.pi*t/steps
    for i in range(0,StatesNumber-1):
        for j in range(0,StatesNumber-1):
            if i == j:
                H[i,j] = H0[i,j] + Hp1[i,j] + Hm1[i,j] + Hp0[i,j]
            elif i>j:
                H[i,j] = H0[i,j] + 2**0.5*np.cos(k)*Hp1[i,j]+1j*2**0.5*np.sin(k)*Hm1[i,j]+(np.cos(k)*(1-1j)+np.sin(k)*(1+1j))*Hp0[i,j]
            elif j>i:
                H[i,j] = H0[i,j] + 2**0.5*np.cos(k)*Hp1[i,j]-1j*2**0.5*np.sin(k)*Hm1[i,j]+(np.cos(k)*(1+j)+np.sin(k)*(1-1j))*Hp0[i,j]
    Energies[:,i], holder = LA.eig(H)

plt.plot(KZ,Energies[1,:])
plt.show()
