"""
Grey Molasses Simulation

Will Lab, Columbia University Physics Department

Date created: 27 March 2018
Last edited: 2 April 2018
Last edited by: Niccolò 
"""

# -*- coding: utf-8 -*-
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
    
#Initialize the unperturbed energies [GHz] - zero set at lowest energy
Ed = 0                                                      #energy of the F=1 manifold 
Eb = 6.83468261090429                                       #energy of the F=2 manifold 
Ee = 4.27167663181519 + 384.2304844685*10**3 - 72.9113*10**(-3)    #energy of the F'=2 manifold 

#Initialize lasers parameters
DeltaRC = 2*np.pi*6834.6*10**(-3)
gamma = 2*np.pi*6.065*10**(-3)
omega_R = 5*gamma+Ee
RabiR = 1.2*gamma
RabiC = 4.2*gamma

#Hamiltonians
H0 = np.matrix(np.zeros((StatesNumber,StatesNumber))) #Initialize the free Hamiltonian
for i in range(0,m_Fdark-1):
    H0[i,i]=Ed
for i in range(m_Fdark,m_Fdark+m_Fbright):
    H0[i,i]=Eb-DeltaRC
for i in range(m_Fdark+m_Fbright,StatesNumber):
    H0[i,i]=Ee-omega_R
H = np.matrix(np.zeros((StatesNumber,StatesNumber)))
Hp1 = np.matrix(np.zeros((StatesNumber,StatesNumber))) #Initialize the interaction Hamiltonian with \sigma_+ light
Hp0 = np.matrix(np.zeros((StatesNumber,StatesNumber))) #Initialize the interaction Hamiltonian with \pi light
Hm1 = np.matrix(np.zeros((StatesNumber,StatesNumber))) #Initialize the interaction Hamiltonian with \sigma_- light

#Brute force interaction matrix elements
Hm1[0,8] = 4**(-0.5)*RabiR
Hm1[1,9] = 8**(-0.5)*RabiR 
Hm1[2,10] = 24**(-0.5)*RabiR 
Hm1[4,8] = -12**(-0.5)*RabiC 
Hm1[5,9] = -8**(-0.5)*RabiC
Hm1[6,10] = -8**(-0.5)*RabiC 
Hm1[7,11] = -12**(-0.5)*RabiC
Hm1[8,0] = 4**(-0.5)*RabiR 
Hm1[9,1] = 8**(-0.5)*RabiR 
Hm1[10,2] = 24**(-0.5)*RabiR 
Hm1[8,4] = -12**(-0.5)*RabiC 
Hm1[9,5] = -8**(-0.5)*RabiC 
Hm1[10,6] = -8**(-0.5)*RabiC
Hm1[11,7] = -12**(-0.5)*RabiC

Hp1[0,10] = 24**(-0.5)*RabiR
Hp1[1,11] = 8**(-0.5)*RabiR
Hp1[2,12] = 4**(-0.5)*RabiR
Hp1[3,9] = 12**(-0.5)*RabiC
Hp1[4,10] = 8**(-0.5)*RabiC
Hp1[5,11] = 8**(-0.5)*RabiC
Hp1[6,12] = 12**(-0.5)*RabiC
Hp1[10,0] = 24**(-0.5)*RabiR
Hp1[11,1] = 8**(-0.5)*RabiR
Hp1[12,2] = 4**(-0.5)*RabiR
Hp1[9,3] = 12**(-0.5)*RabiC
Hp1[10,4] = 8**(-0.5)*RabiC
Hp1[11,5] = 8**(-0.5)*RabiC
Hp1[12,6] = 12**(-0.5)*RabiC

Hp0[0,9] = -8**(-0.5)*RabiR
Hp0[1,10] = -6**(-0.5)*RabiR
Hp0[2,11] = -8**(-0.5)*RabiR
Hp0[3,8] = -6**(-0.5)*RabiC
Hp0[4,9] = -24**(-0.5)*RabiC
Hp0[5,10] = 0*RabiC
Hp0[6,11] = 24**(-0.5)*RabiC
Hp0[7,12] = 6**(-0.5)*RabiC
Hp0[9,0] = -8**(-0.5)*RabiR
Hp0[10,1] = -6**(-0.5)*RabiR
Hp0[11,2] = -8**(-0.5)*RabiR
Hp0[8,3] = -6**(-0.5)*RabiC
Hp0[9,4] = -24**(-0.5)*RabiC
Hp0[10,5] = 0*RabiC
Hp0[11,6] = 24**(-0.5)*RabiC
Hp0[12,7] = 6**(-0.5)*RabiC


DeltaKZ = 0.001
steps = int(np.int(np.pi/(2*DeltaKZ)))
KZ = np.zeros((steps,))
for i in range(0,steps-1):
    KZ[i] = i*DeltaKZ

Energies = np.zeros((StatesNumber,steps))
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
    Energies[:,t], holder = np.linalg.eigh(H)
    H = np.matrix(np.zeros((StatesNumber,StatesNumber)))

plt.plot(KZ,Energies[0,:],'r.',KZ,Energies[1,:],'r.',KZ,Energies[2,:],'r.',KZ,Energies[3,:],'r.',KZ,Energies[4,:],'r.',KZ,Energies[5,:],'r.',KZ,Energies[6,:],'r.',KZ,Energies[7,:],'r.',KZ,Energies[8,:],'r.',KZ,Energies[9,:],'r.',KZ,Energies[10,:],'r.',KZ,Energies[11,:],'r.',KZ,Energies[12,:],'r.')
plt.ylim([-36.1088,-36.1083])
plt.show()
