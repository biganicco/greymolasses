"""
Grey Molasses Simulation

Will Lab, Columbia University Physics Department

Date created: 27 March 2018
Last edited: 25 April 2018
Last edited by: Claire and Niccolò 
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

#Initialize the molasses states
Fdark = 1
Fbright = 2
Fexcited = 2
Jdark = 1/2
Jbright = Jdark
Jexcited = 3/2
INa = 3/2
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
gamma = 9.7946*10**(-3)
D2 = 508.8487162*10**3
F1F1S12 = 1.7716261288
F1hfs = 1.10726633050
F2hfs = 0.66435979830
F2phfs = 15.944*10**(-3)
Ed = 0                                                      #energy of the F=1 manifold 
Eb = F1F1S12                                    #energy of the F=2 manifold 
Ee = D2 + F1hfs - F2phfs    #energy of the F'=2 manifold 

#Initialize lasers parameters
delta = 0.1*gamma
Delta = 2.4*gamma
omega_R = D2 + F1hfs - F2phfs + delta + Delta
omega_C = D2 - F2hfs - F2phfs + Delta
DeltaRC = omega_R-omega_C
RabiR = gamma*(5/2)**0.5
RabiC = gamma*(2*5/2)**0.5
#Hamiltonians
H0 = np.matrix(np.zeros((StatesNumber,StatesNumber),dtype=complex)) #Initialize the free Hamiltonian
for i in range(0,m_Fdark-1):
    H0[i,i]=Ed
for i in range(m_Fdark,m_Fdark+m_Fbright):
    H0[i,i]=Eb-DeltaRC
for i in range(m_Fdark+m_Fbright,StatesNumber):
    H0[i,i]=Ee-omega_R
H = np.matrix(np.zeros((StatesNumber,StatesNumber),dtype=complex))
Hp1 = np.matrix(np.zeros((StatesNumber,StatesNumber),dtype=complex)) #Initialize the interaction Hamiltonian with \sigma_+ light
Hp0 = np.matrix(np.zeros((StatesNumber,StatesNumber),dtype=complex)) #Initialize the interaction Hamiltonian with \pi light
Hm1 = np.matrix(np.zeros((StatesNumber,StatesNumber),dtype=complex)) #Initialize the interaction Hamiltonian with \sigma_- light

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

DeltaKZ = 0.05
Limit = np.pi
steps = np.int(Limit/(DeltaKZ))
KZ = np.zeros((steps,))
for i in range(0,steps-1):
    KZ[i] = i*DeltaKZ
holder = np.zeros((StatesNumber,StatesNumber),dtype=complex)
Energies = np.zeros((StatesNumber,steps),dtype=complex)
EnergiesReal = np.zeros((StatesNumber,steps))
Weights = EnergiesReal = np.zeros((StatesNumber,steps,3))
with open('Eigenenergies and eigenvectors.txt','wb') as ftext:
    for t in range(0,steps):
        k = Limit*t/steps
        for i in range(0,StatesNumber):
            for j in range(0,StatesNumber):
                if i == j:
                    H[i,j] = H0[i,j] + Hp1[i,j] + Hm1[i,j] + Hp0[i,j]
                elif i>j:
                    H[i,j] = H0[i,j] + np.exp(1j*np.pi/4)*(Hp1[i,j]*(np.cos(k)-np.sin(k))+1j*Hm1[i,j]*(np.cos(k)+np.sin(k)))
                elif i<j:
                    H[i,j] = H0[i,j] + np.exp(-1j*np.pi/4)*(Hp1[i,j]*(np.cos(k)-np.sin(k))-1j*Hm1[i,j]*(np.cos(k)+np.sin(k)))
        Energies[:,t], holder = LA.eigh(H)     
        Energy = Energies[:,t]/gamma
        for l in range(0,StatesNumber):
            for y in range(0,StatesNumber):
                if np.absolute(holder[l,y]) <0.01:
                    holder[l,y] = 0
        if t<steps-2:
            H = np.matrix(np.zeros((StatesNumber,StatesNumber),dtype=complex))
        f = np.savetxt(ftext,np.abs(Energies[:,t]),fmt='%.4e',delimiter=',', header='Eigenenergies at kz='+np.str(k)+': ')
        C_B_drk = np.zeros((StatesNumber), dtype = int)
        C_G_brt = np.zeros((StatesNumber), dtype = int)
        C_R_exc = np.zeros((StatesNumber), dtype = int)
        Colour = []
        for l in range(0,StatesNumber):
            Weights[l,t,0] = holder[0,l]*np.conj(holder[0,l])+holder[1,l]*np.conj(holder[1,l])+holder[2,l]*np.conj(holder[2,l])
            Weights[l,t,1] = holder[3,l]*np.conj(holder[3,l])+holder[4,l]*np.conj(holder[4,l])+holder[5,l]*np.conj(holder[5,l])+holder[6,l]*np.conj(holder[6,l])+holder[7,l]*np.conj(holder[7,l])
            Weights[l,t,2] = holder[8,l]*np.conj(holder[8,l])+holder[9,l]*np.conj(holder[9,l])+holder[10,l]*np.conj(holder[10,l])+holder[11,l]*np.conj(holder[11,l])+holder[12,l]*np.conj(holder[12,l])
            C_B_drk[l] = int(round(255*Weights[l,t,0]))
            C_G_brt[l] = int(round(255*Weights[l,t,1]))
            C_R_exc[l] = int(round(255*Weights[l,t,2]))
            Colour.append( '#%02x%02x%02x' % (C_R_exc[l], C_G_brt[l], C_B_drk[l]))
        for l in range(StatesNumber):
            plt.scatter([k], Energy[l], c = Colour[l], marker = "." )
        #print(Colour)
        #plt.plot([k],Energy[0],'r.',[k],Energy[1],'g.',[k],Energy[2],'b.',[k],Energy[3],'k.',[k],Energy[4],'y.',[k],Energy[5],'m.',[k],Energy[6],'r.',[k],Energy[7],'g.',[k],Energy[8],'r.',[k],Energy[9],'g.',[k],Energy[10],'y.',[k],Energy[11],'b.',[k],Energy[12],'k.')
for l in range(StatesNumber):
    for y in range(0,steps-1):
        if np.imag(Energies[l,y])<0.00000001:
            EnergiesReal[l,y] = np.real(Energies[l,y])
        else:
            print('Imaginary energy value found')
EnergiesReal = EnergiesReal/gamma
Energies = Energies/gamma
#plt.plot(KZ,Energies[0,:],'r.',KZ,Energies[1,:],'g.',KZ,Energies[2,:],'b.',KZ,Energies[3,:],'k.',KZ,Energies[4,:],'y.',KZ,Energies[5,:],'m.',KZ,Energies[6,:],'r.',KZ,Energies[7,:],'g.',KZ,Energies[8,:],'r.',KZ,Energies[9,:],'g.',KZ,Energies[10,:],'y.',KZ,Energies[11,:],'b.',KZ,Energies[12,:],'k.')
#plt.plot(KZ,EnergiesReal[0,:],'r.',KZ,EnergiesReal[1,:],'r.',KZ,EnergiesReal[2,:],'r.',KZ,EnergiesReal[3,:],'r.',KZ,EnergiesReal[4,:],'r.',KZ,EnergiesReal[5,:],'r.',KZ,EnergiesReal[6,:],'r.',KZ,EnergiesReal[7,:],'r.',KZ,EnergiesReal[8,:],'r.',KZ,EnergiesReal[9,:],'r.',KZ,EnergiesReal[10,:],'r.',KZ,EnergiesReal[11,:],'r.',KZ,EnergiesReal[12,:],'r.')
#plt.ylim([-36.10855,-36.10835])
#plt.ylim([-0.0001,0.0025])
plt.ylim([-0.2,0.7])

plt.show()  
