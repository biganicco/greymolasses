# -*- coding: utf-8 -*-
"""
Grey Molasses Simulation

Will Lab, Columbia University Physics Department

Date created: 27 March 2018
Last edited: 28 March 2018
Last edited by: Niccolò 
"""

import numpy as np
import matplotlib.pyplot as plt
#import sympy 
from sympy import S
from sympy.physics.wigner import clebsch_gordan as cg
from sympy.physics.quantum.cg import Wigner3j
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
m_Fdark = 2*Fdark + 1
m_Fbright = 2*Fbright + 1
m_Fexcited = 2*Fexcited + 1
StatesNumber = m_Fdark + m_Fbright + m_Fexcited
'''
A = Wigner3j(6, 0, 4, 0, 2, 0).doit()
print(float(A)+0.34)
'''
#Initialize an array containing the state labels for states of the form |j,F,m_F>
StLab = np.zeros((3,StatesNumber)) 
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
Eb = 1.7716261288                                           #energy of the F=2 manifold 
Ee = 1.10726633050 + 508.8487162*10**3 - 15.944*10**(-3)    #energy of the F'=2 manifold 

#Hamiltonians
H0 = np.matrix(np.zeros((StatesNumber,StatesNumber))) #Initialize the free Hamiltonian
for i in range(0,m_Fdark-1):
    H0[i,i]=Ed
for i in range(m_Fdark,m_Fdark+m_Fbright):
    H0[i,i]=Eb
for i in range(m_Fdark+m_Fbright,StatesNumber):
    H0[i,i]=Ee
