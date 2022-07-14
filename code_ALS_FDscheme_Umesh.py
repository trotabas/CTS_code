# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 11:40:17 2020
@author: TROTABAS Baptiste
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import newton
from scipy.optimize import brentq
from scipy import special
from decimal import *
#from matplotlib.ticker import Scalarformatter
#import itertools
from itertools import chain
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
from scipy.sparse import csr_matrix
import scipy.sparse as sparse
from scipy.sparse import linalg 
import scipy.sparse as spsp
from scipy.integrate import quad
import time  
from scipy.interpolate import interp2d
from math import floor, log10

from matplotlib import ticker, cm
from matplotlib.colors import LogNorm
from matplotlib.colors import BoundaryNorm
import pandas as pd

#from Main_function_contourf import *

def cm2inch(value):
    return value/2.54


def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    if exponent is None:
        exponent = int(floor(log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits

    return r"${0:.{2}f}\times 10^{{{1:d}}}$".format(coeff, exponent, precision)

def sci_notation_pow(num, decimal_digits=1, precision=None, exponent=None):
    if exponent is None:
        exponent = int(floor(log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits
    return r"$10^{{{1:d}}}$".format(coeff, exponent, precision)

def where_better(table, element):
    for element_table in table:
        if element_table < element:
            return np.where( table <  element )[0][0]
    return -1
        
        
origin = 'lower'
list_color3 = ["dodgerblue", "darkorange"]
list_color2 = ["green", "red"]
list_color1 = ["saddlebrown", "dodgerblue", "gold","lime", "green", "teal", "navy", "mediumpurple", "orchid", "red"]
list_color = ["dodgerblue", "darkorange", "forestgreen", "black", "chocolate", "blueviolet", "gold","black",
              "gray","gold","lime","saddlebrown", "peru", "gold","lime", "green", "teal", "navy", "mediumpurple", "orchid", "red"]
list_color = ["dodgerblue", "darkorange", "black", "darkviolet", "peru", "blueviolet", "gold","black","gray","gold","lime","saddlebrown", "peru", "gold","lime", "green", "teal", "navy", "mediumpurple", "orchid", "red"]
list_linestyle = ["solid","dashed","dashdot"]
list_marker = ["o","D","s","x","d"]
list_regime = [r'Not Saturated: $\tau_P \simeq 10^{-5}$', r'Partially Saturated: $\tau_P \simeq 10^{-3}$', r'Saturated: $\tau_P \simeq 10^{-2}$', r'Saturated: $\tau_P \simeq 1$']
# %%
Folder = "Test06" 
#-->
InputFolder = "input.dat"
path_input  = "Output/" + Folder + "/" + InputFolder  
#-->
datContent = [i.strip().split() for i in open(path_input).readlines()]
dict_input = {}
for i in open(path_input).readlines():
    x = i.strip().split() 
    if x[0][0] != '/':
        dict_input[x[0]] = np.float64(x[1])
#==============================================================================
#---------------------:PHYSICAL-CONSTANT---------------------------------------
#==============================================================================
NA         = dict_input.get('NA')           # Avogadro  Number      [mol^-1] 
kB         = dict_input.get('kB')           # Boltzmann Constant
CoulombLog = dict_input.get('CoulombLog')   # Coulomnb  Logarithm
eps0       = dict_input.get('eps0')         # Coulomnb  Logarithm
sig0       = dict_input.get('sig0')         # Coulomnb  Logarithm
e          = dict_input.get('e')            # Eletric   Charge      [C]
me         = dict_input.get('me')           # Electron  Mass        [kg]
#==============================================================================
#---------------------:PLASMA-PARAMETERS---------------------------------------
#==============================================================================
mi_amu     = dict_input.get('mi_amu')       # Argon atom mass      [amu] 
mi         = mi_amu*1.6605e-27              # Argon atom mass      [kg] 
eta        = me/mi
Ti = dict_input.get('Ti')                   # Ion       Temperature [eV]
Tn = dict_input.get('Tn')                   # Room      Temperature [K]
Te = dict_input.get('T0')                   # Electron  Temperature [eV]
ne = dict_input.get('n0')                   # Electron  Density     [m^-3]
B  = dict_input.get('B')                    # Magnetic  Field       [mT]
P  = dict_input.get('P')                    # Neutral   Pressure    [Pa]
nN = (1/(kB*Tn))*P                          # Neutral   Density     [m^-3]
#==============================================================================
#---------------------:EMISSIVE-ELECTRODE--------------------------------------
#==============================================================================
Tw    = dict_input.get('Tw')                # Heating    Temperature [K]
W     = dict_input.get('W')                 # Work       Function    [J]
j_eth = 0                                   # Thermionic Emission    [A m^2]
Xi    = 0                                   
phi_e = dict_input.get('phi_e')             # Electrode  Bias        [V]
# =============================================================================
# ---------------------: SHEATH PROPERTIES  -----------------------------------
# =============================================================================
Cs = np.sqrt((e*Te)/mi)                     # Bohm       Velocity    [m s^-1]
Lambda = np.log(np.sqrt(1/(2*np.pi*eta)))   # Sheath     Parameter
jis = (e*ne*Cs)                             # Ion  Sat.  currend  density  [m^-3]

#==============================================================================
#---------------------:GEOMETRICAL-PARAMETERS----------------------------------
#==============================================================================
rg = dict_input.get('rg')                   # Column     Radius      [m]
L = dict_input.get('L')                     # Column     Length      [m]
#==============================================================================
#---------------------:MESHING-PARAMETERS--------------------------------------
#==============================================================================
Nr     = int(dict_input.get('Nr'))          # Radial           Node      
Nre    = int(dict_input.get('Nre'))         # Radial Electrode Node      
Nz     = int(dict_input.get('Nz'))          # Axial            Node   
K      = Nr*Nz                              # Total            Node   
#--> 
step_r = rg/(Nr-1)                          # Radial    step             
step_z = (L/2)/(Nz-1)                       # Axial     step      
#-->
r = np.linspace(0, rg, Nr)                  # Radial    vector  
re = r[Nre-1]
print("re = ", re*1e2, " cm")
z = np.linspace(-L/2, 0, Nz)                # Axial    vector  
#-->
Sr = np.zeros(Nr)                           # Surface vector 
Sr[0]         = np.pi*(step_r/2)**2
Sr[1:Nre-1]  = 2*np.pi*r[1:Nre-1]*step_r
Sr[Nre-1]    = 2*np.pi*re *(step_r/2)
Sr[Nre:-1]   = 2*np.pi*r[Nre:-1]*step_r
Sr[-1]        = 2*np.pi*rg*(step_r/2)

# =============================================================================
# ---------------------: PICTURE  ---------------------------------------------
# =============================================================================
picture_phi = True
picture_Vr  = False
picture_Vz  = False
picture_S   = False

# =============================================================================
# ---------------------: GEOMETRIC PARAMETERS ---------------------------------
# =============================================================================
start_time = time.time()


# INITIALISATION DIRECT METHOD SPARSE
dSjVe = 0
# DIMENSION ------------------------------------------------------------------
#dim_left = 3*Nre + 1*(Nr-1) + 1
dim_left   = 3*Nre + 3*(Nr - Nre - 1) + 1
dim_right  = 3*(Nr-1) + 1
dim_top    = 3*(Nz-2)
dim_bottom = Nz-2
dim_inside = 5*(Nz*Nr-2*Nz-2*(Nr-2))
dim_matrix_A = dim_left + dim_right + dim_top + dim_bottom + dim_inside
row_A  = np.zeros(dim_matrix_A)
col_A  = np.zeros(dim_matrix_A)
data_A = np.zeros(dim_matrix_A)
#------------------------------------------------------------------------------
Alpha_NBC_r1 = (-3)*(1/(2*step_r))
Gamma_P_NBC_r1   =    4*(1/(2*step_r))
Gamma_PP_NBC_r1  = (-1)*(1/(2*step_r))
#~~
Alpha_NBC_z_Left_e1_th = (-1)*((3/(2*step_z)) + dSjVe )
Alpha_NBC_z_Right = (-1)*(3/(2*step_z))
Beta_P_NBC_z      =    4*(1/(2*step_z))
Beta_PP_NBC_z     = (-1)*(1/(2*step_z))

# =============================================================================
# TOP BOUNDARY CONDITION -- NEUMANN (grad Phi |r=0  = 0) 
# =============================================================================
index = np.where(data_A == 0)[0][0]
    #-- row
row_A[0:3*len(np.arange(1,Nz-1)):3]   = np.arange(1,Nz-1)
row_A[1:1+3*len(np.arange(1,Nz-1)):3] = np.arange(1,Nz-1)
row_A[2:2+3*len(np.arange(1,Nz-1)):3] = np.arange(1,Nz-1)
    #-- column
col_A[0:3*len(np.arange(1,Nz-1)):3]     = np.arange(1,Nz-1)
col_A[1:1+3*len(np.arange(1,Nz-1)):3] = np.arange(1,Nz-1) + Nz
col_A[2:2+3*len(np.arange(1,Nz-1)):3] = np.arange(1,Nz-1) + 2*Nz
    #-- data    
data_A[0:3*len(np.arange(1,Nz-1)):3] = Alpha_NBC_r1
data_A[1:1+3*len(np.arange(1,Nz-1)):3] = Gamma_P_NBC_r1
data_A[2:2+3*len(np.arange(1,Nz-1)):3] = Gamma_PP_NBC_r1 

# =============================================================================
# LEFT BC -- 1 -- EMISSIVE + BIASING ELECTRODE 
# =============================================================================
index = np.where(data_A == 0)[0][0]
index_NBC_nonlinear_electrode_1 = np.where(data_A == 0)[0][0] #!!!!!!!!!!!
    #-- row
row_A[index:index+3*len(np.arange(0,Nz*Nre,Nz)):3]     = np.arange(0,Nz*Nre,Nz)
row_A[index+1:index+1+3*len(np.arange(0,Nz*Nre,Nz)):3] = np.arange(0,Nz*Nre,Nz)
row_A[index+2:index+2+3*len(np.arange(0,Nz*Nre,Nz)):3] = np.arange(0,Nz*Nre,Nz)
    #-- column
col_A[index:index+3*len(np.arange(0,Nz*Nre,Nz)):3]     = np.arange(0,Nz*Nre,Nz)
col_A[index+1:index+1+3*len(np.arange(0,Nz*Nre,Nz)):3] = np.arange(0,Nz*Nre,Nz) + 1
col_A[index+2:index+2+3*len(np.arange(0,Nz*Nre,Nz)):3] = np.arange(0,Nz*Nre,Nz) + 2
    #-- data
data_A[index:index+3*len(np.arange(0,Nz*Nre,Nz)):3]     = Alpha_NBC_z_Left_e1_th
data_A[index+1:index+1+3*len(np.arange(0,Nz*Nre,Nz)):3] = Beta_P_NBC_z  
data_A[index+2:index+2+3*len(np.arange(0,Nz*Nre,Nz)):3] = Beta_PP_NBC_z  

# =============================================================================
# LEFT BC -- 1 -- DIRICHLET BC 
# =============================================================================
index = np.where(data_A == 0)[0][0]
    #-- row
row_A[index:index+3*len(np.arange(Nz*Nre,K-Nz,Nz)):3]      = np.arange(Nz*Nre,K-Nz,Nz)
row_A[index+1:index+1+3*len(np.arange(Nz*Nre,K-Nz,Nz)):3]  = np.arange(Nz*Nre,K-Nz,Nz)
row_A[index+2:index+2+3*len(np.arange(Nz*Nre,K-Nz,Nz)):3]  = np.arange(Nz*Nre,K-Nz,Nz)
    #-- column
col_A[index:index+3*len(np.arange(Nz*Nre,K-Nz,Nz)):3]      = np.arange(Nz*Nre,K-Nz,Nz) 
col_A[index+1:index+1+3*len(np.arange(Nz*Nre,K-Nz,Nz)):3]  = np.arange(Nz*Nre,K-Nz,Nz) + 1
col_A[index+2:index+2+3*len(np.arange(Nz*Nre,K-Nz,Nz)):3]  = np.arange(Nz*Nre,K-Nz,Nz) + 2
    #-- data
data_A[index:index+3*len(np.arange(Nz*Nre,K-Nz,Nz)):3]     = Alpha_NBC_z_Left_e1_th 
data_A[index+1:index+1+3*len(np.arange(Nz*Nre,K-Nz,Nz)):3] = Beta_P_NBC_z
data_A[index+2:index+2+3*len(np.arange(Nz*Nre,K-Nz,Nz)):3] = Beta_PP_NBC_z

# =============================================================================
# RIGHT BC -- NEUMANN (grad Phi |z=0  = 0) 
# =============================================================================
index = np.where(data_A == 0)[0][0]
    #-- row
row_A[index:index+3*len(np.arange(Nz-1,K-Nz,Nz)):3]     = np.arange(Nz-1,K-Nz,Nz)
row_A[index+1:index+1+3*len(np.arange(Nz-1,K-Nz,Nz)):3] = np.arange(Nz-1,K-Nz,Nz)
row_A[index+2:index+2+3*len(np.arange(Nz-1,K-Nz,Nz)):3] = np.arange(Nz-1,K-Nz,Nz)
    #-- column
col_A[index:index+3*len(np.arange(Nz-1,K-Nz,Nz)):3]     = np.arange(Nz-1,K-Nz,Nz) 
col_A[index+1:index+1+3*len(np.arange(Nz-1,K-Nz,Nz)):3] = np.arange(Nz-1,K-Nz,Nz) - 1
col_A[index+2:index+2+3*len(np.arange(Nz-1,K-Nz,Nz)):3] = np.arange(Nz-1,K-Nz,Nz) - 2
    #-- data
data_A[index:index+3*len(np.arange(Nz-1,K-Nz,Nz)):3]     = Alpha_NBC_z_Right  
data_A[index+1:index+1+3*len(np.arange(Nz-1,K-Nz,Nz)):3] = Beta_P_NBC_z  
data_A[index+2:index+2+3*len(np.arange(Nz-1,K-Nz,Nz)):3] = Beta_PP_NBC_z    

# =============================================================================
# DOWN BC 
# =============================================================================
index = np.where(data_A == 0)[0][0]
row_A[index:index+Nz] = np.arange(K-Nz,K)
col_A[index:index+Nz] = np.arange(K-Nz,K)
data_A[index:index+Nz] = 1
#--
row_BC = np.zeros(Nr-1) #(Nre+ (Nr-Nr) - 1)
col_BC = np.zeros(Nr-1) #(Nre+ (Nr-Nr) - 1)
data_BC = np.zeros(Nr-1) #(Nre+ (Nr-Nr) - 1)
#--
row_BC[0:Nre] = np.arange(0,Nz*Nre,Nz)
col_BC[0:Nre] = 0   
#--
row_BC[Nre:] = np.arange(Nz*Nre,K-Nz,Nz)
col_BC[Nre:] = 0  
#--
vect_BC_sparse = spsp.csr_matrix((data_BC,(row_BC,col_BC)),shape=(K,1))
#vect_BC_sparse = vect_BC_sparse.todense()

# =============================================================================
# Calcul
# =============================================================================
index_Inside_domain_zone1  = np.where(data_A == 0)[0][0]
index_Inside_domain_zone12 = index_Inside_domain_zone1.copy()  + (Nre-2)*(Nz-2)*5
index_Inside_domain_zone2  = index_Inside_domain_zone12.copy() + (Nz-2)*5

# =============================================================================
# ---------------------: Cyclotron frequency ----------------------------------
# =============================================================================
Omega_i = (e*B)/mi
Omega_e = (e*B)/me

# =============================================================================
# ---------------------: Collision frequency ----------------------------------
# =============================================================================
nu_en = sig0*nN*np.sqrt((8*e*Te)/(np.pi*me))
nu_in = sig0*nN*np.sqrt((8*e*Ti)/(np.pi*mi))
nu_ei = 0 #(e**(5/2)*CoulombLog/(6*np.sqrt(2)*np.pi**(3/2)*eps0**2*np.sqrt(me)))*ne*Te**(-3/2)

# =============================================================================
# ---------------------: MOBILITY and DIFFUSION -------------------------------
# =============================================================================
nu_ei = 0
mu_e_para = e/(me*(nu_ei+nu_en))
D_e_para  = Te*mu_e_para
#-->
mu_e_perp = (me/(e*B**2))*((nu_ei+nu_en)/(1 + (nu_ei+nu_en)**2/Omega_e**2)) 
D_e_perp  = Te*mu_e_perp
#-->
mu_i_para = e/(me*(nu_in)) 
D_i_para  = Te*mu_i_para
#-->
mu_i_perp = (mi/(e*B**2))*((nu_in)/(1 + (nu_in)**2/Omega_i**2))
D_i_perp  = Te*mu_i_perp

# =============================================================================
# ---------------------: CONDUCTIVITY -----------------------------------------
# =============================================================================
sigma_perp0 = e*ne*(mu_i_perp + mu_e_perp)
sigma_para0 = e*ne*(mu_i_para + mu_e_para)
mu          = (sigma_perp0/sigma_para0)
tau         = (L/rg)*np.sqrt(mu)

# =============================================================================
# ---------------------: INSIDE THE DOMAIN - ZONE 1 :  EMISSIVE + BIASED ELECTRODE 
# =============================================================================
vect_fiveScheme_zoneR1 = np.zeros((Nr-2)*(Nz-2)) # np.zeros((Nz*Nr)-2*Nr-2*(Nz-2))
index_vect_fiveScheme_zoneR1 = 0
for i in np.arange(Nz+1,Nz*(Nr-1)-1):
    if i % Nz != 0 and (i+1) % Nz != 0 :
        vect_fiveScheme_zoneR1[index_vect_fiveScheme_zoneR1] = i
        index_vect_fiveScheme_zoneR1 += 1
#-- coef DIRECT
Alpha_DIRECT_zoneR1    = -2*( 1/step_r**2 + 1/(mu*step_z**2) )
Beta_DIRECT_zoneR1     = 1/(mu*step_z**2) 
Gamma_P_DIRECT_zoneR1  = (1/step_r**2 + (1/(2*r[1:Nr-1]*step_r)))
Gamma_M_DIRECT_zoneR1  = (1/step_r**2 - (1/(2*r[1:Nr-1]*step_r)))
index_coef = 0
Alpha_zoneR1 = np.zeros(Nz-2)
Beta_zoneR1  = np.zeros(Nz-2)
Gamma_P_zoneR1 = np.zeros(len(Gamma_P_DIRECT_zoneR1)*(Nz-2))
Gamma_M_zoneR1 = np.zeros(len(Gamma_M_DIRECT_zoneR1)*(Nz-2))
for i in range(len(Gamma_P_DIRECT_zoneR1)):
    for Nzz in np.arange(Nz-2):
#            Alpha_zoneR1[index_coef] = Alpha_DIRECT_zoneR1[i]
#            Beta_zoneR1[index_coef] = Beta_DIRECT_zoneR1[i]
        Gamma_P_zoneR1[index_coef] = Gamma_P_DIRECT_zoneR1[i]
        Gamma_M_zoneR1[index_coef] = Gamma_M_DIRECT_zoneR1[i]
        index_coef += 1

# =============================================================================
# ---------------------: INSIDE THE DOMAIN  -----------------------------------
# =============================================================================
# row
row_A[index_Inside_domain_zone1:index_Inside_domain_zone1+5*len(vect_fiveScheme_zoneR1):5] = vect_fiveScheme_zoneR1
row_A[index_Inside_domain_zone1+1:index_Inside_domain_zone1+1+5*len(vect_fiveScheme_zoneR1):5] = vect_fiveScheme_zoneR1
row_A[index_Inside_domain_zone1+2:index_Inside_domain_zone1+2+5*len(vect_fiveScheme_zoneR1):5] = vect_fiveScheme_zoneR1
row_A[index_Inside_domain_zone1+3:index_Inside_domain_zone1+3+5*len(vect_fiveScheme_zoneR1):5] = vect_fiveScheme_zoneR1
row_A[index_Inside_domain_zone1+4:index_Inside_domain_zone1+4+5*len(vect_fiveScheme_zoneR1):5] = vect_fiveScheme_zoneR1
#-- column
col_A[index_Inside_domain_zone1:index_Inside_domain_zone1+5*len(vect_fiveScheme_zoneR1):5] = vect_fiveScheme_zoneR1
col_A[index_Inside_domain_zone1+1:index_Inside_domain_zone1+1+5*len(vect_fiveScheme_zoneR1):5] = vect_fiveScheme_zoneR1 + 1
col_A[index_Inside_domain_zone1+2:index_Inside_domain_zone1+2+5*len(vect_fiveScheme_zoneR1):5] = vect_fiveScheme_zoneR1 + Nz
col_A[index_Inside_domain_zone1+3:index_Inside_domain_zone1+3+5*len(vect_fiveScheme_zoneR1):5] = vect_fiveScheme_zoneR1 - 1
col_A[index_Inside_domain_zone1+4:index_Inside_domain_zone1+4+5*len(vect_fiveScheme_zoneR1):5] = vect_fiveScheme_zoneR1 - Nz
#-- data
data_A[index_Inside_domain_zone1:index_Inside_domain_zone1+5*len(vect_fiveScheme_zoneR1):5]     = Alpha_DIRECT_zoneR1
data_A[index_Inside_domain_zone1+1:index_Inside_domain_zone1+1+5*len(vect_fiveScheme_zoneR1):5] = Beta_DIRECT_zoneR1
data_A[index_Inside_domain_zone1+2:index_Inside_domain_zone1+2+5*len(vect_fiveScheme_zoneR1):5] = Gamma_P_zoneR1  
data_A[index_Inside_domain_zone1+3:index_Inside_domain_zone1+3+5*len(vect_fiveScheme_zoneR1):5] = Beta_DIRECT_zoneR1
data_A[index_Inside_domain_zone1+4:index_Inside_domain_zone1+4+5*len(vect_fiveScheme_zoneR1):5] = Gamma_M_zoneR1  

# =============================================================================
# ---------------------:
# =============================================================================
jis = (e*ne*Cs)
I_is = np.pi*(rg**2-re**2)*jis
phi_ssh_e1_th = phi_e + Lambda*Te        #      Electrode voltage     [V]
Vp_update  = phi_e  + Lambda*Te          #      Electrode voltage     [V]

# =============================================================================
# ---------------------: SOURCE TERM ------------------------------------------
# =============================================================================
# Source Term - NON-EMISSIVE
Sj_func_eth  = lambda V,x :  (jis/sigma_para0)*(1 + Xi) # - np.exp(Lambda + ((V-x)/Te)) ) #
dSj_func_eth = lambda V,x :  0 #(jis/(Te*sigma_para0))*np.exp(Lambda + ((V-x)/Te)) # 0 #

#--
phi_init_e1_th = phi_ssh_e1_th*np.ones(Nre)
#-->
SjVe = Sj_func_eth(phi_e,phi_init_e1_th) 
#-->
dSjVe = dSj_func_eth(phi_e,phi_init_e1_th)
#---------------------------------------------------------- DIRECT METHOD
err_L2 = 1.0
err_L8 = 1.0
I_null = 1.0
err_SjVe = 1.0
iteration = 0
Eps_L2 = 1e-7
Eps_I = 1e-7
#--

# while err_L2 > Eps_L2: # iteration < 1: # err_L2 > Eps: # gap_L2 > Eps: # gap_L2 > Eps: # iteration < 300: 
#~~ ELECTRODE 1 
Alpha_NBC_z_Left_e1_th = (-1)*((3/(2*step_z)) + dSj_func_eth(phi_e,phi_init_e1_th) )
data_A[index_NBC_nonlinear_electrode_1:index_NBC_nonlinear_electrode_1+3*len(np.arange(0,Nz*Nre,Nz)):3] = Alpha_NBC_z_Left_e1_th
#----------------------------------------------------------------
sparse_matrix_A = spsp.csr_matrix((data_A,(row_A,col_A)),shape=(K,K))
#matrix_A_dense = sparse_matrix_A.todense()
# matrix_A = sparse_matrix_A.todense()
#==== Bondaries Condition Vector -----------------------------------------------------------------------
#-- ELECTRODE 1
data_BC[:Nre] = Sj_func_eth(phi_e,phi_init_e1_th) - dSj_func_eth(phi_e,phi_init_e1_th)*phi_init_e1_th 
#-- 
data_BC[Nre:] = 0
#--
vect_BC_sparse = spsp.csr_matrix((data_BC,(row_BC,col_BC)),shape=(K,1))
#vect_BC_sparse = vect_BC_sparse.todense()

#------------------------- LINEAR SOLVE --------------------------------------------
phi_FDscheme = spsp.linalg.spsolve(sparse_matrix_A, vect_BC_sparse)
#phi_FDscheme = spsp.linalg.bicgstab(sparse_matrix_A, vect_BC_sparse)
phi_FDscheme = phi_FDscheme.reshape(Nr,Nz)
#------------------------- SAVE 
phi_init_e1_th = phi_FDscheme[0:Nre,0]  
Vp_update = phi_FDscheme[Nre-1,0]  
 #--------------------------> CONVERGENCE CRITERIA 
if iteration > 0:
    #----> NORME L2 CRITERIA 
    err_L2 = np.sqrt(np.nansum((phi_FDscheme[1:-1,1:-1]-phi_save[1:-1,1:-1])**2/(phi_FDscheme[1:-1,1:-1]**2)))
#------------------------- Save potential
phi_save = phi_FDscheme.copy()
#------------------------- ITERATION
iteration +=1

print("Time      -> t = ",round((time.time() - start_time),5)," s")
print("Iteration -> I = ", iteration)




# %%
"""
=========================== Determination of perpendicular electric field =================================================================================
"""   
# fig, ax = plt.subplots( constrained_layout = True, figsize = (2*2.20, 2.75))
# ax.plot(r[:Nre]/re, phi_FDscheme[:Nre,0], color = "black")
# ax.set_xlabel(r'$r/r_e$')
# ax.set_ylabel(r'$\textrm{Radial potential profile}$ $\phi_{sh}(r)$ $\mathrm{(V)}$')

E_perp = np.zeros([Nr,Nz])

E_perp[0,:]    = (-1)*(1/(2*step_r))*(-3*phi_FDscheme[0,:] + 4*phi_FDscheme[1,:] - phi_FDscheme[2,:])
E_perp[-1,:]   = (-1)*(-1/(2*step_r))*(-3*phi_FDscheme[Nr-1,:] + 4*phi_FDscheme[Nr-2,:] - phi_FDscheme[Nr-3,:])
E_perp[1:-1,:] = (-1)*(1/(2*step_r))*(phi_FDscheme[2:,:] - phi_FDscheme[0:-2,:])

E_para = np.zeros([Nr,Nz])

E_para[:,0]    = (-1)*(1/(2*step_z))*(-3*phi_FDscheme[:,0] + 4*phi_FDscheme[:,1] - phi_FDscheme[:,2])
E_para[:,-1]   = (-1)*(-1/(2*step_z))*(-3*phi_FDscheme[:,Nr-1] + 4*phi_FDscheme[:,Nr-2] - phi_FDscheme[:,Nr-3])
E_para[:,1:-1] = (-1)*(1/(2*step_z))*(phi_FDscheme[:,2:] - phi_FDscheme[:,0:-2])


# %% Determination of velocity
mu_i_para0 = e/(mi*(nu_in)) 
mu_i_perp0 = (mi/(e*B**2))*((nu_in)/(1 + (nu_in)**2/Omega_i**2)) 
mu_e_perp0 = (me/(e*B**2))*((nu_en + nu_ei)/(1 + (nu_en + nu_ei)**2/Omega_e**2)) 

Vr  = mu_i_perp0*E_perp
Ver = mu_e_perp0*E_perp
Vz  = mu_i_para0*E_para

# %% Determination of current density
ji_perp = e*ne*Vr
je_perp = -e*ne*Ver

# %% Determination of perpendicular conductivity
sigma_perp_FD = (1/E_perp)*(ji_perp - je_perp)
# %%
dr_Vr = np.zeros([Nr,Nz])
dr_Vr[0,:]    = 2*(1/(2*step_r))*(-3*Vr[0,:] + 4*Vr[1,:] - Vr[2,:])
dr_Vr[-1,:]   = (-1/(2*step_r))*(-3*Vr[Nr-1,:] + 4*Vr[Nr-2,:] - Vr[Nr-3,:])
dr_Vr[1:-1,:] = (1/(2*step_r))*(Vr[2:,:] - Vr[0:-2,:])

dz_Vz = np.zeros([Nr,Nz])
dz_Vz[:,0]    = (1/(2*step_z))*(-3*Vz[:,0] + 4*Vz[:,1] - Vz[:,2])
dz_Vz[:,-1]   = (-1/(2*step_z))*(-3*Vz[:,Nr-1] + 4*Vz[:,Nr-2] - Vz[:,Nr-3])
dz_Vz[:,1:-1] = (1/(2*step_z))*(Vz[:,2:] - Vz[:,0:-2])

nu_SN = np.zeros([Nr,Nz])
nu_SN[0,:]  = dr_Vr[0,:]  + dz_Vz[0,:]
nu_SN[1:,:] = dr_Vr[1:,:] + dz_Vz[1:,:]

# %%
if picture_phi:
    X,Y = np.meshgrid(z,r/re)
    fig,ax = plt.subplots()
    fig.suptitle(r'$\textrm{FD Scheme}$')
    plt.gcf().subplots_adjust(right = 0.91, left = 0.125, wspace = 0.25, top = 0.925, hspace = 0.1, bottom = 0.14 )        #------------------------------------
    if np.max(phi_FDscheme) > 0:
        levels_pos = np.linspace(0, np.max(phi_FDscheme),10)
        kw_pos = dict(levels=levels_pos, cmap="Oranges", vmin=0, vmax=np.max(phi_FDscheme), origin='lower')
        CF1_pos = ax.contourf(X,Y, phi_FDscheme[:,:], **kw_pos)
        cbar_pos = fig.colorbar(CF1_pos)

    if np.min(phi_FDscheme) <= 0:
        neg_min = np.min(phi_FDscheme[:,:])
        neg_max = np.max(phi_FDscheme[:,:])
        levels_neg = np.linspace(neg_min, neg_max,10)
        kw_neg = dict(levels=levels_neg, cmap="Blues", vmin=neg_min, vmax=neg_max, origin='lower')       
        CF1_neg = ax.contourf(X,Y, phi_FDscheme[:,:], **kw_neg)
        #=========================> CBAR
    if (np.max(phi_FDscheme) > 0 and np.min(phi_FDscheme) < 0):
        cbar_pos = fig.colorbar(CF1_neg)
        cbar_pos.set_label(r'  $\phi(r,z)$ $\mathrm{(V)}$', rotation = 90)
    if (np.max(phi_FDscheme) <= 0 and np.min(phi_FDscheme) < 0):
        cbar_neg = fig.colorbar(CF1_neg)
        cbar_neg.set_label(r'  $\phi(r,z)$ $\mathrm{(V)}$', rotation = 90)
    if (np.max(phi_FDscheme) > 0 and np.min(phi_FDscheme) >= 0):
        cbar_pos = fig.colorbar(CF1_neg)
        cbar_pos.set_label(r'  $\phi(r,z)$ $\mathrm{(V)}$', rotation = 90)
    
    # current density
    # current_density = np.sqrt(ji_r_BT**2+ji_z_BT**2)
    # lw = 2.5*current_density / current_density.max()
    # ax.streamplot(X_v, Y_v, ji_z_BT, ji_r_BT, density=1, color='coral', linewidth=lw)
    
    # (r,z) limit
    ax.set_xlim([-1,0])
    ax.set_ylim([0, rg/re])
    
    # 
    ax.set_xlabel(r'$z/(L/2)$')
    ax.set_ylabel(r'$r/r_e$')

# %%
if picture_Vr:
    #------------------------------------
    fig,ax = plt.subplots()
    plt.gcf().subplots_adjust(right = 0.95, left = 0.105, wspace = 0.25, top = 0.925, hspace = 0.1, bottom = 0.14 )
    V_max = np.max(Vr[:-1,:])
    V_min = np.min(Vr[:-1,:])
    if V_max > 0:
        pos_min = 0
        pos_max = V_max
        levels_pos = np.linspace(pos_min, pos_max,10)
        kw_pos = dict(levels=levels_pos, cmap="Oranges", vmin=pos_min, vmax=pos_max, origin='lower')
        CF1_pos = ax.contourf(X,Y, Vr[:,:], **kw_pos)
    if V_min < 0:
        neg_min = V_min
        neg_max = 0
        levels_neg = np.linspace(neg_min, neg_max,10)
        kw_neg = dict(levels=levels_neg, cmap="Blues", vmin=neg_min, vmax=neg_max, origin='lower')       
        CF1_neg = ax.contourf(X,Y, Vr[:,:], **kw_neg)
    if (V_max > 0 and V_min < 0):
        cbar_pos = fig.colorbar(CF1_pos)
        cbar_pos.set_label(r'  $Vr(r,z)$ $\mathrm{(m \cdot m^{-1})}$', rotation = 90)
        cbar_neg = fig.colorbar(CF1_neg)

    elif (V_max <= 0 and V_min < 0):
        cbar_neg = fig.colorbar(CF1_neg)
        cbar_neg.set_label(r'  $Vr(r,z)$ $\mathrm{(m \cdot m^{-1})}$', rotation = 90)
    elif (V_max > 0 and V_min >= 0):
        cbar_pos = fig.colorbar(CF1_pos)
        cbar_pos.set_label(r'  $Vr(r,z)$ $\mathrm{(m \cdot m^{-1})}$', rotation = 90)
    ax.set_ylabel(r'$r/r_e$')
    ax.set_xlabel(r'$z/[L/2]$')

# %%
if picture_Vz:
    fig,ax = plt.subplots()
    plt.gcf().subplots_adjust(right = 0.95, left = 0.105, wspace = 0.25, top = 0.925, hspace = 0.1, bottom = 0.14 )
    if np.max(Vz[:,:]) > 0:
        pos_min = 0
        pos_max = np.max(Vz[:,:])
        levels_pos = np.linspace(pos_min, pos_max,10)
        kw_pos = dict(levels=levels_pos, cmap="Oranges", vmin=pos_min, vmax=pos_max, origin='lower')
        CF1_pos = ax.contourf(X,Y, Vz[:,:], **kw_pos)
    if np.min(Vz[:,:]) < 0:
        neg_min = np.min(Vz[:,:])
        neg_max = 0
        levels_neg = np.linspace(neg_min, neg_max,10)
        kw_neg = dict(levels=levels_neg, cmap="Blues", vmin=neg_min, vmax=neg_max, origin='lower')       
        CF1_neg = ax.contourf(X,Y, Vz[:,:], **kw_neg)
    if (np.max(Vz[:,:]) > 0 and np.min(Vz[:,:]) < 0):
        cbar_pos = fig.colorbar(CF1_pos)
        cbar_pos.set_label(r'  $Vz(r,z)$ $\mathrm{(m \cdot m^{-1})}$', rotation = 90)
        cbar_neg = fig.colorbar(CF1_neg)
    elif (np.max(Vz[:,:]) <= 0 and np.min(Vz[:,:]) < 0):
        cbar_neg = fig.colorbar(CF1_neg)
        cbar_neg.set_label(r'  $Vz(r,z)$ $\mathrm{(m \cdot m^{-1})}$', rotation = 90)
    elif (np.max(Vz[:,:]) > 0 and np.min(Vz[:,:]) >= 0):
        cbar_pos = fig.colorbar(CF1_pos)
        cbar_pos.set_label(r'  $Vz(r,z)$ $\mathrm{(m \cdot m^{-1})}$', rotation = 90)
    ax.set_ylabel(r'$r/r_e$')
    ax.set_xlabel(r'$z/[L/2]$')
    
# %%
Src = np.abs(nu_SN/nu_in)
if picture_S:
    levels_log = np.logspace( np.log10(np.max(Src*1e-2)), np.log10(np.max(Src)),10)
    kw = dict(levels=levels_log, locator=ticker.LogLocator(), cmap=cm.YlGnBu )
    
    fig,ax = plt.subplots()
    plt.gcf().subplots_adjust(right = 0.835, left = 0.105, wspace = 0.25, top = 0.925, hspace = 0.1, bottom = 0.14 )
    CF1 = ax.contourf(X,Y, Src[:,:], **kw)
    #=========================> CBAR
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.1)
    cax.yaxis.set_major_locator(ticker.NullLocator())
    
    cbar = fig.colorbar(CF1, cax = cax) #, cax = cax, ticks = levels)
    cbar.set_ticks([])
    loc    = levels_log 
    labels = [ sci_notation(levels_log[i]) for i in range(len(levels_log)) ]

    cbar.set_ticks(loc)
    cbar.set_ticklabels(labels)
    cbar.set_label(r'  $S(r,z)$ $\mathrm{(m^{-3} \cdot s^{-1})}$', rotation = 90)
    ax.set_ylabel(r'$r/r_e$')
    ax.set_xlabel(r'$z/[L/2]$')