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
#---------------------: PHYSICAL-CONSTANT
#============================================================================== 
NA          = dict_input.get('NA')                  #      avogadro number       [mol^-1] 
kB          = dict_input.get('kB')                   #      Boltzmann constant
CoulombLog  = dict_input.get('CoulombLog')         #      Coulomnb Logarithm
eps0        = dict_input.get('eps0')
sig0        = dict_input.get('sig0')
e           = dict_input.get('e')                   #      eletric  charge       [C]
me          = dict_input.get('me')                  #      electron mass         [kg]
#============================================================================== 
#---------------------: PLASMA-PARAMETERS
#============================================================================== 
mi_amu      = dict_input.get('mi_amu')              #      argon atom mass       [amu] 
mi          = mi_amu*1.6605e-27                     #      argon atom mass       [kg] 1.6605e-27
eta         = me/mi
Ti          = dict_input.get('Ti')                  #      ion      temperature  [eV]
Tn          = dict_input.get('Tn')                  #      room     temperature  [K]
Te          = dict_input.get('T0')                  #      ion      temperature  [eV]
ne          = dict_input.get('n0')
B           = dict_input.get('B')
P           = dict_input.get('P') #20 #0.15
nN          = (1/(kB*Tn))*P  
#============================================================================== 
#---------------------: EMISSIVE-ELECTRODE
#============================================================================== 
Tw          = dict_input.get('Tw')  
W           = dict_input.get('W')  
j_eth       = 0
Xi          = 0 
phi_e       = dict_input.get('phi_e')
# =============================================================================
# ---------------------: SHEATH PROPERTIES  
# =============================================================================
Cs          = np.sqrt((e*Te)/mi)                    # Bohm Velocity
Lambda      = np.log(np.sqrt(1/(2*np.pi*eta)))      # sheath parameter
jis         = (e*ne*Cs)
#============================================================================== 
#---------------------: GEOMETRICAL-PARAMETERS
#============================================================================== 
rg          = dict_input.get('rg')
L           = dict_input.get('L')
#============================================================================== 
#---------------------: MESHING-PARAMETERS
#============================================================================== 
Nr          = int(dict_input.get('Nr'))
Nre         = int(dict_input.get('Nre'))
Nz          = int(dict_input.get('Nz'))
K           = Nr*Nz
#-->
step_r      = rg/(Nr-1)
r           = np.linspace(0, rg, Nr)
re          = r[Nre-1]
#-->
step_z      = (L/2)/(Nz-1)
z           = np.linspace(-L/2, 0, Nz)
#============================================================================== 
#---------------------: MESHGRID
#============================================================================== 
r_v         = np.linspace(step_r/2, rg-step_r/2, Nr-1)
X_Vr, Y_Vr  = np.meshgrid(z[1:],r_v/re)
#-->
z_v         = np.linspace(-L/2+step_z/2, 0, Nz-1)
X_Vz, Y_Vz  = np.meshgrid(z_v,r[:-1]/re)
#-->
X_v,Y_v     = np.meshgrid(z_v/(L/2),r_v/re)
#-->
X,Y         = np.meshgrid(z,r/re)
#============================================================================== 
#---------------------: Surface
#============================================================================== 
Sr          = np.zeros(Nr)
Sr[0]       = np.pi*(step_r/2)**2
Sr[1:Nre-1] = 2*np.pi*r[1:Nre-1]*step_r
Sr[Nre-1]   = 2*np.pi*re *(step_r/2)
Sr[Nre:-1]  = 2*np.pi*r[Nre:-1]*step_r
Sr[-1]      = 2*np.pi*rg*(step_r/2)
#-->
Selec       = np.pi*re **2
# =============================================================================
# ---------------------: PICTURE  
# =============================================================================
picture_phi = True
picture_Vr  = False
picture_Vz  = False
picture_S   = False
# =============================================================================
# ---------------------: Cyclotron frequency 
# =============================================================================
Omega_i = (e*B)/mi
Omega_e = (e*B)/me
# =============================================================================
# ---------------------: Collision frequency 
# =============================================================================
nu_en = sig0*nN*np.sqrt((8*e*Te)/(np.pi*me))
nu_in = sig0*nN*np.sqrt((8*e*Ti)/(np.pi*mi))
nu_ei = (e**(5/2)*CoulombLog/(6*np.sqrt(2)*np.pi**(3/2)*eps0**2*np.sqrt(me)))*ne*Te**(-3/2)
# =============================================================================
# #--------------------: Perpendicular Conductivity 
# =============================================================================
eta = me/mi
tilde_nu_N = nu_en/nu_in
tilde_nu_I = nu_ei/nu_in
K0 = (Omega_i/nu_in) / (1 + eta*tilde_nu_I)
K1 = (1 + eta*tilde_nu_I)/( (Omega_i/nu_in)**2 + (1 + eta*tilde_nu_I)**2 )
K2 = 1 / ( eta*tilde_nu_N + eta*tilde_nu_I * (1 - ( (eta*tilde_nu_I)/(1 + eta*tilde_nu_I) )*(1 - K1*K0*(Omega_i/nu_in) ) ) )
K3 = 1 / (eta*tilde_nu_N + K2*(Omega_i/nu_in)**2 + eta*tilde_nu_I*K1*(1 + ((K2*(Omega_i/nu_in)**2)/(1 + eta*tilde_nu_I) ) ) )
#--
sigma_perp0 = ((e**2*ne)/(mi*nu_in))*( K1 - eta*tilde_nu_I*K1*K3*(1 - eta*tilde_nu_I*K1*(1 + ((K2*(Omega_i/nu_in)**2)/(1 + eta*tilde_nu_I)) ) ) -
                                         eta*tilde_nu_I*K0*K1*K2*(Omega_i/nu_in)*( ((eta*tilde_nu_I)/(1 + eta*tilde_nu_I))*K1 + K3*(1 - ((eta*tilde_nu_I)/(1 + eta*tilde_nu_I))*K1)*(1 - eta*tilde_nu_I*K1*(1 + ((K2*(Omega_i/nu_in)**2)/(1 + eta*tilde_nu_I)) ))  ) + 
                                         K3*(1 - eta*tilde_nu_I*K1*(1 + ((K2*(Omega_i/nu_in)**2)/(1 + eta*tilde_nu_I)) ) ) )
# =============================================================================
# ---------------------: Parallel Conductivity -----------------------------------------------------------------------
# =============================================================================
alpha_sigma_para = mi*nu_in / (mi*nu_in + me*nu_ei)
sigma_para0 = (e**2*ne/(mi*nu_in + me*nu_ei))*(1 + (1/(alpha_sigma_para + nu_en/nu_ei))*(((mi*nu_in)/(me*nu_ei)) - alpha_sigma_para) )

# =============================================================================
# ---------------------: MOBILITY and DIFFUSION -----------------------------------------------------------------------
# =============================================================================
nu_ei = 0
mu_e_para = e/(me*(nu_ei+nu_en)) #(e/(me*nu_ei))*(alpha_sigma_para/(alpha_sigma_para + tilde_nu_N)) ;
D_e_para = Te*mu_e_para
#-->
mu_e_perp = (me/(e*B**2))*((nu_ei+nu_en)/(1 + (nu_ei+nu_en)**2/Omega_e**2)) #(e/(mi*nu_in))*K3*(1-eta*tilde_nu_I*K1*(1 + K2*(Omega_i/nu_in)**2/(1+eta*tilde_nu_I)));
D_e_perp  = Te*mu_e_perp
#-->
mu_i_para = e/(me*(nu_in)) #(e/(me*nu_ei))*(alpha_sigma_para/(alpha_sigma_para + tilde_nu_N)) ;
D_i_para = Te*mu_i_para
#-->
mu_i_perp = (mi/(e*B**2))*((nu_in)/(1 + (nu_in)**2/Omega_i**2)) #(e/(mi*nu_in))*K3*(1-eta*tilde_nu_I*K1*(1 + K2*(Omega_i/nu_in)**2/(1+eta*tilde_nu_I)));
D_i_perp  = Te*mu_i_perp
# %%
# =============================================================================
# ---------------------: Surface Element ----------------------------------------------------------------
# =============================================================================
A_z_j  = np.zeros(Nr-1)
A_z_j[0]  = np.pi*(r[0]+(step_r/2))**2
A_z_j[1:] = np.pi*( (r[1:-1]+(step_r/2))**2 - (r[1:-1]-(step_r/2))**2)
#-->
A_r_jp    = np.zeros(Nr-1)
A_r_jp[:] = 2*np.pi*(r[:-1]+(step_r/2))*step_z
#-->
A_r_jm     = np.zeros(Nr-1)
A_r_jm[0]  = 0
A_r_jm[1:] = 2*np.pi*(r[1:-1]-(step_r/2))*step_z



# %%
# INITIALISATION DIRECT METHOD SPARSE
dSjVe = 0
# DIMENSION ------------------------------------------------------------------
dim_sheath = 2*Nre+ 2*(Nr-1-Nre) + 1;
dim_mid    = 4*(Nr-1) + 3;
dim_axisym = 4*(Nz-2);
dim_ground = Nz-2;
dim_inside = 5*(Nz*Nr-2*Nz-2*(Nr-2));
dim_matrix_A = dim_sheath + dim_mid + dim_axisym + dim_ground + dim_inside;

row_A  = np.zeros(dim_matrix_A)
col_A  = np.zeros(dim_matrix_A)
data_A = np.zeros(dim_matrix_A)


# %%
# =============================================================================
# AXISYMMETRY BOUNDARY CONDITION -- NEUMANN (grad Phi |r=0  = 0) ----------------------------------------------------------------
# =============================================================================
#-->
sigma_z = e*ne*(mu_e_para + mu_i_para)
sigma_r = e*ne*(mu_e_perp + mu_i_perp)


C_ij  = -(1./step_z)*A_z_j[0]*2*sigma_z - (1./step_r)*A_r_jp[0]*sigma_r
C_ipj =  (1./step_z)*A_z_j[0]*sigma_z
C_imj =  (1./step_z)*A_z_j[0]*sigma_z
C_ijp =  (1./step_r)*A_r_jp[0]*sigma_r
#-->
index = np.where(data_A == 0)[0][0]
    #-- row
row_A[0:4*len(np.arange(1,Nz-1)):4]   = np.arange(1,Nz-1)
row_A[1:1+4*len(np.arange(1,Nz-1)):4] = np.arange(1,Nz-1)
row_A[2:2+4*len(np.arange(1,Nz-1)):4] = np.arange(1,Nz-1)
row_A[3:3+4*len(np.arange(1,Nz-1)):4] = np.arange(1,Nz-1)
    #-- column
col_A[0:4*len(np.arange(1,Nz-1)):4]   = np.arange(1,Nz-1)
col_A[1:1+4*len(np.arange(1,Nz-1)):4] = np.arange(1,Nz-1) + 1
col_A[2:2+4*len(np.arange(1,Nz-1)):4] = np.arange(1,Nz-1) + Nz
col_A[3:3+4*len(np.arange(1,Nz-1)):4] = np.arange(1,Nz-1) - 1
    #-- data    
data_A[0:4*len(np.arange(1,Nz-1)):4]   = C_ij 
data_A[1:1+4*len(np.arange(1,Nz-1)):4] = C_ipj
data_A[2:2+4*len(np.arange(1,Nz-1)):4] = C_ijp
data_A[3:3+4*len(np.arange(1,Nz-1)):4] = C_imj
# %%
# =============================================================================
# SHEATH PLAN -- 1 -- EMISSIVE + BIASING ELECTRODE ----------------------------------------------------------------
# =============================================================================
index = np.where(data_A == 0)[0][0]
index_NBC_nonlinear_electrode_1 = np.where(data_A == 0)[0][0] #!!!!!!!!!!!
C_ij  = -(1./step_z)*sigma_z
C_ipj =  (1./step_z)*sigma_z
#-- row
row_A[index:index+2*len(np.arange(0,Nz*Nre,Nz)):2]     = np.arange(0,Nz*Nre,Nz)
row_A[index+1:index+1+2*len(np.arange(0,Nz*Nre,Nz)):2] = np.arange(0,Nz*Nre,Nz)
#-- column
col_A[index:index+2*len(np.arange(0,Nz*Nre,Nz)):2]     = np.arange(0,Nz*Nre,Nz)
col_A[index+1:index+1+2*len(np.arange(0,Nz*Nre,Nz)):2] = np.arange(0,Nz*Nre,Nz) + 1
#-- data
data_A[index:index+2*len(np.arange(0,Nz*Nre,Nz)):2]     = C_ij
data_A[index+1:index+1+2*len(np.arange(0,Nz*Nre,Nz)):2] = C_ipj

# =============================================================================
# SHEATH PLAN -- 2 -- VOLTAGE DROP ----------------------------------------------------------------
# =============================================================================
C_ij  = -(1./step_z)*sigma_z
C_ipj =  (1./step_z)*sigma_z

index = np.where(data_A == 0)[0][0]

#-- row
row_A[index:index+2*len(np.arange(Nz*Nre,K-Nz,Nz)):2]     = np.arange(Nz*Nre,K-Nz,Nz)
row_A[index+1:index+1+2*len(np.arange(Nz*Nre,K-Nz,Nz)):2] = np.arange(Nz*Nre,K-Nz,Nz)
#-- column
col_A[index:index+2*len(np.arange(Nz*Nre,K-Nz,Nz)):2]     = np.arange(Nz*Nre,K-Nz,Nz)
col_A[index+1:index+1+2*len(np.arange(Nz*Nre,K-Nz,Nz)):2] = np.arange(Nz*Nre,K-Nz,Nz) + 1
#-- data
data_A[index:index+2*len(np.arange(Nz*Nre,K-Nz,Nz)):2]     = C_ij
data_A[index+1:index+1+2*len(np.arange(Nz*Nre,K-Nz,Nz)):2] = C_ipj
# %%
# =============================================================================
# MID PLAN -- NEUMANN (grad Phi |z=0  = 0) ----------------------------------------------------------------
# =============================================================================
C_ij  = -(1./step_z)*A_z_j[0]*sigma_z - (1./step_r)*(A_r_jp[0]/2)*sigma_r
C_imj =  (1./step_z)*A_z_j[0]*sigma_z
C_ijp =  (1./step_r)*(A_r_jp[0]/2)*sigma_r

index = np.where(data_A == 0)[0][0]
row_A[index:index+3:3]     = (Nz-1)
row_A[index+1:index+1+3:3] = (Nz-1)
row_A[index+2:index+2+3:3] = (Nz-1)
    #-- column
col_A[index:index+3:3]     = (Nz-1) 
col_A[index+1:index+1+3:3] = (Nz-1) + Nz
col_A[index+2:index+2+3:3] = (Nz-1) - 1
    #-- data
data_A[index:index+3:3]     = C_ij
data_A[index+1:index+1+3:3] = C_ijp
data_A[index+2:index+2+3:3] = C_imj
# %%
#------------------------------------------------->
C_ij  = -(1./step_z)*A_z_j[1:]*sigma_z - (1./step_r)*sigma_r*((A_r_jp[1:]/2)+(A_r_jm[1:]/2))
C_imj =  (1./step_z)*A_z_j[1:]*sigma_z
C_ijp =  (1./step_r)*(A_r_jp[1:]/2)*sigma_r
C_ijm =  (1./step_r)*(A_r_jm[1:]/2)*sigma_r
index = np.where(data_A == 0)[0][0]
    #-- row
row_A[index:index+4*len(np.arange(2*Nz-1,K-Nz,Nz)):4]     = np.arange(2*Nz-1,K-Nz,Nz)
row_A[index+1:index+1+4*len(np.arange(2*Nz-1,K-Nz,Nz)):4] = np.arange(2*Nz-1,K-Nz,Nz)
row_A[index+2:index+2+4*len(np.arange(2*Nz-1,K-Nz,Nz)):4] = np.arange(2*Nz-1,K-Nz,Nz)
row_A[index+3:index+3+4*len(np.arange(2*Nz-1,K-Nz,Nz)):4] = np.arange(2*Nz-1,K-Nz,Nz)
    #-- column
col_A[index:index+4*len(np.arange(2*Nz-1,K-Nz,Nz)):4]     = np.arange(2*Nz-1,K-Nz,Nz) 
col_A[index+1:index+1+4*len(np.arange(2*Nz-1,K-Nz,Nz)):4] = np.arange(2*Nz-1,K-Nz,Nz) + Nz
col_A[index+2:index+2+4*len(np.arange(2*Nz-1,K-Nz,Nz)):4] = np.arange(2*Nz-1,K-Nz,Nz) - 1
col_A[index+3:index+3+4*len(np.arange(2*Nz-1,K-Nz,Nz)):4] = np.arange(2*Nz-1,K-Nz,Nz) - Nz
    #-- data
data_A[index:index+4*len(np.arange(2*Nz-1,K-Nz,Nz)):4]     = C_ij
data_A[index+1:index+1+4*len(np.arange(2*Nz-1,K-Nz,Nz)):4] = C_ijp
data_A[index+2:index+2+4*len(np.arange(2*Nz-1,K-Nz,Nz)):4] = C_imj
data_A[index+3:index+3+4*len(np.arange(2*Nz-1,K-Nz,Nz)):4] = C_ijm


# =============================================================================
# GROUND BC ----------------------------------------------------------------
# =============================================================================
index = np.where(data_A == 0)[0][0]
row_A[index:index+Nz] = np.arange(K-Nz,K)
col_A[index:index+Nz] = np.arange(K-Nz,K)
data_A[index:index+Nz] = 1



# %%
# =============================================================================
# ---------------------: INSIDE THE DOMAIN - ZONE 1 :  EMISSIVE + BIASED ELECTRODE ><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
# =============================================================================
Z,R = np.meshgrid(z, r)

vect_fiveScheme_zoneR1 = np.zeros((Nr-2)*(Nz-2)) # np.zeros((Nz*M)-2*M-2*(Nz-2))
index_vect_fiveScheme_zoneR1 = 0
for i in np.arange(Nz+1,Nz*(Nr-1)-1):
    if i % Nz != 0 and (i+1) % Nz != 0 :
        vect_fiveScheme_zoneR1[index_vect_fiveScheme_zoneR1] = i
        index_vect_fiveScheme_zoneR1 += 1
# %%
AZJ = np.pi*( (R[1:-1,1:-1]+(step_r/2))**2 - (R[1:-1,1:-1]-(step_r/2))**2)
ARJp = 2*np.pi*(R[1:-1, 1:-1]+(step_r/2))*step_z
ARJm = 2*np.pi*(R[1:-1, 1:-1]-(step_r/2))*step_z
# %%
#-- coef DIRECT
Cij  = (-(1./step_z)*AZJ*2*sigma_z - (1./step_r)*sigma_r*(ARJp+ARJm))
Cipj = (1./step_z)*AZJ*sigma_z
Cimj = (1./step_z)*AZJ*sigma_z
Cijp = (1./step_r)*sigma_r*ARJp
Cijm = (1./step_r)*sigma_r*ARJm
# %%
# ---------------------: INSIDE THE DOMAIN  -----------------------------------------------------------------------
index = np.where(data_A == 0)[0][0]
# row
row_A[index:index+5*len(vect_fiveScheme_zoneR1):5]     = vect_fiveScheme_zoneR1
row_A[index+1:index+1+5*len(vect_fiveScheme_zoneR1):5] = vect_fiveScheme_zoneR1
row_A[index+2:index+2+5*len(vect_fiveScheme_zoneR1):5] = vect_fiveScheme_zoneR1
row_A[index+3:index+3+5*len(vect_fiveScheme_zoneR1):5] = vect_fiveScheme_zoneR1
row_A[index+4:index+4+5*len(vect_fiveScheme_zoneR1):5] = vect_fiveScheme_zoneR1
#-- column
col_A[index:index+5*len(vect_fiveScheme_zoneR1):5]     = vect_fiveScheme_zoneR1
col_A[index+1:index+1+5*len(vect_fiveScheme_zoneR1):5] = vect_fiveScheme_zoneR1 + 1
col_A[index+2:index+2+5*len(vect_fiveScheme_zoneR1):5] = vect_fiveScheme_zoneR1 + Nz
col_A[index+3:index+3+5*len(vect_fiveScheme_zoneR1):5] = vect_fiveScheme_zoneR1 - 1
col_A[index+4:index+4+5*len(vect_fiveScheme_zoneR1):5] = vect_fiveScheme_zoneR1 - Nz
#-- data
data_A[index:index+5*len(vect_fiveScheme_zoneR1):5]     = Cij.reshape(1, (Nz-2)*(Nr-2) )
data_A[index+1:index+1+5*len(vect_fiveScheme_zoneR1):5] = Cipj.reshape(1, (Nz-2)*(Nr-2) )
data_A[index+2:index+2+5*len(vect_fiveScheme_zoneR1):5] = Cijp.reshape(1, (Nz-2)*(Nr-2) )
data_A[index+3:index+3+5*len(vect_fiveScheme_zoneR1):5] = Cimj.reshape(1, (Nz-2)*(Nr-2) )
data_A[index+4:index+4+5*len(vect_fiveScheme_zoneR1):5] = Cijm.reshape(1, (Nz-2)*(Nr-2) )


# %%
# =============================================================================
# MATRIX A----------------------------------------------------------------
# =============================================================================
sparse_matrix_A = spsp.csr_matrix((data_A,(row_A,col_A)),shape=(K,K))
# matrix_A_Python = sparse_matrix_A.todense()

# %%


# %%
row_BC  = np.zeros(Nre) 
col_BC  = np.zeros(Nre) 
data_BC = np.zeros(Nre) 
#--
row_BC[:]  = np.arange(0,Nz*Nre,Nz)
col_BC[:]  = 0  
data_BC[:] = jis
#--
vect_b = spsp.csr_matrix((data_BC,(row_BC,col_BC)),shape=(K,1))
# vect_b = vect_b.todense()
# =============================================================================
#         LINEAR SOLVE 
# =============================================================================
phi_FVscheme = spsp.linalg.spsolve(sparse_matrix_A, vect_b)
phi_FVscheme = phi_FVscheme.reshape(Nr,Nz)
# %%--> 
# sigma_r = 0.00244301
R_perp = np.log(rg/re)/(np.pi*L*sigma_r)
Iis = np.sum(jis*Sr[:Nre])
chi = Te/(Iis*R_perp)
phi_th = -(1/chi)*(1 + (1/(2*chi*np.log(rg/re))))
print("theortical solution: ", phi_th, "V")


# %%
"""
=========================== Determination of perpendicular electric field =================================================================================
"""   
# fig, ax = plt.subplots( constrained_layout = True, figsize = (2*2.20, 2.75))
# ax.plot(r[:Nre]/re, phi_FVscheme[:Nre,0], color = "black")
# ax.set_xlabel(r'$r/r_e$')
# ax.set_ylabel(r'$\textrm{Radial potential profile}$ $\phi_{sh}(r)$ $\mathrm{(V)}$')

E_perp_FV = np.zeros([Nr-1,Nz-1])
E_perp_FV[:,:] = (-1)*(1/(step_r))*(phi_FVscheme[1:,1:] - phi_FVscheme[:-1,1:])

E_para_FV = np.zeros([Nr-1,Nz-1])
E_para_FV[:,:] = (-1)*(1/(step_z))*(phi_FVscheme[:-1,1:] - phi_FVscheme[:-1,:-1])


# %% Determination of velocity
mu_i_para0 = e/(mi*(nu_in)) 
mu_i_perp0 = (mi/(e*B**2))*((nu_in)/(1 + (nu_in)**2/Omega_i**2)) 

# Vr_FV = np.zeros([Nr,Nz])
Vr_FV = mu_i_perp0*E_perp_FV

# Vz_FV = np.zeros([Nr,Nz])
Vz_FV = mu_i_para0*E_para_FV


# %%
"""
=========================== Determination of current density =================================================================================
"""    
# Parallel ion current density
ji_r_FV = np.zeros([Nr-1, Nz-1]) # ji_perp_BT = e n Vr_FV

for i in np.arange(Nz-1):
    for j in np.arange(Nr-1):
        nv_r = max(Vr_FV[j,i], 0)*ne + min(Vr_FV[j,i], 0)*ne
        ji_r_FV[j,i] = e*nv_r


# Parallel ion current density
ji_z_FV = np.zeros([Nr-1, Nz-1]) # ji_para_BT = e n Vz_FV

for i in np.arange(Nz-1):
    for j in np.arange(Nr-1):
        nv_z= max(Vz_FV[j,i], 0)*ne + min(Vz_FV[j,i], 0)*ne 
        ji_z_FV[j,i] = e*nv_z
# %%
"""
=========================== Determination of perpendicular conductivity =================================================================================
"""    
sigma_perp_FV = (ji_r_FV/E_perp_FV) + e*ne*mu_e_perp


# %%
#-->
nu_en = sig0*nN*np.sqrt((8*e*Te)/(np.pi*me))
nu_ei = np.zeros([Nr, Nz]) #(e**(5/2)*CoulombLog/(6*np.sqrt(2)*np.pi**(3/2)*eps0**2*np.sqrt(me)))*ne*Te**(-3/2)
mu_e_perp = (me/(e*B**2))*((nu_ei+nu_en)/(1 + (nu_ei+nu_en)**2/Omega_e**2)) 
#-->
# for i in np.arange(Nz):
#     for j in np.arange(Nr):
#         if i == 0:
#             mean_Vr = 0
#         if j == 0:
#             mean_Vr = Vr_FV[j,i-1]
#         elif j == Nr-1:
#             mean_Vr = Vr_FV[j-1,i-1]
#         elif (i > 0 and j != 0 and j != Nr-1):
#             mean_Vr = (1/2)*(Vr_FV[j,i-1] + Vr_FV[j-1,i-1])
        
#         sigma_perp_BT[j,i] = (-e*ne*mean_Vr/((-1)*E_perp_FV[j,i])) + e*ne*mu_e_perp[j,i]
#-->
eta = me/mi
tilde_nu_N = nu_en/nu_in
tilde_nu_I = nu_ei/nu_in
K0 = (Omega_i/nu_in) / (1 + eta*tilde_nu_I)
K1 = (1 + eta*tilde_nu_I)/( (Omega_i/nu_in)**2 + (1 + eta*tilde_nu_I)**2 )
K2 = 1 / ( eta*tilde_nu_N + eta*tilde_nu_I * (1 - ( (eta*tilde_nu_I)/(1 + eta*tilde_nu_I) )*(1 - K1*K0*(Omega_i/nu_in) ) ) )
K3 = 1 / (eta*tilde_nu_N + K2*(Omega_i/nu_in)**2 + eta*tilde_nu_I*K1*(1 + ((K2*(Omega_i/nu_in)**2)/(1 + eta*tilde_nu_I) ) ) )
#--
sigma_perp = ((e**2*ne)/(mi*nu_in))*( K1 - eta*tilde_nu_I*K1*K3*(1 - eta*tilde_nu_I*K1*(1 + ((K2*(Omega_i/nu_in)**2)/(1 + eta*tilde_nu_I)) ) ) -
                                          eta*tilde_nu_I*K0*K1*K2*(Omega_i/nu_in)*( ((eta*tilde_nu_I)/(1 + eta*tilde_nu_I))*K1 + K3*(1 - ((eta*tilde_nu_I)/(1 + eta*tilde_nu_I))*K1)*(1 - eta*tilde_nu_I*K1*(1 + ((K2*(Omega_i/nu_in)**2)/(1 + eta*tilde_nu_I)) ))  ) + 
                                          K3*(1 - eta*tilde_nu_I*K1*(1 + ((K2*(Omega_i/nu_in)**2)/(1 + eta*tilde_nu_I)) ) ) )

Sigma_Pedersen = (e**2*ne/me)*((eta*nu_in)/(Omega_i**2 + nu_in**2))


# %%
# list_color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'darkorange', 'forestgreen', 'darkred', 'royalblue', 'gold', 'plum']
# fig, ax = plt.subplots( constrained_layout = True, figsize = (2*2.20, 2.75))
# for index, i in enumerate(np.arange(1,16)):
#     ax.scatter((-1)*E_perp_FV[1:-1,i], sigma_perp_FV, s = 10, c = list_color[index], marker = "o")
#     ax.axhline( y = Sigma_Pedersen, color = "crimson", linestyle = "dashed")
# ax.set_xlabel(r'$\partial \phi/\partial r$ $\mathrm{(V \cdot m^{-1})}$')
# ax.set_ylabel(r'$\sigma_\perp$ $\mathrm{(\Omega^{-1} \cdot m^{-1})}$')
# vareps = 1e-5
# ax.set_ylim([Sigma_Pedersen-vareps, Sigma_Pedersen+vareps])
# ax.set_yscale('log')

# %%
"""
=========================== PICTURE =================================================================================
"""
if picture_Vr:
    #------------------------------------
    fig,ax = plt.subplots()
    plt.gcf().subplots_adjust(right = 0.95, left = 0.105, wspace = 0.25, top = 0.925, hspace = 0.1, bottom = 0.14 )
    V_max = np.max(Vr_FV[:-1,:])
    V_min = np.min(Vr_FV[:-1,:])
    if V_max > 0:
        pos_min = 0
        pos_max = V_max
        levels_pos = np.linspace(pos_min, pos_max,10)
        kw_pos = dict(levels=levels_pos, cmap="Oranges", vmin=pos_min, vmax=pos_max, origin='lower')
        CF1_pos = ax.contourf(X_v,Y_v, Vr_FV[:,:], **kw_pos)
    if V_min < 0:
        neg_min = V_min
        neg_max = 0
        levels_neg = np.linspace(neg_min, neg_max,10)
        kw_neg = dict(levels=levels_neg, cmap="Blues", vmin=neg_min, vmax=neg_max, origin='lower')       
        CF1_neg = ax.contourf(X_v,Y_v, Vr_FV[:,:], **kw_neg)
    if (V_max > 0 and V_min < 0):
        cbar_pos = fig.colorbar(CF1_pos)
        cbar_pos.set_label(r'  $V_r(r,z)$ $\mathrm{(m \cdot m^{-1})}$', rotation = 90)
        cbar_neg = fig.colorbar(CF1_neg)

    elif (V_max <= 0 and V_min < 0):
        cbar_neg = fig.colorbar(CF1_neg)
        cbar_neg.set_label(r'  $V_r(r,z)$ $\mathrm{(m \cdot m^{-1})}$', rotation = 90)
    elif (V_max > 0 and V_min >= 0):
        cbar_pos = fig.colorbar(CF1_pos)
        cbar_pos.set_label(r'  $V_r(r,z)$ $\mathrm{(m \cdot m^{-1})}$', rotation = 90)
    ax.set_ylabel(r'$r/r_e$')
    ax.set_xlabel(r'$z/[L/2]$')

     
# -------------------------------------------------------------------------------------------------------
if picture_Vz:
    fig,ax = plt.subplots()
    plt.gcf().subplots_adjust(right = 0.95, left = 0.105, wspace = 0.25, top = 0.925, hspace = 0.1, bottom = 0.14 )
    if np.max(Vz_FV[:,:]) > 0:
        pos_min = 0
        pos_max = np.max(Vz_FV[:,:])
        levels_pos = np.linspace(pos_min, pos_max,10)
        kw_pos = dict(levels=levels_pos, cmap="Oranges", vmin=pos_min, vmax=pos_max, origin='lower')
        CF1_pos = ax.contourf(X_v,Y_v, Vz_FV[:,:], **kw_pos)
    if np.min(Vz_FV[:,:]) < 0:
        neg_min = np.min(Vz_FV[:,:])
        neg_max = 0
        levels_neg = np.linspace(neg_min, neg_max,10)
        kw_neg = dict(levels=levels_neg, cmap="Blues", vmin=neg_min, vmax=neg_max, origin='lower')       
        CF1_neg = ax.contourf(X_v,Y_v, Vz_FV[:,:], **kw_neg)
    if (np.max(Vz_FV[:,:]) > 0 and np.min(Vz_FV[:,:]) < 0):
        cbar_pos = fig.colorbar(CF1_pos)
        cbar_pos.set_label(r'  $V_z(r,z)$ $\mathrm{(m \cdot m^{-1})}$', rotation = 90)
        cbar_neg = fig.colorbar(CF1_neg)
    elif (np.max(Vz_FV[:,:]) <= 0 and np.min(Vz_FV[:,:]) < 0):
        cbar_neg = fig.colorbar(CF1_neg)
        cbar_neg.set_label(r'  $V_z(r,z)$ $\mathrm{(m \cdot m^{-1})}$', rotation = 90)
    elif (np.max(Vz_FV[:,:]) > 0 and np.min(Vz_FV[:,:]) >= 0):
        cbar_pos = fig.colorbar(CF1_pos)
        cbar_pos.set_label(r'  $V_z(r,z)$ $\mathrm{(m \cdot m^{-1})}$', rotation = 90)
    ax.set_ylabel(r'$r/r_e$')
    ax.set_xlabel(r'$z/[L/2]$')
    
    #-------------------------------------------------------------------------------------------------------


if picture_phi:
    fig,ax = plt.subplots()
    fig.suptitle(r'$\textrm{FV Scheme}$')
    plt.gcf().subplots_adjust(right = 0.91, left = 0.125, wspace = 0.25, top = 0.925, hspace = 0.1, bottom = 0.14 )        #------------------------------------
    if np.max(phi_FVscheme) > 0:
        levels_pos = np.linspace(0, np.max(phi_FVscheme),10)
        kw_pos = dict(levels=levels_pos, cmap="Oranges", vmin=0, vmax=np.max(phi_FVscheme), origin='lower')
        CF1_pos = ax.contourf(X,Y, phi_FVscheme[:,:], **kw_pos)
        cbar_pos = fig.colorbar(CF1_pos)

    if np.min(phi_FVscheme) < 0:
        neg_min = np.min(phi_FVscheme[:,:])
        neg_max = np.max(phi_FVscheme[:,:])
        levels_neg = np.linspace(neg_min, neg_max,10)
        kw_neg = dict(levels=levels_neg, cmap="Blues", vmin=neg_min, vmax=neg_max, origin='lower')       
        CF1_neg = ax.contourf(X,Y, phi_FVscheme[:,:], **kw_neg)
        #=========================> CBAR
    if (np.max(phi_FVscheme) > 0 and np.min(phi_FVscheme) < 0):
        cbar_pos = fig.colorbar(CF1_neg)
        cbar_pos.set_label(r'  $\phi(r,z)$ $\mathrm{(V)}$', rotation = 90)
    if (np.max(phi_FVscheme) <= 0 and np.min(phi_FVscheme) < 0):
        cbar_neg = fig.colorbar(CF1_neg)
        cbar_neg.set_label(r'  $\phi(r,z)$ $\mathrm{(V)}$', rotation = 90)
    if (np.max(phi_FVscheme) > 0 and np.min(phi_FVscheme) >= 0):
        cbar_pos = fig.colorbar(CF1_neg)
        cbar_pos.set_label(r'  $\phi(r,z)$ $\mathrm{(V)}$', rotation = 90)
    
    # current density
    current_density = np.sqrt(ji_r_FV**2+ji_z_FV**2)
    lw = 1*current_density / current_density.max()
    ax.streamplot(X_v, Y_v, ji_z_FV, ji_r_FV, density=1, color='coral', linewidth=lw)
    
    # (r,z) limit
    ax.set_xlim([-1,0])
    ax.set_ylim([0, rg/re])
    
    # 
    ax.set_xlabel(r'$z/(L/2)$')
    ax.set_ylabel(r'$r/r_e$')
# %%
# if Read_Source_Term:
#     levels_log = np.logspace( np.log10(np.max(Src*1e-2)), np.log10(np.max(Src)),10)
#     kw = dict(levels=levels_log, locator=ticker.LogLocator(), cmap=cm.YlGnBu )
    
#     fig,ax = plt.subplots()
#     plt.gcf().subplots_adjust(right = 0.835, left = 0.105, wspace = 0.25, top = 0.925, hspace = 0.1, bottom = 0.14 )
#     CF1 = ax.contourf(X,Y, Src[:,:], **kw)
#     #=========================> CBAR
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="2.5%", pad=0.1)
#     cax.yaxis.set_major_locator(ticker.NullLocator())
    
#     cbar = fig.colorbar(CF1, cax = cax) #, cax = cax, ticks = levels)
#     cbar.set_ticks([])
#     loc    = levels_log 
#     labels = [ sci_notation(levels_log[i]) for i in range(len(levels_log)) ]

#     cbar.set_ticks(loc)
#     cbar.set_ticklabels(labels)
#     cbar.set_label(r'  $S(r,z)$ $\mathrm{(m^{-3} \cdot s^{-1})}$', rotation = 90)
#     ax.set_ylabel(r'$r/r_e$')
#     ax.set_xlabel(r'$z/[L/2]$')




