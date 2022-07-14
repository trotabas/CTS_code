# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 11:40:17 2020
@author: TROTABAS Baptiste
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import floor, log10
#from Main_function_contourf import *
import csv
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib import ticker, cm
from matplotlib.colors import LogNorm
from matplotlib.colors import BoundaryNorm
import pandas as pd
from scipy import interpolate
# from scipy.interpolate import interp2d

def cm2inch(value):
    return value/2.54

def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    if exponent is None:
        exponent = int(floor(log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits

    return r"${0:.{2}f}\cdot 10^{{{1:d}}}$".format(coeff, exponent, precision)

def sci_contour(num, decimal_digits=0, precision=None, exponent=None):
    if exponent is None:
        exponent = int(floor(log10(abs(num))))
    coeff = round(num / float(10**exponent), 0)
    if precision is None:
        precision = decimal_digits

    return coeff*10**exponent

def sci_output(num, decimal_digits=1, precision=None, exponent=None):
    if exponent is None:
        exponent = int(floor(log10(abs(num))))
    if precision is None:
        precision = decimal_digits
    if exponent < 0:
        if np.abs(exponent) < 10:
            return (r'1e-0%s' %np.abs(exponent) )
        else:
            return (r'1e%s' % exponent )

def where_better(table, element):
    for element_table in table:
        if element_table < element:
            return np.where( table <  element )[0][0]
    return -1

origin = "lower"        
#Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata = dict(artis='btrotabas'),bitrate=-1)
dpi = 300
# %%
"""
============== FOLDERNAME ========================================================================================
1  - 
==================================================================================================================
"""
Folder = "Test03" 
#-->
Read_CTS = True
path     = "Output/" + Folder + "/Final/"
#-->
Read_GH    = False
#-->
path_input  = "Output/"+ Folder +"/"
InputFolder = "input.dat"
csv.register_dialect('skip_space', skipinitialspace=True)
#-->
path_graph = "Graph/"+Folder+"/Version/";
#-->
Read_Source_Term = False
#-->
picture_Vr     = True
picture_Vtheta = False
picture_Vz     = False
picture_ne     = False
picture_phi    = True
#-->
animation_ne     = False
animation_phi    = False
animation_Vr     = False
animation_Vtheta = False
animation_Vz     = False

# %%
datContent = [i.strip().split() for i in open(path_input + InputFolder).readlines()]
dict_input = {}
for i in open(path_input + InputFolder).readlines():
    x = i.strip().split() 
    if x[0][0] != '/':
        dict_input[x[0]] = np.float64(x[1])
#==============================================================================================================
#---------------------:PHYSICAL-CONSTANT-----------------------------------------------------------------------
#==============================================================================================================
NA         = dict_input.get('NA')      #      avogadro number       [mol^-1] 
kB         = dict_input.get('kB')    #      Boltzmann constant
CoulombLog = dict_input.get('CoulombLog')         #      Coulomnb Logarithm
eps0       = dict_input.get('eps0')
sig0       = dict_input.get('sig0')
e          = dict_input.get('e')           #      eletric  charge       [C]
me         = dict_input.get('me')       #      electron mass         [kg]
#==============================================================================================================
#---------------------:PLASMA-PARAMETERS-----------------------------------------------------------------------
#==============================================================================================================
mi_amu     = dict_input.get('mi_amu')            #      argon atom mass       [amu] 
mi         = mi_amu*1.6605e-27 #      argon atom mass       [kg] 1.6605e-27
eta        = me/mi
Ti = dict_input.get('Ti')               #      ion      temperature  [eV]
Tn = dict_input.get('Tn')                #      room     temperature  [K]
Te = 3                  #      ion      temperature  [eV]
ne = 1e18 #1e18
B  = dict_input.get('B')
P  = dict_input.get('P') #20 #0.15
nN = (1/(kB*Tn))*P  
#==============================================================================================================
#---------------------:EMISSIVE-ELECTRODE----------------------------------------------------------------------
#==============================================================================================================
Tw    = dict_input.get('Tw')  
W     = dict_input.get('W')  
j_eth = 0
phi_e = dict_input.get('phi_e')
#==============================================================================================================
#---------------------:GEOMETRICAL-PARAMETERS----------------------------------------------------------------------
#==============================================================================================================
rg = dict_input.get('rg')
L = dict_input.get('L')
#==============================================================================================================
#---------------------:MESHING-PARAMETERS----------------------------------------------------------------------
#==============================================================================================================
Nr     = int(dict_input.get('Nr'))
Nre    = 16 #int(dict_input.get('Nre'))
Nz     = int(dict_input.get('Nz'))

#-->
step_r = rg/(Nr-1)
r = np.linspace(0, rg, Nr)
#-->
re = r[Nre-1]
#-->
step_z = (L/2)/(Nz-1)
z = np.linspace(-L/2, 0, Nz)
#-->
K=Nr*Nz
#==============================================================================================================
#---------------------:TIME-SETTING----------------------------------------------------------------------
#==============================================================================================================
step_t = dict_input.get('step_t')
# max_iteration = int(dict_input.get('max_iteration'))
iteration_save = int(dict_input.get('iteration_save'))
#-->
max_iteration = int(1888*iteration_save+1)
# mesh: 31x31, Ieth = 15A, max_iteration = 536
# mesh: 31x31, Ieth = 25A, max_iteration = 535
#==============================================================================================================
#---------------------:SOURCE-TERM----------------------------------------------------------------------
#==============================================================================================================
S = 0
# =============================================================================
# ---------------------: SHEATH PROPERTIES  -----------------------------------------------------------------
# =============================================================================
Cs = np.sqrt((e*Te)/mi)                   # Bohm Velocity
Lambda = np.log(np.sqrt(1/(2*np.pi*eta))) # sheath parameter
jis = (e*ne*Cs)

# =============================================================================
# ---------------------: Cyclotron frequency ----------------------------------------------------------------
# =============================================================================
Omega_i = (e*B)/mi
Omega_e = (e*B)/me
# =============================================================================
# ---------------------: Collision frequency ----------------------------------------------------------------
# =============================================================================
nu_en = sig0*nN*np.sqrt((8*e*Te)/(np.pi*me))
nu_in = sig0*nN*np.sqrt((8*e*Ti)/(np.pi*mi))
nu_ei = (e**(5/2)*CoulombLog/(6*np.sqrt(2)*np.pi**(3/2)*eps0**2*np.sqrt(me)))*ne*Te**(-3/2)
# =============================================================================
# ---------------------: MOBILITY and DIFFUSION -----------------------------------------------------------------------
# =============================================================================
mu_e_para0 = e/(me*(nu_ei+nu_en)) 
D_e_para0 = Te*mu_e_para0
#-->
mu_e_perp_CTS0 = (me/(e*B**2))*((nu_ei+nu_en)/(1 + (nu_ei+nu_en)**2/Omega_e**2)) 
D_E_perp_CTS0  = Te*mu_e_perp_CTS0
#-->
mu_i_para0 = e/(mi*(nu_in)) 
mu_i_perp0 = (mi/(e*B**2))*((nu_in)/(1 + (nu_in)**2/Omega_i**2)) 
# =============================================================================
# #--------------------: Conductivity -----------------------------------------------------------------------
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
#-->
alpha_sigma_para = mi*nu_in / (mi*nu_in + me*nu_ei)
sigma_para0 = (e**2*ne/(mi*nu_in + me*nu_ei))*(1 + (1/(alpha_sigma_para + nu_en/nu_ei))*(((mi*nu_in)/(me*nu_ei)) - alpha_sigma_para) )
#-->
tau = (L/rg)*np.sqrt(sigma_perp0/sigma_para0)


# %%
#-->
X,Y = np.meshgrid(z/(L/2),r/re)
#-->
r_v = np.linspace(step_r/2, rg-step_r/2, Nr-1)
z_v = np.linspace(-L/2+step_z/2, 0, Nz-1)
X_v,Y_v = np.meshgrid(z_v/(L/2),r_v/re)
#-->
dim_for_scalar_field = np.array([int((max_iteration-1)/iteration_save)+1, Nr, Nz])
phi = np.zeros(dim_for_scalar_field)
ne = np.zeros(dim_for_scalar_field)
Src = np.zeros([Nr, Nz])
#--> Separator
if (Read_CTS):
    phi_CTS    = np.zeros([Nr, Nz])
    n_CTS     = np.zeros([Nr, Nz])
    Vr_CTS     = np.zeros([Nr-1, Nz-1])
    Vtheta_CTS = np.zeros([Nr-1, Nz-1])
    Vz_CTS     = np.zeros([Nr-1, Nz-1])
if (Read_GH):
    phi_GH    = np.zeros([Nr, Nz])
    ne_GH     = np.zeros([Nr, Nz])
    Vr_GH     = np.zeros([Nr-1, Nz-1])
    Vtheta_GH = np.zeros([Nr-1, Nz-1])
    Vz_GH     = np.zeros([Nr-1, Nz-1])
    phi_GH_inv    = np.zeros([Nr, Nz])
    ne_GH_inv     = np.zeros([Nr, Nz])
    Vr_GH_inv     = np.zeros([Nr-1, Nz-1])
    Vtheta_GH_inv = np.zeros([Nr-1, Nz-1])
    Vz_GH_inv     = np.zeros([Nr-1, Nz-1])
#--> mean density
ne_mean = np.zeros(int((max_iteration-1)/iteration_save)+1)
ne_diff = np.zeros(int((max_iteration-1)/iteration_save))
#-->
dim_for_current_density = np.array([int((max_iteration-1)/iteration_save)+1, Nre])
j_z_sh_electrode_CTS_iteration    = np.zeros(dim_for_current_density)
j_z_sh_electrode_BC_CTS_iteration = np.zeros(dim_for_current_density)
# j_z_sh_drop         = np.zeros(np.array([int((max_iteration-1)/iteration_save)+1, int(Nr-Nre)]))
#-->
dim_for_vector_field = np.array([int((max_iteration-1)/iteration_save)+1, Nr-1, Nz-1])
Vr      = np.zeros(dim_for_vector_field)
Vtheta  = np.zeros(dim_for_vector_field)
Vz      = np.zeros(dim_for_vector_field)
#-->
ne_anim   = np.zeros([Nr, Nz])
phi_anim  = np.zeros([Nr, Nz])
Vr_anim      = np.zeros([Nr-1, Nz-1])
Vtheta_anim  = np.zeros([Nr-1, Nz-1])
Vz_anim      = np.zeros([Nr-1, Nz-1])
#-->
#---------------------> Surface
Sr = np.zeros(Nr)
Sr[0]         = np.pi*(step_r/2)**2
Sr[1:Nre-1]  = 2*np.pi*r[1:Nre-1]*step_r
Sr[Nre-1]    = 2*np.pi*re*(step_r/2)
Sr[Nre:-1]   = 2*np.pi*r[Nre:-1]*step_r
Sr[-1]        = 2*np.pi*rg*(step_r/2)

# %%
#-----> MPS-Potential (iteration)  <------------------------------------------------------
if (Read_CTS):
    with open(path + 'S_CTS.csv', 'r') as f:
        reader_S = csv.reader(f , delimiter=' ', dialect='skip_space')
        entries_S  = list(reader_S)
        #-->
        for i in np.arange(Nz):
            for j in np.arange(Nr):
                Src[j, i] = np.float64(entries_S[j+1][i])
        print("max[S(r,z)] = ", np.max(Src), " m-3 s-1")
        
      

    with open(path + 'phi_CTS.csv', 'r') as f:
        reader_phi=csv.reader(f , delimiter=' ', dialect='skip_space')
        entries_phi = list(reader_phi)
        #-->
        for i in np.arange(Nz):
            for j in np.arange(Nr):
                phi_CTS[j, i] = np.float64(entries_phi[j+1][i])
   
    with open(path + 'ne_CTS.csv', 'r') as f:
        reader_ne=csv.reader(f , delimiter=' ', dialect='skip_space')
        entries_ne = list(reader_ne)
        #-->
        for i in np.arange(Nz):
            for j in np.arange(Nr):
                n_CTS[j, i] = np.float64(entries_ne[j+1][i])
                
    with open(path + 'Vr_CTS.csv', 'r') as f:
        reader_Vr  = csv.reader(f , delimiter=' ', dialect='skip_space')
        entries_Vr = list(reader_Vr)
        #-->
        for i in np.arange(Nz-1):
            for j in np.arange(Nr-1):
                Vr_CTS[j, i] = np.float64(entries_Vr[j+1][i])
   
    with open(path + 'Vtheta_CTS.csv', 'r') as f:
        reader_Vtheta  = csv.reader(f , delimiter=' ', dialect='skip_space')
        entries_Vtheta = list(reader_Vtheta)
        #-->
        for i in np.arange(Nz-1):
            for j in np.arange(Nr-1):
                Vtheta_CTS[j, i] = np.float64(entries_Vtheta[j+1][i])
                
    with open(path + 'Vz_CTS.csv', 'r') as f:
        reader_Vz  = csv.reader(f , delimiter=' ', dialect='skip_space')
        entries_Vz = list(reader_Vz)
        #-->
        for i in np.arange(Nz-1):
            for j in np.arange(Nr-1):
                Vz_CTS[j, i] = np.float64(entries_Vz[j+1][i])   
      
# %%
if (Read_GH):      
    file = path + 'phi_GH.csv'
    with open(file, 'r') as f:
        reader=csv.reader(f , delimiter=' ', dialect='skip_space')
        entries = list(reader)
        #-->
        for i in np.arange(Nz):
            for j in np.arange(Nr):
                phi_GH_inv[j,i] = np.float64(entries[j+1][i])
                
    file = path + 'ne_GH.csv'
    with open(file, 'r') as f:
        reader=csv.reader(f , delimiter=' ', dialect='skip_space')
        entries = list(reader)
        #-->
        for i in np.arange(Nz):
            for j in np.arange(Nr):
                ne_GH_inv[j,i] = np.float64(entries[j+1][i])    

    for i in np.arange(Nz):
        cpt = -1
        for j in np.arange(Nr):
            phi_GH[cpt,i] = phi_GH_inv[j,i]
            ne_GH[cpt,i] = ne_GH_inv[j,i]
            cpt += -1
            
    with open(path + 'Vr_GH.csv', 'r') as f:
        reader_Vr  = csv.reader(f , delimiter=' ', dialect='skip_space')
        entries_Vr = list(reader_Vr)
        #-->
        for i in np.arange(Nz-1):
            for j in np.arange(Nr-1):
                Vr_GH_inv[j, i] = float(entries_Vr[j+1][i])
   
    with open(path + 'Vtheta_GH.csv', 'r') as f:
        reader_Vtheta  = csv.reader(f , delimiter=' ', dialect='skip_space')
        entries_Vtheta = list(reader_Vtheta)
        #-->
        for i in np.arange(Nz-1):
            for j in np.arange(Nr-1):
                Vtheta_GH_inv[j, i] = float(entries_Vtheta[j+1][i])
                
    with open(path + 'Vz_GH.csv', 'r') as f:
        reader_Vz  = csv.reader(f , delimiter=' ', dialect='skip_space')
        entries_Vz = list(reader_Vz)
        #-->
        for i in np.arange(Nz-1):
            for j in np.arange(Nr-1):
                Vz_GH_inv[j, i] = float(entries_Vz[j+1][i])   
                
    for i in np.arange(Nz-1):
        cpt = -1
        for j in np.arange(Nr-1):
            Vr_GH[cpt,i] = Vr_GH_inv[j,i]
            Vtheta_GH[cpt,i] = Vtheta_GH_inv[j,i]
            Vz_GH[cpt,i] = Vz_GH_inv[j,i]
            cpt += -1
# %%
if (Read_CTS and Read_GH):
    err_phi    = np.sum(np.abs(phi_GH    - phi_CTS))    / np.sum(np.abs(phi_GH))     
    err_ne     = np.sum(np.abs(ne_GH     - n_CTS))     / np.sum(np.abs(ne_GH))     
    err_Vr     = np.sum(np.abs(Vr_GH     - Vr_CTS))     / np.sum(np.abs(Vr_GH))     
    err_Vtheta = np.sum(np.abs(Vtheta_GH - Vtheta_CTS)) / np.sum(np.abs(Vtheta_GH))     
    err_Vz     = np.sum(np.abs(Vz_GH     - Vz_CTS))     / np.sum(np.abs(Vz_GH)) 
    print(Folder)
    print("relative error: ")
    print("             phi ", round(err_phi*100,3), " %")
    print("             ne ",  round(err_ne*100,3), " %")
    print("             Vr ",  round(err_Vr*100,3), " %")
    print("             Vtheta ", round(err_Vtheta*100,3), " %")
    print("             Vz ",     round(err_Vz*100,3), " %")

# %%
# =============================================================================
# ---------------------: Cyclotron frequency ----------------------------------------------------------------
# =============================================================================
Omega_i = (e*B)/mi
Omega_e = (e*B)/me
# =============================================================================
# ---------------------: Collision frequency ----------------------------------------------------------------
# =============================================================================
nu_en = sig0*nN*np.sqrt((8*e*Te)/(np.pi*me))
nu_in = sig0*nN*np.sqrt((8*e*Ti)/(np.pi*mi))
nu_ei = (e**(5/2)*CoulombLog/(6*np.sqrt(2)*np.pi**(3/2)*eps0**2*np.sqrt(me)))*n_CTS*Te**(-3/2)

# =============================================================================
# ---------------------: MOBILITY and DIFFUSION -----------------------------------------------------------------------
# =============================================================================
nu_ei = 0 # (e**(5/2)*CoulombLog/(6*np.sqrt(2)*np.pi**(3/2)*eps0**2*np.sqrt(me)))*n_CTS*Te**(-3/2)
mu_E_para_CTS = e/(me*(nu_ei+nu_en)) #(e/(me*nu_ei))*(alpha_sigma_para/(alpha_sigma_para + tilde_nu_N)) ;
D_E_para_CTS  = Te*mu_E_para_CTS
#-->
mu_e_perp_CTS = (me/(e*B**2))*((nu_ei+nu_en)/(1 + (nu_ei+nu_en)**2/Omega_e**2)) #(e/(mi*nu_in))*K3*(1-eta*tilde_nu_I*K1*(1 + K2*(Omega_i/nu_in)**2/(1+eta*tilde_nu_I)));
D_E_perp_CTS  = Te*mu_e_perp_CTS
#-->
mu_i_para = e/(me*(nu_in)) #(e/(me*nu_ei))*(alpha_sigma_para/(alpha_sigma_para + tilde_nu_N)) ;
D_i_para = Te*mu_i_para
#-->
mu_i_perp = (mi/(e*B**2))*((nu_in)/(1 + (nu_in)**2/Omega_i**2)) #(e/(mi*nu_in))*K3*(1-eta*tilde_nu_I*K1*(1 + K2*(Omega_i/nu_in)**2/(1+eta*tilde_nu_I)));
D_i_perp  = Te*mu_i_perp

# =============================================================================
# ---------------------: CONDUCTIVITY -----------------------------------------
# =============================================================================
sigma_perp0 = e*n_CTS*(mu_i_perp + mu_e_perp_CTS)
sigma_para0 = e*n_CTS*(mu_i_para + mu_E_para_CTS)
# %%
"""
=========================== Determination of perpendicular electric field =================================================================================
"""   
E_perp_CTS = np.zeros([Nr-1,Nz-1])
E_perp_CTS[:,:] = (-1)*(1/(step_r))*(phi_CTS[1:,1:] - phi_CTS[:-1,1:])
func_E_perp_CTS      = interpolate.interp1d(r_v, E_perp_CTS[:,0], kind='cubic')

E_perp_CTS_v2           = np.zeros([Nr,Nz])
E_perp_CTS_v2[0,:]      = (-1)*(1/(2*step_r))*(-3*phi_CTS[0,:] + 4*phi_CTS[1,:] - phi_CTS[2,:])
E_perp_CTS_v2[-1,:]     = (-1)*(-1/(2*step_r))*(-3*phi_CTS[Nr-1,:] + 4*phi_CTS[Nr-2,:] - phi_CTS[Nr-3,:])
E_perp_CTS_v2[1:-1,:]   = (-1)*(1/(2*step_r))*(phi_CTS[2:,:] - phi_CTS[0:-2,:])
func_E_perp_CTS_V2      = interpolate.interp1d(r, E_perp_CTS_v2[:,0], kind='cubic')
# %%
fig, ax = plt.subplots( constrained_layout = True) #, figsize = (2*2.20, 2.75))
ax2 = ax.twinx()

ax.plot( r/re, phi_CTS[:,0], color = "black", linestyle = "solid",  label = r'$\phi_{sh}(r)$')
ax2.plot( r_v/re, E_perp_CTS[:,0], color = "gray",  linestyle = "solid", label = r'$E_{sh,\perp}(r)$')
ax2.plot( r/re, E_perp_CTS_v2[:,0], color = "crimson",  linestyle = "dotted", label = r'$E_{sh,\perp}(r)$')
ax.set_xlabel(r'$r/r_e$')
ax.set_ylabel(r'$\phi_{sh}(r)$', color = "black")
ax2.set_ylabel(r'$E_{sh,\perp}(r)$', color = "gray")

ax.legend(loc=0, frameon = False)

# %%
"""
=========================== Determination of current density =================================================================================
"""    
# Parallel ion current density
ji_r_CTS = np.zeros([Nr-1, Nz-1]) # ji_r= e n Vr
for i in np.arange(Nz-1):
    for j in np.arange(Nr-1):
        nv_r = max(Vr_CTS[j,i], 0)*n_CTS[j-1,i] + min(Vr_CTS[j,i], 0)*n_CTS[j,i] 
        ji_r_CTS[j,i] = e*nv_r

# Parallel ion current density
ji_z_CTS = np.zeros([Nr-1, Nz-1]) # ji_para_CTS = e n Vz

for i in np.arange(Nz-1):
    for j in np.arange(Nr-1):
        nv_z= max(Vz_CTS[j,i], 0)*n_CTS[j,i-1] + min(Vz_CTS[j,i], 0)*n_CTS[j,i] 
        ji_z_CTS[j,i] = e*nv_z
        
# ===========================       
sigma_e_r_CTS = np.mean(e*n_CTS*mu_e_perp_CTS)
# ===========================        
Vr_CTS_f                = interpolate.interp1d(r_v, Vr_CTS[:,0], kind='cubic')
n_f                     = interpolate.interp1d(r, n_CTS[:,0], kind='cubic')
func_ji_r_CTS           = lambda r : e*n_f(r)*Vr_CTS_f(r)
func_sigma_perp_CTS     = lambda r : func_ji_r_CTS(r)/func_E_perp_CTS(r) + sigma_e_r_CTS
# %%
"""
=========================== Determination of perpendicular conductivity =================================================================================
"""  
sigma_perp_CTS = (ji_r_CTS/E_perp_CTS) + sigma_e_r_CTS  


sigma_Pedersen = (e**2*n_CTS/me)*((eta*nu_in)/(Omega_i**2 + nu_in**2))
# np.mean(sigma_perp0)
errA_mean_sigma_perp = np.mean(np.abs(np.mean(sigma_perp0)*np.ones([Nr-1, Nz-1]) - sigma_perp_CTS))
errR_L1_sigma_perp = np.sum(np.abs(np.mean(sigma_perp0)*np.ones([Nr-1, Nz-1]) - sigma_perp_CTS)) / np.sum(np.abs(np.mean(sigma_perp0)*np.ones([Nr-1, Nz-1])))

# %%
list_color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'darkorange', 'forestgreen', 'darkred', 'royalblue', 'gold', 'plum']
fig, ax = plt.subplots( constrained_layout = True) #, figsize = (2*2.20, 2.75))
# fig.suptitle(r'$\delta_A ( < \sigma_\perp > ) = %s $' % errA_mean_sigma_perp)
fig.suptitle(r'$\delta_R L_1 ( \sigma_\perp ) = %s $' % errR_L1_sigma_perp)
for index, i in enumerate(np.arange(1,16)):
    ax.scatter((-1)*E_perp_CTS[:,i], sigma_perp_CTS[:,i], s = 10, c = list_color[index], marker = "o")
# ax.plot(-1*func_E_perp_CTS_V2(r_v), func_sigma_perp_CTS(r_v), color = "black", linestyle = "dotted")

ax.axhline( y = np.mean(sigma_perp0), color = "crimson", linestyle = "dashed")
ax.set_xlabel(r'$\partial \phi/\partial r$ $\mathrm{(V \cdot m^{-1})}$')
ax.set_ylabel(r'$\sigma_\perp$ $\mathrm{(\Omega^{-1} \cdot m^{-1})}$')
vareps = 1e-5
# ax.set_ylim([Sigma_Pedersen-vareps, Sigma_Pedersen+vareps])
# ax.set_yscale('log')
# %%
fig, ax = plt.subplots( constrained_layout = True) #, figsize = (2*2.20, 2.75))
fig.suptitle(r'$\delta_R L_1 ( \sigma_\perp ) = %s $' % errR_L1_sigma_perp)
ax2 = ax.twinx()
ax.plot(r_v, sigma_perp_CTS[:,0], color = "black")
ax2.plot(r_v, E_perp_CTS[:,0], color = "gray")
ax.set_xlabel(r'$\partial \phi/\partial r$ $\mathrm{(V \cdot m^{-1})}$')
ax.set_ylabel(r'$\sigma_\perp$ $\mathrm{(\Omega^{-1} \cdot m^{-1})}$')
ax.axhline( y = np.mean(sigma_perp0), color = "crimson", linestyle = "dashed")


# %%
delta_sigma = (sigma_perp_CTS - sigma_perp0[1:,1:]) / np.max(np.abs(sigma_perp_CTS - sigma_perp0[1:,1:]))

levels_log = np.linspace(np.min(delta_sigma), np.max(delta_sigma), 10)
kw = dict(levels=10,  cmap=cm.YlGnBu )
fig,ax = plt.subplots()
plt.gcf().subplots_adjust(right = 0.835, left = 0.105, wspace = 0.25, top = 0.925, hspace = 0.1, bottom = 0.14 )
CF1 = ax.contourf(X_v,Y_v, delta_sigma, **kw)
#=========================> CBAR
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.yaxis.set_major_locator(ticker.NullLocator())

cbar = fig.colorbar(CF1, cax = cax) #, cax = cax, ticks = levels)
cbar.set_ticks([])
loc    = levels_log 
labels = [ sci_notation(levels_log[i]) for i in range(len(levels_log)) ]

cbar.set_ticks(loc)
# cbar.set_ticklabels(labels)
cbar.set_label(r'$(\sigma_\perp^{(num)} - \sigma_\perp) / \max | \sigma_\perp^{(num)} - \sigma_\perp |$', rotation = 90)
ax.set_ylabel(r'$r/r_e$')
ax.set_xlabel(r'$z/[L/2]$')

# current density
current_density = np.sqrt(ji_r_CTS**2+ji_z_CTS**2)
lw = 0.5*current_density / current_density.max()
ax.streamplot(X_v, Y_v, ji_z_CTS, ji_r_CTS, density=1, color='coral', linewidth=lw)

# x_position = array([0.   , 0.025, 0.05 , 0.075, 0.1  , 0.125, 0.15 , 0.175, 0.2  ])/re
# my_xticks = [r'$0$', r'$50$', r'$100$', r'$150$', r'$200$', r'$250$', r'$290$']
# ax0.set_yticks(x_position)
# ax0.set_yticklabels(my_xticks)

ax.set_xlim([-1, 0])
ax.set_ylim([0, rg/re])

# %%
"""
=========================== PICTURE =================================================================================
"""
vect_iteration = np.arange(0, max_iteration, iteration_save)
index_t = -1
if Read_CTS:
    print("Vr, positive value: ", np.max(Vr[index_t,:,:]) > 0)  

# %%-->
if Read_CTS:
    if picture_Vr:
        #------------------------------------
        fig,ax = plt.subplots()
        plt.gcf().subplots_adjust(right = 0.95, left = 0.105, wspace = 0.25, top = 0.925, hspace = 0.1, bottom = 0.14 )
        V_max = np.max(Vr_CTS[:-1,:])
        V_min = np.min(Vr_CTS[:-1,:])
        if V_max > 0:
            pos_min = 0
            pos_max = V_max
            levels_pos = np.linspace(pos_min, pos_max,10)
            kw_pos = dict(levels=levels_pos, cmap="Oranges", vmin=pos_min, vmax=pos_max, origin='lower')
            CF1_pos = ax.contourf(X_v,Y_v, Vr_CTS[:,:], **kw_pos)
        if V_min < 0:
            neg_min = V_min
            neg_max = 0
            levels_neg = np.linspace(neg_min, neg_max,10)
            kw_neg = dict(levels=levels_neg, cmap="Blues", vmin=neg_min, vmax=neg_max, origin='lower')       
            CF1_neg = ax.contourf(X_v,Y_v, Vr_CTS[:,:], **kw_neg)
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
    
    
if Read_CTS:
    if picture_Vtheta:
        # #-------------------------------------------------------------------------------------------------------
        fig,ax = plt.subplots()
        plt.gcf().subplots_adjust(right = 0.95, left = 0.105, wspace = 0.25, top = 0.925, hspace = 0.1, bottom = 0.14 )
        if np.max(Vtheta_CTS[:,:]) > 0:
            pos_min = 0
            pos_max = np.max(Vtheta_CTS[:,:])
            levels_pos = np.linspace(pos_min, pos_max,10)
            kw_pos = dict(levels=levels_pos, cmap="Oranges", vmin=pos_min, vmax=pos_max, origin='lower')
            CF1_pos = ax.contourf(X_v,Y_v, Vtheta_CTS[:,:], **kw_pos)
        if np.min(Vtheta_CTS[:,:]) < 0:
            neg_min = np.min(Vtheta_CTS[:,:])
            neg_max = 0
            levels_neg = np.linspace(neg_min, neg_max,10)
            kw_neg = dict(levels=levels_neg, cmap="Blues", vmin=neg_min, vmax=neg_max, origin='lower')       
            CF1_neg = ax.contourf(X_v,Y_v, Vtheta_CTS[:,:], **kw_neg)
        if (np.max(Vtheta_CTS[:,:]) > 0 and np.min(Vtheta_CTS[:,:]) < 0):
            cbar_pos = fig.colorbar(CF1_pos)
            cbar_pos.set_label(r'  $V_\theta(r,z)$ $\mathrm{(m \cdot m^{-1})}$', rotation = 90)
            cbar_neg = fig.colorbar(CF1_neg)
        elif (np.max(Vtheta_CTS[:,:]) <= 0 and np.min(Vtheta_CTS[:,:]) < 0):
            cbar_neg = fig.colorbar(CF1_neg)
            cbar_neg.set_label(r'  $V_\theta(r,z)$ $\mathrm{(m \cdot m^{-1})}$', rotation = 90)
        elif (np.max(Vtheta_CTS[:,:]) > 0 and np.min(Vtheta_CTS[:,:]) >= 0):
            cbar_pos = fig.colorbar(CF1_pos)
            cbar_pos.set_label(r'  $V_\theta(r,z)$ $\mathrm{(m \cdot m^{-1})}$', rotation = 90)
        ax.set_ylabel(r'$r/r_e$')
        ax.set_xlabel(r'$z/[L/2]$')

   
    # #-------------------------------------------------------------------------------------------------------
if Read_CTS:
    if picture_Vz:
        fig,ax = plt.subplots()
        plt.gcf().subplots_adjust(right = 0.95, left = 0.105, wspace = 0.25, top = 0.925, hspace = 0.1, bottom = 0.14 )
        if np.max(Vz_CTS[:,:]) > 0:
            pos_min = 0
            pos_max = np.max(Vz_CTS[:,:])
            levels_pos = np.linspace(pos_min, pos_max,10)
            kw_pos = dict(levels=levels_pos, cmap="Oranges", vmin=pos_min, vmax=pos_max, origin='lower')
            CF1_pos = ax.contourf(X_v,Y_v, Vz_CTS[:,:], **kw_pos)
        if np.min(Vz_CTS[:,:]) < 0:
            neg_min = np.min(Vz_CTS[:,:])
            neg_max = 0
            levels_neg = np.linspace(neg_min, neg_max,10)
            kw_neg = dict(levels=levels_neg, cmap="Blues", vmin=neg_min, vmax=neg_max, origin='lower')       
            CF1_neg = ax.contourf(X_v,Y_v, Vz_CTS[:,:], **kw_neg)
        if (np.max(Vz_CTS[:,:]) > 0 and np.min(Vz_CTS[:,:]) < 0):
            cbar_pos = fig.colorbar(CF1_pos)
            cbar_pos.set_label(r'  $V_z(r,z)$ $\mathrm{(m \cdot m^{-1})}$', rotation = 90)
            cbar_neg = fig.colorbar(CF1_neg)
        elif (np.max(Vz_CTS[:,:]) <= 0 and np.min(Vz_CTS[:,:]) < 0):
            cbar_neg = fig.colorbar(CF1_neg)
            cbar_neg.set_label(r'  $V_z(r,z)$ $\mathrm{(m \cdot m^{-1})}$', rotation = 90)
        elif (np.max(Vz_CTS[:,:]) > 0 and np.min(Vz_CTS[:,:]) >= 0):
            cbar_pos = fig.colorbar(CF1_pos)
            cbar_pos.set_label(r'  $V_z(r,z)$ $\mathrm{(m \cdot m^{-1})}$', rotation = 90)
        ax.set_ylabel(r'$r/r_e$')
        ax.set_xlabel(r'$z/[L/2]$')
    #-------------------------------------------------------------------------------------------------------

# %%
if Read_CTS:
    if picture_ne:
        levels_log = np.logspace( np.log10(np.max(n_CTS*1e-2)), np.log10(np.max(n_CTS)),30)
        kw = dict(levels=levels_log, locator=ticker.LogLocator(), cmap=cm.YlGnBu )
        fig,ax = plt.subplots()
        plt.gcf().subplots_adjust(right = 0.835, left = 0.105, wspace = 0.25, top = 0.925, hspace = 0.1, bottom = 0.14 )
        CF1 = ax.contourf(X,Y, n_CTS[:,:], **kw)
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
        cbar.set_label(r'  $n_{e}(r,z)$ $\mathrm{(m^{-3})}$', rotation = 90)
        ax.set_ylabel(r'$r/r_e$')
        ax.set_xlabel(r'$z/[L/2]$')
        
        # current density
        current_density = np.sqrt(ji_r_CTS**2+ji_z_CTS**2)
        lw = 2.5*current_density / current_density.max()
        ax.streamplot(X_v, Y_v, ji_z_CTS, ji_r_CTS, density=1, color='coral', linewidth=lw)
    
# %%
if Read_CTS:
    if picture_phi:
        fig,ax = plt.subplots()
        plt.gcf().subplots_adjust(right = 0.975, left = 0.075, wspace = 0.25, top = 0.95, hspace = 0.1, bottom = 0.14 )        #------------------------------------
        if np.max(phi_CTS) > 0:
            levels_pos = np.linspace(0, np.max(phi_CTS),10)
            kw_pos = dict(levels=levels_pos, cmap="Oranges", vmin=0, vmax=np.max(phi_CTS), origin='lower')
            CF1_pos = ax.contourf(X,Y, phi_CTS[:,:], **kw_pos)
            cbar_pos = fig.colorbar(CF1_pos)
    
        if np.min(phi_CTS) < 0:
            neg_min = np.min(phi_CTS[:,:])
            neg_max = np.max(phi_CTS[:,:])
            levels_neg = np.linspace(neg_min, neg_max,10)
            kw_neg = dict(levels=levels_neg, cmap="Blues", vmin=neg_min, vmax=neg_max, origin='lower')       
            CF1_neg = ax.contourf(X,Y, phi_CTS[:,:], **kw_neg)
            #=========================> CBAR
        if (np.max(phi_CTS) > 0 and np.min(phi_CTS) < 0):
            cbar_pos = fig.colorbar(CF1_neg)
            cbar_pos.set_label(r'  $\phi(r,z)$ $\mathrm{(V)}$', rotation = 90)
        if (np.max(phi_CTS) <= 0 and np.min(phi_CTS) < 0):
            cbar_neg = fig.colorbar(CF1_neg)
            cbar_neg.set_label(r'  $\phi(r,z)$ $\mathrm{(V)}$', rotation = 90)
        if (np.max(phi_CTS) > 0 and np.min(phi_CTS) >= 0):
            cbar_pos = fig.colorbar(CF1_neg)
            cbar_pos.set_label(r'  $\phi(r,z)$ $\mathrm{(V)}$', rotation = 90)
        
        # current density
        current_density = np.sqrt(ji_r_CTS**2+ji_z_CTS**2)
        lw = 1*current_density / current_density.max()
        ax.streamplot(X_v, Y_v, ji_z_CTS, ji_r_CTS, density=1, color='coral', linewidth=lw)
        
        # (r,z) limit
        ax.set_xlim([-1,0])
        ax.set_ylim([0, rg/re])
        
        # 
        ax.set_xlabel(r'$z/(L/2)$')
        ax.set_ylabel(r'$r/r_e$')
# %%
if Read_Source_Term:
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




# %%

# fig, ax = plt.subplots()
# x = np.linspace(0, (1/np.exp(Lambda))-1e-10, 101)
# phi_p = -Te*np.log( (1/np.exp(Lambda)) - x)
# ax.plot(x, phi_p)

# x = np.linspace((1/np.exp(Lambda))-1e-4, 1, 101)
# psi_e = 30
# phi_p = -Te*np.log((np.exp(Lambda)**-1*(1+x))/(1+x*np.exp(psi_e)))
# ax.plot(x, phi_p)
# ax.set_xscale('log')
# ax.set_xlabel(r'$A_E/A_W$')
# ax.set_ylabel(r'$\phi_p$ $\mathrm{(V)}$')
# plt.gcf().subplots_adjust(right = 0.99, left = 0.105, wspace = 0.25, top = 0.925, hspace = 0.1, bottom = 0.14 )












