#!/usr/bin/env python
from __future__ import division, print_function
from __future__ import division, print_function
import os
os.environ["OMP_NUM_THREADS"] = "10" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "10"# export OPENBLAS_NUM_THREADS=4 
import matplotlib
matplotlib.use('Agg')

import scipy.interpolate
import pycbc.types # TimeSeries
import pycbc.filter
import pycbc.psd



import numpy as np
from harmonics import sYlm
import numpy as np
import matplotlib.pyplot as plt
import os
from importlib.machinery import SourceFileLoader
from scipy.interpolate import interp1d
from numpy import pi

import pandas as pd

## constants
G = 6.67408*(10.**(-11))
cc = 299792458.
Msun = 1.98855*(10.**30)
Rsun = Msun*G/cc**2
Megapc = 3.08*(10.**22)

def spherical_harmonics(angle,mode):
    l, m = mode
    Yp = np.real(sYlm(-2,l,m,angle,0)+(-1)**l*sYlm(-2,l,-m,angle,0))
    Yc = np.real(sYlm(-2,l,m,angle,0)-(-1)**l*sYlm(-2,l,-m,angle,0))
    return Yp, Yc

## load qnm numerical data
qnm_interp_dict = {}
qnm_data_dict = {(2,2):'/THEORY/USERS/swetha.bhagwat/DATA/Ringdown_trial_setup/Merger_Ringdown_Test/Modules/n1l2m2.dat',(2,1):'/THEORY/USERS/swetha.bhagwat/DATA/Ringdown_trial_setup/Merger_Ringdown_Test/Modules/n1l2m1.dat',(3,3):'/THEORY/USERS/swetha.bhagwat/DATA/Ringdown_trial_setup/Merger_Ringdown_Test/Modules/n1l3m3.dat',(4,4):'/THEORY/USERS/swetha.bhagwat/DATA/Ringdown_trial_setup/Merger_Ringdown_Test/Modules/n1l4m4.dat'}

def interp_qnm(mode):
    df = pd.read_csv(qnm_data_dict[mode],sep=' ',\
        engine='python',names=['spin','omegaR','omegaIm','arg1','arg2'])
    x, y1, y2 = np.array(df['spin']), np.array(df['omegaR']), np.array(-df['omegaIm'])
    outR, outI = interp1d(x,y1,'cubic'), interp1d(x,y2,'cubic')
    qnm_interp_dict[mode] = (outR,outI)
    return None

for k in qnm_data_dict.keys():
    interp_qnm(k)

def qnm_Kerr(mass,spin,mode,method='numerical'):
    conversion_factor = Rsun/cc

    if method == 'numerical':
    ## interpolate modes from data given in
    ## https://pages.jh.edu/~eberti2/ringdown/
        omegaR = qnm_interp_dict[mode][0](spin)/mass/conversion_factor
        omegaI = qnm_interp_dict[mode][1](spin)/mass/conversion_factor
        tau = 1/omegaI

    elif method == 'fit':
    ## use qnm fits from
    ## https://arxiv.org/abs/gr-qc/0512160
        coeff = {}
        coeff[(2,1)] = [0.6,-0.2339,0.4175,-0.3,2.3561,-0.2277]
        coeff[(2,2)] = [1.5251,-1.1568,0.1292,0.7,1.4187,-0.4990]
        coeff[(3,3)] = [1.8956,-1.3043,0.1818,0.9,2.3430,-0.4810]
        coeff[(4,4)] = [2.3,-1.5056,0.2244,1.1929,3.1191,-0.4825]

        f = coeff[mode][:3]
        q = coeff[mode][3:]

        omegaR = (f[0]+f[1]*(1-spin)**f[2])/mass
        omegaR /= conversion_factor
        Q = (q[0]+q[1]*(1-spin)**q[2])
        tau = 2*Q/omegaR

    return omegaR/2/np.pi, tau


def qnm_amplitudes(q,mode,method='numerical'):
    A = {}

    if method == 'fit':
    ## use fits from https://arxiv.org/abs/1111.5819
        eta = q/(1+q)**2
        A[(2,2)] = 0.864*eta
        A[(2,1)] = 0.52*(1-4*eta)**0.71*A[(2,2)]
        A[(3,3)] = 0.44*(1-4*eta)**0.45*A[(2,2)]
        A[(4,4)] = (5.4*(eta-0.22)**2+0.04)*A[(2,2)]

    elif method == 'numerical':
    ## use fits from
    ## in their updated form
        eta = q/(1+q)**2
        A[(2,2)] = 0.864*eta
        A[(2,1)] = A[(2,2)]*(0.472881 - 1.1035/q + 1.03775/q**2 - 0.407131/q**3)
        A[(3,3)] = A[(2,2)]*(0.433253 - 0.555401/q + 0.0845934/q**2 + 0.0375546/q**3)

    return A[mode]


def amplitude_Ratio(q,dev):
        A = {}
        eta = q/(1+q)**2
        A[(2,2)] = 0.864*eta
        A[(3,3)] = A[(2,2)]*((0.433253 - 0.555401/q + 0.0845934/q**2 + 0.0375546/q**3)*(1.+(dev/100.)))
        
        return A[(3,3)]/A[(2,2)]


def qnm_phases(q,mode,phase,method='numerical'):
    if method == 'numerical':
    ## use fits from
    ## https://arxiv.org/abs/2005.03260
    ## in their updated form
        P = {}
        P[(2,2)] = phase
        P[(2,1)] = P[(2,2)] - (1.80298 - 9.70704/(9.77376 + q**2))
        P[(3,3)] = P[(2,2)] - (2.63521 + 8.09316/(8.32479 + q**2))
        out = P[mode]

    elif method == 'fit':
        out = mode[1]*phase

    return out

def qnm_phases_33(q,phase):
    return phase - (2.63521 + 8.09316/(8.32479 + q**2))
 


def theta_phi_to_ra_dec(theta, phi, gmst):
    ra = phi + gmst
    dec = np.pi / 2 - theta
    return ra, dec





# define the time-domain model
def RDwaveform(times,massBH,spinBH,qBH,iotaBH,invDistanceBH):
    t0 = times[0]
    dt = times[1]-times[0]
    method='numerical'
    distance=1./invDistanceBH
    conversion_factor = massBH*Rsun/distance/Megapc
    modes = [(2,2),(3,3)]
    phase=0.0
    hp = np.zeros_like(times)
    hc = np.zeros_like(times)
    for mode in modes:
        A = qnm_amplitudes(qBH,mode,method=method)
        freq, tau = qnm_Kerr(massBH,spinBH,mode,method=method)
        Yp, Yc = spherical_harmonics(iotaBH,mode)
        phi = qnm_phases(qBH,mode,phase,method=method)
        hp += A*np.exp(-(times-t0)/tau)*Yp*np.cos(2*np.pi*freq*(times-t0) - phi)
#         hc += A*np.exp(-(times-t0)/tau)*Yc*np.sin(2*np.pi*freq*(times-t0) - phi)

    hp *= conversion_factor
#     hc *= conversion_factor
#     print(hp)
    
    return pycbc.types.TimeSeries(hp,delta_t=dt)



def spin_fit(q):
  ## from https://arxiv.org/abs/1106.1021
    eta = q/(1+q)**2
    spin = 2*np.sqrt(3)*eta - 3.871*eta**2 + 4.028*eta**3
    return spin
    
    
def qnm_amplitudes_x_PercentMoreThanGR(x,q,mode,method='numerical'):
    A = {}

    if method == 'fit':
    ## use fits from https://arxiv.org/abs/1111.5819
        eta = q/(1+q)**2
        A[(2,2)] = 0.864*eta
        A[(2,1)] = 0.52*(1-4*eta)**0.71*A[(2,2)]
        A[(3,3)] = 0.44*(1-4*eta)**0.45*A[(2,2)]
        A[(4,4)] = (5.4*(eta-0.22)**2+0.04)*A[(2,2)]
        A[(2,1)] = A[(2,1)] + ((x/100.)*A[(2,1)])
        A[(3,3)] = A[(3,3)] + ((x/100.)*A[(3,3)])

    elif method == 'numerical':
    ## use fits from
    ## in their updated form
        eta = q/(1+q)**2
        A[(2,2)] = 0.864*eta
        A[(3,3)] = A[(2,2)]*((0.433253 - 0.555401/q + 0.0845934/q**2 + 0.0375546/q**3)*(1.+(x/100.)))

    return A[mode]


def qnm_phases_abs_y_dev(y,q,mode,phase,method='numerical'):
    if method == 'numerical':
    ## use fits from
    ## https://arxiv.org/abs/2005.03260
    ## in their updated form
        P = {}
        P[(2,2)] = phase
        P[(2,1)] = P[(2,2)] - ((1.80298 - 9.70704/(9.77376 + q**2)))+y
        P[(3,3)] = P[(2,2)] - ((2.63521 + 8.09316/(8.32479 + q**2)))+y
        out = P[mode]

    elif method == 'fit':
        out = mode[1]*phase

    return out


def RDwaveform_x_PercentDev(x,y,times,massBH,spinBH,qBH,iotaBH,invDistanceBH):
    t0 = times[0]
    dt = times[1]-times[0]
    method='numerical'
    distance=1./invDistanceBH
    conversion_factor = massBH*Rsun/distance/Megapc
    modes = [(2,2),(3,3)]
    phase=0.0
    nonGR_hp = np.zeros_like(times)
    hc = np.zeros_like(times)
    for mode in modes:
#         print(mode)
        A = qnm_amplitudes_x_PercentMoreThanGR(x,qBH,mode,method=method)
#         print(A)
        freq, tau = qnm_Kerr(massBH,spinBH,mode,method=method)
        Yp, Yc = spherical_harmonics(iotaBH,mode)
        phi = qnm_phases_abs_y_dev(y,qBH,mode,phase,method=method)
        nonGR_hp += A*np.exp(-(times-t0)/tau)*Yp*np.cos(2*np.pi*freq*(times-t0) - phi)
#         if mode==(3,3):
#             plt.plot(times,nonGR_hp,ls='--',label=mode)
# #         hc += A*np.exp(-(times-t0)/tau)*Yc*np.sin(2*np.pi*freq*(times-t0) - phi)

    nonGR_hp *= conversion_factor
#     hc *= conversion_factor
#     print(hp)
    return pycbc.types.TimeSeries(nonGR_hp,delta_t=dt)
    
    
sampling_rate = 4096
dt = 1/sampling_rate
bins = 256
times = np.linspace(0,bins*dt,bins)


def match(h1,h2):
    return pycbc.filter.match(h1,h2)[0]
    
       
mass_range=np.arange(10.,70.)
q_range=np.arange(1.,7.,0.1)
spin_range=np.arange(0.,0.9,0.05)
phase_modification=np.arange(0,2.*np.pi,np.pi/60.,)
percent_Amp_modification=np.arange(0.,100.,5.)


dev_ar=[]
dev_phi=[]

difference_in_spin_rec=[]
difference_in_mass_rec=[]
difference_in_q_rec=[]
best_match=[]

real_mf=50.
real_q=4.
real_spin=spin_fit(4.)

for delta_phi in phase_modification:
    for delta_ar in percent_Amp_modification:
        print(delta_phi,delta_ar)
        h_nonGR=RDwaveform_x_PercentDev(delta_ar,delta_phi,times,real_mf,real_spin,real_q,np.pi/3.,1000)
        match_list=[]
        q_list=[]
        spin_list=[]
        mass_list=[]

        for q_temp in q_range:
            for m_temp in mass_range:
                for spin_temp in spin_range:
                    h_temp=RDwaveform(times,m_temp,spin_temp,q_temp,np.pi/3.,1000)
                    match_temp=match(h_nonGR,h_temp)
            
                    mass_list=np.append(mass_list,m_temp)
                    spin_list=np.append(spin_list,spin_temp)
                    q_list=np.append(q_list,q_temp)
                    match_list=np.append(match_list,match_temp)
                    
        maximum_match=np.max(match_list)
        arg_max=np.argmax(match_list)
        bestmatch_mass=mass_list[arg_max]
        bestmatch_spin=spin_list[arg_max]
        bestmatch_q=q_list[arg_max]
                    
        dev_ar=np.append(dev_ar,delta_ar)
        dev_phi=np.append(dev_phi,delta_phi)
        best_match=np.append(best_match,maximum_match)
        difference_in_spin_rec=np.append(difference_in_spin_rec,real_spin-bestmatch_spin)
        difference_in_mass_rec=np.append(difference_in_mass_rec,real_mf-bestmatch_mass)
        difference_in_q_rec=np.append(difference_in_q_rec,real_q-bestmatch_q)
                    
np.save('correct_finer_difference_in_spin_rec_q4.npy',difference_in_spin_rec)                 
np.save('correct_finer_difference_in_mass_rec_q4.npy',difference_in_mass_rec) 
np.save('correct_finer_difference_in_q_rec_q4.npy',difference_in_q_rec) 
np.save('correct_finer_dev_ar_q4.npy',dev_ar) 
np.save('correct_finer_dev_phi_q4.npy',dev_phi) 
np.save('correct_finer_best_match_q4.npy',best_match) 
