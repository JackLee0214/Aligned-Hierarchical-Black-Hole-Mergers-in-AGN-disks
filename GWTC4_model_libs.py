import bilby
import numpy as np
import pickle
from pandas.core.frame import DataFrame
import h5py
from astropy.cosmology import Planck15
from bilby.hyper.likelihood import HyperparameterLikelihood
from bilby.core.prior import Interped, Uniform, LogUniform
from bilby.core.prior import TruncatedGaussian as TG
from bilby.core.prior import Beta as Bt
from bilby.core.prior import PowerLaw
from scipy.interpolate import RegularGridInterpolator, interp1d
import astropy.units as u
import sys
from scipy.special._ufuncs import xlogy, erf
import json
from scipy.interpolate import CubicSpline
import os

use_kde=0
#########################
#kde_model
#########################
def TG_kde(x,mu,sig,min,max,sig_kde=None):
    if sig_kde==None:
        sig_kde=np.array([0.5*np.std(x,axis=-1)*x.shape[-1]**(-1/5.)]).T
    norm = np.sqrt(2.*np.pi*(sig_kde**2+sig**2))*(erf((max-mu)/np.sqrt(2.*sig**2)) - erf((min-mu)/np.sqrt(2.*sig**2)))
    kde_integral = (erf((sig_kde**2*(max-mu)+sig**2*(max-x))/np.sqrt(2.*sig_kde**2*sig**2*(sig_kde**2+sig**2)))\
                        - erf((sig_kde**2*(min-mu)+sig**2*(min-x))/np.sqrt(2.*sig_kde**2*sig**2*(sig_kde**2+sig**2))))\
                    *np.exp(-(x-mu)**2/(2.*(sig_kde**2+sig**2)))/norm

    return kde_integral

if os.path.isdir('/public23/home/sc60972/Spin_Ori/data'):
    inject_dir='/public23/home/sc60972/Spin_Ori/data/'
elif os.path.isdir('/Users/liyinjie/Desktop/Works/Globle_data/GW_data'):
    inject_dir='/Users/liyinjie/Desktop/Works/Globle_data/GW_data/'
elif os.path.isdir('/public1/home/m8s000968/GWdata/post'):
    inject_dir='/public1/home/m8s000968/GWdata/post/'
else:
    print(f"文件夹不存在!")
################################################################################################
#redshift
#################################################################################################
fdVcdz=interp1d(np.linspace(0,5,10000),4*np.pi*Planck15.differential_comoving_volume(np.linspace(0,5,10000)).to(u.Gpc**3/u.sr).value)
zs=np.linspace(1e-100,2.9,2000)
dVdzs=fdVcdz(zs)
logdVdzs=np.log(dVdzs)
def llh_z(z,gamma):
    norm=np.sum((1+zs)**(gamma-1)*dVdzs)*2.9/2000.
    norm0=np.sum((1+zs)**(-1)*dVdzs)*2.9/2000.
    return np.where((z>0) & (z<2.9), (1+z)**gamma/norm*norm0 , 1e-100)

def p_z(z,gamma):
    norm=np.sum((1+zs)**(gamma-1)*dVdzs)*2.9/2000.
    p = (1+z)**(gamma-1)*fdVcdz(z)/norm
    return np.where((z>0) & (z<2.9), p , 1e-100)

def log_N(T,lgR0,gamma):
    return np.log(T) + lgR0/np.log10(np.e) + np.logaddexp.reduce((gamma-1)*np.log(zs+1) + logdVdzs) + np.log(2.9/2000)

##### MD function
def phi(z,gamma,k,zp):
    return (1+(1+zp)**(-gamma-k))*(1+z)**gamma/(1+((1+z)/(1+zp))**(gamma+k))
    
def MD_llh_z(z,gamma,k,zp):
    un_normd_ps=phi(zs,gamma,k,zp)*dVdzs/(1+zs)
    norm=np.sum(un_normd_ps)*2.9/2000.
    norm0=np.sum((1+zs)**(-1)*dVdzs)*2.9/2000.
    return np.where((z>0) & (z<2.9), phi(z,gamma,k,zp)/norm*norm0 , 1e-100)

def MD_p_z(z,gamma,k,zp):
    un_normd_ps=phi(zs,gamma,k,zp)*dVdzs/(1+zs)
    norm=np.sum(un_normd_ps)*2.9/2000.
    p = phi(z,gamma,k,zp)/(1+z)*fdVcdz(z)/norm
    return np.where((z>0) & (z<2.9), p , 1e-100)

def MD_dN_dz(z,lgR0,gamma,k,zp,T=1):
    return 10**lgR0 * T * phi(z,gamma,k,zp)*fdVcdz(z)/(1+z)

def MD_log_N(lgR0,gamma,k,zp,T=1):
    un_normd_ps=phi(zs,gamma,k,zp)*dVdzs/(1+zs)
    norm=np.sum(un_normd_ps)*2.9/2000.
    return np.log(T) + lgR0/np.log10(np.e) + np.log(norm)

def MD_N(lgR0,gamma,k,zp,T=1):
    un_normd_ps=phi(zs,gamma,k,zp)*dVdzs/(1+zs)
    norm=np.sum(un_normd_ps)*2.9/2000.
    return T * 10**lgR0 *norm

def R_z(z,lgR0,gamma,k,zp):
    return 10**lgR0*phi(z,gamma,k,zp)
####################################
#Three component spin magnitude
####################################
from model_libs_GWTC3 import Double_mass, Double_ma, PS_mass, spin_a, PS_ma, Default_ct, Double_mact_pair, Single_mact_pair_nospin, Double_mact_pair_nospin, Single_mass_pair, Double_mass_pair_un,\
    spin_ct

if use_kde:
    def PS_ma(m1,a1,alpha,mmin,mmax,delta,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,mu_a,sigma_a,amin,amax):
        pdf=PS_mass(m1,alpha,mmin,mmax,delta,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12)*TG_kde(a1,mu_a,sigma_a,amin,amax)
        return pdf

def Three_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,\
                    alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,r3):
    p1=Double_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2)*(1-r3)
    p2=PS_mass(m1,alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12)*r3
    return p1+p2

def Three_mass_pair_un(m1,m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,\
                    alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,r3,beta):
    pm1=Three_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,\
                    alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,r3)
    pm2=Three_mass(m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,\
                    alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,r3)
    pdf = pm1*pm2*(m2/m1)**beta
    return np.where((m2<m1), pdf , 1e-100)

def Three_ma(m1,a1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,\
                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,mu_a3,sigma_a3,amin3,amax3,r3):
    p1=Double_ma(m1,a1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2)*(1-r3)
    p2=PS_ma(m1,a1,alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,mu_a3,sigma_a3,amin3,amax3)*r3
    return p1+p2

def Three_ma_pair_ct_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,\
                                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,mu_a3,sigma_a3,amin3,amax3,r3,\
                                            beta,mu_t,sigma_t,zmin,zeta):
    pma1=Three_ma(m1,a1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,\
                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,mu_a3,sigma_a3,amin3,amax3,r3)
    pma2=Three_ma(m2,a2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,\
                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,mu_a3,sigma_a3,amin3,amax3,r3)
    pct=Default_ct(ct1,ct2,mu_t,sigma_t,zeta,zmin)
    pdf = pma1*pma2*pct*(m2/m1)**beta
    return np.where((m2<m1), pdf , 1e-100)


def Three_ma_pair_ct_nospin(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,\
                                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,mu_a3,sigma_a3,amin3,amax3,r3,\
                                            beta,mu_t,sigma_t,zmin,zeta):
    m1_sam = np.linspace(2,200,500)
    m2_sam = np.linspace(2,200,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Three_mass_pair_un(x,y,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,\
                    alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,r3,beta)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Three_mass_pair_un(m1,m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,\
                    alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,r3,beta)/AMP1
    pspin = 1/4.
    pdf=pdf*pspin
    return np.where((m2<m1), pdf , 1e-100)

def Three_ma_pair_ct(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,\
                                    alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,mu_a3,sigma_a3,amin3,amax3,r3,\
                                    beta,mu_t,sigma_t,zmin,zeta):
    m1_sam = np.linspace(2,200,500)
    m2_sam = np.linspace(2,200,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Three_mass_pair_un(x,y,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,\
                    alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,r3,beta)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Three_ma_pair_ct_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,\
                                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,mu_a3,sigma_a3,amin3,amax3,r3,\
                                            beta,mu_t,sigma_t,zmin,zeta)/AMP1
    return pdf
   
def hyper_Three_ma_pair_ct(dataset,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,\
                                    alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,mu_a3,sigma_a3,amin3,amax3,r3,\
                                    beta,mu_t,sigma_t,zmin,zeta,gamma):
    z = dataset['z']
    a1,a2,ct1,ct2 = dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1,m2 = dataset['m1'], dataset['m2']
    hp = Three_ma_pair_ct(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,\
                                    alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,mu_a3,sigma_a3,amin3,amax3,r3,\
                                    beta,mu_t,sigma_t,zmin,zeta)*llh_z(z,gamma)
    return hp
####################################
#nonparametric pairing

def spline_f(q,nq1,nq2,nq3,nq4):
    xi=np.linspace(0,1,4)
    yi=np.array([nq1,nq2,nq3,nq4])
    cs = CubicSpline(xi,yi,bc_type='natural')
    pq = np.exp(cs(q)*(q<1))
    return pq

def spline_pair(q,beta,nq1,nq2,nq3,nq4):
    q_sam = np.linspace(0,1,100)
    norm = np.sum(spline_f(q_sam,nq1,nq2,nq3,nq4)*(q_sam**beta))/100.
    return spline_f(q,nq1,nq2,nq3,nq4)*q**beta/norm

def Three_mass_nonppair_un(m1,m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,\
                    alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,r3,beta,nq1,nq2,nq3,nq4):
    pm1=Three_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,\
                    alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,r3)
    pm2=Three_mass(m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,\
                    alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,r3)
    pdf = pm1*pm2*(m2/m1)**beta*spline_f(m2/m1,nq1,nq2,nq3,nq4)
    return np.where((m2<m1), pdf , 1e-100)

def Three_ma_nonppair_ct_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,\
                                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,mu_a3,sigma_a3,amin3,amax3,r3,\
                                            beta,nq1,nq2,nq3,nq4,mu_t,sigma_t,zmin,zeta):
    pma1=Three_ma(m1,a1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,\
                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,mu_a3,sigma_a3,amin3,amax3,r3)
    pma2=Three_ma(m2,a2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,\
                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,mu_a3,sigma_a3,amin3,amax3,r3)
    pct=Default_ct(ct1,ct2,mu_t,sigma_t,zeta,zmin)
    pdf = pma1*pma2*pct*(m2/m1)**beta*spline_f(m2/m1,nq1,nq2,nq3,nq4)
    return np.where((m2<m1), pdf , 1e-100)


def Three_ma_nonppair_ct_nospin(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,\
                                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,mu_a3,sigma_a3,amin3,amax3,r3,\
                                            beta,nq1,nq2,nq3,nq4,mu_t,sigma_t,zmin,zeta):
    m1_sam = np.linspace(2,200,500)
    m2_sam = np.linspace(2,200,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Three_mass_nonppair_un(x,y,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,\
                    alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,r3,beta,nq1,nq2,nq3,nq4)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Three_mass_nonppair_un(m1,m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,\
                    alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,r3,beta,nq1,nq2,nq3,nq4)/AMP1
    pspin = 1/4.
    pdf=pdf*pspin
    return np.where((m2<m1), pdf , 1e-100)

def Three_ma_nonppair_ct(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,\
                                    alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,mu_a3,sigma_a3,amin3,amax3,r3,\
                                    beta,nq1,nq2,nq3,nq4,mu_t,sigma_t,zmin,zeta):
    m1_sam = np.linspace(2,200,500)
    m2_sam = np.linspace(2,200,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Three_mass_nonppair_un(x,y,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,\
                    alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,r3,beta,nq1,nq2,nq3,nq4)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Three_ma_nonppair_ct_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,\
                                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,mu_a3,sigma_a3,amin3,amax3,r3,\
                                            beta,nq1,nq2,nq3,nq4,mu_t,sigma_t,zmin,zeta)/AMP1
    return pdf
   
def hyper_Three_ma_nonppair_ct(dataset,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,\
                                    alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,mu_a3,sigma_a3,amin3,amax3,r3,\
                                    beta,nq1,nq2,nq3,nq4,mu_t,sigma_t,zmin,zeta,gamma):
    z = dataset['z']
    a1,a2,ct1,ct2 = dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1,m2 = dataset['m1'], dataset['m2']
    hp = Three_ma_nonppair_ct(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,\
                                    alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,mu_a3,sigma_a3,amin3,amax3,r3,\
                                    beta,nq1,nq2,nq3,nq4,mu_t,sigma_t,zmin,zeta)*llh_z(z,gamma)
    return hp


##########################
#spin
##########################

def Aligned_ct12(ct1,ct2,sigma_t,zmin=-1,mu_t=1):
    align=TG(mu_t,sigma_t,zmin,1)
    return align.prob(ct1)*align.prob(ct2)

def Aligned_spline_ct(ct,nx1,nx2,nx3):
    xi=np.linspace(0,1,3)
    yi=np.array([nx1,nx2,nx3])
    cs = CubicSpline(xi,yi,bc_type='natural')
    xx=np.linspace(0,1,1000)
    yy=np.exp(cs(xx))*(xx>0)*(xx<1)
    norm=np.sum(yy)*1./1000.
    px = np.exp(cs(ct))*(ct>0)*(ct<1)/norm
    return px

############################
# the high-spin pop

def spline_ct(ct,nx1,nx2,nx3,nx4):
    xi=np.linspace(-1,1,4)
    yi=np.array([nx1,nx2,nx3,nx4])
    cs = CubicSpline(xi,yi,bc_type='natural')
    xx=np.linspace(-1,1,1000)
    yy=np.exp(cs(xx)*(xx>-1)*(xx<1))
    norm=np.sum(yy)*2./1000.
    px = np.exp(cs(ct)*(ct>-1)*(ct<1))/norm
    return px

def spline_a(a,nx1,nx2,nx3,nx4,nx5,amin,amax):
    xi=np.linspace(0,1,5)
    yi=np.array([nx1,nx2,nx3,nx4,nx5])
    cs = CubicSpline(xi,yi,bc_type='natural')
    xx=np.linspace(0,1,1000)
    yy=np.exp(cs(xx)*(xx>amin)*(xx<amax))
    norm=np.sum(yy)*1./1000.
    px = np.exp(cs(a)*(a>amin)*(a<amax))/norm
    return px

def Double_ma_spline_ct(m1,a1,ct1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,zmin1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2):
    p1=PS_ma(m1,a1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,mu_a1,sigma_a1,amin1,amax1)*(1-r2)*spin_ct(ct1,mu_t1,sigma_t1,zeta1,zmin1)
    p2=PS_ma(m1,a1,alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,mu_a2,sigma_a2,amin2,amax2)*r2*spline_ct(ct1,nx1,nx2,nx3,nx4)
    return p1+p2

def Double_ma_spline_ct_pair_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t,sigma_t,zeta,zmin,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta):
    pma1=Double_ma_spline_ct(m1,a1,ct1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t,sigma_t,zeta,zmin,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2)
    pma2=Double_ma_spline_ct(m2,a2,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t,sigma_t,zeta,zmin,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2)
    pdf = pma1*pma2*(m2/m1)**beta
    return np.where((m2<m1), pdf , 1e-100)

def Double_ma_spline_ct_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t,sigma_t,zeta,zmin,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta):
    m1_sam = np.linspace(2,200,500)
    m2_sam = np.linspace(2,200,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Double_mass_pair_un(x,y,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Double_ma_spline_ct_pair_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t,sigma_t,zeta,zmin,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta)/AMP1
    return pdf

def hyper_Double_ma_spline_ct_pair(dataset,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t,sigma_t,zeta,zmin,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta,gamma):
    z = dataset['z']
    a1,a2,ct1,ct2=dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1=dataset['m1']
    m2=dataset['m2']
    hp = Double_ma_spline_ct_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t,sigma_t,zeta,zmin,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta)*llh_z(z,gamma)
    return hp

def Double_ma_spline_ct_pair_nospin(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t,sigma_t,zeta,zmin,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta):
    return Double_mact_pair_nospin(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,beta,mu_t,sigma_t,zmin,zeta)

####################################
#Spin orientation vs mass
####################################

#The low spin pop
def Aligned_pop(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,beta,sigma_t):
    pm=Single_mass_pair(m1,m2, alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,beta)
    pact=spin_a(a1,mu_a1,sigma_a1,amin1,amax1)*spin_a(a2,mu_a1,sigma_a1,amin1,amax1)*Aligned_ct12(ct1,ct2,sigma_t,zmin=0)
    pdf = pm*pact
    return pdf

def Field_Dyn(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta,\
                                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,beta3,sigma_t3,r3):
    p1=Double_ma_spline_ct_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,1,10,0,-1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta)*(1-r3)
    p2=Aligned_pop(m1,m2,a1,a2,ct1,ct2,alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,amin1,mu_a1,sigma_a1,amax1,beta3,sigma_t3)*r3
    return p1+p2

def hyper_Field_Dyn(dataset,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta,\
                                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,beta3,sigma_t3,r3,gamma):
    z = dataset['z']
    a1,a2,ct1,ct2 = dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1,m2 = dataset['m1'], dataset['m2']
    hp = Field_Dyn(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta,\
                                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,beta3,sigma_t3,r3)*llh_z(z,gamma)
    return hp

def Field_Dyn_nospin(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta,\
                                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,beta3,sigma_t3,r3):
    p1 = Single_mact_pair_nospin(m1,m2,a1,a2,ct1,ct2,alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,amin1,mu_a1,sigma_a1,amax1,beta3,1,sigma_t3,-1,1)*r3
    p2 = Double_mact_pair_nospin(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,beta,1,10,0,-1)*(1-r3)
    return p1+p2

######################################
#Nearly aligned pop
def NearlyAligned_pop(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,beta,sigma_t,mu_t):
    pm=Single_mass_pair(m1,m2, alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,beta)
    pact=spin_a(a1,mu_a1,sigma_a1,amin1,amax1)*spin_a(a2,mu_a1,sigma_a1,amin1,amax1)*Aligned_ct12(ct1,ct2,sigma_t,zmin=-1,mu_t=mu_t)
    pdf = pm*pact
    return pdf

def NearlyAligned_Field_Dyn(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta,\
                                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,beta3,sigma_t3,mu_t3,r3):
    p1=Double_ma_spline_ct_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,1,10,0,-1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta)*(1-r3)
    p2=NearlyAligned_pop(m1,m2,a1,a2,ct1,ct2,alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,amin1,mu_a1,sigma_a1,amax1,beta3,sigma_t3,mu_t3)*r3
    return p1+p2

def hyper_NearlyAligned_Field_Dyn(dataset,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta,\
                                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,beta3,sigma_t3,mu_t3,r3,gamma):
    z = dataset['z']
    a1,a2,ct1,ct2 = dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1,m2 = dataset['m1'], dataset['m2']
    hp = NearlyAligned_Field_Dyn(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta,\
                                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,beta3,sigma_t3,mu_t3,r3)*llh_z(z,gamma)
    return hp

def NearlyAligned_Field_Dyn_nospin(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta,\
                                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,beta3,sigma_t3,mu_t3,r3):
    p1 = Single_mact_pair_nospin(m1,m2,a1,a2,ct1,ct2,alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,amin1,mu_a1,sigma_a1,amax1,beta3,mu_t3,sigma_t3,-1,1)*r3
    p2 = Double_mact_pair_nospin(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,beta,1,10,0,-1)*(1-r3)
    return p1+p2


######################################
#The aligned spline ct
def Aligned_spline_pop(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,beta,nt1,nt2,nt3):
    pm=Single_mass_pair(m1,m2, alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,beta)
    pact=spin_a(a1,mu_a1,sigma_a1,amin1,amax1)*spin_a(a2,mu_a1,sigma_a1,amin1,amax1)*Aligned_spline_ct(ct1,nt1,nt2,nt3)*Aligned_spline_ct(ct2,nt1,nt2,nt3)
    pdf = pm*pact
    return pdf

def Field_spline_Dyn(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta,\
                                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,beta3,nt1,nt2,nt3,r3):
    p1=Double_ma_spline_ct_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,1,10,0,-1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta)*(1-r3)
    p2=Aligned_spline_pop(m1,m2,a1,a2,ct1,ct2,alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,amin1,mu_a1,sigma_a1,amax1,beta3,nt1,nt2,nt3)*r3
    return p1+p2

def hyper_Field_spline_Dyn(dataset,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta,\
                                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,beta3,nt1,nt2,nt3,r3,gamma):
    z = dataset['z']
    a1,a2,ct1,ct2 = dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1,m2 = dataset['m1'], dataset['m2']
    hp = Field_spline_Dyn(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta,\
                                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,beta3,nt1,nt2,nt3,r3)*llh_z(z,gamma)
    return hp

def Field_Dyn_spline_nospin(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta,\
                                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,beta3,nt1,nt2,nt3,r3):
    p1 = Single_mact_pair_nospin(m1,m2,a1,a2,ct1,ct2,alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,amin1,mu_a1,sigma_a1,amax1,beta3,1,1,-1,1)*r3
    p2 = Double_mact_pair_nospin(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,beta,1,10,0,-1)*(1-r3)
    return p1+p2

####################################
#Introduce redshift distribution
####################################

def hyper_Field_Dyn_two_z(dataset,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta,\
                                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,beta3,sigma_t3,lgR0,gamma,k,zp,lgR3,gamma3,k3,zp3):
    z = dataset['z']
    a1,a2,ct1,ct2 = dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1,m2 = dataset['m1'], dataset['m2']
    z_prior = dataset['z_prior']
    r3=MD_N(lgR3,gamma3,k3,zp3)/(MD_N(lgR3,gamma3,k3,zp3)+MD_N(lgR0,gamma,k,zp))
    p1=Double_ma_spline_ct_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,1,10,0,-1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta)*(1-r3)*MD_p_z(z,gamma,k,zp)/z_prior
    p2=Aligned_pop(m1,m2,a1,a2,ct1,ct2,alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,amin1,mu_a1,sigma_a1,amax1,beta3,sigma_t3)*r3*MD_p_z(z,gamma3,k3,zp3)/z_prior
    return p1+p2

def Field_Dyn_two_z_selection(m1,m2,a1,a2,ct1,ct2,z,Tobs,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta,\
                                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,beta3,sigma_t3,lgR0,gamma,k,zp,lgR3,gamma3,k3,zp3):
    N1=MD_N(lgR3,gamma3,k3,zp3,Tobs)
    N2=MD_N(lgR0,gamma,k,zp,Tobs)
    log_N_tot = np.log(N1+N2)
    dN_dm_ds_dz1 = Single_mact_pair_nospin(m1,m2,a1,a2,ct1,ct2,alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,amin1,mu_a1,sigma_a1,amax1,beta3,1,sigma_t3,-1,1)*MD_dN_dz(z,lgR3,gamma3,k3,zp3,T=Tobs)
    dN_dm_ds_dz2 = Double_mact_pair_nospin(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,beta,1,10,0,-1)*MD_dN_dz(z,lgR0,gamma,k,zp,T=Tobs)
    return np.log(dN_dm_ds_dz1+dN_dm_ds_dz2), log_N_tot

#####
#NearlyAligned

def hyper_NearlyAligned_Field_Dyn_two_z(dataset,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta,\
                                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,beta3,sigma_t3,mu_t3,lgR0,gamma,k,zp,lgR3,gamma3,k3,zp3):
    z = dataset['z']
    a1,a2,ct1,ct2 = dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1,m2 = dataset['m1'], dataset['m2']
    z_prior = dataset['z_prior']
    r3=MD_N(lgR3,gamma3,k3,zp3)/(MD_N(lgR3,gamma3,k3,zp3)+MD_N(lgR0,gamma,k,zp))
    p1=Double_ma_spline_ct_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,1,10,0,-1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta)*(1-r3)*MD_p_z(z,gamma,k,zp)/z_prior
    p2=NearlyAligned_pop(m1,m2,a1,a2,ct1,ct2,alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,amin1,mu_a1,sigma_a1,amax1,beta3,sigma_t3,mu_t3)*r3*MD_p_z(z,gamma3,k3,zp3)/z_prior
    return p1+p2

def NearlyAligned_Field_Dyn_two_z_selection(m1,m2,a1,a2,ct1,ct2,z,Tobs,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta,\
                                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,beta3,sigma_t3,mu_t3,lgR0,gamma,k,zp,lgR3,gamma3,k3,zp3):
    N1=MD_N(lgR3,gamma3,k3,zp3,Tobs)
    N2=MD_N(lgR0,gamma,k,zp,Tobs)
    log_N_tot = np.log(N1+N2)
    dN_dm_ds_dz1 = Single_mact_pair_nospin(m1,m2,a1,a2,ct1,ct2,alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,amin1,mu_a1,sigma_a1,amax1,beta3,1,sigma_t3,-1,1)*MD_dN_dz(z,lgR3,gamma3,k3,zp3,T=Tobs)
    dN_dm_ds_dz2 = Double_mact_pair_nospin(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,beta,1,10,0,-1)*MD_dN_dz(z,lgR0,gamma,k,zp,T=Tobs)
    return np.log(dN_dm_ds_dz1+dN_dm_ds_dz2), log_N_tot

######################################
#The aligned spline ct

def hyper_Field_spline_Dyn_two_z(dataset,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta,\
                                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,beta3,nt1,nt2,nt3,lgR0,gamma,k,zp,lgR3,gamma3,k3,zp3):
    z = dataset['z']
    a1,a2,ct1,ct2 = dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1,m2 = dataset['m1'], dataset['m2']
    z_prior = dataset['z_prior']
    r3=MD_N(lgR3,gamma3,k3,zp3)/(MD_N(lgR3,gamma3,k3,zp3)+MD_N(lgR0,gamma,k,zp))
    p1=Double_ma_spline_ct_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,1,10,0,-1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta)*(1-r3)*MD_p_z(z,gamma,k,zp)/z_prior
    p2=Aligned_spline_pop(m1,m2,a1,a2,ct1,ct2,alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,amin1,mu_a1,sigma_a1,amax1,beta3,nt1,nt2,nt3)*r3*MD_p_z(z,gamma3,k3,zp3)/z_prior
    return p1+p2

def Field_spline_Dyn_two_z_selection(m1,m2,a1,a2,ct1,ct2,z,Tobs,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,nx1,nx2,nx3,nx4,r2,beta,\
                                            alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,beta3,nt1,nt2,nt3,lgR0,gamma,k,zp,lgR3,gamma3,k3,zp3):
    N1=MD_N(lgR3,gamma3,k3,zp3,Tobs)
    N2=MD_N(lgR0,gamma,k,zp,Tobs)
    log_N_tot = np.log(N1+N2)
    dN_dm_ds_dz1 = Single_mact_pair_nospin(m1,m2,a1,a2,ct1,ct2,alpha3,mmin3,mmax3,delta3,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,amin1,mu_a1,sigma_a1,amax1,beta3,1,1,-1,1)*MD_dN_dz(z,lgR3,gamma3,k3,zp3,T=Tobs)
    dN_dm_ds_dz2 = Double_mact_pair_nospin(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,beta,1,10,0,-1)*MD_dN_dz(z,lgR0,gamma,k,zp,T=Tobs)
    return np.log(dN_dm_ds_dz1+dN_dm_ds_dz2), log_N_tot

############################
# Nonparametric model

def Double_nonpara_mact(m1,a1,ct1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,na1,na2,na3,na4,na5,amin1,amax1,nt1,nt2,nt3,nt4,\
                                  alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,oa1,oa2,oa3,oa4,oa5,amin2,amax2,ot1,ot2,ot3,ot4,r2):
    p1=PS_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12)*(1-r2)*spline_ct(ct1,nt1,nt2,nt3,nt4)*spline_a(a1,na1,na2,na3,na4,na5,amin1,amax1)
    p2=PS_mass(m1,alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12)*r2*spline_ct(ct1,ot1,ot2,ot3,ot4)*spline_a(a1,oa1,oa2,oa3,oa4,oa5,amin2,amax2)
    return p1+p2

def Double_nonpara_mact_pair_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,na1,na2,na3,na4,na5,amin1,amax1,nt1,nt2,nt3,nt4,\
                                  alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,oa1,oa2,oa3,oa4,oa5,amin2,amax2,ot1,ot2,ot3,ot4,r2,beta):
    pma1=Double_nonpara_mact(m1,a1,ct1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,na1,na2,na3,na4,na5,amin1,amax1,nt1,nt2,nt3,nt4,\
                                  alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,oa1,oa2,oa3,oa4,oa5,amin2,amax2,ot1,ot2,ot3,ot4,r2)
    pma2=Double_nonpara_mact(m2,a2,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,na1,na2,na3,na4,na5,amin1,amax1,nt1,nt2,nt3,nt4,\
                                  alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,oa1,oa2,oa3,oa4,oa5,amin2,amax2,ot1,ot2,ot3,ot4,r2)
    pdf = pma1*pma2*(m2/m1)**beta
    return np.where((m2<m1), pdf , 1e-100)

def Double_nonpara_mact_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,na1,na2,na3,na4,na5,amin1,amax1,nt1,nt2,nt3,nt4,\
                                  alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,oa1,oa2,oa3,oa4,oa5,amin2,amax2,ot1,ot2,ot3,ot4,r2,beta):
    m1_sam = np.linspace(2,200,500)
    m2_sam = np.linspace(2,200,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Double_mass_pair_un(x,y,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Double_nonpara_mact_pair_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,na1,na2,na3,na4,na5,amin1,amax1,nt1,nt2,nt3,nt4,\
                                  alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,oa1,oa2,oa3,oa4,oa5,amin2,amax2,ot1,ot2,ot3,ot4,r2,beta)/AMP1
    return pdf

def hyper_Double_nonpara_mact_pair(dataset,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,na1,na2,na3,na4,na5,amin1,amax1,nt1,nt2,nt3,nt4,\
                                  alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,oa1,oa2,oa3,oa4,oa5,amin2,amax2,ot1,ot2,ot3,ot4,r2,beta,gamma):
    z = dataset['z']
    a1,a2,ct1,ct2=dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1=dataset['m1']
    m2=dataset['m2']
    hp = Double_nonpara_mact_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,na1,na2,na3,na4,na5,amin1,amax1,nt1,nt2,nt3,nt4,\
                                  alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,oa1,oa2,oa3,oa4,oa5,amin2,amax2,ot1,ot2,ot3,ot4,r2,beta)*llh_z(z,gamma)
    return hp

def Double_nonpara_mact_pair_nospin(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,na1,na2,na3,na4,na5,amin1,amax1,nt1,nt2,nt3,nt4,\
                                  alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,oa1,oa2,oa3,oa4,oa5,amin2,amax2,ot1,ot2,ot3,ot4,r2,beta):
    m1_sam = np.linspace(2,200,500)
    m2_sam = np.linspace(2,200,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Double_mass_pair_un(x,y,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Double_mass_pair_un(m1,m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta)/AMP1
    return pdf


########################
#priors
########################

def Triple_constraint(params):
    params['amax1-mua1']=np.sign(params['amax1']-params['mu_a1'])-1
    params['pop1_scale']=np.sign(params['mmax1']-params['mmin1']-20)-1 

    params['mua2-amin2']=np.sign(params['mu_a2']-params['amin2'])-1
    params['amax2-mua2']=np.sign(params['amax2']-params['mu_a2'])-1
    params['amax2-amin2']=np.sign(params['amax2']-params['amin2']-0.2)-1
    params['pop2_scale']=np.sign(params['mmax2']-params['mmin2']-20)-1  

    params['mua3-amin3']=np.sign(params['mu_a3']-params['amin3'])-1
    params['amax3-mua3']=np.sign(params['amax3']-params['mu_a3'])-1
    params['pop3_scale']=np.sign(params['mmax3']-params['mmin3']-20)-1  
    params['amax3-amin3']=np.sign(params['amax3']-params['amin3']-0.2)-1

    params['mua2-mua3']=np.sign(params['mu_a2']-params['mu_a3'])-1  
    params['mua3-mua1']=np.sign(params['mu_a3']-params['mu_a1'])-1  

    return params

def default_tilt_priors():
    priors=dict(    beta = Uniform(0,6,'beta','$\\beta$'),
                    mu_t = 1,
                    sigma_t = Uniform(0.1, 4., 'sigma_t', '$\\sigma_{\\rm t}$'),
                    zmin = -1,
                    zeta = Uniform(0,1,'zeta','$\\zeta$'))
    return priors

#priors
def Triple_priors(conversion_function=Triple_constraint):
    priors=bilby.prior.PriorDict(conversion_function=conversion_function)
    priors.update(default_tilt_priors())
    priors.update(dict(
                    delta1 =Uniform(1,10),
                    mmin1 = Uniform(2., 50., 'mmin1', '$m_{\\rm min,1}$'),
                    mmax1 = Uniform(20., 200, 'mmax1', '$m_{\\rm max,1}$'),
                    alpha1 = Uniform(-4, 8., 'alpha1', '$\\alpha,1$'),
                    mu_a1 = Uniform(0., 1., 'mu_a1', '$\\mu_{\\rm a,1}$'),
                    sigma_a1 = Uniform(0.05, 0.5, 'sigma_a1', '$\\sigma_{\\rm a,1}$'),
                    amin1 = 0,
                    amax1 = Uniform(0.2,1,'amax1', '$a_{\\rm max,1}$'),
                    
                    delta2 = Uniform(1,10),
                    mmin2 = Uniform(2., 50., 'mmin2', '$m_{\\rm min,2}$'),
                    mmax2 = Uniform(20., 200, 'mmax2', '$m_{\\rm max,2}$'),
                    alpha2 = Uniform(-4., 8., 'alpha2', '$\\alpha_2$'),
                    amin2 = Uniform(0,0.8, 'amin2', '$a_{\\rm min,2}$'),
                    mu_a2 = Uniform(0, 1., 'mu_a2', '$mu_{\\rm a,2}$'),
                    sigma_a2 = Uniform(0.05, 0.5, 'sigma_a2', '$\\sigma_{\\rm a,2}$'),
                    amax2 = Uniform(0.2,1, 'amax2', '$a_{\\rm max,2}$'),
                    r2 = Uniform(0,1, 'r2', '$r_2$'),
                    
                    delta3 = Uniform(1,10),
                    mmin3 = Uniform(2., 50., 'mmin3', '$m_{\\rm min,3}$'),
                    mmax3 = Uniform(20., 200, 'mmax3', '$m_{\\rm max,3}$'),
                    alpha3 = Uniform(-4., 8., 'alpha3', '$\\alpha_3$'),
                    amin3 = Uniform(0,0.8, 'amin3', '$a_{\\rm min,3}$'),
                    mu_a3 = Uniform(0, 1., 'mu_a3', '$mu_{\\rm a,3}$'),
                    sigma_a3 = Uniform(0.05, 0.5, 'sigma_a3', '$\\sigma_{\\rm a,3}$'),
                    amax3 = Uniform(0.2,1, 'amax3', '$a_{\\rm max,3}$'),
                    r3 = Uniform(0,1, 'r3', '$r_3$'),

                    lgR0 = Uniform(0,3,'lgR0','$log_{10}~R_0$'),
                    gamma= Uniform(-2,7,'gamma','$\\gamma$')
                 ))
                     
    priors.update({'n'+str(i+1): TG(0,1,-100,100,name='n'+str(i+1))  for i in np.arange(12)})
    priors.update({'n1':0,'n'+str(12): 0})
    priors.update({'o'+str(i+1): TG(0,1,-100,100,name='o'+str(i+1))  for i in np.arange(12)})
    priors.update({'o1':0,'o'+str(12): 0})
    priors.update({'q'+str(i+1): TG(0,1,-100,100,name='q'+str(i+1))  for i in np.arange(12)})
    priors.update({'q1':0,'q'+str(12): 0})

    #priors.update({'constraint'+str(i+1):bilby.prior.Constraint(minimum=-0.1, maximum=0.1) for i in np.arange(7)})
    return priors


########################
#fix some parameters
########################

def reduced_Double_constraint_GWTC4(params):
    params['pop1_scale']=np.sign(params['mmax1']-params['mmin1']-20)-1 
    params['pop2_scale']=np.sign(params['mmax2']-params['mmin2']-20)-1  
    try:
        params['mua2-mua1']=np.sign(params['mu_a2']-params['mu_a1'])-1  
    except:
        pass
    return params

def Fixed_Double_priors(priors,add_label):
    priors.update({'o'+str(i+1): 0  for i in np.arange(5)})
    priors.update({'n11':0, 'n10':0})
    priors.update(dict(
                amax2 = 1,
                amin1 = 0
                #,amax1 = 1
             ))
    add_label+='_fixedge'
    priors.update({key:bilby.prior.Constraint(minimum=-0.1, maximum=0.1) for key in ['pop1_scale','pop2_scale','mua2-mua1']})
    return priors, add_label

def Double_nonpara_priors(conversion_function=reduced_Double_constraint_GWTC4):
    priors=bilby.prior.PriorDict(conversion_function=conversion_function)
    priors.update(dict(
                    beta = Uniform(0,6,'beta','$\\beta$'),
                    delta1 =Uniform(1,10),
                    mmin1 = Uniform(2., 50., 'mmin1', '$m_{\\rm min,1}$'),
                    mmax1 = Uniform(20., 200, 'mmax1', '$m_{\\rm max,1}$'),
                    alpha1 = Uniform(-4, 8., 'alpha1', '$\\alpha,1$'),
                    amin1 = 0,
                    amax1 = Uniform(0.2,1,'amax1', '$a_{\\rm max,1}$'),
                    
                    delta2 = Uniform(1,20),
                    mmin2 = Uniform(2., 50., 'mmin2', '$m_{\\rm min,2}$'),
                    mmax2 = Uniform(20., 200, 'mmax2', '$m_{\\rm max,2}$'),
                    alpha2 = Uniform(-4., 8., 'alpha2', '$\\alpha_2$'),
                    amin2 = Uniform(0,0.8, 'amin2', '$a_{\\rm min,2}$'),
                    amax2 = 1,
                    r2 = Uniform(0,1, 'r2', '$r_2$'),
                    lgR0 = Uniform(0,3,'lgR0','$log_{10}~R_0$'),
                    gamma= Uniform(-2,7,'gamma','$\\gamma$')))
    priors.update({'n'+str(i+1): TG(0,1,-100,100,name='n'+str(i+1))  for i in np.arange(12)})
    priors.update({'n1':0,'n'+str(12): 0})
    priors.update({'o'+str(i+1): TG(0,1,-100,100,name='o'+str(i+1))  for i in np.arange(12)})
    priors.update({'o1':0,'o'+str(12): 0})

    priors.update({'o'+str(i+1): 0  for i in np.arange(5)})
    priors.update({'n11':0, 'n10':0})

    priors.update({'na'+str(i+1): TG(0,2,-100,100,name='na'+str(i+1))  for i in np.arange(5)})
    priors.update({'oa'+str(i+1): TG(0,2,-100,100,name='oa'+str(i+1))  for i in np.arange(5)})
    priors.update({'nt'+str(i+1): TG(0,1,-100,100,name='nt'+str(i+1))  for i in np.arange(4)})
    priors.update({'ot'+str(i+1): TG(0,1,-100,100,name='ot'+str(i+1))  for i in np.arange(4)})
    priors.update({'oa1':-10,'na5': -10})

    priors.update({key:bilby.prior.Constraint(minimum=-0.1, maximum=0.1) for key in ['pop1_scale','pop2_scale']})
    return priors


def Fixed_triple_priors(priors,add_label,fix_node=True,fix_aminmax=True,zero_spin=False,sig=0.05):
    if fix_node:
        priors.update({'o'+str(i+1): 0  for i in np.arange(5)})
        priors.update({'n11':0, 'q11':0, 'n10':0, 'q10':0})
    if fix_aminmax:
        priors.update(dict(
                    amax3 = 1,
                    amin3 = 0,

                    amax2 = 1,

                    amin1 = 0
                    #,amax1 = 1
                 ))
        add_label+='_fixedge'
    priors.update({key:bilby.prior.Constraint(minimum=-0.1, maximum=0.1) for key in ['pop1_scale','pop2_scale','pop3_scale','mua2-mua1','mua1-mua3']})
    if zero_spin:
        priors.update(dict(
                    mu_a3 = 0,
                    sigma_a3 = sig))
        priors.pop('mua1-mua3')
        add_label+='_zerospin'
        add_label+='_sig3_{}'.format(str(priors['sigma_a3'])[2:])
    return priors, add_label

def reduced_Triple_constraint(params):
    params['pop1_scale']=np.sign(params['mmax1']-params['mmin1']-20)-1 
    params['pop2_scale']=np.sign(params['mmax2']-params['mmin2']-20)-1  
    params['pop3_scale']=np.sign(params['mmax3']-params['mmin3']-20)-1  
    params['mua2-mua1']=np.sign(params['mu_a2']-params['mu_a1'])-1  
    try: 
        params['mua1-mua3']=np.sign(params['mu_a1']-params['mu_a3'])-1  
    except:
        pass
    return params

def Field_Dyn_priors(add_label):
    priors=Triple_priors(reduced_Triple_constraint)
    priors, add_label=Fixed_triple_priors(priors,add_label,zero_spin=False)
    for key in ['amin3','mu_a3','sigma_a3','amax3','mu_t','sigma_t','zmin','zeta']:
        priors.pop(key)
    priors['beta3'] = Uniform(0,6,'beta3','$\\beta_3$')
    priors['sigma_t3'] = Uniform(0.1, 1, 'sigma_t3', '$\\sigma_{\\rm t,3}$')
    priors['amax1'] = Uniform(0.2,1,'amax1', '$a_{\\rm max,1}$')
    priors.update({'nx'+str(i+1): TG(0,1,-100,100,name='nx'+str(i+1))  for i in np.arange(4)})
    try:
        priors.pop('mua1-mua3')
    except:
        pass
    return priors, add_label

def Field_Dyn_two_z_priors(add_label):
    priors, add_label = Field_Dyn_priors(add_label)
    priors.pop('r3')
    priors.update(dict(
                        lgR0 = Uniform(-2,2,'lgR0','$log_{10}~R_0$'),
                    gamma= Uniform(-2,7,'gamma','$\\gamma$'),
                    zp = 2,
                    k = 2,
                    lgR3 = Uniform(-2,2,'lgR3','$log_{10}~R_3$'),
                    gamma3= Uniform(-2,7,'gamma3','$\\gamma_3$'),
                    zp3 = 2,
                    k3 = 2,
                    ))
    return priors, add_label

def add_nonppair_priors(priors):
    priors.update({'nq'+str(i+1): TG(0,1,-100,100)  for i in np.arange(4)})
    priors.update({'nq1':0, 'nq2':0})
    return priors

def add_Field_spline_priors(priors):
    priors.pop('sigma_t3')
    priors.update({'nt1':TG(0,1,-100,100,name='nt1'), 'nt2':TG(0,1,-100,100,name='nt2'), 'nt3':TG(0,1,-100,100,name='nt3')})
    return priors



#####################
#injection campaign


def p_a(a):
    xx=np.linspace(0,1,500)
    yy=np.exp(-2*xx**2)
    norm=np.sum(yy)*1./500.
    return np.exp(-2*a**2)/norm

def p_ct(ct):
    return 0.35+0.3*(1+ct)**3/4

def p_t(t):
    return np.sin(t)*p_ct(np.cos(t))

def p_phi(phi):
    return 1/2./np.pi

def lnp_spin(a1,a2,t1,t2,phi1,phi2):
    return np.log(p_a(a1))+np.log(p_a(a2))+np.log(p_t(t1))+np.log(p_t(t2))+np.log(p_phi(phi1))+np.log(p_phi(phi2))

path = inject_dir+"GWTC-4/mixture-semi_o1_o2-real_o3_o4a-polar_spins_20250503134659UTC.hdf"
with h5py.File(path, "r") as f:
    Tobs=f.attrs['total_analysis_time']/(365.25*24*3600)
    Ndraw = f.attrs['total_generated']
    events = f["events"][:]
    meta = dict(f.attrs.items())

m1_inj = np.array(events['mass1_source'])
m2_inj = np.array(events['mass2_source'])
z_inj = np.array(events['redshift'])
a1_inj = np.array(events['spin1_magnitude'])
a2_inj = np.array(events['spin2_magnitude'])
t1_inj = np.array(events['spin1_polar_angle'])
t2_inj = np.array(events['spin2_polar_angle'])
phi1_inj = np.array(events['spin1_azimuthal_angle'])
phi2_inj = np.array(events['spin2_azimuthal_angle'])
ct1_inj = np.cos(t1_inj)
ct2_inj = np.cos(t2_inj)
weights = np.array(events["weights"])
ln_ws = np.log(weights)
min_far = np.min([events["%s_far"%search] for search in meta["searches"]], axis=0)
detected = min_far < 1.0 # /year
lnp_draw = np.array(events['lnpdraw_mass1_source_mass2_source_redshift_spin1_magnitude_spin1_polar_angle_spin1_azimuthal_angle_spin2_magnitude_spin2_polar_angle_spin2_azimuthal_angle'])
#lnp_draw = lnp_draw + ln_ws
lnp_draw = lnp_draw - ln_ws

sel_indx=np.where((m2_inj>2) & (m1_inj<300))

m1_inj=m1_inj[sel_indx]
m2_inj=m2_inj[sel_indx]
z_inj=z_inj[sel_indx]
a1_inj=a1_inj[sel_indx]
a2_inj=a2_inj[sel_indx]
ct1_inj=ct1_inj[sel_indx]
ct2_inj=ct2_inj[sel_indx]
t1_inj=t1_inj[sel_indx]
t2_inj=t2_inj[sel_indx]
phi1_inj=phi1_inj[sel_indx]
phi2_inj=phi2_inj[sel_indx]

detected=detected[sel_indx]
lnp_draw=lnp_draw[sel_indx]


log_pspin = lnp_spin(a1_inj,a2_inj,t1_inj,t2_inj,phi1_inj,phi2_inj)
logpdraw=lnp_draw-log_pspin+np.log(1./4.)

detection_selector = detected

log1pz_inj = np.log1p(z_inj)
logdVdz_inj = np.log(4*np.pi) + np.log(Planck15.differential_comoving_volume(z_inj).to(u.Gpc**3/u.sr).value)

#This selection effect not accounts for spin distribution
def Rate_selection_function_with_uncertainty(Nobs,mass_spin_model,lgR0,gamma,**kwargs):
    log_dNdz = lgR0/np.log10(np.e) + (gamma-1)*log1pz_inj + logdVdz_inj
    log_dNdmds = np.log(mass_spin_model(m1_inj,m2_inj,a1_inj,a2_inj,ct1_inj,ct2_inj,**kwargs))
    log_dNdzdmds = np.where(detection_selector, log_dNdz+log_dNdmds, np.NINF)
    log_Nexp = np.log(Tobs) + np.logaddexp.reduce(log_dNdzdmds - logpdraw) - np.log(Ndraw)
    log_N_tot = log_N(Tobs,lgR0,gamma)
    term1 = Nobs*log_N_tot
    term2 = -np.exp(log_Nexp)
    selection=term1 + term2
    logmu=log_Nexp-log_N_tot 
    varsel= np.sum(np.exp(2*(np.log(Tobs)+log_dNdzdmds - logpdraw-log_N_tot- np.log(Ndraw))))-np.exp(2*logmu)/Ndraw
    total_vars=Nobs**2 * varsel / np.exp(2*logmu)
    Neff=np.exp(2*logmu)/varsel
    return selection, total_vars, Neff


#This is selection effect not account for spin distribution
def Rate_mass_z_selection(Nobs,mass_z_model,**kwargs):
    log_dN_dmdsdz,log_N_tot=mass_z_model(m1_inj,m2_inj,a1_inj,a2_inj,ct1_inj,ct2_inj,z_inj,Tobs,**kwargs)
    log_dNdzdmds = np.where(detection_selector, log_dN_dmdsdz, np.NINF)
    log_Ndraw=np.log(Ndraw)
    log_Nexp = np.logaddexp.reduce(log_dNdzdmds - logpdraw) - log_Ndraw
    term1 = Nobs*log_N_tot
    term2 = -np.exp(log_Nexp)
    selection = term1 + term2
    logmu = log_Nexp-log_N_tot 
    varsel= np.sum(np.exp(2*(np.log(Tobs)+log_dNdzdmds - logpdraw - log_N_tot - log_Ndraw)))-np.exp(2*logmu)/Ndraw
    total_vars=Nobs**2 * varsel / np.exp(2*logmu)
    Neff=np.exp(2*logmu)/varsel
    return selection, total_vars, Neff

def Observable_Rate(Double_mass_pair_branch,lgR0,gamma,**kwargs):
    log_dNdz = lgR0/np.log10(np.e) + (gamma-1)*log1pz_inj + logdVdz_inj
    p11,p12,p21,p22 = Double_mass_pair_branch(m1_inj,m2_inj,**kwargs)
    logp11,logp12,logp21,logp22 = np.log(p11)+log_dNdz- logpdraw,np.log(p12)+log_dNdz- logpdraw,\
            np.log(p21)+log_dNdz- logpdraw,np.log(p22)+log_dNdz- logpdraw
    detection_selectors=np.array([detection_selector,detection_selector,detection_selector,detection_selector])
    log_dNdzdmds = np.where(detection_selectors, np.array([logp11,logp12,logp21,logp22]), np.NINF)
    log_Nexp = np.log(Tobs) + np.logaddexp.reduce(log_dNdzdmds,axis=1) - np.log(Ndraw)
    log_N_tot = log_N(Tobs,lgR0,gamma)
    logmu=log_Nexp-log_N_tot 
    return np.exp(logmu)
'''
idx=np.where((m1_inj>20)*(a1_inj>0.4))
m1_inj=m1_inj[idx]
a1_inj=a1_inj[idx]
a2_inj=a2_inj[idx]
ct1_inj=ct1_inj[idx]
ct2_inj=ct2_inj[idx]
'''
'''
pct1_inj=p_ct(ct1_inj)
pct2_inj=p_ct(ct2_inj)
pct_inj=np.append(pct1_inj,pct2_inj)
ct_inj=np.append(ct1_inj,ct2_inj)
plt.hist(ct_inj,weights=1./pct_inj,histtype='step',density=1,label='observable')
plt.plot([-1,1],[0.5,0.5],label='injection')
plt.legend()

y,_=np.histogram(ct_inj,weights=1./pct_inj,density=1,bins=100)
x=np.linspace(-1,1,100)
f=interp1d(x,y)
xx=np.linspace(-1,1,5000)
yy=f(xx)
norm=np.sum(yy)*2./5000
def f_normed(x):
    return f(x)/norm
'''