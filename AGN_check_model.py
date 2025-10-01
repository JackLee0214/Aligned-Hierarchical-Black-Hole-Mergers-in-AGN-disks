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


####################################

from model_libs_GWTC3 import Double_mass, Double_ma, PS_mass, spin_a, PS_ma, Default_ct, Double_mact_pair, Single_mact_pair_nospin, Double_mact_pair_nospin, Single_mass_pair, Double_mass_pair_un,\
    spin_ct

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

############################
#nonparametric spin 

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
    yy=np.exp(cs(xx))*(xx>amin)*(xx<amax)
    norm=np.sum(yy)*1./1000.
    px = np.exp(cs(a))*(a>amin)*(a<amax)/norm
    return px

def anti_aligned(ct,sigma_t,zeta,r_anti):
    p_al=TG(1,sigma_t,-1,1).prob(ct)*(1-r_anti)+TG(-1,sigma_t,-1,1).prob(ct)*r_anti
    p_un=Uniform(-1,1).prob(ct)
    return p_al*zeta+p_un*(1-zeta)

###############
#parametric ct

def Double_ma_para_ct(m1,a1,ct1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,zmin1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,mu_t2,sigma_t2,zeta2,zmin2,r2):
    p1=PS_ma(m1,a1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,mu_a1,sigma_a1,amin1,amax1)*(1-r2)*spin_ct(ct1,mu_t1,sigma_t1,zeta1,zmin1)
    p2=PS_ma(m1,a1,alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,mu_a2,sigma_a2,amin2,amax2)*r2*spin_ct(ct1,mu_t2,sigma_t2,zeta2,zmin2)
    return p1+p2

def Double_ma_para_ct_anti_aligned(m1,a1,ct1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,zmin1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,mu_t2,sigma_t2,zeta2,zmin2,r_anti2,r2):
    p1=PS_ma(m1,a1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,mu_a1,sigma_a1,amin1,amax1)*(1-r2)*spin_ct(ct1,mu_t1,sigma_t1,zeta1,zmin1)
    p2=PS_ma(m1,a1,alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,mu_a2,sigma_a2,amin2,amax2)*r2*anti_aligned(ct1,sigma_t2,zeta2,r_anti2)
    return p1+p2

#############################################
#combined
#############################################

#nonppair
def Double_ma_para_ct_nonppair_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,zmin1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,mu_t2,sigma_t2,zeta2,zmin2,r2,beta,nq1,nq2,nq3,nq4):
    pma1=Double_ma_para_ct(m1,a1,ct1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,zmin1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,mu_t2,sigma_t2,zeta2,zmin2,r2)
    pma2=Double_ma_para_ct(m2,a2,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,zmin1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,mu_t2,sigma_t2,zeta2,zmin2,r2)
    pdf = pma1*pma2*spline_pair(m2/m1,beta,nq1,nq2,nq3,nq4)
    return np.where((m2<m1), pdf , 1e-100)

def Double_mass_nonppair_un(m1,m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta,nq1,nq2,nq3,nq4):
    pm1=Double_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2)
    pm2=Double_mass(m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2)
    pdf = pm1*pm2*spline_pair(m2/m1,beta,nq1,nq2,nq3,nq4)
    return np.where((m2<m1), pdf , 1e-100)

def Double_ma_para_ct_nonppair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,zmin1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,mu_t2,sigma_t2,zeta2,zmin2,r2,beta,nq1,nq2,nq3,nq4):
    m1_sam = np.linspace(2,200,500)
    m2_sam = np.linspace(2,200,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Double_mass_nonppair_un(x,y,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta,nq1,nq2,nq3,nq4)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Double_ma_para_ct_nonppair_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,zmin1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,mu_t2,sigma_t2,zeta2,zmin2,r2,beta,nq1,nq2,nq3,nq4)/AMP1
    return pdf

def hyper_Double_ma_para_ct_nonppair(dataset,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,zmin1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,mu_t2,sigma_t2,zeta2,zmin2,r2,beta,nq1,nq2,nq3,nq4,gamma):
    z = dataset['z']
    a1,a2,ct1,ct2=dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1=dataset['m1']
    m2=dataset['m2']
    hp = Double_ma_para_ct_nonppair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,zmin1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,mu_t2,sigma_t2,zeta2,zmin2,r2,beta,nq1,nq2,nq3,nq4)*llh_z(z,gamma)
    return hp

def Double_ma_para_ct_nonppair_nospin(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,zmin1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,mu_t2,sigma_t2,zeta2,zmin2,r2,beta,nq1,nq2,nq3,nq4):
    m1_sam = np.linspace(2,200,500)
    m2_sam = np.linspace(2,200,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Double_mass_nonppair_un(x,y,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta,nq1,nq2,nq3,nq4)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Double_mass_nonppair_un(m1,m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta,nq1,nq2,nq3,nq4)/AMP1*1./4.
    return pdf

######################
#Lowspin anti-aligned

def anti_aligned_lowspin(ct,mu_t,sigma_t,zeta,r_anti):
    p_al=TG(mu_t,sigma_t,-1,1).prob(ct)*(1-r_anti)+TG(-1,sigma_t,-1,1).prob(ct)*r_anti
    p_un=Uniform(-1,1).prob(ct)
    return p_al*zeta+p_un*(1-zeta)


def Double_ma_para_ct_anti_aligned_lowspin(m1,a1,ct1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,r_anti1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,mu_t2,sigma_t2,zeta2,zmin2,r_anti2,r2):
    p1=PS_ma(m1,a1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,mu_a1,sigma_a1,amin1,amax1)*(1-r2)*anti_aligned_lowspin(ct1,mu_t1,sigma_t1,zeta1,r_anti1)
    p2=PS_ma(m1,a1,alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,mu_a2,sigma_a2,amin2,amax2)*r2*anti_aligned(ct1,sigma_t2,zeta2,r_anti2)
    return p1+p2

def Double_ma_para_ct_anti_lowspin_nonppair_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,r_anti1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,mu_t2,sigma_t2,zeta2,zmin2,r_anti2,r2,beta,nq1,nq2,nq3,nq4):
    pma1=Double_ma_para_ct_anti_aligned_lowspin(m1,a1,ct1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,r_anti1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,mu_t2,sigma_t2,zeta2,zmin2,r_anti2,r2)
    pma2=Double_ma_para_ct_anti_aligned_lowspin(m2,a2,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,r_anti1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,mu_t2,sigma_t2,zeta2,zmin2,r_anti2,r2)
    pdf = pma1*pma2*spline_pair(m2/m1,beta,nq1,nq2,nq3,nq4)
    return np.where((m2<m1), pdf , 1e-100)

def Double_ma_para_ct_anti_lowspin_nonppair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,r_anti1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,mu_t2,sigma_t2,zeta2,zmin2,r_anti2,r2,beta,nq1,nq2,nq3,nq4):
    m1_sam = np.linspace(2,200,500)
    m2_sam = np.linspace(2,200,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Double_mass_nonppair_un(x,y,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta,nq1,nq2,nq3,nq4)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Double_ma_para_ct_anti_lowspin_nonppair_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,r_anti1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,mu_t2,sigma_t2,zeta2,zmin2,r_anti2,r2,beta,nq1,nq2,nq3,nq4)/AMP1
    return pdf

def hyper_Double_ma_para_ct_anti_lowspin_nonppair(dataset,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,r_anti1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,mu_t2,sigma_t2,zeta2,zmin2,r_anti2,r2,beta,nq1,nq2,nq3,nq4,gamma):
    z = dataset['z']
    a1,a2,ct1,ct2=dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1=dataset['m1']
    m2=dataset['m2']
    hp = Double_ma_para_ct_anti_lowspin_nonppair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,r_anti1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,mu_t2,sigma_t2,zeta2,zmin2,r_anti2,r2,beta,nq1,nq2,nq3,nq4)*llh_z(z,gamma)
    return hp

def Double_ma_para_ct_anti_lowspin_nonppair_nospin(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,mu_t1,sigma_t1,zeta1,r_anti1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,mu_t2,sigma_t2,zeta2,zmin2,r_anti2,r2,beta,nq1,nq2,nq3,nq4):
    m1_sam = np.linspace(2,200,500)
    m2_sam = np.linspace(2,200,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Double_mass_nonppair_un(x,y,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta,nq1,nq2,nq3,nq4)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Double_mass_nonppair_un(m1,m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta,nq1,nq2,nq3,nq4)/AMP1*1./4.
    return pdf

############################
# Nonparametric model

def Double_nonpara_mact(m1,a1,ct1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,na1,na2,na3,na4,na5,amin1,amax1,nt1,nt2,nt3,nt4,\
                                  alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,oa1,oa2,oa3,oa4,oa5,amin2,amax2,ot1,ot2,ot3,ot4,r2):
    p1=PS_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12)*(1-r2)*spline_ct(ct1,nt1,nt2,nt3,nt4)*spline_a(a1,na1,na2,na3,na4,na5,amin1,amax1)
    p2=PS_mass(m1,alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12)*r2*spline_ct(ct1,ot1,ot2,ot3,ot4)*spline_a(a1,oa1,oa2,oa3,oa4,oa5,amin2,amax2)
    return p1+p2

def Double_nonpara_mact_nonppair_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,na1,na2,na3,na4,na5,amin1,amax1,nt1,nt2,nt3,nt4,\
                                  alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,oa1,oa2,oa3,oa4,oa5,amin2,amax2,ot1,ot2,ot3,ot4,r2,beta,nq1,nq2,nq3,nq4):
    pma1=Double_nonpara_mact(m1,a1,ct1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,na1,na2,na3,na4,na5,amin1,amax1,nt1,nt2,nt3,nt4,\
                                  alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,oa1,oa2,oa3,oa4,oa5,amin2,amax2,ot1,ot2,ot3,ot4,r2)
    pma2=Double_nonpara_mact(m2,a2,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,na1,na2,na3,na4,na5,amin1,amax1,nt1,nt2,nt3,nt4,\
                                  alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,oa1,oa2,oa3,oa4,oa5,amin2,amax2,ot1,ot2,ot3,ot4,r2)
    pdf = pma1*pma2*spline_pair(m2/m1,beta,nq1,nq2,nq3,nq4)
    return np.where((m2<m1), pdf , 1e-100)

def Double_nonpara_mact_nonppair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,na1,na2,na3,na4,na5,amin1,amax1,nt1,nt2,nt3,nt4,\
                                  alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,oa1,oa2,oa3,oa4,oa5,amin2,amax2,ot1,ot2,ot3,ot4,r2,beta,nq1,nq2,nq3,nq4):
    m1_sam = np.linspace(2,200,500)
    m2_sam = np.linspace(2,200,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Double_mass_nonppair_un(x,y,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta,nq1,nq2,nq3,nq4)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Double_nonpara_mact_nonppair_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,na1,na2,na3,na4,na5,amin1,amax1,nt1,nt2,nt3,nt4,\
                                  alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,oa1,oa2,oa3,oa4,oa5,amin2,amax2,ot1,ot2,ot3,ot4,r2,beta,nq1,nq2,nq3,nq4)/AMP1
    return pdf

def hyper_Double_nonpara_mact_nonppair(dataset,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,na1,na2,na3,na4,na5,amin1,amax1,nt1,nt2,nt3,nt4,\
                                  alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,oa1,oa2,oa3,oa4,oa5,amin2,amax2,ot1,ot2,ot3,ot4,r2,beta,nq1,nq2,nq3,nq4,gamma):
    z = dataset['z']
    a1,a2,ct1,ct2=dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1=dataset['m1']
    m2=dataset['m2']
    hp = Double_nonpara_mact_nonppair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,na1,na2,na3,na4,na5,amin1,amax1,nt1,nt2,nt3,nt4,\
                                  alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,oa1,oa2,oa3,oa4,oa5,amin2,amax2,ot1,ot2,ot3,ot4,r2,beta,nq1,nq2,nq3,nq4)*llh_z(z,gamma)
    return hp

def Double_nonpara_mact_nonppair_nospin(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,na1,na2,na3,na4,na5,amin1,amax1,nt1,nt2,nt3,nt4,\
                                  alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,oa1,oa2,oa3,oa4,oa5,amin2,amax2,ot1,ot2,ot3,ot4,r2,beta,nq1,nq2,nq3,nq4):
    m1_sam = np.linspace(2,200,500)
    m2_sam = np.linspace(2,200,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Double_mass_nonppair_un(x,y,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta,nq1,nq2,nq3,nq4)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Double_mass_nonppair_un(m1,m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta,nq1,nq2,nq3,nq4)/AMP1*1./4.
    return pdf

########################
#priors
########################
def reduced_Double_constraint_GWTC4(params):
    params['pop1_scale']=np.sign(params['mmax1']-params['mmin1']-20)-1 
    params['pop2_scale']=np.sign(params['mmax2']-params['mmin2']-20)-1  
    try:
        params['mua2-mua1']=np.sign(params['mu_a2']-params['mu_a1'])-1  
    except:
        pass
    return params

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

    priors.update({'na'+str(i+1): TG(0,2,-100,100,name='na'+str(i+1))  for i in np.arange(5)})
    priors.update({'oa'+str(i+1): TG(0,2,-100,100,name='oa'+str(i+1))  for i in np.arange(5)})
    priors.update({'nt'+str(i+1): TG(0,1,-100,100,name='nt'+str(i+1))  for i in np.arange(4)})
    priors.update({'ot'+str(i+1): TG(0,1,-100,100,name='ot'+str(i+1))  for i in np.arange(4)})
    priors.update({'oa1':-10,'na5': -10})

    priors.update({key:bilby.prior.Constraint(minimum=-0.1, maximum=0.1) for key in ['pop1_scale','pop2_scale']})
    return priors

def add_nonppair_priors(priors):
    priors.update({'nq'+str(i+1): TG(0,1,-100,100)  for i in np.arange(4)})
    priors.update({'nq1':0})
    return priors

def Double_para_ct_priors(conversion_function=reduced_Double_constraint_GWTC4):
    priors=bilby.prior.PriorDict(conversion_function=conversion_function)
    priors.update(dict(
                    beta = Uniform(0,6,'beta','$\\beta$'),

                    delta1 =Uniform(1,10),
                    mmin1 = Uniform(2., 50., 'mmin1', '$m_{\\rm min,1}$'),
                    mmax1 = Uniform(20., 200, 'mmax1', '$m_{\\rm max,1}$'),
                    alpha1 = Uniform(-4, 8., 'alpha1', '$\\alpha,1$'),
                    
                    delta2 = Uniform(1,20),
                    mmin2 = Uniform(2., 50., 'mmin2', '$m_{\\rm min,2}$'),
                    mmax2 = Uniform(20., 200, 'mmax2', '$m_{\\rm max,2}$'),
                    alpha2 = Uniform(-4., 8., 'alpha2', '$\\alpha_2$'),
                    r2 = Uniform(0,1, 'r2', '$r_2$'),

                    mu_t1 = Uniform(-1, 1, 'mu_t1', '$\\mu_{\\rm t1}$'),
                    sigma_t1 = Uniform(0.1, 4., 'sigma_t1', '$\\sigma_{\\rm t1}$'),
                    zmin1 = -1,
                    zeta1 = Uniform(0,1,'zeta1','$\\zeta_1$'),

                    mu_t2 = Uniform(-1, 1, 'mu_t2', '$\\mu_{\\rm t2}$'),
                    sigma_t2 = Uniform(0.1, 4., 'sigma_t2', '$\\sigma_{\\rm t2}$'),
                    zmin2 = -1,
                    zeta2 = Uniform(0,1,'zeta2','$\\zeta_2$'),

                    mu_a1 = Uniform(0., 1., 'mu_a1', '$\\mu_{\\rm a,1}$'),
                    sigma_a1 = Uniform(0.05, 0.5, 'sigma_a1', '$\\sigma_{\\rm a,1}$'),
                    amin1 = 0,
                    amax1 = Uniform(0.2,1,'amax1', '$a_{\\rm max,1}$'),
                    
                    amin2 = Uniform(0,0.8, 'amin2', '$a_{\\rm min,2}$'),
                    mu_a2 = Uniform(0, 1., 'mu_a2', '$mu_{\\rm a,2}$'),
                    sigma_a2 = Uniform(0.05, 0.5, 'sigma_a2', '$\\sigma_{\\rm a,2}$'),
                    amax2 = 1,

                    lgR0 = Uniform(0,3,'lgR0','$log_{10}~R_0$'),
                    gamma= Uniform(-2,7,'gamma','$\\gamma$')))
    priors.update({'n'+str(i+1): TG(0,1,-100,100,name='n'+str(i+1))  for i in np.arange(12)})
    priors.update({'n1':0,'n'+str(12): 0})
    priors.update({'o'+str(i+1): TG(0,1,-100,100,name='o'+str(i+1))  for i in np.arange(12)})
    priors.update({'o1':0,'o'+str(12): 0})

    priors.update({key:bilby.prior.Constraint(minimum=-0.1, maximum=0.1) for key in ['pop1_scale','pop2_scale','mua2-mua1']})
    return priors



