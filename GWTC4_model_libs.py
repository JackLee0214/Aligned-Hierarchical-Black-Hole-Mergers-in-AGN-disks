import bilby
import numpy as np
import h5py
from astropy.cosmology import Planck15
from bilby.core.prior import Uniform, TruncatedGaussian as TG
from scipy.interpolate import interp1d, CubicSpline
import astropy.units as u
import os

########################################
#specify the injection dir in your computer
if os.path.isdir('/public23/home/sc60972/Spin_Ori/data'):
    inject_dir='/public23/home/sc60972/Spin_Ori/data/'
elif os.path.isdir('/Users/liyinjie/Desktop/Works/Globle_data/GW_data'):
    inject_dir='/Users/liyinjie/Desktop/Works/Globle_data/GW_data/'
elif os.path.isdir('/public1/home/m8s000968/GWdata/post'):
    inject_dir='/public1/home/m8s000968/GWdata/post/'
else:
    print(f"injection file not exist!")
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

def O3p_a(a):
    return 1

def O3p_ct(ct):
    return 1/2

def O3p_t(t):
    return np.sin(t)*O3p_ct(np.cos(t))

def O3lnp_spin(a1,a2,t1,t2,phi1,phi2):
    return np.log(O3p_a(a1))+np.log(O3p_a(a2))+np.log(O3p_t(t1))+np.log(O3p_t(t2))+np.log(p_phi(phi1))+np.log(p_phi(phi2))


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

O4_min_far = np.min([events["%s_far"%search] for search in ['o4a_cwb-bbh', 'o4a_gstlal', 'o4a_mbta','o4a_pycbc']], axis=0)
O3_min_far = np.min([events["%s_far"%search] for search in ['o3_cwb','o3_pycbc_bbh', 'o3_gstlal', 'o3_mbta', 'o3_pycbc_hyperbank']], axis=0)
############################################################
#lnp_spin for O4a, and np.log(1./4.) for O3
log_pspin = lnp_spin(a1_inj,a2_inj,t1_inj,t2_inj,phi1_inj,phi2_inj)*(O4_min_far<1)+O3lnp_spin(a1_inj,a2_inj,t1_inj,t2_inj,phi1_inj,phi2_inj)*(O3_min_far<1)
logpdraw=lnp_draw-log_pspin+np.log(1./4.)

#exclude NS with has spin magnitude U(0,0.4)
sel_indx=np.where((m2_inj>3) & (m1_inj<300))

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
logpdraw=logpdraw[sel_indx]

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
