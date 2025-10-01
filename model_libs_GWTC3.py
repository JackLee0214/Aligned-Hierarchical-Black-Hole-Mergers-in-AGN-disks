import numpy as np
from scipy.interpolate import CubicSpline
import bilby
from bilby.core.prior import TruncatedGaussian as TG
from bilby.core.prior import Uniform
from scipy.special import erf

###########################################################################################################
#
#mass and spin
#
###########################################################################################################

# Analytic Gaussian
def TG_analy(x,mu,sig,min,max):
    norm = (erf((max - mu) / 2 ** 0.5 / sig) - erf(
            (min - mu) / 2 ** 0.5 / sig)) / 2
    pdf = np.exp(-(mu - x) ** 2 / (2 * sig ** 2)) / (2 * np.pi) ** 0.5 \
            / sig / norm
    return np.where((min<x) & (x<max), pdf , 1e-10000)


############
#mass
############
def smooth(m,mmin,delta):
    A = (m-mmin == 0.)*1e-10 + (m-mmin)
    B = (m-mmin-delta == 0.)*1e-10 + abs(m-mmin-delta)
    f_m_delta = delta/A - delta/B
    return (np.exp(f_m_delta) + 1.)**(-1.)*(m<=(mmin+delta))+1.*(m>(mmin+delta))

def PL(m1,mmin,mmax,alpha,delta):
    norm=(mmax**(1-alpha)-mmin**(1-alpha))/(1-alpha)
    pdf = m1**(-alpha)/norm*smooth(m1,mmin,delta)
    return np.where((mmin<m1) & (m1<mmax), pdf , 1e-10000)

def PS_mass(m1,alpha,mmin,mmax,delta,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12):
    xi=np.exp(np.linspace(np.log(6),np.log(80),12))
    yi=np.array([n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12])
    cs = CubicSpline(xi,yi,bc_type='natural')
    xx=np.linspace(2,200,2000)
    yy=np.exp(cs(xx)*(xx>6)*(xx<80))*PL(xx,mmin,mmax,alpha,delta)
    norm=np.sum(yy)*198./2000.
    pm1 = np.exp(cs(m1)*(m1>6)*(m1<80))*PL(m1,mmin,mmax,alpha,delta)/norm
    return pm1
        
####################################
#spin
####################################

#magnitude
def spin_a(a1,mu_a,sigma_a,amin,amax):
    return TG(mu_a,sigma_a,amin,amax).prob(a1)

#cosine tilt angle
def Default_ct(ct1,ct2,mu_t,sigma_t,zeta,zmin):
    return TG(mu_t,sigma_t,zmin,1).prob(ct1)*TG(mu_t,sigma_t,zmin,1).prob(ct2)*zeta+\
        Uniform(-1,1).prob(ct1)*Uniform(-1,1).prob(ct2)*(1-zeta)
#cosine tilt angle
def spin_ct(ct1,mu_t,sigma_t,zeta,zmin):
    return TG(mu_t,sigma_t,zmin,1).prob(ct1)*zeta+Uniform(-1,1).prob(ct1)*(1-zeta)
    
####################################
#only mass model
####################################
#Single
def Single_mass_pair_un(m1,m2, alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,beta):
    pm1 = PS_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12)
    pm2 = PS_mass(m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12)
    pdf = pm1*pm2*(m2/m1)**beta
    return np.where((m2<m1), pdf , 1e-10000)
    
def Single_mass_pair(m1,m2, alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,beta):
    m1_sam = np.linspace(2,200,500)
    m2_sam = np.linspace(2,200,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Single_mass_pair_un(x,y, alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,beta)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Single_mass_pair_un(m1,m2, alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,beta)/AMP1
    return pdf

#Double
def Double_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                    alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2):
    p1=PS_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12)*(1-r2)
    p2=PS_mass(m1,alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12)*r2
    return p1+p2

def Double_mass_pair_un(m1,m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta):
    pm1=Double_mass(m1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2)
    pm2=Double_mass(m2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2)
    pdf = pm1*pm2*(m2/m1)**beta
    return np.where((m2<m1), pdf , 1e-10000)
        
########################
#mass vs spin model
########################

#Double
def PS_ma(m1,a1,alpha,mmin,mmax,delta,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,mu_a,sigma_a,amin,amax):
    pdf=PS_mass(m1,alpha,mmin,mmax,delta,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12)*spin_a(a1,mu_a,sigma_a,amin,amax)
    return pdf

def Double_ma(m1,a1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2):
    p1=PS_ma(m1,a1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,mu_a1,sigma_a1,amin1,amax1)*(1-r2)
    p2=PS_ma(m1,a1,alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,mu_a2,sigma_a2,amin2,amax2)*r2
    return p1+p2

def Double_mact_pair_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,beta,mu_t,sigma_t,zmin,zeta):
    pma1=Double_ma(m1,a1,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                        alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2)
    pma2=Double_ma(m2,a2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                        alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2)
    pct=Default_ct(ct1,ct2,mu_t,sigma_t,zeta,zmin)
    pdf = pma1*pma2*pct*(m2/m1)**beta
    return np.where((m2<m1), pdf , 1e-100)

def Double_mact_pair(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,beta,mu_t,sigma_t,zmin,zeta):
    m1_sam = np.linspace(2,200,500)
    m2_sam = np.linspace(2,200,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    pgrid1 = Double_mass_pair_un(x,y,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,\
                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,r2,beta)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    AMP1 = np.sum(pgrid1*dx*dy)
    pdf = Double_mact_pair_un(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,beta,mu_t,sigma_t,zmin,zeta)/AMP1
    return pdf

def Double_mact_pair_nospin(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,\
                                            alpha2,mmin2,mmax2,delta2,o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,amin2,mu_a2,sigma_a2,amax2,r2,beta,mu_t,sigma_t,zmin,zeta):
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
    pdf_spin = 1/4.
    return pdf*pdf_spin

#Single
def Single_mact_pair_nospin(m1,m2,a1,a2,ct1,ct2,alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,amin1,mu_a1,sigma_a1,amax1,beta,mu_t,sigma_t,zmin,zeta,mu_a2=None,md=None):
    pdf = Single_mass_pair(m1,m2, alpha1,mmin1,mmax1,delta1,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,beta)*1/4.
    return pdf
    
#########################################################################################################
#
#conversion function
#
#########################################################################################################

def Double_constraint(params):
    params['constraint1']=np.sign(params['amax1']-params['mu_a1'])-1
    params['constraint2']=np.sign(params['mu_a2']-params['mu_a1'])-1
    params['constraint3']=np.sign(params['mu_a2']-params['amin2'])-1
    params['constraint4']=np.sign(params['amax2']-params['mu_a2'])-1
    params['constraint5']=np.sign(params['amax2']-params['amin2']-0.2)-1
    params['constraint6']=np.sign(params['mmax2']-params['mmin2']-20)-1
    params['constraint7']=np.sign(params['mmax1']-params['mmin1']-20)-1
    
    return params

#priors
def Double_priors(Double_constraint=Double_constraint):
    priors=bilby.prior.PriorDict(conversion_function=Double_constraint)
    priors.update(dict(
                    delta1=Uniform(1,10),
                    mmin1 = Uniform(2., 50., 'mmin1', '$m_{\\rm min,1}$'),
                    mmax1 = Uniform(20., 200, 'mmax1', '$m_{\rm max,1}$'),
                    alpha1 = Uniform(-4, 8., 'alpha1', '$\\alpha,1$'),
                    mu_a1 = Uniform(0., 1., 'mu_a1', '$\\mu_{\\rm a,1}$'),
                    sigma_a1 = Uniform(0.05, 0.5, 'sigma_a1', '$\\sigma_{\\rm a,1}$'),
                    amin1 = 0,
                    amax1 = Uniform(0.2,1,'amax1', '$a_{\rm max,1}$'),
                    
                    delta2=Uniform(1,10),
                    mmin2 = Uniform(2., 50., 'mmin2', '$m_{\rm min,2}$'),
                    mmax2 = Uniform(20., 200, 'mmax2', '$m_{\rm max,2}$'),
                    alpha2 = Uniform(-4., 8., 'alpha2', '$\\alpha_2$'),
                    amin2 = Uniform(0,0.8, 'amin2', '$a_{\rm min,2}$'),
                    mu_a2 = Uniform(0, 1., 'mu_a2', '$mu_{\\rm a,2}$'),
                    sigma_a2 = Uniform(0.05, 0.5, 'sigma_a2', '$\\sigma_{\\rm a,2}$'),
                    amax2 = Uniform(0.2,1, 'amax2', '$a_{\rm max,2}$'),
                    r2 = Uniform(0,1, 'r2', '$r_2$'),

                    beta = Uniform(0,6,'beta','$\\beta$'),
                    mu_t = 1,
                    sigma_t = Uniform(0.1, 4., 'sigma_t', '$\\sigma_{\\rm t}$'),
                    zmin = -1,
                    zeta = Uniform(0,1,'zeta','$\\zeta$'),

                    lgR0 = Uniform(0,3,'lgR0','$log_{10}~R_0$'),
                    gamma=2.7
                 ))
                     
    priors.update({'n'+str(i+1): TG(0,1,-100,100,name='n'+str(i+1))  for i in np.arange(12)})
    priors.update({'n1':0,'n'+str(12): 0})
    priors.update({'o'+str(i+1): TG(0,1,-100,100,name='o'+str(i+1))  for i in np.arange(12)})
    priors.update({'o1':0,'o'+str(12): 0})
    priors.update({'constraint'+str(i+1):bilby.prior.Constraint(minimum=-0.1, maximum=0.1) for i in np.arange(7)})
    return priors

#fix params to reduce model
def Fixed_Double_priors(priors,add_label,fix_node=True,fix_aminmax=None):
    if fix_node:
        priors.update({'o'+str(i+1): 0  for i in np.arange(6)})
    if fix_aminmax:
        priors.update(dict(
                    amax1 = 1,
                    amin2 = 0,
                    amax2 = 1
                                  ))
        priors.pop('constraint1')
        priors.pop('constraint3')
        priors.pop('constraint4')
        priors.pop('constraint5')
        add_label+='_fixedge'
    return priors, add_label