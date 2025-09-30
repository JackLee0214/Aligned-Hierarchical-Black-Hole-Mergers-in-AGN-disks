import bilby
from bilby.core.sampler import run_sampler
import numpy as np
import h5py
import pickle
from bilby.core.result import read_in_result as rr
import json
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import pickle
from bilby.hyper.likelihood import HyperparameterLikelihood
from bilby.core.prior import TruncatedGaussian as TG

from bilby.core.prior import Interped, Uniform, LogUniform, DeltaFunction
import sys

from model_libs_GWTC3 import Double_mact_pair_nospin,hyper_Double,Double_priors, Single_mact_pair_nospin,hyper_Single,Single_priors
import os

from GWTC4_model_libs import reduced_Double_constraint_GWTC4, hyper_Double_ma_spline_ct_pair, Double_ma_spline_ct_pair_nospin
from GWTC4_model_libs import Double_nonpara_priors, Double_nonpara_mact_pair_nospin, hyper_Double_nonpara_mact_pair
from GWTC4_model_libs import Rate_selection_function_with_uncertainty


from AGN_check_model import hyper_Double_ma_para_ct_pair, Double_ma_para_ct_pair_nospin, Double_para_ct_priors, Fixed_Double_priors,\
    hyper_Double_ma_para_ct_nonppair, Double_ma_para_ct_nonppair_nospin, add_nonppair_priors, hyper_Double_ma_modify_ct_nonppair, Double_ma_modify_ct_nonppair_nospin,\
    Double_nonpara_mact_nonppair_nospin, hyper_Double_nonpara_mact_nonppair, Single_nonpara_mact_nonppair_nospin, hyper_Single_nonpara_mact_nonppair, Single_nonpara_priors,\
    Double_nonpara_mact_alignedpop2_nonppair_nospin, hyper_Double_nonpara_mact_alignedpop2_nonppair,\
    hyper_Double_ma_para_ct_anti_nonppair, Double_ma_para_ct_anti_nonppair_nospin,\
    hyper_Double_ma_para_ct_anti_lowspin_nonppair, Double_ma_para_ct_anti_lowspin_nonppair_nospin



Neff_obs_thr=20

run=1

computer='papp_cloud'
if os.path.isdir('/public23/home/sc60972/GWdata'):
    data_dir='/public23/home/sc60972/GWdata/'
elif os.path.isdir('/Users/liyinjie/Desktop/Works/Globle_data/GW_data'):
    data_dir='/Users/liyinjie/Desktop/Works/Globle_data/GW_data/'
    computer='mac'
elif os.path.isdir('/public1/home/m8s000968/GWdata/post'):
    data_dir='/public1/home/m8s000968/GWdata/post/'
else:
    print(f"文件夹不存在!")


outdir='Search_AGNdisk_test'
label=['Double_para_ct_anti_nonppair','Double_nonpara_alignedpop2_nonppair','Single_nonpara_nonppair','Double_nonpara_nonppair','Double_para_ct_nonppair','Double_iso_ct_nonppair'][int(sys.argv[1])]
add_label=''
sampler='pymultinest'
#sampler='inessai'
dyn_sample='hslice'
importance_nest_sampler=False
#dyn_sample='rwalk'
#dyn_sample=None
npool=None
#npool=64

All_aligned=1
fix_mut=1
zero_spin=1
iso=0

selection_function=Rate_selection_function_with_uncertainty

if label=='Double_nonpara_nonppair':
    priors=Double_nonpara_priors()
    priors=add_nonppair_priors(priors)
    mass_model= Double_nonpara_mact_nonppair_nospin
    hyper_prior= hyper_Double_nonpara_mact_nonppair
    if iso:
        priors.update({'ot'+str(i+1): 0  for i in np.arange(4)})
        add_label+='_Pop2iso'
    fixnode=0
    if not fixnode:
        priors.update({'n'+str(i+1): TG(0,1,-100,100,name='n'+str(i+1))  for i in np.arange(12)})
        priors.update({'n1':0,'n'+str(12): 0})
        priors.update({'o'+str(i+1): TG(0,1,-100,100,name='o'+str(i+1))  for i in np.arange(12)})
        priors.update({'o1':0,'o'+str(12): 0})
        add_label+='notfix_node'
elif label=='Double_para_ct_nonppair':
    hyper_prior=hyper_Double_ma_para_ct_nonppair
    mass_model=Double_ma_para_ct_nonppair_nospin
    priors=Double_para_ct_priors(reduced_Double_constraint_GWTC4)
    #priors=Double_priors()
    priors, add_label=Fixed_Double_priors(priors,add_label)
    if All_aligned:
        priors['zeta2'] = 1
        add_label+='_Aligned'
        if fix_mut:
            priors['mu_t2'] = 1
            add_label+='_fixmut'
    priors=add_nonppair_priors(priors)
elif label=='Double_para_ct_anti_nonppair':
    mass_model= Double_ma_para_ct_anti_nonppair_nospin
    hyper_prior= hyper_Double_ma_para_ct_anti_nonppair
    priors=Double_para_ct_priors(reduced_Double_constraint_GWTC4)
    priors, add_label=Fixed_Double_priors(priors,add_label)
    priors=add_nonppair_priors(priors)
    priors['mu_t2'] = 1
    priors['sigma_t2'] = Uniform(0.1,4)
    priors['zeta2'] = Uniform(0,1)
    priors['r_anti2'] = Uniform(0,1)
    priors['zmin2'] = -1
    lowspin_anti=1
    if lowspin_anti:
        priors['r_anti1']=Uniform(0,1)
        priors.pop('zmin1')
        add_label+='_lowspin_anti'
        mass_model= Double_ma_para_ct_anti_lowspin_nonppair_nospin
        hyper_prior= hyper_Double_ma_para_ct_anti_lowspin_nonppair
elif label=='Double_iso_ct_nonppair':
    hyper_prior=hyper_Double_ma_para_ct_nonppair
    mass_model=Double_ma_para_ct_nonppair_nospin
    priors=Double_para_ct_priors(reduced_Double_constraint_GWTC4)
    #priors=Double_priors()
    priors, add_label=Fixed_Double_priors(priors,add_label)
    priors['mu_t2'] = 1
    priors['sigma_t2'] = 100
    priors['zeta2'] = 0
    priors=add_nonppair_priors(priors)


priors['gamma']=Uniform(-2,7,'gamma',r'$\gamma$')
    

#read data
#read data
with open(data_dir+'GWTC-3/GWTC3_BBH_Mixed_5000.pickle', 'rb') as fp:
    samples3, evidences3 = pickle.load(fp)
Nobs3=len(samples3)
ln_evidences3=[np.log(ev) for ev in evidences3]
print('number of events in GWTC-3:',Nobs3)

with open(data_dir+'GWTC-4/O4a_BBH_Mixed5000_Nobs_83.pickle', 'rb') as fp:
    samples4, ln_evidences4 = pickle.load(fp)
Nobs4=len(samples4)
print('number of events in O4a:',Nobs4)

samples=samples3+samples4
ln_evidences=ln_evidences3+ln_evidences4
Nobs=len(samples)
print('number of events in GWTC-4:',Nobs)


class Hyper_selection_with_var(HyperparameterLikelihood):

    def convert_params(self):
        #self.parameters=convert_params(self.parameters)
        #self.parameters['constraint']=0
        #self.parameters.pop('constraint')
        pass

    def likelihood_ratio_obs_var(self):
        weights = np.nan_to_num(self.hyper_prior.prob(self.data) / self.data['prior'])
        expectations = np.nan_to_num(np.mean(weights, axis=-1))
        if np.any(expectations==0.):
            nan_count = np.count_nonzero(expectations==0.)
            #print(3.1, nan_count)
            return 100+np.square(nan_count), -1e4 - np.square(nan_count), -2.e10
        else:
            square_expectations = np.mean(weights**2, axis=-1)
            variances = (square_expectations - expectations**2) / (
                self.samples_per_posterior * expectations**2
            )
            variances = np.nan_to_num(variances)
            if np.any(variances==0.):
                #print(3.2, np.count_nonzero(variances==0.))
                return 100 + np.square(np.count_nonzero(variances==0.)),  -1e4 , -2.e10
            else:
                variance = np.sum(variances)
                Neffs = expectations**2/square_expectations*self.samples_per_posterior
                Neffmin = np.min(Neffs)
                return variance, Neffmin, np.sum(np.log(expectations))
    """
    def log_likelihood(self):
        self.hyper_prior.parameters.update(self.parameters)
        
        #self.convert_params()
        obs_vars, obs_Neff, llhr= self.likelihood_ratio_obs_var()
        #print('obs_vars:',obs_vars,'obs_Neff:',obs_Neff,'llhr:',llhr)
        if (obs_Neff>=Neff_obs_thr):
            selection, sel_vars, sel_Neff = selection_function(self.n_posteriors, mass_model, **self.parameters)
            #print('sel_Neff:',sel_Neff)
            if (sel_Neff>=4*self.n_posteriors):
                #print(self.parameters)
                #print(1, self.noise_log_likelihood()+llhr + selection)
                return self.noise_log_likelihood() + llhr + selection
            else:
                #print(2, - 2.e10 - np.square(sel_Neff - 4*self.n_posteriors))
                return - 2.e10 - np.square(sel_Neff - 4*self.n_posteriors)  
        else:
            #print(3, - 4.e10 - np.square(obs_Neff - Neff_obs_thr))
            return - 4.e10 - np.square(obs_Neff - Neff_obs_thr)
    """    
    def log_likelihood(self):
        self.hyper_prior.parameters.update(self.parameters)
        self.convert_params()
        obs_vars, obs_Neff, llhr= self.likelihood_ratio_obs_var()
        if (obs_vars<1):
            selection, sel_vars, sel_Neff = selection_function(self.n_posteriors, mass_model, **self.parameters)
            if (sel_vars+obs_vars<1):
                #print(self.parameters)
                to_ret = self.noise_log_likelihood() + llhr + selection
                #print(1, to_ret)
                return to_ret
            else:
                #print(2, - 2.e10 - np.square(100*(sel_vars+obs_vars-1)))
                return - 2.e10 - np.square(100*(sel_vars+obs_vars-1))
        else:
            #print(3, - 4.e10 - np.square(100*obs_vars - 100))
            return - 4.e10 - np.square(100*obs_vars - 100)
            
bilby.core.utils.setup_logger(outdir=outdir, label=label+add_label)

all_keys=priors.keys()
search_keys=[]
key_dict={}
for key in all_keys:
    prior=priors[key]
    try:
        a=prior.sample(1)
        if a==None:
            print('constraint parameters:',key)
        else:
            search_keys.append(key)
    except: 
        key_dict.update({key:prior})
        print('fixed parameters:',key)
print('search parameters in order:',search_keys)
search_keys.append('log_likelihood')
key_dict['search_keys']=search_keys
import pickle
with open(outdir+'/'+label+add_label+'_keys_dict.pkl', 'wb') as f:  # 注意 'wb' 二进制模式
    pickle.dump(key_dict, f)

if computer=='mac':
    if sampler=='pymultinest':
        os.environ['DYLD_LIBRARY_PATH'] = '/Users/liyinjie/Desktop/Works/Globle_data/code/MultiNest/lib'
if __name__ == '__main__':
    if run:
        hp_likelihood = Hyper_selection_with_var(posteriors=samples, hyper_prior=hyper_prior, log_evidences=ln_evidences, max_samples=1e+100)
        result = run_sampler(likelihood=hp_likelihood, priors=priors, sampler=sampler, nlive=1000,npool=npool,\
                    sample=dyn_sample,importance_nest_sampling=importance_nest_sampler,\
                    use_ratio=False, outdir=outdir, label=label+add_label)
    else:
        result=rr('./{}/{}_result.json'.format(outdir,label+add_label))

    node_paras=[]
    for a in ['n','o','q','nx']:
        for b in np.arange(12):
            node_paras.append(a+str(b+1))
    plot_paras=[key for key in result.search_parameter_keys if key not in node_paras]
    if 'mmax3' in plot_paras:
        mass_paras=['beta', 'mmin1', 'mmax1', 'alpha1', 'mmin2', 'mmax2', 'alpha2', 'r2', 'mmin3', 'mmax3', 'alpha3', 'r3', 'lgR0', 'gamma']
        print('ploting mass')
        result.plot_corner(quantiles=[0.05, 0.95],parameters=mass_paras,filename='./{}/{}_mass_corner.pdf'.format(outdir,label+add_label),smooth=1.,color='skyblue')
        spin_paras=['sigma_t', 'zeta', 'mu_a2',  'r2','sigma_a2', 'mu_a3', 'sigma_a3', 'r3']
        print('ploting spin')
        result.plot_corner(quantiles=[0.05, 0.95],parameters=spin_paras,filename='./{}/{}_spin_corner.pdf'.format(outdir,label+add_label),smooth=1.,color='pink')
    print('ploting full')

    result.plot_corner(quantiles=[0.05, 0.95],parameters=plot_paras,filename='./{}/{}_corner.pdf'.format(outdir,label+add_label),smooth=1.,color='green')
    
