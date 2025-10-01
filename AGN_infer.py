import bilby
from bilby.core.sampler import run_sampler
import numpy as np
import pickle
from bilby.core.result import read_in_result as rr
import matplotlib.pyplot as plt
from bilby.hyper.likelihood import HyperparameterLikelihood
from bilby.core.prior import TruncatedGaussian as TG
from bilby.core.prior import Uniform
import sys
import os
import pickle

from GWTC4_model_libs import Rate_selection_function_with_uncertainty
from AGN_check_model import hyper_Double_ma_para_ct_nonppair, Double_ma_para_ct_nonppair_nospin, Double_para_ct_priors,\
                             hyper_Double_nonpara_mact_nonppair, Double_nonpara_mact_nonppair_nospin, Double_nonpara_priors,\
                             hyper_Double_ma_para_ct_anti_lowspin_nonppair, Double_ma_para_ct_anti_lowspin_nonppair_nospin, \
                             add_nonppair_priors, reduced_Double_constraint_GWTC4
selection_function=Rate_selection_function_with_uncertainty

########################################
#specify the data dir in your computer
########################################
computer='papp_cloud'
if os.path.isdir('/public23/home/sc60972/GWdata'):
    data_dir='/public23/home/sc60972/GWdata/'
elif os.path.isdir('/Users/liyinjie/Desktop/Works/Globle_data/GW_data'):
    data_dir='/Users/liyinjie/Desktop/Works/Globle_data/GW_data/'
    computer='mac'
elif os.path.isdir('/public1/home/m8s000968/GWdata/post'):
    data_dir='/public1/home/m8s000968/GWdata/post/'
else:
    print(f"file not exists")

outdir='Search_AGNdisk'
########################################
#specify a model
########################################
#following models are for nonparametric model, Gaussain, with anti aligned model, isotropic model respectively

label=['Double_nonpara','Double_para_Gaussian','Double_para_anti','Double_iso'][int(sys.argv[1])]

add_label=''
sampler='pymultinest'
dyn_sample='hslice'
importance_nest_sampler=False
npool=None

########################################
#the priors
########################################
if label=='Double_nonpara':
    mass_model= Double_nonpara_mact_nonppair_nospin
    hyper_prior= hyper_Double_nonpara_mact_nonppair
    priors=Double_nonpara_priors()

elif label=='Double_para_Gaussian':
    hyper_prior=hyper_Double_ma_para_ct_nonppair
    mass_model=Double_ma_para_ct_nonppair_nospin
    priors=Double_para_ct_priors(reduced_Double_constraint_GWTC4)
    priors['zeta2'] = 1
    priors['mu_t2'] = 1
elif label=='Double_iso':
    hyper_prior=hyper_Double_ma_para_ct_nonppair
    mass_model=Double_ma_para_ct_nonppair_nospin
    priors=Double_para_ct_priors(reduced_Double_constraint_GWTC4)
    priors['mu_t2'] = 1
    priors['sigma_t2'] = 100
    priors['zeta2'] = 0
elif label=='Double_para_anti':
    mass_model= Double_ma_para_ct_anti_lowspin_nonppair_nospin
    hyper_prior= hyper_Double_ma_para_ct_anti_lowspin_nonppair
    priors=Double_para_ct_priors(reduced_Double_constraint_GWTC4)
    priors['mu_t2'] = 1
    priors['sigma_t2'] = Uniform(0.1,4)
    priors['zeta2'] = Uniform(0,1)
    priors['r_anti2'] = Uniform(0,1)
    priors['zmin2'] = -1
    priors['r_anti1']=Uniform(0,1)
    priors.pop('zmin1')

priors=add_nonppair_priors(priors)

########################################
#read data
########################################
with open(data_dir+'GWTC-3/GWTC3_BBH_Mixed_5000.pickle', 'rb') as fp:
    samples3, evidences3 = pickle.load(fp)
Nobs3=len(samples3)
ln_evidences3=[np.log(ev) for ev in evidences3]
print('number of events in GWTC-3:',Nobs3)

with open(data_dir+'GWTC-4/O4a_BBH_Mixed5000_Nobs_84.pickle', 'rb') as fp:
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


if computer=='mac':
    if sampler=='pymultinest':
        os.environ['DYLD_LIBRARY_PATH'] = '/Users/liyinjie/Desktop/Works/Globle_data/code/MultiNest/lib'
if __name__ == '__main__':

    hp_likelihood = Hyper_selection_with_var(posteriors=samples, hyper_prior=hyper_prior, log_evidences=ln_evidences, max_samples=1e+100)
    result = run_sampler(likelihood=hp_likelihood, priors=priors, sampler=sampler, nlive=1000,npool=npool,\
                sample=dyn_sample,importance_nest_sampling=importance_nest_sampler,\
                use_ratio=False, outdir=outdir, label=label+add_label)

