
## `Aligned Hierarchical Black Hole Mergers in AGN disks revealed by GWTC-4`


These files include the codes and data to re-produce the results of the work  _Aligned Hierarchical Black Hole Mergers in AGN disks revealed by GWTC-4_, arXiv: [2509.23897](https://arxiv.org/abs/2509.23897)
, [Yin-Jie Li](https://inspirehep.net/authors/1838354) ,  [Yuan-Zhu Wang](https://inspirehep.net/authors/1664025),  [Shao-Peng Tang](https://inspirehep.net/authors/1838355) , and [Yi-Zhong Fan](https://inspirehep.net/authors/1040745)

#### Main requirements
- [BILBY](https://git.ligo.org/lscsoft/bilby)
- [PyMultiNest](https://johannesbuchner.github.io/PyMultiNest/install.html)

#### Data
The events posterior samples are adopted from the [Gravitational Wave Open Science Center](https://www.gw-openscience.org/eventapi/html/GWTC/), here `C01:Mixed` samples are used for analysis and stored in `GWTC3_BBH_Mixed_5000.pickle`. 
We adopt 5000 samples for per event. For events with initial sample sizes less than 5000, the posterior samples are reused.

The O4a data are stored in `O4a_BBH_Mixed5000_Nobs_84.pickle`. 

 
 
#### Hierarchical Bayesian inference
- Specify the `data_dir` and `inject_dir` in the `AGN_infer.py`  and `AGN_check_model.py`
- Inference with our main model: run the python script `AGN_infer.py` , and specify the model in the script.

  
#### Acknowledgements
The  publicly available code [GWPopulation](https://github.com/ColmTalbot/gwpopulation) is referenced to calculate the variance of log-likelihood in the Monte Carlo integrals, and the [FigureScript](https://dcc.ligo.org/public/0171/P2000434/003/Produce-Figures.ipynb) from [LIGO Document P2000434](https://dcc.ligo.org/LIGO-P2000434/public) is referenced to produced figures in this work.


  


