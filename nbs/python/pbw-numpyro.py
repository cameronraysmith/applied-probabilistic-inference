# ---
# jupyter:
#   celltoolbar: Slideshow
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,md,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.9.4
#   rise:
#     scroll: true
#     theme: black
#   toc-autonumbering: true
#   toc-showcode: false
#   toc-showmarkdowntxt: false
#   toc-showtags: false
# ---

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# <center><font size="+3">Introductory review of applied probabilistic inference</font></center>

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# # References

# %% [markdown] {"slideshow": {"slide_type": "fragment"}}
# The following notes borrow heavily from and are *thoroughly* based on Michael Betancourt's developments that put forth a principled Bayesian workflow. The following references were integrated to produce this document:
#
# * Betancourt, Michael (2019). Probabilistic Modeling and Statistical Inference. Retrieved from https://github.com/betanalpha/knitr_case_studies/tree/master/modeling_and_inference, commit b474ec.
# * Betancourt, Michael (2020). Towards A Principled Bayesian Workflow (RStan). Retrieved from https://github.com/betanalpha/knitr_case_studies/tree/master/principled_bayesian_workflow, commit 23eb26.
# * [betanalpha/knitr_case_studies](https://github.com/betanalpha/knitr_case_studies)
# * [lstmemery/principled-bayesian-workflow-pymc3](https://github.com/lstmemery/principled-bayesian-workflow-pymc3)
# * [bayespy documentation](https://github.com/bayespy/bayespy/blob/develop/doc/source/user_guide/quickstart.rst)
# * [tikz-bayesnet](https://github.com/jluttine/tikz-bayesnet) library [technical report](https://github.com/jluttine/tikz-bayesnet/blob/master/dietz-techreport.pdf)
#
# The implementation of the modelling and inference translated from [lstmemery's pymc3 implementation of Betancourt's principled Bayesian workflow](https://github.com/lstmemery/principled-bayesian-workflow-pymc3) to [numpyro](http://num.pyro.ai/en/stable/) is by [Du Phan](https://fehiepsi.github.io/).

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# # Plotting setup

# %% [markdown]
# This plotting script relies on an installation of the `Latin Modern` fonts. You can comment it out if you do not want to use it, but should still
# ```python
# import matplotlib.pyplot as plt
# ```

# %% {"slideshow": {"slide_type": "fragment"}, "tags": []}
# %run -i 'plotting.py'

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# # Modeling process

# %% [markdown] {"slideshow": {"slide_type": "subslide"}}
# ## Observing the world through the lens of probability

# %% [markdown] {"slideshow": {"slide_type": "subslide"}}
# ### Systems, environments, and observations
#
# <div>
# <center>    
# <img src="https://github.com/betanalpha/knitr_case_studies/raw/master/modeling_and_inference/figures/inferential_config/observational_process/multiple_probes/multiple_probes.png" alt="Drawing" width="90%"/>
# <img src="https://github.com/betanalpha/knitr_case_studies/raw/master/modeling_and_inference/figures/inferential_config/observational_process/multiple_observational_processes/multiple_observational_processes.png" alt="Drawing" width="90%"/></center>
# </div>

# %% [markdown] {"slideshow": {"slide_type": "subslide"}}
# ### The space of observational models and the true data generating process

# %% [markdown] {"slideshow": {"slide_type": "fragment"}, "tags": []}
# #### The observational model
# * observation space: $Y$
# * arbitrary points in the observation space: $y$
# * explicitly realized observations from the observational process $\tilde{y}$
# * data generating process: a probability distribution over the observation space
# * space of all data generating processes: $\mathcal{P}$
# * observational model vs model configuration space: the subspace, $\mathcal{S} \subset \mathcal{P}$, of data generating processes considered in any particular application
# * parametrization: a map from a model configuration space $\mathcal{S}$ to a parameter space $\mathcal{\Theta}$ assigning to each model configuration $s \in \mathcal{S}$ a parameter $\theta \in \mathcal{\Theta}$
# * probability density for an observational model: $\pi_{\mathcal{S}}(y; s)$ in general using the parametrization to assign $\pi_{\mathcal{S}}(y; \theta)$

# %% [markdown] {"slideshow": {"slide_type": "subslide"}}
# <div>
# <center>    
# <img src="https://github.com/betanalpha/knitr_case_studies/raw/master/modeling_and_inference/figures/small_world/small_world/small_world.png" alt="Drawing" width="45%"/>
# <img src="https://github.com/betanalpha/knitr_case_studies/raw/master/modeling_and_inference/figures/small_world/small_world_one/small_world_one.png" alt="Drawing" width="45%"/>
# </center>
# </div>

# %% [markdown] {"slideshow": {"slide_type": "subslide"}}
# #### The true data generating process
# * true data generating process: $\pi^{\dagger}$ is the probability distribution that exactly captures the observational process in a given application

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# ## The practical reality of model construction

# %% [markdown] {"slideshow": {"slide_type": "fragment"}}
# <div>
# <center>    
# <img src="https://github.com/betanalpha/knitr_case_studies/raw/master/modeling_and_inference/figures/small_world/small_world_two/small_world_two.png" alt="Drawing" width="75%"/></center>
# </div>

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# ## The process of inference

# %% [markdown] {"slideshow": {"slide_type": "subslide"}}
# <div>
# <center>    
# <img src="https://github.com/betanalpha/knitr_case_studies/raw/master/modeling_and_inference/figures/inferential_config/model_config/model_config5/model_config5.png" alt="Drawing" width="90%"/></center>
# </div>

# %% [markdown] {"slideshow": {"slide_type": "fragment"}}
# How can we do our best to validate this process works as close as possible to providing a high quality mirror for natural systems?

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Workflow overview

# %% [markdown] {"slideshow": {"slide_type": "subslide"}}
# <div>
# <center>    
# <img src="https://github.com/betanalpha/knitr_case_studies/raw/master/principled_bayesian_workflow/figures/workflow/all/all.png" alt="Drawing" width="70%"/></center>
# </div>

# %% [markdown] {"slideshow": {"slide_type": "subslide"}}
# ## Example generative models

# %% [markdown] {"slideshow": {"slide_type": "subslide"}}
# ### Univariate normal model

# %% [markdown] {"slideshow": {"slide_type": "fragment"}}
# From a very simple perspective, generative modeling refers to the situation in which we develop a candidate probabilistic specification of the process from which our data are generated. Usually this will include the specification of prior distributions over all first-order parameters.

# %% [markdown] {"slideshow": {"slide_type": "fragment"}}
# <div>
# <center>    
# <img src="https://www.bayespy.org/_images/tikz-57bc0c88a2974f4c1e2335fe9edb88ff2efdf970.png" style="background-color:white;" alt="Drawing" width="10%"/></center>
# </div>

# %% [markdown] {"slideshow": {"slide_type": "fragment"}}
# \begin{split}
# p(\mathbf{y}|\mu,\tau) &= \prod^{9}_{n=0} \mathcal{N}(y_n|\mu,\tau) \\
# p(\mu) &= \mathcal{N}(\mu|0,10^{-6}) \\
# p(\tau) &= \mathcal{G}(\tau|10^{-6},10^{-6})
# \end{split}

# %% [markdown] {"slideshow": {"slide_type": "fragment"}}
# This comes from the library [bayespy](https://github.com/bayespy/bayespy/blob/develop/doc/source/user_guide/quickstart.rst). The best description we are aware of regarding the syntax and semantics of graphical models via factor graph notation is in the [tikz-bayesnet](https://github.com/jluttine/tikz-bayesnet) library [technical report](https://github.com/jluttine/tikz-bayesnet/blob/master/dietz-techreport.pdf).

# %% [markdown] {"slideshow": {"slide_type": "subslide"}}
# ### Multivariate normal models

# %% [markdown] {"slideshow": {"slide_type": "fragment"}}
# <div>
# <center>    
# <img src="https://www.bayespy.org/_images/tikz-80a1db369be1f25b61ceacfff551dae2bdd331c3.png" style="background-color:white;" alt="Drawing" width="10%"/></center>
# </div>

# %% [markdown] {"slideshow": {"slide_type": "fragment"}}
# $$\mathbf{y}_{mn} \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{\Lambda}),\qquad m=0,\ldots,9, \quad n=0,\ldots,29.$$

# %% [markdown] {"slideshow": {"slide_type": "subslide"}}
# <div>
# <center>    
# <img src="https://www.bayespy.org/_images/tikz-97236981a2be663d10ade1ad85caa727621615db.png" style="background-color:white;" alt="Drawing" width="20%"/></center>
# </div>

# %% [markdown] {"slideshow": {"slide_type": "fragment"}}
# $$\mathbf{y}_{mn} \sim \mathcal{N}(\boldsymbol{\mu}_m,
# \mathbf{\Lambda}_n),\qquad m=0,\ldots,9, \quad n=0,\ldots,29.$$
#
# Note that these are for illustrative purposes of the manner in which our data can share parameters and we have not yet defined priors over our parameters.

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# # "Build, compute, critique, repeat": Box's loop in iteration through Betancourt's principled Bayesian workflow

# %% [markdown]
# ## Setup

# %% [markdown]
# ### Load libraries

# %% {"slideshow": {"slide_type": "fragment"}}
# # %pylab inline
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import numpy as np
# plt.style.use(['seaborn-talk'])
# plt.rcParams["figure.figsize"] = (10,8)

import arviz as az
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
print(numpyro.__version__)
print(jax.__version__)
print(az.__version__)

numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)

# %% [markdown] {"slideshow": {"slide_type": "fragment"}}
# ### define colors

# %% {"slideshow": {"slide_type": "fragment"}}
c_light ="#DCBCBC"
c_light_highlight ="#C79999"
c_mid ="#B97C7C"
c_mid_highlight ="#A25050"
c_dark ="#8F2727"
c_dark_highlight ="#7C0000"

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Poisson process for arbitrary detector count data
# <!-- 4.1 -->

# %% [markdown]
# Here we build a candidate model that generates (Poisson) counts that may explain what we observe in our sample data.

# %% [markdown] {"slideshow": {"slide_type": "subslide"}}
# ### Sample data

# %% {"slideshow": {"slide_type": "fragment"}, "tags": []}
df = pd.read_csv('data.csv')
print(df.head(8))
df.shape

# %% [markdown] {"slideshow": {"slide_type": "subslide"}}
# ### Generative model specification

# %% [markdown]
# #### Prior

# %% {"slideshow": {"slide_type": "fragment"}, "tags": []}
lbda  = np.linspace(0, 20, num=int(20/0.001))

plt.plot(lbda, stats.norm(loc=0,scale=6.44787).pdf(lbda), c=c_dark_highlight, lw=2)
plt.xlabel("lambda"); plt.ylabel("Prior Density"); plt.yticks([]);

lbda99 = np.linspace(0, 15, num=int(15/0.001))
plt.fill_between(lbda99,0.,y2=stats.norm(loc=0,scale=6.44787).pdf(lbda99),color=c_dark);

# !mkdir -p ./fig/
plt.savefig("fig/prior-density-lambda.svg", bbox_inches="tight");

# %%
# !inkscape fig/prior-density-lambda.svg --export-filename=fig/prior-density-lambda.pdf 2>/dev/null

# %% [markdown]
# #### Model

# %% [markdown] {"slideshow": {"slide_type": "subslide"}}
# In this case, the candidate _complete Bayesian model_ under consideration is given by
#
# $$
# \pi( y_{1}, \ldots, y_{N}, \lambda )
# =
# \left[ \prod_{n = 1}^{N} \text{Poisson} (y_{n} \mid \lambda) \right]
# \cdot \text{HalfNormal} (\lambda \mid 6).
# $$

# %% [markdown] {"slideshow": {"slide_type": "fragment"}}
# <div>
# <center>    
# <img src="https://github.com/betanalpha/knitr_case_studies/raw/master/principled_bayesian_workflow/figures/iter1/dgm/dgm.png" alt="Drawing" width="40%"/></center>
# </div>

# %%
N = 1000
R = 500

def model(y=None):
    lbda = numpyro.sample("lbda", dist.HalfNormal(6.44787))
    return numpyro.sample("y", dist.Poisson(lbda).expand([N]), obs=y)


# %% [markdown]
# #### Simulation

# %% {"tags": []}
trace = Predictive(model, {}, num_samples=R)(jax.random.PRNGKey(0))

# %% {"slideshow": {"slide_type": "fragment"}, "tags": []}
simu_lbdas = trace['lbda']
simu_ys = trace['y']

# %% {"tags": []}
print(simu_lbdas[0:9])
print(simu_lbdas.shape)

# %% {"tags": []}
print(simu_ys[0:9])
print(simu_ys.shape)

# %% [markdown] {"slideshow": {"slide_type": "subslide"}}
# ### Plot prior predictive distribution

# %% {"slideshow": {"slide_type": "fragment"}, "tags": []}
x_max = 30
bins = np.arange(0,x_max)
bin_interp = np.linspace(0,x_max-1,num=(x_max-1)*10)
hists = np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], 1, simu_ys)

prctiles = np.percentile(hists,np.linspace(10,90,num=9),axis=0)
prctiles_interp = np.repeat(prctiles, 10,axis=1)

# %% {"slideshow": {"slide_type": "fragment"}, "tags": []}
for i,color in enumerate([c_light,c_light_highlight,c_mid,c_mid_highlight]):
    plt.fill_between(bin_interp,prctiles_interp[i,:],prctiles_interp[-1-i,:],alpha=1.0,color=color);


plt.plot(bin_interp,prctiles_interp[4,:],color=c_dark_highlight);
plt.axvline(x=25,ls='-',lw=2,color='k');
plt.xlabel('y');
plt.title('Prior predictive distribution');

# %% [markdown] {"slideshow": {"slide_type": "subslide"}}
# ### Fit to simulated data

# %% [markdown]
# [Betancourt, 2020](https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html#Step_Nine:_Fit_Simulated_Ensemble60) performs this for each `y` in trace.

# %% {"tags": []}
mcmc = MCMC(NUTS(model), num_warmup=4 * R, num_samples=R, num_chains=2)
mcmc.run(jax.random.PRNGKey(1), y=simu_ys[-1, :])
trace = mcmc.get_samples(group_by_chain=True)

# %% {"tags": []}
az.plot_trace(trace);

# %% [markdown]
# <div>
# <center>    
# <img src="https://github.com/betanalpha/knitr_case_studies/raw/master/principled_bayesian_workflow/figures/eye_chart/prior_post_regimes/prior_post_regimes.png" alt="Drawing" width="70%"/></center>
# </div>

# %% [markdown]
# <div>
# <center>    
# <img src="https://github.com/betanalpha/knitr_case_studies/raw/master/principled_bayesian_workflow/figures/eye_chart/eye_chart_regimes.png" alt="Drawing" width="70%"/></center>
# </div>

# %% [markdown]
# Posterior z-score
#
# $$
# z[f \mid \tilde{y}, \theta^{\dagger}] =
# \frac{ \mathbb{E}_{\mathrm{post}}[f \mid \tilde{y}] - f(\theta^{\dagger}) }
# { \mathbb{E}_{\mathrm{post}}[f \mid \tilde{y} ] },
# $$

# %% [markdown]
# Posterior contraction
# $$
# c[f \mid \tilde{y}] = 1 -
# \frac{ \mathbb{V}_{\mathrm{post}}[f \mid \tilde{y} ] }
# { \mathbb{V}_{\mathrm{prior}}[f \mid \tilde{y} ] },
# $$

# %% {"slideshow": {"slide_type": "fragment"}, "tags": []}
# Compute rank of prior draw with respect to thinned posterior draws
sbc_rank = np.sum(simu_lbdas < trace['lbda'][::2])

# %% {"slideshow": {"slide_type": "subslide"}, "tags": []}
# posterior sensitivities analysis
s = numpyro.diagnostics.summary(trace)["lbda"]
post_mean_lbda = s["mean"]
post_sd_lbda = s["std"]
prior_sd_lbda = 6.44787
# z_score = np.abs((post_mean_lbda - simu_lbdas) / post_sd_lbda)
z_score = (post_mean_lbda - simu_lbdas) / post_sd_lbda
shrinkage = 1 - (post_sd_lbda / prior_sd_lbda ) ** 2

# %%
post_mean_lbda

# %% {"slideshow": {"slide_type": "fragment"}, "tags": []}
plt.plot(shrinkage*np.ones(len(z_score)),z_score,'o',c="#8F272720");
plt.xlim(0,1.01); plt.xlabel('Posterior shrinkage'); plt.ylabel('Posterior z-score');

# %% [markdown] {"slideshow": {"slide_type": "subslide"}}
# ### Fit observations and evaluate

# %% {"slideshow": {"slide_type": "fragment"}}
df = pd.read_csv('data.csv')
data_ys = df[df['data']=='y']['value'].values

# %%
mcmc = MCMC(NUTS(model), num_warmup=4 * R, num_samples=R, num_chains=4)
mcmc.run(jax.random.PRNGKey(2), y=data_ys)
trace = mcmc.get_samples(group_by_chain=True)

# %%
az.plot_posterior(trace, kind="hist");

# %%
ppc = Predictive(model, mcmc.get_samples())(jax.random.PRNGKey(3))

# %% {"slideshow": {"slide_type": "fragment"}}
x_max = 30
bins = np.arange(0,x_max)
bin_interp = np.linspace(0,x_max-1,num=(x_max-1)*10)
hists = np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], 1, ppc['y'])

prctiles = np.percentile(hists,np.linspace(10,90,num=9),axis=0)
prctiles_interp = np.repeat(prctiles, 10,axis=1)

data_hist = np.histogram(data_ys,bins=bins)[0]
data_hist_interp = np.repeat(data_hist, 10)

# %% {"slideshow": {"slide_type": "subslide"}, "tags": []}
for i,color in enumerate([c_light,c_light_highlight,c_mid,c_mid_highlight]):
    plt.fill_between(bin_interp,prctiles_interp[i,:],prctiles_interp[-1-i,:],alpha=1.0,color=color);


plt.plot(bin_interp,prctiles_interp[4,:],color=c_dark_highlight);
plt.plot(bin_interp,data_hist_interp,color='black');
plt.axvline(x=25,ls='-',lw=2,color='k');
plt.xlabel('y');
plt.title('Posterior predictive distribution');

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Accounting for measurement device failure
#
# <!-- 4.2 -->

# %% [markdown]
# ### Updating model specification

# %% [markdown]
# Recall the specification of our first attempt to model the detector count data with a [simple Poisson process model](poisson-process-for-arbitrary-detector-count-data)
# ```python
# N = 1000
# R = 500
#
# def model(y=None):
#     lbda = numpyro.sample("lbda", dist.HalfNormal(6.44787))
#     return numpyro.sample("y", dist.Poisson(lbda).expand([N]), obs=y)
# ```
# Here we adapt our likelihood to include a so-called "zero-inflated Poisson distribution" to account for the presence of many $0$ counts that may derive from malfunctioning detector devices.

# %% {"slideshow": {"slide_type": "subslide"}, "tags": []}
N = 1000
R = 1000

def model2(y=None):
    theta = numpyro.sample("theta", dist.Beta(1, 1))
    lambda_ = numpyro.sample("lambda", dist.HalfNormal(6.44787))
    return numpyro.sample(
        "y", dist.ZeroInflatedPoisson(rate=lambda_, gate=1 - theta).expand([N]), obs=y)


# %% [markdown]
# ### Simulating the updated model

# %% {"tags": []}
trace = Predictive(model2, {}, num_samples=R)(jax.random.PRNGKey(0))

# %% {"slideshow": {"slide_type": "subslide"}, "tags": []}
trace["theta"][:10]

# %% {"slideshow": {"slide_type": "fragment"}, "tags": []}
trace["lambda"][:10]

# %% {"slideshow": {"slide_type": "fragment"}, "tags": []}
simu_ys = trace["y"]
simu_ys

# %% [markdown]
# What is the fraction of zero values in this simulated data?

# %% {"tags": []}
simu_ys[simu_ys < 0.001].size / simu_ys.size

# %% {"slideshow": {"slide_type": "fragment"}, "tags": []}
print(simu_ys.shape)
np.count_nonzero(simu_ys, axis=1).mean()

# %% {"slideshow": {"slide_type": "subslide"}, "tags": []}
x_max = 30
bins = np.arange(0 ,x_max)
bin_interp = np.linspace(0,x_max-1,num=(x_max-1)*10)

hists = np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], 1, simu_ys)

prctiles = np.percentile(hists,np.linspace(10, 90,num=9),axis=0)
prctiles_interp = np.repeat(prctiles, 10,axis=1)


for i, color in enumerate([c_light, c_light_highlight, c_mid, c_mid_highlight]):
    plt.fill_between(bin_interp, prctiles_interp[i, :],
                     prctiles_interp[-1 - i, :],
                     alpha = 1.0,
                     color = color);


plt.plot(bin_interp,prctiles_interp[4,:],color=c_dark_highlight);
plt.axvline(x=25,ls='-',lw=2,color='k');
plt.xlabel('y');
plt.title('Prior predictive distribution');

# %% {"slideshow": {"slide_type": "fragment"}, "tags": []}
simu_ys[simu_ys > 25].size / simu_ys.size

# %% [markdown] {"slideshow": {"slide_type": "subslide"}}
# ### Fit Simulated Observations and Evaluate 

# %% {"slideshow": {"slide_type": "fragment"}, "tags": []}
N = 1000
R = 1000

mcmc = MCMC(NUTS(model2), num_warmup=R, num_samples=R, num_chains=2)
mcmc.run(jax.random.PRNGKey(1), y=simu_ys[-1, :])
trace_fit = mcmc.get_samples(group_by_chain=True)

# %% {"tags": []}
az.plot_trace(trace_fit);

# %% {"tags": []}
numpyro.diagnostics.print_summary(trace_fit)

# %% {"slideshow": {"slide_type": "subslide"}, "tags": []}
import pickle
with open("fit_data2.pkl", "wb+") as buffer:
    pickle.dump({"model": model2, "trace": trace_fit}, buffer)

# %% {"tags": []}
ppc = Predictive(model2, mcmc.get_samples())(jax.random.PRNGKey(3))

# %% {"slideshow": {"slide_type": "fragment"}, "tags": []}
x_max = 30
bins = np.arange(0,x_max)
bin_interp = np.linspace(0,x_max-1,num=(x_max-1)*10)
hists = np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], 1, ppc['y'])

prctiles = np.percentile(hists,np.linspace(10,90,num=9),axis=0)
prctiles_interp = np.repeat(prctiles, 10,axis=1)

data_hist = np.histogram(data_ys,bins=bins)[0]
data_hist_interp = np.repeat(data_hist, 10)

# %% {"slideshow": {"slide_type": "subslide"}, "tags": []}
for i,color in enumerate([c_light,c_light_highlight,c_mid,c_mid_highlight]):
    plt.fill_between(bin_interp,prctiles_interp[i,:],prctiles_interp[-1-i,:],alpha=1.0,color=color);


plt.plot(bin_interp,prctiles_interp[4,:],color=c_dark_highlight);
plt.plot(bin_interp,data_hist_interp,color='black');
plt.axvline(x=25,ls='-',lw=2,color='k');
plt.xlabel('y');
plt.title('Posterior predictive distribution');

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# # Section 4.3

# %% [markdown] {"slideshow": {"slide_type": "subslide"}}
# ## Build a generative model

# %% [markdown]
# Build a model that generates zero-inflated Poisson counts

# %% {"slideshow": {"slide_type": "fragment"}}
lbda  = np.linspace(0, 20, num=int(20/0.001))
pdf = stats.invgamma(3.48681,scale=9.21604)
plt.plot(lbda, pdf.pdf(lbda), c=c_dark_highlight, lw=2)
plt.xlabel("lambda"); plt.ylabel("Prior Density"); plt.yticks([]);

lbda99 = np.linspace(1, 15, num=int(15/0.001))

plt.fill_between(lbda99,0.,y2=pdf.pdf(lbda99),color=c_dark);

# %% {"slideshow": {"slide_type": "subslide"}}
theta  = np.linspace(0, 1, num=int(1/0.001))
pdf = stats.beta(2.8663,2.8663)
plt.plot(theta, pdf.pdf(theta), c=c_dark_highlight, lw=2)
plt.xlabel("theta"); plt.ylabel("Prior Density"); plt.yticks([]);

theta99 = np.linspace(0.1, 0.9, num=int(0.8/0.001))

plt.fill_between(theta99,0.,y2=pdf.pdf(theta99),color=c_dark);

# %% {"slideshow": {"slide_type": "subslide"}}
#WORKING

N = 1000
R = 1000

def model3(y=None):
    lbda = numpyro.sample("lbda", dist.InverseGamma(3.48681, 9.21604))
    theta = numpyro.sample("theta", dist.Beta(2.8663, 2.8663))  
    return numpyro.sample(
        "y", dist.ZeroInflatedPoisson(rate=lbda, gate=1 - theta).expand([N]), obs=y)


# %% {"slideshow": {"slide_type": "fragment"}}
trace = Predictive(model3, {}, num_samples=R)(jax.random.PRNGKey(0))

# %% {"slideshow": {"slide_type": "fragment"}}
simu_lbdas = trace['lbda']
simu_thetas = trace['theta']
simu_ys = trace['y']

# %% [markdown] {"slideshow": {"slide_type": "subslide"}}
# ## Plot prior predictive distribution

# %% {"slideshow": {"slide_type": "fragment"}}
x_max = 30
bins = np.arange(0,x_max)
bin_interp = np.linspace(0,x_max-1,num=(x_max-1)*10)
hists = np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], 1, simu_ys)

prctiles = np.percentile(hists,np.linspace(10,90,num=9),axis=0)
prctiles_interp = np.repeat(prctiles, 10,axis=1)

# %% {"slideshow": {"slide_type": "fragment"}}
for i,color in enumerate([c_light,c_light_highlight,c_mid,c_mid_highlight]):
    plt.fill_between(bin_interp,prctiles_interp[i,:],prctiles_interp[-1-i,:],alpha=1.0,color=color);


plt.plot(bin_interp,prctiles_interp[4,:],color=c_dark_highlight);
plt.axvline(x=25,ls='-',lw=2,color='k');
plt.xlabel('y');
plt.title('Prior predictive distribution');

# %% [markdown] {"slideshow": {"slide_type": "subslide"}}
# ## Fit to simulated data

# %% [markdown]
# In the example, Betancourt performs this for each `y` in trace. Here we only compute this for one element of the trace.

# %%
N = 1000
R = 1000

mcmc = MCMC(NUTS(model3), num_warmup=4 * R, num_samples=R, num_chains=2)
mcmc.run(jax.random.PRNGKey(1), y=simu_ys[:, -1])
trace = mcmc.get_samples(group_by_chain=True)

# %%
az.plot_trace(trace);

# %% {"slideshow": {"slide_type": "fragment"}}
# Compute rank of prior draw with respect to thinned posterior draws
sbc_rank = np.sum(simu_lbdas < trace['lbda'][::2])

# %% {"slideshow": {"slide_type": "subslide"}}
# posterior sensitivities analysis
s = numpyro.diagnostics.summary(trace)['lbda']
post_mean_lbda = s['mean']
post_sd_lbda = s['std']
prior_sd_lbda = 6.44787
z_score = np.abs((post_mean_lbda - simu_lbdas) / post_sd_lbda)
shrinkage = 1 - (post_sd_lbda / prior_sd_lbda ) ** 2

# %% {"slideshow": {"slide_type": "fragment"}}
plt.plot(shrinkage*np.ones(len(z_score)),z_score,'o',c="#8F272720");
plt.xlim(0,1.01); plt.xlabel('Posterior shrinkage'); plt.ylabel('Posterior z-score');

# %% [markdown] {"slideshow": {"slide_type": "subslide"}}
# ## Fit observations and evaluate

# %% {"slideshow": {"slide_type": "fragment"}}
df = pd.read_csv('data.csv')
data_ys = df[df['data']=='y']['value'].values

# %% {"slideshow": {"slide_type": "fragment"}}
mcmc = MCMC(NUTS(model3), num_warmup=4 * R, num_samples=R, num_chains=4)
mcmc.run(jax.random.PRNGKey(2), y=data_ys)
trace = mcmc.get_samples(group_by_chain=True)

# %%
az.plot_posterior(trace, var_names="lbda");

# %%
ppc = Predictive(model3, mcmc.get_samples())(jax.random.PRNGKey(3))

# %% {"slideshow": {"slide_type": "fragment"}}
x_max = 30
bins = np.arange(0,x_max)
bin_interp = np.linspace(0,x_max-1,num=(x_max-1)*10)
hists = np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], 1, ppc['y'])

prctiles = np.percentile(hists,np.linspace(10,90,num=9),axis=0)
prctiles_interp = np.repeat(prctiles, 10,axis=1)

data_hist = np.histogram(data_ys,bins=bins)[0]
data_hist_interp = np.repeat(data_hist, 10)

# %% {"slideshow": {"slide_type": "subslide"}}
for i,color in enumerate([c_light,c_light_highlight,c_mid,c_mid_highlight]):
    plt.fill_between(bin_interp,prctiles_interp[i,:],prctiles_interp[-1-i,:],alpha=1.0,color=color);


plt.plot(bin_interp,prctiles_interp[4,:],color=c_dark_highlight);
plt.plot(bin_interp,data_hist_interp,color='black');
plt.axvline(x=25,ls='-',lw=2,color='k');
plt.xlabel('y');
plt.title('Posterior predictive distribution');

# %% {"slideshow": {"slide_type": "subslide"}, "tags": []}
# posterior sensitivities analysis
s = numpyro.diagnostics.summary(trace)['lbda']
post_mean_lbda = s['mean']
post_sd_lbda = s['std']
prior_sd_lbda = 6.44787
z_score = np.abs((post_mean_lbda - simu_lbdas) / post_sd_lbda)
shrinkage = 1 - (post_sd_lbda / prior_sd_lbda ) ** 2

# %% {"slideshow": {"slide_type": "fragment"}, "tags": []}
plt.plot(shrinkage*np.ones(len(z_score)),z_score,'o',c="#8F272720");
plt.xlim(0,1.01); plt.xlabel('Posterior shrinkage'); plt.ylabel('Posterior z-score');


# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# # Section 4.4

# %% {"slideshow": {"slide_type": "subslide"}}
def rv_truncated_poisson(mu, mx, size=None):
    mu = np.asarray(mu)
    mx = np.asarray(mx)
    dist = stats.distributions.poisson(mu)

    lower_cdf = 0.
    upper_cdf = dist.cdf(mx)
    nrm = upper_cdf - lower_cdf
    sample = np.random.random(size) * nrm + lower_cdf

    return dist.ppf(sample)


def rv_truncated_zip(args):
    rate, gate, high, shape = args
    g = rv_truncated_poisson(rate, high, size=shape)
    return g * (np.random.random(shape) > gate)


class TruncatedZeroInflatedPoisson(dist.Distribution):

    def __init__(self, rate, gate, high, validate_args=None):
        self.rate, self.gate, self.high = rate, gate, high
        batch_shape = jax.lax.broadcast_shapes(
            jnp.shape(rate), jnp.shape(gate), jnp.shape(high))
        super().__init__(batch_shape, validate_args=None)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape
        samples = jax.experimental.host_callback.call(
            rv_truncated_zip, (self.rate, self.gate, self.high, shape),
            result_shape=jax.ShapeDtypeStruct(shape, jnp.result_type(float)))
        return samples.astype(jnp.result_type(int))

    def log_prob(self, value):
        upper_cdf = jax.scipy.special.gammaincc(self.high + 1, self.rate)
        log_prob = dist.Poisson(self.rate).log_prob(value) - jnp.log(upper_cdf)
        log_prob = jnp.log1p(-self.gate) + log_prob
        return jnp.where(value == 0, jnp.log(self.gate + jnp.exp(log_prob)), log_prob)


# %% {"slideshow": {"slide_type": "subslide"}}
N = 1000
R = 1000

def model4(y=None):
    lbda = numpyro.sample("lbda", dist.InverseGamma(3.48681, 9.21604))
    psi = numpyro.sample("psi", dist.Beta(2.8663, 2.8663))  
    return numpyro.sample(
        "y",
        TruncatedZeroInflatedPoisson(rate=lbda, gate=1 - psi, high=14).expand([N]),
        obs=y)


# %%
trace = Predictive(model4, {}, num_samples=1000)(jax.random.PRNGKey(0))

# %% {"slideshow": {"slide_type": "fragment"}}
simu_lbdas = trace['lbda']
simu_thetas = trace['psi']
simu_ys = trace['y']

# %% {"slideshow": {"slide_type": "subslide"}}
x_max = 30
bins = np.arange(0,x_max)
bin_interp = np.linspace(0,x_max-1,num=(x_max-1)*10)
hists = np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], 1, simu_ys)

prctiles = np.percentile(hists,np.linspace(10,90,num=9),axis=0)
prctiles_interp = np.repeat(prctiles, 10,axis=1)

# %% {"slideshow": {"slide_type": "fragment"}}
for i,color in enumerate([c_light,c_light_highlight,c_mid,c_mid_highlight]):
    plt.fill_between(bin_interp,prctiles_interp[i,:],prctiles_interp[-1-i,:],alpha=1.0,color=color);


plt.plot(bin_interp,prctiles_interp[4,:],color=c_dark_highlight);
plt.axvline(x=25,ls='-',lw=2,color='k');
plt.xlabel('y');
plt.title('Prior predictive distribution');

# %% {"slideshow": {"slide_type": "subslide"}}
N = 1000
R = 1000

mcmc = MCMC(NUTS(model4), num_warmup=4 * R, num_samples=R, num_chains=4)
mcmc.run(jax.random.PRNGKey(1), y=data_ys)
trace = mcmc.get_samples(group_by_chain=True)

# %%
az.plot_posterior(trace);

# %%
ppc = Predictive(model4, mcmc.get_samples())(jax.random.PRNGKey(2))

# %% {"slideshow": {"slide_type": "subslide"}}
x_max = 30
bins = np.arange(0,x_max)
bin_interp = np.linspace(0,x_max-1,num=(x_max-1)*10)
hists = np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], 1, ppc['y'])

prctiles = np.percentile(hists,np.linspace(10,90,num=9),axis=0)
prctiles_interp = np.repeat(prctiles, 10,axis=1)

data_hist = np.histogram(data_ys,bins=bins)[0]
data_hist_interp = np.repeat(data_hist, 10)

# %% {"slideshow": {"slide_type": "fragment"}, "tags": []}
for i,color in enumerate([c_light,c_light_highlight,c_mid,c_mid_highlight]):
    plt.fill_between(bin_interp,prctiles_interp[i,:],prctiles_interp[-1-i,:],alpha=1.0,color=color);


plt.plot(bin_interp,prctiles_interp[4,:],color=c_dark_highlight);
plt.plot(bin_interp,data_hist_interp,color='black');
plt.axvline(x=25,ls='-',lw=2,color='k');
plt.xlabel('y');
plt.title('Posterior predictive distribution');
