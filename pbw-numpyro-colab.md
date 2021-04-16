---
jupyter:
  celltoolbar: Slideshow
  jupytext:
    cell_metadata_json: true
    formats: ipynb,md,py:percent
    notebook_metadata_filter: all
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.9.2
  rise:
    scroll: true
    theme: black
  toc-autonumbering: true
  toc-showcode: false
  toc-showmarkdowntxt: false
  toc-showtags: false
---

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
<center><font size="+3">Introductory review of applied probabilistic inference</font></center>
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# References
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
The following notes borrow heavily from and are *thoroughly* based on Michael Betancourt's developments that put forth a principled Bayesian workflow. The following references were integrated to produce this document:

* Betancourt, Michael (2019). Probabilistic Modeling and Statistical Inference. Retrieved from https://github.com/betanalpha/knitr_case_studies/tree/master/modeling_and_inference, commit b474ec.
* Betancourt, Michael (2020). Towards A Principled Bayesian Workflow (RStan). Retrieved from https://github.com/betanalpha/knitr_case_studies/tree/master/principled_bayesian_workflow, commit 23eb26.
* [betanalpha/knitr_case_studies](https://github.com/betanalpha/knitr_case_studies)
* [lstmemery/principled-bayesian-workflow-pymc3](https://github.com/lstmemery/principled-bayesian-workflow-pymc3)
* [bayespy documentation](https://github.com/bayespy/bayespy/blob/develop/doc/source/user_guide/quickstart.rst)
* [tikz-bayesnet](https://github.com/jluttine/tikz-bayesnet) library [technical report](https://github.com/jluttine/tikz-bayesnet/blob/master/dietz-techreport.pdf)

The implementation of the modelling and inference translated from [lstmemery's pymc3 implementation of Betancourt's principled Bayesian workflow](https://github.com/lstmemery/principled-bayesian-workflow-pymc3) to [numpyro](http://num.pyro.ai/en/stable/) is by [Du Phan](https://fehiepsi.github.io/).
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Setup
<!-- #endregion -->

## Install libraries

```python slideshow={"slide_type": "fragment"} tags=[]
# %run -i 'plotting.py'
```

```python
# !apt-get install -y fonts-lmodern
!pip install -q arviz numpyro
```

## Add latin modern fonts

```python
import matplotlib.pyplot as plt
import matplotlib.font_manager
```

```python
fonts_path = "/usr/share/texmf/fonts/opentype/public/lm/" #ubuntu
# fonts_path = "~/Library/Fonts/" # macos
# fonts_path = "/usr/share/fonts/OTF/" # arch
matplotlib.font_manager.fontManager.addfont(fonts_path + "lmsans10-regular.otf")
matplotlib.font_manager.fontManager.addfont(fonts_path + "lmroman10-regular.otf")
```

## Set matplotlib to use latin modern fonts

```python
from IPython.display import set_matplotlib_formats
#%matplotlib inline
set_matplotlib_formats('svg') # use SVG backend to maintain vectorization
plt.style.use('default') #reset default parameters
# https://stackoverflow.com/a/3900167/446907
plt.rcParams.update({'font.size': 16,
                     'font.family': ['sans-serif'],
                     'font.serif': ['Latin Modern Roman'] + plt.rcParams['font.serif'],
                     'font.sans-serif': ['Latin Modern Sans'] + plt.rcParams['font.sans-serif']})
```

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Modeling process
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "subslide"}} -->
## Observing the world through the lens of probability
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "subslide"}} -->
### Systems, environments, and observations

<div>
<center>    
<img src="https://github.com/betanalpha/knitr_case_studies/raw/master/modeling_and_inference/figures/inferential_config/observational_process/multiple_probes/multiple_probes.png" alt="Drawing" width="90%"/>
<img src="https://github.com/betanalpha/knitr_case_studies/raw/master/modeling_and_inference/figures/inferential_config/observational_process/multiple_observational_processes/multiple_observational_processes.png" alt="Drawing" width="90%"/></center>
</div>
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "subslide"}} -->
### The space of observational models and the true data generating process
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "fragment"}, "tags": []} -->
#### The observational model
* observation space: $Y$
* arbitrary points in the observation space: $y$
* explicitly realized observations from the observational process $\tilde{y}$
* data generating process: a probability distribution over the observation space $\pi \colon Y \rightarrow [0,1]$
* space of all data generating processes: $\mathcal{P}$
* observational model vs model configuration space: the subspace, $\mathcal{S} \subset \mathcal{P}$, of data generating processes considered in any particular application
* parametrization: a map from a model configuration space $\mathcal{S}$ to a parameter space $\mathcal{\Theta}$ assigning to each model configuration $s \in \mathcal{S}$ a parameter $\theta \in \mathcal{\Theta}$
* probability density for an observational model: $\pi_{\mathcal{S}}(y; s)$ in general using the parametrization to assign $\pi_{\mathcal{S}}(y; \theta)$
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "subslide"}} -->
<div>
<center>    
<img src="https://github.com/betanalpha/knitr_case_studies/raw/master/modeling_and_inference/figures/small_world/small_world/small_world.png" alt="Drawing" width="45%"/>
<img src="https://github.com/betanalpha/knitr_case_studies/raw/master/modeling_and_inference/figures/small_world/small_world_one/small_world_one.png" alt="Drawing" width="45%"/>
</center>
</div>
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "subslide"}} -->
#### The true data generating process
* true data generating process: $\pi^{\dagger}$ is the probability distribution that exactly captures the observational process in a given application
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
## The practical reality of model construction
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
<div>
<center>    
<img src="https://github.com/betanalpha/knitr_case_studies/raw/master/modeling_and_inference/figures/small_world/small_world_two/small_world_two.png" alt="Drawing" width="75%"/></center>
</div>
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
## The process of inference
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "subslide"}} -->
<div>
<center>    
<img src="https://github.com/betanalpha/knitr_case_studies/raw/master/modeling_and_inference/figures/inferential_config/model_config/model_config5/model_config5.png" alt="Drawing" width="90%"/></center>
</div>
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
How can we do our best to validate this process works as close as possible to providing a high quality mirror for natural systems?
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
## Workflow overview
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "subslide"}} -->
<div>
<center>    
<img src="https://github.com/betanalpha/knitr_case_studies/raw/master/principled_bayesian_workflow/figures/workflow/all/all.png" alt="Drawing" width="70%"/></center>
</div>
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "subslide"}} -->
## Example generative models
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "subslide"}} -->
### Univariate normal model
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
From a very simple perspective, generative modeling refers to the situation in which we develop a candidate probabilistic specification of the process from which our data are generated. Usually this will include the specification of prior distributions over all first-order parameters.
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
<div>
<center>    
<img src="https://www.bayespy.org/_images/tikz-57bc0c88a2974f4c1e2335fe9edb88ff2efdf970.png" style="background-color:white;" alt="Drawing" width="10%"/></center>
</div>
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
\begin{split}
p(\mathbf{y}|\mu,\tau) &= \prod^{9}_{n=0} \mathcal{N}(y_n|\mu,\tau) \\
p(\mu) &= \mathcal{N}(\mu|0,10^{-6}) \\
p(\tau) &= \mathcal{G}(\tau|10^{-6},10^{-6})
\end{split}
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
This comes from the library [bayespy](https://github.com/bayespy/bayespy/blob/develop/doc/source/user_guide/quickstart.rst). The best description we are aware of regarding the syntax and semantics of graphical models via factor graph notation is in the [tikz-bayesnet](https://github.com/jluttine/tikz-bayesnet) library [technical report](https://github.com/jluttine/tikz-bayesnet/blob/master/dietz-techreport.pdf).
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "subslide"}} -->
### Multivariate normal models
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
<div>
<center>    
<img src="https://www.bayespy.org/_images/tikz-80a1db369be1f25b61ceacfff551dae2bdd331c3.png" style="background-color:white;" alt="Drawing" width="10%"/></center>
</div>
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
$$\mathbf{y}_{mn} \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{\Lambda}),\qquad m=0,\ldots,9, \quad n=0,\ldots,29.$$
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "subslide"}} -->
<div>
<center>    
<img src="https://www.bayespy.org/_images/tikz-97236981a2be663d10ade1ad85caa727621615db.png" style="background-color:white;" alt="Drawing" width="20%"/></center>
</div>
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
$$\mathbf{y}_{mn} \sim \mathcal{N}(\boldsymbol{\mu}_m,
\mathbf{\Lambda}_n),\qquad m=0,\ldots,9, \quad n=0,\ldots,29.$$

Note that these are for illustrative purposes of the manner in which our data can share parameters and we have not yet defined priors over our parameters.
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# "Build, compute, critique, repeat": Box's loop in iteration through Betancourt's principled Bayesian workflow
<!-- #endregion -->

## Setup


### Load libraries

```python slideshow={"slide_type": "fragment"}
# %pylab inline
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
```

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
### define colors
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
c_light ="#DCBCBC"
c_light_highlight ="#C79999"
c_mid ="#B97C7C"
c_mid_highlight ="#A25050"
c_dark ="#8F2727"
c_dark_highlight ="#7C0000"
```

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
## Poisson process for arbitrary detector count data
<!-- 4.1 -->
<!-- #endregion -->

Here we build a candidate model that generates (Poisson) counts that may explain what we observe in our sample data.

<!-- #region {"slideshow": {"slide_type": "subslide"}} -->
### Sample data
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"} tags=[]
df = pd.read_csv('https://raw.githubusercontent.com/cameronraysmith/applied-probabilistic-inference/master/data.csv')
print(df.head(8))
df.shape
```

<!-- #region {"slideshow": {"slide_type": "subslide"}} -->
### Generative model specification
<!-- #endregion -->

#### Prior

```python slideshow={"slide_type": "fragment"} tags=[]
lbda  = np.linspace(0, 20, num=int(20/0.001))

plt.plot(lbda, stats.norm(loc=0,scale=6.44787).pdf(lbda), c=c_dark_highlight, lw=2)
plt.xlabel("lambda"); plt.ylabel("Prior Density"); plt.yticks([]);

lbda99 = np.linspace(0, 15, num=int(15/0.001))
plt.fill_between(lbda99,0.,y2=stats.norm(loc=0,scale=6.44787).pdf(lbda99),color=c_dark);

!mkdir -p ./fig/
plt.savefig("fig/prior-density-lambda.svg", bbox_inches="tight");
```

```python
!inkscape fig/prior-density-lambda.svg --export-filename=fig/prior-density-lambda.pdf 2>/dev/null
```

#### Model

<!-- #region {"slideshow": {"slide_type": "subslide"}} -->
In this case, the candidate _complete Bayesian model_ under consideration is given by

$$
\pi( y_{1}, \ldots, y_{N}, \lambda )
=
\left[ \prod_{n = 1}^{N} \text{Poisson} (y_{n} \mid \lambda) \right]
\cdot \text{HalfNormal} (\lambda \mid 6).
$$
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
<div>
<center>    
<img src="https://github.com/betanalpha/knitr_case_studies/raw/master/principled_bayesian_workflow/figures/iter1/dgm/dgm.png" alt="Drawing" width="40%"/></center>
</div>
<!-- #endregion -->

```python
N = 1000
R = 500

def model(y=None):
    lbda = numpyro.sample("lbda", dist.HalfNormal(6.44787))
    return numpyro.sample("y", dist.Poisson(lbda).expand([N]), obs=y)
```

#### Simulation

```python
trace = Predictive(model, {}, num_samples=R)(jax.random.PRNGKey(0))
```

```python slideshow={"slide_type": "fragment"}
simu_lbdas = trace['lbda']
simu_ys = trace['y']
```

```python
print(simu_lbdas[0:9])
print(simu_lbdas.shape)
```

```python
print(simu_ys[0:9])
print(simu_ys.shape)
```

<!-- #region {"slideshow": {"slide_type": "subslide"}} -->
### Plot prior predictive distribution
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
x_max = 30
bins = np.arange(0,x_max)
bin_interp = np.linspace(0,x_max-1,num=(x_max-1)*10)
hists = np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], 1, simu_ys)

prctiles = np.percentile(hists,np.linspace(10,90,num=9),axis=0)
prctiles_interp = np.repeat(prctiles, 10,axis=1)
```

```python slideshow={"slide_type": "fragment"} tags=[]
for i,color in enumerate([c_light,c_light_highlight,c_mid,c_mid_highlight]):
    plt.fill_between(bin_interp,prctiles_interp[i,:],prctiles_interp[-1-i,:],alpha=1.0,color=color);


plt.plot(bin_interp,prctiles_interp[4,:],color=c_dark_highlight);
plt.axvline(x=25,ls='-',lw=2,color='k');
plt.xlabel('y');
plt.title('Prior predictive distribution');
```

<!-- #region {"slideshow": {"slide_type": "subslide"}} -->
### Fit to simulated data
<!-- #endregion -->

[Betancourt, 2020](https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html#Step_Nine:_Fit_Simulated_Ensemble60) performs this for each `y` in trace.

```python
mcmc = MCMC(NUTS(model), num_warmup=4 * R, num_samples=R, num_chains=2)
mcmc.run(jax.random.PRNGKey(1), y=simu_ys[-1, :])
trace = mcmc.get_samples(group_by_chain=True)
```

```python
az.plot_trace(trace);
```

<!-- #region {"slideshow": {"slide_type": "subslide"}} -->
### Fit observations and evaluate
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
# df = pd.read_csv('data.csv')
data_ys = df[df['data']=='y']['value'].values
```

```python
mcmc = MCMC(NUTS(model), num_warmup=4 * R, num_samples=R, num_chains=4)
mcmc.run(jax.random.PRNGKey(2), y=data_ys)
trace = mcmc.get_samples(group_by_chain=True)
```

```python
az.plot_posterior(trace, kind="hist");
```

```python
ppc = Predictive(model, mcmc.get_samples())(jax.random.PRNGKey(3))
```

```python slideshow={"slide_type": "fragment"}
x_max = 30
bins = np.arange(0,x_max)
bin_interp = np.linspace(0,x_max-1,num=(x_max-1)*10)
hists = np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], 1, ppc['y'])

prctiles = np.percentile(hists,np.linspace(10,90,num=9),axis=0)
prctiles_interp = np.repeat(prctiles, 10,axis=1)

data_hist = np.histogram(data_ys,bins=bins)[0]
data_hist_interp = np.repeat(data_hist, 10)
```

```python slideshow={"slide_type": "subslide"} tags=[]
for i,color in enumerate([c_light,c_light_highlight,c_mid,c_mid_highlight]):
    plt.fill_between(bin_interp,prctiles_interp[i,:],prctiles_interp[-1-i,:],alpha=1.0,color=color);


plt.plot(bin_interp,prctiles_interp[4,:],color=c_dark_highlight);
plt.plot(bin_interp,data_hist_interp,color='black');
plt.axvline(x=25,ls='-',lw=2,color='k');
plt.xlabel('y');
plt.title('Posterior predictive distribution');
```

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
## Account for measurement device failure

<!-- 4.2 -->
<!-- #endregion -->

### Update the generative model specification

<!-- #region -->
Recall the specification of our first attempt to model the detector count data with a [simple Poisson process model](poisson-process-for-arbitrary-detector-count-data)
```python
N = 1000
R = 500

def model(y=None):
    lbda = numpyro.sample("lbda", dist.HalfNormal(6.44787))
    return numpyro.sample("y", dist.Poisson(lbda).expand([N]), obs=y)
```
Here we adapt our likelihood to include a so-called "zero-inflated Poisson distribution" to account for the presence of many $0$ counts that may derive from malfunctioning detector devices.
<!-- #endregion -->

```python slideshow={"slide_type": "subslide"} tags=[]
N = 1000
R = 1000

def model2(y=None):
    theta = numpyro.sample("theta", dist.Beta(1, 1))
    lambda_ = numpyro.sample("lambda", dist.HalfNormal(6.44787))
    return numpyro.sample(
        "y", dist.ZeroInflatedPoisson(rate=lambda_, gate=1 - theta).expand([N]), obs=y)
```

### Simulate the updated model

```python tags=[]
trace = Predictive(model2, {}, num_samples=R)(jax.random.PRNGKey(0))
```

```python slideshow={"slide_type": "subslide"}
trace["theta"][:10]
```

```python slideshow={"slide_type": "fragment"}
trace["lambda"][:10]
```

```python slideshow={"slide_type": "fragment"} tags=[]
simu_ys = trace["y"]
simu_ys
```

What is the fraction of zero values in this simulated data?

```python
simu_ys[simu_ys < 0.001].size / simu_ys.size
```

```python slideshow={"slide_type": "fragment"} tags=[]
print(simu_ys.shape)
np.count_nonzero(simu_ys, axis=1).mean()
```

```python slideshow={"slide_type": "subslide"}
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
```

```python slideshow={"slide_type": "fragment"}
simu_ys[simu_ys > 25].size / simu_ys.size
```

<!-- #region {"slideshow": {"slide_type": "subslide"}} -->
### Fit Simulated Observations and Evaluate 
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"} tags=[]
N = 1000
R = 1000

mcmc = MCMC(NUTS(model2), num_warmup=R, num_samples=R, num_chains=2)
mcmc.run(jax.random.PRNGKey(1), y=simu_ys[-1, :])
trace_fit = mcmc.get_samples(group_by_chain=True)
```

```python tags=[]
az.plot_trace(trace_fit);
```

```python tags=[]
numpyro.diagnostics.print_summary(trace_fit)
```

```python slideshow={"slide_type": "subslide"} tags=[]
import pickle
with open("fit_data2.pkl", "wb+") as buffer:
    pickle.dump({"model": model2, "trace": trace_fit}, buffer)
```

<!-- #region {"tags": []} -->
### Fit observations and evaluate
<!-- #endregion -->

```python tags=[]
mcmc = MCMC(NUTS(model2), num_warmup=4 * R, num_samples=R, num_chains=4)
mcmc.run(jax.random.PRNGKey(2), y=data_ys)
trace = mcmc.get_samples(group_by_chain=True)
```

```python
az.plot_posterior(trace, kind="hist");
```

```python tags=[]
ppc = Predictive(model2, mcmc.get_samples())(jax.random.PRNGKey(3))
```

```python slideshow={"slide_type": "fragment"} tags=[]
x_max = 30
bins = np.arange(0,x_max)
bin_interp = np.linspace(0,x_max-1,num=(x_max-1)*10)
hists = np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], 1, ppc['y'])

prctiles = np.percentile(hists,np.linspace(10,90,num=9),axis=0)
prctiles_interp = np.repeat(prctiles, 10,axis=1)

data_hist = np.histogram(data_ys,bins=bins)[0]
data_hist_interp = np.repeat(data_hist, 10)
```

```python slideshow={"slide_type": "subslide"} tags=[]
for i,color in enumerate([c_light,c_light_highlight,c_mid,c_mid_highlight]):
    plt.fill_between(bin_interp,prctiles_interp[i,:],prctiles_interp[-1-i,:],alpha=1.0,color=color);


plt.plot(bin_interp,prctiles_interp[4,:],color=c_dark_highlight);
plt.plot(bin_interp,data_hist_interp,color='black');
plt.axvline(x=25,ls='-',lw=2,color='k');
plt.xlabel('y');
plt.title('Posterior predictive distribution');
```

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
## Account for the distinction between functioning and malfunctioning measurement devices
<!-- 4.3 -->
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "subslide"}} -->
### Update the generative model
<!-- #endregion -->

<!-- #region -->
Build a model that generates zero-inflated Poisson counts. For reference, the model from the second attempt is
```python
N = 1000
R = 1000

def model2(y=None):
    lambda_ = numpyro.sample("lambda", dist.HalfNormal(6.44787))
    theta = numpyro.sample("theta", dist.Beta(1, 1))
    return numpyro.sample(
        "y", dist.ZeroInflatedPoisson(rate=lambda_, gate=1 - theta).expand([N]), obs=y)
```
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
lbda  = np.linspace(0, 20, num=int(20/0.001))
pdf = stats.invgamma(3.48681,scale=9.21604)
plt.plot(lbda, pdf.pdf(lbda), c=c_dark_highlight, lw=2)
plt.xlabel("lambda"); plt.ylabel("Prior Density"); plt.yticks([]);

lbda99 = np.linspace(1, 15, num=int(15/0.001))

plt.fill_between(lbda99,0.,y2=pdf.pdf(lbda99),color=c_dark);
```

```python slideshow={"slide_type": "subslide"}
theta  = np.linspace(0, 1, num=int(1/0.001))
pdf = stats.beta(2.8663,2.8663)
plt.plot(theta, pdf.pdf(theta), c=c_dark_highlight, lw=2)
plt.xlabel("theta"); plt.ylabel("Prior Density"); plt.yticks([]);

theta99 = np.linspace(0.1, 0.9, num=int(0.8/0.001))

plt.fill_between(theta99,0.,y2=pdf.pdf(theta99),color=c_dark);
```

```python slideshow={"slide_type": "subslide"}
#WORKING

N = 1000
R = 1000

def model3(y=None):
    lbda = numpyro.sample("lbda", dist.InverseGamma(3.48681, 9.21604))
    theta = numpyro.sample("theta", dist.Beta(2.8663, 2.8663))  
    return numpyro.sample(
        "y", dist.ZeroInflatedPoisson(rate=lbda, gate=1 - theta).expand([N]), obs=y)
```

```python slideshow={"slide_type": "fragment"}
trace = Predictive(model3, {}, num_samples=R)(jax.random.PRNGKey(0))
```

```python slideshow={"slide_type": "fragment"}
simu_lbdas = trace['lbda']
simu_thetas = trace['theta']
simu_ys = trace['y']
```

<!-- #region {"slideshow": {"slide_type": "subslide"}} -->
### Plot prior predictive distribution
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
x_max = 30
bins = np.arange(0,x_max)
bin_interp = np.linspace(0,x_max-1,num=(x_max-1)*10)
hists = np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], 1, simu_ys)

prctiles = np.percentile(hists,np.linspace(10,90,num=9),axis=0)
prctiles_interp = np.repeat(prctiles, 10,axis=1)
```

```python slideshow={"slide_type": "fragment"}
for i,color in enumerate([c_light,c_light_highlight,c_mid,c_mid_highlight]):
    plt.fill_between(bin_interp,prctiles_interp[i,:],prctiles_interp[-1-i,:],alpha=1.0,color=color);


plt.plot(bin_interp,prctiles_interp[4,:],color=c_dark_highlight);
plt.axvline(x=25,ls='-',lw=2,color='k');
plt.xlabel('y');
plt.title('Prior predictive distribution');
```

<!-- #region {"slideshow": {"slide_type": "subslide"}} -->
### Fit to simulated data
<!-- #endregion -->

In the example, Betancourt performs this for each `y` in trace. Here we only compute this for one element of the trace.

```python
N = 1000
R = 1000

mcmc = MCMC(NUTS(model3), num_warmup=4 * R, num_samples=R, num_chains=2)
mcmc.run(jax.random.PRNGKey(1), y=simu_ys[:, -1])
trace = mcmc.get_samples(group_by_chain=True)
```

```python
az.plot_trace(trace);
```

<!-- #region {"slideshow": {"slide_type": "subslide"}} -->
### Fit observations and evaluate
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
# df = pd.read_csv('data.csv')
data_ys = df[df['data']=='y']['value'].values
```

```python slideshow={"slide_type": "fragment"}
mcmc = MCMC(NUTS(model3), num_warmup=4 * R, num_samples=R, num_chains=4)
mcmc.run(jax.random.PRNGKey(2), y=data_ys)
trace = mcmc.get_samples(group_by_chain=True)
```

```python
az.plot_posterior(trace, var_names="lbda");
```

```python
ppc = Predictive(model3, mcmc.get_samples())(jax.random.PRNGKey(3))
```

```python slideshow={"slide_type": "fragment"}
x_max = 30
bins = np.arange(0,x_max)
bin_interp = np.linspace(0,x_max-1,num=(x_max-1)*10)
hists = np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], 1, ppc['y'])

prctiles = np.percentile(hists,np.linspace(10,90,num=9),axis=0)
prctiles_interp = np.repeat(prctiles, 10,axis=1)

data_hist = np.histogram(data_ys,bins=bins)[0]
data_hist_interp = np.repeat(data_hist, 10)
```

```python slideshow={"slide_type": "subslide"}
for i,color in enumerate([c_light,c_light_highlight,c_mid,c_mid_highlight]):
    plt.fill_between(bin_interp,prctiles_interp[i,:],prctiles_interp[-1-i,:],alpha=1.0,color=color);


plt.plot(bin_interp,prctiles_interp[4,:],color=c_dark_highlight);
plt.plot(bin_interp,data_hist_interp,color='black');
plt.axvline(x=25,ls='-',lw=2,color='k');
plt.xlabel('y');
plt.title('Posterior predictive distribution');
```

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
## Account for upper limit of detection
<!-- #endregion -->

<!-- #region -->
The results of our third attempt identified the missing component of our prior was an upper threshold beyond which detectors were unable to register counts. Our model was
```python
#WORKING

N = 1000
R = 1000

def model3(y=None):
    lbda = numpyro.sample("lbda", dist.InverseGamma(3.48681, 9.21604))
    theta = numpyro.sample("theta", dist.Beta(2.8663, 2.8663))  
    return numpyro.sample(
        "y", dist.ZeroInflatedPoisson(rate=lbda, gate=1 - theta).expand([N]), obs=y)
```
Now, we implement a means of truncating the zero-inflated Poisson distribution to reflect this newly identified information.
<!-- #endregion -->

```python slideshow={"slide_type": "subslide"}
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
```

```python slideshow={"slide_type": "subslide"}
N = 1000
R = 1000

def model4(y=None):
    lbda = numpyro.sample("lbda", dist.InverseGamma(3.48681, 9.21604))
    psi = numpyro.sample("psi", dist.Beta(2.8663, 2.8663))  
    return numpyro.sample(
        "y",
        TruncatedZeroInflatedPoisson(rate=lbda, gate=1 - psi, high=14).expand([N]),
        obs=y)
```

```python
trace = Predictive(model4, {}, num_samples=1000)(jax.random.PRNGKey(0))
```

```python slideshow={"slide_type": "fragment"}
simu_lbdas = trace['lbda']
simu_thetas = trace['psi']
simu_ys = trace['y']
```

```python slideshow={"slide_type": "subslide"}
x_max = 30
bins = np.arange(0,x_max)
bin_interp = np.linspace(0,x_max-1,num=(x_max-1)*10)
hists = np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], 1, simu_ys)

prctiles = np.percentile(hists,np.linspace(10,90,num=9),axis=0)
prctiles_interp = np.repeat(prctiles, 10,axis=1)
```

```python slideshow={"slide_type": "fragment"}
for i,color in enumerate([c_light,c_light_highlight,c_mid,c_mid_highlight]):
    plt.fill_between(bin_interp,prctiles_interp[i,:],prctiles_interp[-1-i,:],alpha=1.0,color=color);


plt.plot(bin_interp,prctiles_interp[4,:],color=c_dark_highlight);
plt.axvline(x=25,ls='-',lw=2,color='k');
plt.xlabel('y');
plt.title('Prior predictive distribution');
```

```python slideshow={"slide_type": "subslide"}
N = 1000
R = 1000

mcmc = MCMC(NUTS(model4), num_warmup=4 * R, num_samples=R, num_chains=4)
mcmc.run(jax.random.PRNGKey(1), y=data_ys)
trace = mcmc.get_samples(group_by_chain=True)
```

```python
az.plot_posterior(trace);
```

```python
ppc = Predictive(model4, mcmc.get_samples())(jax.random.PRNGKey(2))
```

```python slideshow={"slide_type": "subslide"}
x_max = 30
bins = np.arange(0,x_max)
bin_interp = np.linspace(0,x_max-1,num=(x_max-1)*10)
hists = np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], 1, ppc['y'])

prctiles = np.percentile(hists,np.linspace(10,90,num=9),axis=0)
prctiles_interp = np.repeat(prctiles, 10,axis=1)

data_hist = np.histogram(data_ys,bins=bins)[0]
data_hist_interp = np.repeat(data_hist, 10)
```

```python slideshow={"slide_type": "fragment"} tags=[]
for i,color in enumerate([c_light,c_light_highlight,c_mid,c_mid_highlight]):
    plt.fill_between(bin_interp,prctiles_interp[i,:],prctiles_interp[-1-i,:],alpha=1.0,color=color);


plt.plot(bin_interp,prctiles_interp[4,:],color=c_dark_highlight);
plt.plot(bin_interp,data_hist_interp,color='black');
plt.axvline(x=25,ls='-',lw=2,color='k');
plt.xlabel('y');
plt.title('Posterior predictive distribution');
```
