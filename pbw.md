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
      format_version: '1.2'
      jupytext_version: 1.6.0
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
    version: 3.8.5
  rise:
    scroll: true
    theme: black
  toc-autonumbering: true
---

# Workflow overview


<div>
<center>    
<img src="https://github.com/betanalpha/knitr_case_studies/raw/master/principled_bayesian_workflow/figures/workflow/all/all.png" alt="Drawing" width="90%"/></center>
</div>

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Load libraries
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
# %pylab inline
import matplotlib.pyplot as plt
import pymc3 as pm
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import theano.tensor as T
import theano
plt.style.use(['seaborn-talk'])
plt.rcParams["figure.figsize"] = (10,8)
print(pm.__version__)
print(theano.__version__)
```

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
## define colors
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
# Section 3.1

Build a model that generates Poisson counts
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "subslide"}} -->
## Build a generative model
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
lbda  = np.linspace(0, 20, num=int(20/0.001))

plt.plot(lbda, stats.norm(loc=0,scale=6.44787).pdf(lbda), c=c_dark_highlight, lw=2)
plt.xlabel("lambda"); plt.ylabel("Prior Density"); plt.yticks([]);


lbda99 = np.linspace(0, 15, num=int(15/0.001))



plt.fill_between(lbda99,0.,y2=stats.norm(loc=0,scale=6.44787).pdf(lbda99),color=c_dark)
```

```python slideshow={"slide_type": "subslide"}
#WORKING

model = pm.Model()
N = 1000
R = 500
with model:
    lbda = pm.HalfNormal("lbda",sd=6.44787)
    
    y = pm.Poisson("y",mu=lbda,shape=(N,),observed=None)
    
```

```python slideshow={"slide_type": "fragment"}
with model:
    trace = pm.sample_prior_predictive(samples=R)
```

```python slideshow={"slide_type": "fragment"}
simu_lbdas = trace['lbda']
simu_ys = trace['y']
```

<!-- #region {"slideshow": {"slide_type": "subslide"}} -->
## Plot prior predictive distribution
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
## Fit to simulated data

In example Betancourt performs this for each `y` in trace. Here we just do it for one.
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
model = pm.Model()
with model:
    lbda = pm.HalfNormal("lbda",sd=6.44787)
    
    y = pm.Poisson("y",mu=lbda,shape=(N,),observed=simu_ys[-1,:])
    
    trace = pm.sample(draws=R,tune=4*R)
   
```

```python slideshow={"slide_type": "fragment"}
pm.plots.traceplot(trace);
```

```python slideshow={"slide_type": "fragment"}
# Compute rank of prior draw with respect to thinned posterior draws
sbc_rank = np.sum(simu_lbdas < trace['lbda'][::2])


```

```python slideshow={"slide_type": "subslide"}
# posterior sensitivities analysis
s = pm.stats.summary(trace,varnames=['lbda'])
post_mean_lbda = s['mean'].values
post_sd_lbda = s['sd'].values
prior_sd_lbda = 6.44787
z_score = np.abs((post_mean_lbda - simu_lbdas) / post_sd_lbda)
shrinkage = 1 - (post_sd_lbda / prior_sd_lbda ) ** 2
```

```python slideshow={"slide_type": "fragment"}
plt.plot(shrinkage[0]*np.ones(len(z_score)),z_score,'o',c="#8F272720");
plt.xlim(0,1.01); plt.xlabel('Posterior shrinkage'); plt.ylabel('Posterior z-score');
```

<!-- #region {"slideshow": {"slide_type": "subslide"}} -->
## Fit observations and evaluate
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
df = pd.read_csv('data.csv')
data_ys = df[df['data']=='y']['value'].values
```

```python slideshow={"slide_type": "fragment"}
model = pm.Model()
with model:
    lbda = pm.HalfNormal("lbda",sd=6.44787)
    
    y = pm.Poisson("y",mu=lbda,shape=(N,),observed=data_ys)
    
    trace = pm.sample(draws=R,tune=4*R,chains=4)
```

```python slideshow={"slide_type": "subslide"}
pm.plots.plot_posterior(trace,varnames=['lbda']);
```

```python slideshow={"slide_type": "subslide"}
with model:
     ppc = pm.sample_ppc(trace)
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
# Section 3.2
<!-- #endregion -->

```python slideshow={"slide_type": "subslide"}
generative_ensemble2 = pm.Model()

N = 1000
R = 1000

with generative_ensemble2:
    theta = pm.Beta(name="theta", alpha = 1, beta = 1)
    lambda_ = pm.HalfNormal(name="lambda", sd = 6.44787)
    y = pm.ZeroInflatedPoisson(name = "y", psi = theta, theta = lambda_, shape = (N,))
```

```python slideshow={"slide_type": "fragment"}
with generative_ensemble2:
    trace = pm.sample_prior_predictive(samples=R)
```

```python slideshow={"slide_type": "subslide"}
trace["theta"][:10]
```

```python slideshow={"slide_type": "fragment"}
trace["lambda"][:10]
```

```python slideshow={"slide_type": "fragment"}
simu_ys = trace["y"]
simu_ys
```

```python slideshow={"slide_type": "fragment"}
np.count_nonzero(simu_ys, axis=0).std()
```

```python slideshow={"slide_type": "subslide"}
x_max = 30
bins = np.arange(0 ,x_max)
bin_interp = np.linspace(0,x_max-1,num=(x_max-1)*10)

hists = np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], 1, simu_ys.T)

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
## Fit Simulated Observations and Evaluate 
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
fit_data2 = pm.Model()

N = 1000
R = 1000

with fit_data2:
    theta = pm.Beta(name="theta", alpha = 1, beta = 1)
    lambda_ = pm.HalfNormal(name="lambda", sd = 6.44787)
    y = pm.ZeroInflatedPoisson(name = "y", 
                               psi = theta, 
                               theta = lambda_, 
                               shape = (N,),
                               observed=simu_ys[-1,:])
```

```python slideshow={"slide_type": "fragment"}
with fit_data2:
    trace_fit = pm.sample(R)
```

```python slideshow={"slide_type": "subslide"}
pm.plots.traceplot(trace_fit)
```

```python slideshow={"slide_type": "fragment"}
pm.summary(trace_fit, varnames=["theta", "lambda"]).round(2)
```

```python slideshow={"slide_type": "skip"}

```

```python slideshow={"slide_type": "subslide"}
import pickle
with open("fit_data2.pkl", "wb+") as buffer:
    pickle.dump({"model": fit_data2, "trace": trace_fit}, buffer)
```

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Section 3.3

Build a model that generates zero-inflated Poisson counts
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "subslide"}} -->
## Build a generative model
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
lbda  = np.linspace(0, 20, num=int(20/0.001))
pdf = stats.invgamma(3.48681,scale=9.21604)
plt.plot(lbda, pdf.pdf(lbda), c=c_dark_highlight, lw=2)
plt.xlabel("lambda"); plt.ylabel("Prior Density"); plt.yticks([]);


lbda99 = np.linspace(1, 15, num=int(15/0.001))



plt.fill_between(lbda99,0.,y2=pdf.pdf(lbda99),color=c_dark)
```

```python slideshow={"slide_type": "subslide"}
theta  = np.linspace(0, 1, num=int(1/0.001))
pdf = stats.beta(2.8663,2.8663)
plt.plot(theta, pdf.pdf(theta), c=c_dark_highlight, lw=2)
plt.xlabel("theta"); plt.ylabel("Prior Density"); plt.yticks([]);


theta99 = np.linspace(0.1, 0.9, num=int(0.8/0.001))



plt.fill_between(theta99,0.,y2=pdf.pdf(theta99),color=c_dark)
```

```python slideshow={"slide_type": "subslide"}
#WORKING

model = pm.Model()
N = 1000
R = 1000
with model:
    lbda = pm.InverseGamma("lbda",alpha=3.48681,beta=9.21604)
    theta = pm.Beta("theta",alpha=2.8663,beta=2.8663)
    
    y = pm.ZeroInflatedPoisson("y",psi=theta,theta=lbda,shape=N)
    
```

```python slideshow={"slide_type": "fragment"}
# Note this breaks when N != R
with model:
    trace = pm.sample_prior_predictive(samples=R)
```

```python slideshow={"slide_type": "fragment"}
simu_lbdas = trace['lbda']
simu_thetas = trace['theta']
simu_ys = trace['y']
```

<!-- #region {"slideshow": {"slide_type": "subslide"}} -->
## Plot prior predictive distribution
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
x_max = 30
bins = np.arange(0,x_max)
bin_interp = np.linspace(0,x_max-1,num=(x_max-1)*10)
hists = np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], 0, simu_ys)

prctiles = np.percentile(hists,np.linspace(10,90,num=9),axis=1)
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
## Fit to simulated data

In example Betancourt performs this for each `y` in trace. Here we just do it for one.
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
model = pm.Model()
with model:
    lbda = pm.InverseGamma("lbda",alpha=3.48681,beta=9.21604)
    theta = pm.Beta("theta",alpha=2.8663,beta=2.8663)
    
    y = pm.ZeroInflatedPoisson("y",psi=theta,theta=lbda,shape=N,observed=simu_ys[:,-1])
    
    trace = pm.sample(draws=R,tune=4*R)
   
```

```python slideshow={"slide_type": "fragment"}
pm.plots.traceplot(trace);
```

```python slideshow={"slide_type": "fragment"}
# Compute rank of prior draw with respect to thinned posterior draws
sbc_rank = np.sum(simu_lbdas < trace['lbda'][::2])


```

```python slideshow={"slide_type": "subslide"}
# posterior sensitivities analysis
s = pm.stats.summary(trace,varnames=['lbda'])
post_mean_lbda = s['mean'].values
post_sd_lbda = s['sd'].values
prior_sd_lbda = 6.44787
z_score = np.abs((post_mean_lbda - simu_lbdas) / post_sd_lbda)
shrinkage = 1 - (post_sd_lbda / prior_sd_lbda ) ** 2
```

```python slideshow={"slide_type": "fragment"}
plt.plot(shrinkage[0]*np.ones(len(z_score)),z_score,'o',c="#8F272720");
plt.xlim(0,1.01); plt.xlabel('Posterior shrinkage'); plt.ylabel('Posterior z-score');
```

<!-- #region {"slideshow": {"slide_type": "subslide"}} -->
## Fit observations and evaluate
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
df = pd.read_csv('data.csv')
data_ys = df[df['data']=='y']['value'].values
```

```python slideshow={"slide_type": "fragment"}
model = pm.Model()
with model:
    lbda = pm.InverseGamma("lbda",alpha=3.48681,beta=9.21604)
    theta = pm.Beta("theta",alpha=2.8663,beta=2.8663)
    
    y = pm.ZeroInflatedPoisson("y",psi=theta,theta=lbda,shape=N,observed=data_ys)
    
    trace = pm.sample(draws=R,tune=4*R,chains=4)
```

```python slideshow={"slide_type": "subslide"}
pm.plots.plot_posterior(trace,varnames=['lbda']);
```

```python slideshow={"slide_type": "subslide"}
with model:
     ppc = pm.sample_ppc(trace)
```

```python slideshow={"slide_type": "fragment"}
x_max = 30
bins = np.arange(0,x_max)
bin_interp = np.linspace(0,x_max-1,num=(x_max-1)*10)
hists = np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], 0, ppc['y'])

prctiles = np.percentile(hists,np.linspace(10,90,num=9),axis=1)
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
# Section 3.4
<!-- #endregion -->

```python slideshow={"slide_type": "subslide"}
from pymc3.distributions.distribution import generate_samples,draw_values,Discrete
from pymc3.distributions.discrete import Poisson

def rv_truncated_poisson(mu,mx, size=None):
    mu = np.asarray(mu)
    mx = np.asarray(mx)
    dist = stats.distributions.poisson(mu)

    lower_cdf = 0.
    upper_cdf = dist.cdf(mx)
    nrm = upper_cdf - lower_cdf
    sample = np.random.random_sample(size) * nrm + lower_cdf

    return dist.ppf(sample)

class TruncatedZeroInflatedPoisson(Discrete):

    def __init__(self, mu, mx, psi, *args, **kwargs):
        super(TruncatedZeroInflatedPoisson, self).__init__(*args, **kwargs)
        self.mu  = tt.as_tensor_variable(mu)
        self.mx = tt.as_tensor_variable(mx)
        self.psi = tt.as_tensor_variable(psi)
        self.mode = tt.floor(mu).astype('int32')


    def random(self, point=None, size=None):
        mu, psi, mx = draw_values([self.mu, self.psi, self.mx], point=point, size=size)
        g = generate_samples(rv_truncated_poisson, mu,mx,
                             dist_shape=self.shape,
                             size=size)
        return g * (np.random.random(np.squeeze(g.shape)) < psi)

    def logp(self, value):
        psi = self.psi
        mu = self.mu
        mx = self.mx
        poisson = pm.Poisson.dist(mu)
        logp_val = tt.switch(
            tt.gt(value, 0),
            tt.log(psi) + poisson.logp(value),
            pm.math.logaddexp(tt.log1p(-psi), tt.log(psi) - mu))

        return pm.distributions.dist_math.bound(
            logp_val,
            0 <= value,
            value <= mx,
            0 <= psi, psi <= 1,
            0 <= mu)
```

```python slideshow={"slide_type": "subslide"}
model = pm.Model()
N = 1000
R = 1000
with model:
    lbda = pm.InverseGamma("lbda",alpha=3.48681,beta=9.21604)
    psi = pm.Beta("psi",alpha=2.8663,beta=2.8663)
    
    y = TruncatedZeroInflatedPoisson("y",psi=psi,mu=lbda,mx=15.,shape=N)
```

```python slideshow={"slide_type": "fragment"}
with model:
    trace = pm.sample_prior_predictive(samples=1000)
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
hists = np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], 0, simu_ys)

prctiles = np.percentile(hists,np.linspace(10,90,num=9),axis=1)
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
model = pm.Model()
N = 1000
R = 1000
with model:
    lbda = pm.InverseGamma("lbda",alpha=3.48681,beta=9.21604)
    psi = pm.Beta("psi",alpha=2.8663,beta=2.8663)
    
    y = TruncatedZeroInflatedPoisson("y",psi=psi,mu=lbda,mx=14.,shape=N,observed=data_ys)
    trace = pm.sample(draws=R,tune=4*R,chains=4)    
```

```python slideshow={"slide_type": "fragment"}
pm.plots.plot_posterior(trace);
```

```python slideshow={"slide_type": "fragment"}
with model:
     ppc = pm.sample_ppc(trace)
```

```python slideshow={"slide_type": "subslide"}
x_max = 30
bins = np.arange(0,x_max)
bin_interp = np.linspace(0,x_max-1,num=(x_max-1)*10)
hists = np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], 0, ppc['y'])

prctiles = np.percentile(hists,np.linspace(10,90,num=9),axis=1)
prctiles_interp = np.repeat(prctiles, 10,axis=1)

data_hist = np.histogram(data_ys,bins=bins)[0]
data_hist_interp = np.repeat(data_hist, 10)
```

```python slideshow={"slide_type": "fragment"}
for i,color in enumerate([c_light,c_light_highlight,c_mid,c_mid_highlight]):
    plt.fill_between(bin_interp,prctiles_interp[i,:],prctiles_interp[-1-i,:],alpha=1.0,color=color);


plt.plot(bin_interp,prctiles_interp[4,:],color=c_dark_highlight);
plt.plot(bin_interp,data_hist_interp,color='black');
plt.axvline(x=25,ls='-',lw=2,color='k');
plt.xlabel('y');
plt.title('Posterior predictive distribution');
```
