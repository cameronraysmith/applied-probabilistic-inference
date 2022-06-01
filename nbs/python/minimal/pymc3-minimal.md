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
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3 (ipykernel)
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
    version: 3.10.4
  rise:
    scroll: true
    theme: black
  toc-autonumbering: true
  toc-showcode: false
  toc-showmarkdowntxt: false
  widgets:
    application/vnd.jupyter.widget-state+json:
      state: {}
      version_major: 2
      version_minor: 0
---

# Minimal example for PyMC3


## Setup


### Import libraries

```python
import arviz as az
import numpy as np
import pymc3 as pm

az.style.use("arviz-darkgrid")
```

```python
print(pm.__version__)
print(az.__version__)
```

### Setup plotting

```python slideshow={"slide_type": "fragment"} tags=[]
import matplotlib.font_manager
import matplotlib.pyplot as plt

# import matplotlib_inline
```

```python slideshow={"slide_type": "fragment"} tags=[]
# fonts_path = "/usr/share/texmf/fonts/opentype/public/lm/" #ubuntu
# fonts_path = "~/Library/Fonts/" # macos
fonts_path = "/usr/share/fonts/OTF/"  # arch
matplotlib.font_manager.fontManager.addfont(fonts_path + "lmsans10-regular.otf")
matplotlib.font_manager.fontManager.addfont(fonts_path + "lmroman10-regular.otf")
```

```python slideshow={"slide_type": "fragment"} tags=[]
# https://stackoverflow.com/a/36622238/446907
%config InlineBackend.figure_formats = ['svg']
```

```python slideshow={"slide_type": "fragment"} tags=[]
plt.style.use("default")  # reset default parameters
# https://stackoverflow.com/a/3900167/446907
plt.rcParams.update(
    {
        "font.size": 16,
        "font.family": ["sans-serif"],
        "font.serif": ["Latin Modern Roman"] + plt.rcParams["font.serif"],
        "font.sans-serif": ["Latin Modern Sans"] + plt.rcParams["font.sans-serif"],
    }
)
```

## Execute


### Define sample data

```python
N_obs = 100
```

```python
observations = np.random.randn(N_obs)
```

### Define model

```python tags=[]
with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    sd = pm.HalfNormal("sd", sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=sd, observed=observations)
```

```python
pm.model_to_graphviz(model)
```

### Fit model

```python tags=[]
with model:
    prior = pm.sample_prior_predictive()
    trace = pm.sample(1000, tune=500, cores=4, return_inferencedata=False)
    posterior_predictive = pm.sample_posterior_predictive(trace)
```

### Organize output data

```python tags=[]
with model:
    data = az.from_pymc3(
        model=model,
        trace=trace,
        prior=prior,
        posterior_predictive=posterior_predictive,
    )
```

```python tags=[]
posterior_predictive["obs"].shape
```

```python
[v.shape for k,v in prior.items()]
```

```python tags=[]
prior["obs"].shape
```

```python tags=[]
prior["mu"].shape
```

```python tags=[]
data
```

### Evaluate model


#### Plot autocorrelation to evaluate MCMC chain mixing

```python tags=[]
with model:
    az.plot_autocorr(trace, var_names=["mu", "sd"])
```

#### Plot prior and posterior predictive distributions

```python
data.prior_predictive
```

```python
data.posterior_predictive
```

```python tags=[]
az.plot_ppc(
    data,
    group="prior",
    data_pairs={"obs": "obs"},
    kind="cumulative",
    num_pp_samples=100,
    random_seed=7,
)
az.plot_ppc(
    data,
    group="posterior",
    data_pairs={"obs": "obs"},
    kind="cumulative",
    num_pp_samples=100,
    random_seed=7,
);
```

```python tags=[]
print(data.prior_predictive.sizes["chain"])
data.prior_predictive.sizes["draw"]
```

```python
print(data.posterior_predictive.sizes["chain"])
data.posterior_predictive.sizes["draw"]
```

```python tags=[]
az.plot_ppc(
    data, 
    group="prior", 
    data_pairs={"obs": "obs"}, 
    num_pp_samples=100, 
    random_seed=7
)
az.plot_ppc(
    data,
    group="posterior",
    data_pairs={"obs": "obs"},
    num_pp_samples=100,
    random_seed=7,
);
```

#### Characterize posterior distribution

```python tags=[]
az.plot_forest(data)
az.plot_trace(data)
az.plot_posterior(data)
```
