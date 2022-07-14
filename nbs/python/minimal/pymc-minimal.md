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
      jupytext_version: 1.14.0
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
    version: 3.10.5
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

# Minimal example for PyMC


## Setup


### Import libraries

```python
import arviz as az
import numpy as np
import pymc as pm

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

```python tags=[]
N_obs = 100
```

```python tags=[]
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
    idata = pm.sample_prior_predictive()
    idata.extend(pm.sample(1000, tune=500, cores=4))
    idata.extend(pm.sample_posterior_predictive(idata))
```

```python tags=[]
type(idata)
```

### Organize output data

```python tags=[]
idata
```

### Evaluate model


#### Plot autocorrelation to evaluate MCMC chain mixing

```python tags=[]
with model:
    az.plot_autocorr(idata, var_names=["mu", "sd"])
```

#### Plot prior and posterior predictive distributions

```python tags=[]
idata.prior_predictive
```

```python tags=[]
idata.posterior_predictive
```

```python tags=[]
az.plot_ppc(
    idata,
    group="prior",
    data_pairs={"obs": "obs"},
    kind="cumulative",
    num_pp_samples=100,
    random_seed=7,
)
az.plot_ppc(
    idata,
    group="posterior",
    data_pairs={"obs": "obs"},
    kind="cumulative",
    num_pp_samples=100,
    random_seed=7,
);
```

```python tags=[]
print(idata.prior_predictive.sizes["chain"])
idata.prior_predictive.sizes["draw"]
```

```python tags=[]
print(idata.posterior_predictive.sizes["chain"])
idata.posterior_predictive.sizes["draw"]
```

```python tags=[]
az.plot_ppc(
    idata, 
    group="prior", 
    data_pairs={"obs": "obs"}, 
    num_pp_samples=100, 
    random_seed=7
)
az.plot_ppc(
    idata,
    group="posterior",
    data_pairs={"obs": "obs"},
    num_pp_samples=100,
    random_seed=7,
);
```

#### Characterize posterior distribution

```python tags=[]
az.plot_forest(idata)
az.plot_trace(idata)
az.plot_posterior(idata)
```
