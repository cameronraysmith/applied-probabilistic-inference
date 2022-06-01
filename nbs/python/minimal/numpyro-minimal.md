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

# Minimal example in numpyro


## Setup


### Import libraries

```python tags=[]
from inspect import getmembers
from pprint import pprint
from types import FunctionType

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

# az.style.use("arviz-darkgrid")
```

```python tags=[]
numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)
```

```python tags=[]
print(numpyro.__version__)
print(jax.__version__)
# print(pm.__version__)
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

### Utility functions

```python
def attributes(obj):
    disallowed_names = {
        name for name, value in getmembers(type(obj)) if isinstance(value, FunctionType)
    }
    return {
        name: getattr(obj, name)
        for name in dir(obj)
        if name[0] != "_" and name not in disallowed_names and hasattr(obj, name)
    }


def print_attributes(obj):
    pprint(attributes(obj))
```

## Execute


### Define sample data

```python
N_obs = 100
```

```python tags=[]
observations = np.random.randn(N_obs)
```

### Define model

```python
def model(obs=None):
    mu = numpyro.sample("mu", dist.Normal(1, 2))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=obs)
```

```python tags=[]
numpyro.render_model(
    model, model_args=(observations,), render_distributions=True, render_params=True
)
```

### Fit model

```python tags=[]
R = 1000
```

```python tags=[]
# Start from this source of randomness. We will split keys for subsequent operations.
rng_key = jax.random.PRNGKey(0)
rng_key, rng_key_ = jax.random.split(rng_key)
```

```python tags=[]
kernel = NUTS(model)
mcmc = MCMC(
    kernel, num_warmup=500, num_samples=R, num_chains=4, chain_method="parallel"
)
```

```python tags=[]
mcmc.run(rng_key_, observations)
```

```python tags=[]
mcmc.print_summary()
```

```python tags=[]
posterior_samples = mcmc.get_samples(group_by_chain=False)
```

```python tags=[]
rng_key, rng_key_ = jax.random.split(rng_key)
posterior_predictive = Predictive(model, posterior_samples)
posterior_predictions = jax.lax.map(posterior_predictive,jax.random.split(rng_key_, N_obs))
```

```python tags=[]
rng_key, rng_key_ = jax.random.split(rng_key)
prior_predictive = Predictive(model, num_samples=500)
prior_predictions = jax.lax.map(prior_predictive,jax.random.split(rng_key_, N_obs))
```

```python tags=[]
# posterior_predictions["obs"] = jax.numpy.transpose(posterior_predictions["obs"])
posterior_predictions = {k: jax.numpy.transpose(v) for k,v in posterior_predictions.items()}
# prior_predictions["obs"] = jax.numpy.transpose(prior_predictions["obs"])
prior_predictions = {k: jax.numpy.transpose(v) for k,v in prior_predictions.items()}
```

```python tags=[]
posterior_predictions["obs"].shape
```

```python tags=[]
print(prior_predictions["obs"].shape)
[v.shape for k,v in prior_predictions.items()]
```

<!-- #region {"tags": []} -->
### Organize output data
<!-- #endregion -->

```python tags=[]
type(mcmc)
```

```python tags=[]
# print_attributes(mcmc)
# print_attributes(mcmc.last_state)
# print_attributes(mcmc.last_state.adapt_state)
```

```python tags=[]
data = az.from_numpyro(
    mcmc,
    prior=prior_predictions,
    posterior_predictive=posterior_predictions,
)
```

```python tags=[]
data
```

### Evaluate model


#### Plot autocorrelation to evaluate MCMC chain mixing

```python tags=[]
# with model:
az.plot_autocorr(data, var_names=["mu", "sigma"])
```

#### Plot prior and posterior predictive distributions

```python tags=[]
ax_pr_pred_cum = az.plot_ppc(
    data,
    group="prior",
    data_pairs={"obs": "obs"},
    kind="cumulative",
    num_pp_samples=100,
    random_seed=7,
)
ax_pr_pred_cum.set_xlim([-7,5.5])
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
ax_pr_pred = az.plot_ppc(
    data,
    group="prior",
    data_pairs={"obs": "obs"},
    num_pp_samples=100,
    random_seed=7,
)
ax_pr_pred.set_xlim([-5,5])
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

```python

```
