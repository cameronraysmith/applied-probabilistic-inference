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
    display_name: api
    language: python
    name: api
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.10.9
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

# Minimal example in pyro

<!-- #region {"tags": [], "jp-MarkdownHeadingCollapsed": true} -->
## Debug
<!-- #endregion -->

```python tags=[]
# may need development version of pyro
# when running on python 3.10
# see: https://github.com/pyro-ppl/pyro/pull/3101
# !sudo pip install git+https://github.com/pyro-ppl/pyro.git
```

```python tags=[]
# # importing os module 
# import os
# import pprint
  
# # Get the list of user's
# # environment variables
# env_var = os.environ
  
# # Print the list of user's
# # environment variables
# print("User's Environment variable:")
# pprint.pprint(dict(env_var), width = 1)
```

```python tags=[]
# %%bash

# which python
# python --version
# echo ${PATH}
# echo ${LD_LIBRARY_PATH}
```

```python
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
```

## Setup

```python tags=[]
USE_CUDA = False
TORCH_DETERMINISTIC = True
```

<!-- #region {"tags": []} -->
### Import libraries
<!-- #endregion -->

```python tags=[]
import os
```

```python tags=[]
if TORCH_DETERMINISTIC:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print(os.environ["CUBLAS_WORKSPACE_CONFIG"])
```

```python tags=[]
from inspect import getmembers
from pprint import pprint
from types import FunctionType

import arviz as az
import numpy as np
import torch
```

```python tags=[]
torch.use_deterministic_algorithms(TORCH_DETERMINISTIC)
```

```python tags=[]
SEED = 1234
```

```python tags=[]
np.random.seed(seed=SEED);
torch.manual_seed(SEED);
```

```python tags=[]
import pyro
import pyro.distributions as dist

from pyro.infer import MCMC, NUTS, Predictive
import platform
```

```python tags=[]
print(pyro.settings.get())
```

```python tags=[]
print(platform.python_version())
print(pyro.__version__)
print(torch.__version__)
print(az.__version__)
```

```python tags=[]
if not USE_CUDA:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

```python tags=[]
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

<!-- #region {"jp-MarkdownHeadingCollapsed": true, "tags": []} -->
### Setup plotting
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"} tags=[]
import matplotlib.font_manager
import matplotlib.pyplot as plt

# import matplotlib_inline
```

```python slideshow={"slide_type": "fragment"} tags=[]
fonts_path = "/usr/share/texmf/fonts/opentype/public/lm/" #ubuntu
# fonts_path = "~/Library/Fonts/" # macos
# fonts_path = "/usr/share/fonts/OTF/"  # arch
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

<!-- #region {"jp-MarkdownHeadingCollapsed": true, "tags": []} -->
### Utility functions
<!-- #endregion -->

```python tags=[]
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

```python tags=[]
N_obs = 100
```

```python tags=[]
# device=torch.device("cpu")
# observations = dist.Normal(0, 1).sample([N_obs])
observations = torch.randn(
    N_obs, 
    # names=(None,),
    # device=device,
)
```

```python tags=[]
observations
```

### Define model

```python tags=[]
def model(obs=None):
    mu = pyro.sample("mu", dist.Normal(0, 1))
    sigma = pyro.sample("sigma", dist.HalfNormal(1))
    with pyro.plate("N_obs", N_obs):
        pyro.sample("obs", dist.Normal(mu, sigma), obs=obs)
```

```python tags=[]
pyro.render_model(
    model, 
    model_args=(observations,), 
    render_distributions=True, 
    render_params=True,
)
```

### Fit model

```python tags=[]
R = 1000
```

```python tags=[]
prior_predictive = Predictive(model, num_samples=500)
prior_predictions = prior_predictive()
```

```python tags=[]
kernel = NUTS(model, jit_compile=False)
```

```python tags=[]
mcmc = MCMC(
    kernel, 
    warmup_steps=500, 
    num_samples=R, 
    num_chains=4, 
    # mp_context="spawn"
)
```

```python tags=[]
mcmc.run(observations)
```

```python tags=[]
posterior_samples = mcmc.get_samples(group_by_chain=False)
```

```python tags=[]
posterior_predictive = Predictive(model, posterior_samples)
posterior_predictions = posterior_predictive()
```

```python tags=[]
[v.shape for k, v in posterior_predictions.items()]
```

```python tags=[]
prior_predictive = Predictive(model, num_samples=500)
prior_predictions = prior_predictive()
```

```python tags=[]
[v.shape for k, v in prior_predictions.items()]
```

<!-- #region {"tags": []} -->
### Organize output data
<!-- #endregion -->

```python tags=[]
data = az.from_pyro(
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
az.plot_autocorr(data, var_names=["mu", "sigma"]);
```

#### Plot prior and posterior predictive distributions

```python tags=[]
ax_pr_pred = az.plot_ppc(
    data,
    group="prior",
    data_pairs={"obs": "obs"},
    num_pp_samples=100,
    random_seed=7,
)
ax_pr_pred.set_xlim([-5, 5])
az.plot_ppc(
    data,
    group="posterior",
    data_pairs={"obs": "obs"},
    num_pp_samples=100,
    random_seed=7,
);
```

```python tags=[]
ax_pr_pred_cum = az.plot_ppc(
    data,
    group="prior",
    data_pairs={"obs": "obs"},
    kind="cumulative",
    num_pp_samples=100,
    random_seed=7,
)
ax_pr_pred_cum.set_xlim([-7, 5.5])
az.plot_ppc(
    data,
    group="posterior",
    data_pairs={"obs": "obs"},
    kind="cumulative",
    num_pp_samples=100,
    random_seed=7,
);
```

#### Characterize posterior distribution

```python tags=[]
az.plot_forest(data);
az.plot_trace(data);
az.plot_posterior(data);
```

```python

```
