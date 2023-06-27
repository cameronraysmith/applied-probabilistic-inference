---
jupyter:
  celltoolbar: Slideshow
  environment:
    kernel: api
    name: pytorch-gpu.1-12.m100
    type: gcloud
    uri: gcr.io/deeplearning-platform-release/pytorch-gpu.1-12:m100
  jupytext:
    cell_metadata_json: true
    formats: ipynb,md,py:percent
    notebook_metadata_filter: all
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.6
  kernelspec:
    display_name: api
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
    version: 3.10.12
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

<!-- #region {"jp-MarkdownHeadingCollapsed": true} -->
## Debug
<!-- #endregion -->

```python
# may need development version of pyro
# when running on python 3.10
# see: https://github.com/pyro-ppl/pyro/pull/3101
# !sudo pip install git+https://github.com/pyro-ppl/pyro.git
```

```python
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

```python
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

```python
USE_CUDA = False
TORCH_DETERMINISTIC = True
```

### Import libraries

```python
import os
```

```python
if TORCH_DETERMINISTIC:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print(os.environ["CUBLAS_WORKSPACE_CONFIG"])
```

```python
from inspect import getmembers
from pprint import pprint
from types import FunctionType

import arviz as az
import numpy as np
import torch
```

```python
torch.use_deterministic_algorithms(TORCH_DETERMINISTIC)
```

```python
SEED = 1234
```

```python
np.random.seed(seed=SEED);
torch.manual_seed(SEED);
```

```python
import pyro
import pyro.distributions as dist

from pyro.infer import MCMC, NUTS, Predictive
import platform
```

```python
print(pyro.settings.get())
```

```python
print(platform.python_version())
print(pyro.__version__)
print(torch.__version__)
print(az.__version__)
```

```python
if not USE_CUDA:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

```python
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

<!-- #region {"jp-MarkdownHeadingCollapsed": true} -->
### Setup plotting
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
import matplotlib.font_manager
import matplotlib.pyplot as plt

# import matplotlib_inline
```

```python slideshow={"slide_type": "fragment"}
fonts_path = "/usr/share/texmf/fonts/opentype/public/lm/" #ubuntu
# fonts_path = "~/Library/Fonts/" # macos
# fonts_path = "/usr/share/fonts/OTF/"  # arch
matplotlib.font_manager.fontManager.addfont(fonts_path + "lmsans10-regular.otf")
matplotlib.font_manager.fontManager.addfont(fonts_path + "lmroman10-regular.otf")
```

```python slideshow={"slide_type": "fragment"}
# https://stackoverflow.com/a/36622238/446907
%config InlineBackend.figure_formats = ['svg']
```

```python slideshow={"slide_type": "fragment"}
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

<!-- #region {"jp-MarkdownHeadingCollapsed": true} -->
### Utility functions
<!-- #endregion -->

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

```python
# device=torch.device("cpu")
# observations = dist.Normal(0, 1).sample([N_obs])
observations = torch.randn(
    N_obs, 
    # names=(None,),
    # device=device,
)
```

```python
observations
```

### Define model

```python
def model(obs=None):
    mu = pyro.sample("mu", dist.Normal(0, 1))
    sigma = pyro.sample("sigma", dist.HalfNormal(1))
    with pyro.plate("N_obs", N_obs):
        pyro.sample("obs", dist.Normal(mu, sigma), obs=obs)
```

```python
pyro.render_model(
    model, 
    model_args=(observations,), 
    render_distributions=True, 
    render_params=True,
)
```

### Fit model

```python
R = 1000
```

```python
prior_predictive = Predictive(model, num_samples=500)
prior_predictions = prior_predictive()
```

```python
kernel = NUTS(model, jit_compile=False)
```

```python
mcmc = MCMC(
    kernel, 
    warmup_steps=500, 
    num_samples=R, 
    num_chains=4, 
    # mp_context="spawn"
)
```

```python
mcmc.run(observations)
```

```python
posterior_samples = mcmc.get_samples(group_by_chain=False)
```

```python
posterior_predictive = Predictive(model, posterior_samples)
posterior_predictions = posterior_predictive()
```

```python
[v.shape for k, v in posterior_predictions.items()]
```

```python
prior_predictive = Predictive(model, num_samples=500)
prior_predictions = prior_predictive()
```

```python
[v.shape for k, v in prior_predictions.items()]
```

### Organize output data

```python
data = az.from_pyro(
    mcmc,
    prior=prior_predictions,
    posterior_predictive=posterior_predictions,
)
```

```python
data
```

### Evaluate model


#### Plot autocorrelation to evaluate MCMC chain mixing

```python
az.plot_autocorr(data, var_names=["mu", "sigma"]);
```

#### Plot prior and posterior predictive distributions

```python
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

```python
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

```python
az.plot_forest(data);
az.plot_trace(data);
az.plot_posterior(data);
```

```python

```
