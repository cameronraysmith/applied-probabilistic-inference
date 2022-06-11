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
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
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
#     version: 3.10.4
#   rise:
#     scroll: true
#     theme: black
#   toc-autonumbering: true
#   toc-showcode: false
#   toc-showmarkdowntxt: false
#   widgets:
#     application/vnd.jupyter.widget-state+json:
#       state: {}
#       version_major: 2
#       version_minor: 0
# ---

# %% [markdown]
# # Minimal example in pyro

# %% [markdown]
# ## Setup

# %% [markdown]
# ### Import libraries

# %%
# may need development version of pyro
# when running on python 3.10
# see: https://github.com/pyro-ppl/pyro/pull/3101
# # !sudo pip install git+https://github.com/pyro-ppl/pyro.git

# %% {"tags": []}
from inspect import getmembers
from pprint import pprint
from types import FunctionType

import arviz as az
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import MCMC, NUTS, Predictive
import platform

# az.style.use("arviz-darkgrid")

# %% {"tags": []}
# pyro.set_platform("cpu")
# pyro.set_host_device_count(4)

# %% {"tags": []}
print(platform.python_version())
print(pyro.__version__)
print(torch.__version__)
print(az.__version__)

# %% [markdown]
# ### Setup plotting

# %% {"slideshow": {"slide_type": "fragment"}, "tags": []}
import matplotlib.font_manager
import matplotlib.pyplot as plt

# import matplotlib_inline

# %% {"slideshow": {"slide_type": "fragment"}, "tags": []}
# fonts_path = "/usr/share/texmf/fonts/opentype/public/lm/" #ubuntu
# fonts_path = "~/Library/Fonts/" # macos
fonts_path = "/usr/share/fonts/OTF/"  # arch
matplotlib.font_manager.fontManager.addfont(fonts_path + "lmsans10-regular.otf")
matplotlib.font_manager.fontManager.addfont(fonts_path + "lmroman10-regular.otf")

# %% {"slideshow": {"slide_type": "fragment"}, "tags": []}
# https://stackoverflow.com/a/36622238/446907
# %config InlineBackend.figure_formats = ['svg']

# %% {"slideshow": {"slide_type": "fragment"}, "tags": []}
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


# %% [markdown]
# ### Utility functions

# %% {"tags": []}
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


# %% [markdown]
# ## Execute

# %% [markdown]
# ### Define sample data

# %% {"tags": []}
N_obs = 100

# %% {"tags": []}
# observations = dist.Normal(0, 1).sample([N_obs])
observations = torch.randn(N_obs, names=(None,))


# %% [markdown]
# ### Define model

# %% {"tags": []}
def model(obs=None):
    mu = pyro.sample("mu", dist.Normal(0, 1))
    sigma = pyro.sample("sigma", dist.HalfNormal(1))
    with pyro.plate("N_obs", N_obs):
        pyro.sample("obs", dist.Normal(mu, sigma), obs=obs)


# %% {"tags": []}
pyro.render_model(
    model, model_args=(observations,), render_distributions=True, render_params=True
)

# %% [markdown]
# ### Fit model

# %% {"tags": []}
R = 1000

# %% {"tags": []}
kernel = NUTS(model, jit_compile=False)

# %% {"tags": []}
mcmc = MCMC(kernel, warmup_steps=500, num_samples=R, num_chains=4)

# %% {"tags": []}
mcmc.run(observations)

# %% {"tags": []}
posterior_samples = mcmc.get_samples(group_by_chain=False)

# %% {"tags": []}
# rng_key, rng_key_ = jax.random.split(rng_key)
posterior_predictive = Predictive(model, posterior_samples)
posterior_predictions = posterior_predictive()

# %% {"tags": []}
[v.shape for k, v in posterior_predictions.items()]

# %% {"tags": []}
# rng_key, rng_key_ = jax.random.split(rng_key)
prior_predictive = Predictive(model, num_samples=500)
prior_predictions = prior_predictive()

# %% {"tags": []}
[v.shape for k, v in prior_predictions.items()]

# %% [markdown] {"tags": []}
# ### Organize output data

# %% {"tags": []}
data = az.from_pyro(
    mcmc,
    prior=prior_predictions,
    posterior_predictive=posterior_predictions,
)

# %% {"tags": []}
data

# %% [markdown]
# ### Evaluate model

# %% [markdown]
# #### Plot autocorrelation to evaluate MCMC chain mixing

# %% {"tags": []}
az.plot_autocorr(data, var_names=["mu", "sigma"])

# %% [markdown]
# #### Plot prior and posterior predictive distributions

# %% {"tags": []}
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

# %% {"tags": []}
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

# %% [markdown]
# #### Characterize posterior distribution

# %% {"tags": []}
az.plot_forest(data)
az.plot_trace(data)
az.plot_posterior(data)

# %%
