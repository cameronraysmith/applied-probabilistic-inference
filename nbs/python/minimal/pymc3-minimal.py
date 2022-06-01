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
# # Minimal example for PyMC3

# %% [markdown]
# ## Setup

# %% [markdown]
# ### Import libraries

# %%
import arviz as az
import numpy as np
import pymc3 as pm

az.style.use("arviz-darkgrid")

# %%
print(pm.__version__)
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
# ## Execute

# %% [markdown]
# ### Define sample data

# %%
N_obs = 100

# %%
observations = np.random.randn(N_obs)

# %% [markdown]
# ### Define model

# %% {"tags": []}
with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    sd = pm.HalfNormal("sd", sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=sd, observed=observations)

# %%
pm.model_to_graphviz(model)

# %% [markdown]
# ### Fit model

# %% {"tags": []}
with model:
    prior = pm.sample_prior_predictive()
    trace = pm.sample(1000, tune=500, cores=4, return_inferencedata=False)
    posterior_predictive = pm.sample_posterior_predictive(trace)

# %% [markdown]
# ### Organize output data

# %% {"tags": []}
with model:
    data = az.from_pymc3(
        model=model,
        trace=trace,
        prior=prior,
        posterior_predictive=posterior_predictive,
    )

# %% {"tags": []}
posterior_predictive["obs"].shape

# %%
[v.shape for k,v in prior.items()]

# %% {"tags": []}
prior["obs"].shape

# %% {"tags": []}
prior["mu"].shape

# %% {"tags": []}
data

# %% [markdown]
# ### Evaluate model

# %% [markdown]
# #### Plot autocorrelation to evaluate MCMC chain mixing

# %% {"tags": []}
with model:
    az.plot_autocorr(trace, var_names=["mu", "sd"])

# %% [markdown]
# #### Plot prior and posterior predictive distributions

# %%
data.prior_predictive

# %%
data.posterior_predictive

# %% {"tags": []}
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

# %% {"tags": []}
print(data.prior_predictive.sizes["chain"])
data.prior_predictive.sizes["draw"]

# %%
print(data.posterior_predictive.sizes["chain"])
data.posterior_predictive.sizes["draw"]

# %% {"tags": []}
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

# %% [markdown]
# #### Characterize posterior distribution

# %% {"tags": []}
az.plot_forest(data)
az.plot_trace(data)
az.plot_posterior(data)
