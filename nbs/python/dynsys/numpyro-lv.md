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

# Dynamical systems example in numpyro

<!-- #region {"jp-MarkdownHeadingCollapsed": true} -->
## Debug
<!-- #endregion -->

<!-- #region {"jp-MarkdownHeadingCollapsed": true} -->
## Setup
<!-- #endregion -->

### Import libraries

```python
import platform
from inspect import getmembers
from pprint import pprint
from types import FunctionType

import arviz as az
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
```

```python
from jax.experimental.ode import odeint
from numpyro.examples.datasets import LYNXHARE, load_dataset
from numpyro.infer import MCMC, NUTS, Predictive
```

```python
import warnings

warnings.filterwarnings("ignore")
```

```python
numpyro.set_platform("cpu")
numpyro.set_host_device_count(8)
```

```python
print(platform.python_version())
print(numpyro.__version__)
print(jax.__version__)
print(az.__version__)
```

```python
# print("Numpyro platform:", numpyro.get_platform())
print("JAX backend:", jax.devices())
```

### Setup plotting

```python slideshow={"slide_type": "fragment"}
import matplotlib.font_manager
import matplotlib.pyplot as plt

# import matplotlib_inline
```

```python slideshow={"slide_type": "fragment"}
fonts_path = "/usr/share/texmf/fonts/opentype/public/lm/"  # ubuntu
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
_, fetch = load_dataset(LYNXHARE, shuffle=False)
year, data = fetch()  # data is in hare -> lynx order
```

```python
type(year), year.shape, year.dtype
```

```python
type(data), data.shape, data.dtype
```

### Define model


#### Simulate dynamical system

```python
def dz_dt(z, t, theta):
    """
    Lotka–Volterra equations. Real positive parameters `alpha`, `beta`, `gamma`, `delta`
    describes the interaction of two species.
    """
    u = z[0]
    v = z[1]
    alpha, beta, gamma, delta = (
        theta[..., 0],
        theta[..., 1],
        theta[..., 2],
        theta[..., 3],
    )
    du_dt = (alpha - beta * v) * u
    dv_dt = (-gamma + delta * u) * v
    return jnp.stack([du_dt, dv_dt])
```

#### Define probabilistic model

```python
def model(N, y=None):
    """
    :param int N: number of measurement times
    :param numpy.ndarray y: measured populations with shape (N, 2)
    """
    # initial population
    z_init = numpyro.sample("z_init", dist.LogNormal(jnp.log(10), 1).expand([2]))
    # measurement times
    ts = jnp.arange(float(N))
    # parameters alpha, beta, gamma, delta of dz_dt
    theta = numpyro.sample(
        "theta",
        dist.TruncatedNormal(
            low=0.0,
            loc=jnp.array([1.0, 0.05, 1.0, 0.05]),
            scale=jnp.array([0.5, 0.05, 0.5, 0.05]),
        ),
    )
    # integrate dz/dt, the result will have shape N x 2
    z = odeint(dz_dt, z_init, ts, theta, rtol=1e-6, atol=1e-5, mxstep=1000)
    # measurement errors
    sigma = numpyro.sample("sigma", dist.LogNormal(-1, 1).expand([2]))
    # measured populations
    numpyro.sample("y", dist.LogNormal(jnp.log(z), sigma), obs=y)
```

```python
numpyro.render_model(
    model,
    model_args=(
        data.shape[0],
        data,
    ),
    render_distributions=True,
    render_params=True,
)
```

#### Plot priors

```python
# Start from this source of randomness. We will split keys for subsequent operations.
rng_key = jax.random.PRNGKey(0)
rng_key, rng_key_ = jax.random.split(rng_key)
```

```python
rng_key, rng_key_ = jax.random.split(rng_key)
prior_predictive = Predictive(model, num_samples=1000)
prior_predictions = prior_predictive(rng_key_, data.shape[0])
```

```python
idata_prior = az.from_numpyro(
    posterior=None,
    prior=prior_predictions,
    posterior_predictive={"y": prior_predictions["y"]},
)
import xarray as xr

observed_data = xr.Dataset(
    {"y": (["y_dim_0", "y_dim_1"], data)},
    coords={"y_dim_0": range(data.shape[0]), "y_dim_1": range(data.shape[1])},
)
idata_prior.add_groups(observed_data=observed_data)
```

```python
observed_y = idata_prior.observed_data["y"]
prior_samples = idata_prior.prior["y"]

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True)

ax1.plot(observed_y[:, 0], label="hare", color="green")
ax1.plot(observed_y[:, 1], label="lynx", color="gray")

selected_indices = np.random.choice(prior_samples.shape[1], 20, replace=False)
max_val = 0
for i in selected_indices:
    ax1.plot(prior_samples[0, i, :, 0], color="green", alpha=0.1)
    ax1.plot(prior_samples[0, i, :, 1], color="gray", alpha=0.1)
    # max_val = max(max_val, prior_samples[0, i, :, 0].max(), prior_samples[0, i, :, 1].max())

max_val = observed_y.max()
ax1.set_ylim([-0.01, max_val * 1.1])

ax2.plot(observed_y[:, 0], label="hare", color="green")
ax2.plot(observed_y[:, 1], label="lynx", color="gray")
for i in selected_indices:
    ax2.plot(prior_samples[0, i, :, 0], color="green", alpha=0.1)
    ax2.plot(prior_samples[0, i, :, 1], color="gray", alpha=0.1)

ax2.set_yscale("log")

ax1.set_ylabel("Population number (linear)")
ax2.set_xlabel("Time (years)")
ax2.set_ylabel("(log)")
ax1.legend()
ax1.set_title("Observed Data and Prior Sample Trajectories")

plt.tight_layout()
plt.show()
```

```python
idata_prior
```

```python
light_gray = (0.7, 0.7, 0.7)
```

```python
az.plot_posterior(
    idata_prior,
    var_names=["sigma"],
    group="prior",
    kind="hist",
    round_to=2,
    hdi_prob=0.89,
    color=light_gray,
);
```

```python
az.plot_posterior(
    idata_prior,
    var_names=["theta"],
    grid=(2, 2),
    group="prior",
    kind="hist",
    round_to=2,
    hdi_prob=0.89,
    color=light_gray,
);
```

```python
az.plot_posterior(
    idata_prior,
    var_names=["z_init"],
    grid=(1, 2),
    group="prior",
    kind="hist",
    round_to=2,
    hdi_prob=0.89,
    color=light_gray,
);
```

### Fit model

```python
R = 1000
```

```python
kernel = NUTS(model, dense_mass=True)
mcmc = MCMC(
    kernel, num_warmup=1000, num_samples=R, num_chains=1, chain_method="parallel"
)
```

```python
mcmc.run(rng_key_, N=data.shape[0], y=data)
```

```python
mcmc.print_summary()
```

```python
posterior_samples = mcmc.get_samples(group_by_chain=False)
```

```python
rng_key, rng_key_ = jax.random.split(rng_key)
posterior_predictive = Predictive(model, posterior_samples)
posterior_predictions = posterior_predictive(rng_key_, data.shape[0])
```

```python
[v.shape for k, v in posterior_predictions.items()]
```

```python
rng_key, rng_key_ = jax.random.split(rng_key)
prior_predictive = Predictive(model, num_samples=1000)
prior_predictions = prior_predictive(rng_key_, data.shape[0])
```

```python
[v.shape for k, v in prior_predictions.items()]
```

### Organize output data

```python
type(mcmc)
```

```python
inferencedata = az.from_numpyro(
    mcmc,
    prior=prior_predictions,
    posterior_predictive=posterior_predictions,
)
```

```python
inferencedata
```

### Evaluate model


#### Plot autocorrelation to evaluate MCMC chain mixing

```python
# with model:
az.plot_autocorr(inferencedata, var_names=["sigma", "theta"]);
```

#### Plot prior and posterior predictive distributions

```python
ax_ppc = az.plot_ts(idata_prior, y="y", plot_dim="y_dim_0")
for ax in ax_ppc[0]:
    ax.set_xlim([-1, 93])
    ax.set_ylim([-1, 200])
```

```python

ax_ppc_log = az.plot_ts(idata_prior, y="y", plot_dim="y_dim_0")
for ax in ax_ppc_log[0]:
    ax.set_yscale("log")
```

```python
ax_postpc = az.plot_ts(inferencedata, y="y", plot_dim="y_dim_0")
for ax in ax_postpc[0]:
    ax.set_xlim([-1, 93])
    ax.set_ylim([-1, 200])
```

```python
ax_postpc_log = az.plot_ts(inferencedata, y="y", plot_dim="y_dim_0")
for ax in ax_postpc_log[0]:
    ax.set_yscale("log")
```

#### Characterize posterior distribution

```python
az.plot_posterior(
    inferencedata, var_names=["sigma"], grid=(1, 2), kind="hist", color=light_gray
)
az.plot_posterior(
    inferencedata, var_names=["theta"], grid=(2, 2), kind="hist", color=light_gray
)
az.plot_posterior(
    inferencedata, var_names=["z_init"], grid=(1, 2), kind="hist", color=light_gray
);
```

```python
az.plot_forest(inferencedata, var_names=["theta"])
az.plot_forest(inferencedata, var_names=["sigma"])
az.plot_forest(inferencedata, var_names=["z_init"]);
```

```python
az.plot_trace(inferencedata, rug=True);
```

```python

```
