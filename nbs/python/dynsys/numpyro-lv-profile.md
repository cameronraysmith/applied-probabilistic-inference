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
```

```python
from jax.experimental.ode import odeint
from numpyro.examples.datasets import LYNXHARE, load_dataset
from numpyro.infer import MCMC, NUTS, Predictive
from diffrax import diffeqsolve, ODETerm, Tsit5, Dopri5, BacksolveAdjoint, SaveAt, PIDController
```

```python
jax.config.update("jax_enable_x64", True)
numpyro.util.enable_x64()
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
def dz_dt(t, z, theta):
    """
    Lotka–Volterra equations. Real positive parameters `alpha`, `beta`, `gamma`, `delta`
    describes the interaction of two species.
    """
    u, v = z
    alpha, beta, gamma, delta = theta
    du = (alpha - beta * v) * u
    dv = (-gamma + delta * u) * v
    d_z = du, dv
    return d_z
```

```python
term = ODETerm(dz_dt)
t0 = 0
t1 = data.shape[0]
dt0 = 0.01
y0 = (10.0, 10.0)
args = (1.0, 0.05, 1.0, 0.05)
# saveat = SaveAt(ts=jnp.linspace(t0,t1,1000))
saveat = SaveAt(ts=jnp.arange(float(t1)))
stepsize_controller = PIDController(rtol=1e-6, atol=1e-5)
solution = diffeqsolve(
    term,
    solver=Dopri5(),
    t0=t0,
    t1=t1,
    dt0=dt0,
    y0=y0,
    args=args,
    saveat=saveat,
    stepsize_controller=stepsize_controller,
    adjoint=BacksolveAdjoint(),
    max_steps=1000,
)
```

```python
plt.plot(solution.ts, solution.ys[0], label="prey", marker=".", ms=12, color="green")
plt.plot(solution.ts, solution.ys[1], label="predator", marker=".", ms=12, color="gray")
plt.xlabel("time (years)")
plt.ylabel("population (thousands)")
plt.legend(loc="upper right")
plt.show()
```

#### Define probabilistic model

```python
def model(N, y=None):
    """
    :param int N: number of measurement times
    :param numpy.ndarray y: measured populations with shape (N, 2)
    """
    z_init = numpyro.sample("z_init", dist.LogNormal(jnp.log(10), 1).expand([2]))
    ts = jnp.arange(float(N))
    theta = numpyro.sample(
        "theta",
        dist.TruncatedNormal(
            low=5e-3,
            loc=jnp.array([1.0, 0.05, 1.0, 0.05]),
            scale=jnp.array([0.5, 0.05, 0.5, 0.05]),
        ),
    )

    term = ODETerm(dz_dt)
    t0 = 0.0
    t1 = float(N)
    dt0 = 0.01
    saveat = SaveAt(ts=ts)
    stepsize_controller = PIDController(rtol=1e-6, atol=1e-5)
    
    y0 = tuple(z_init)
    args = tuple(theta)
    
    solution = diffeqsolve(
        term,
        solver=Dopri5(),
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=args,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=int(1e9),
    )

    z = jnp.stack(solution.ys, axis=-1)
    positive_mask = z > 1e-10
    log_z = jnp.where(positive_mask, jnp.log(z), -1e10)

    sigma = numpyro.sample("sigma", dist.LogNormal(-1, 1).expand([2]))
    numpyro.sample("y", dist.LogNormal(log_z, sigma), obs=y)
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

ax1.plot(observed_y[:, 0], label="prey", color="green")
ax1.plot(observed_y[:, 1], label="predator", color="gray")

selected_indices = np.random.choice(prior_samples.shape[1], 20, replace=False)
max_val = 0
for i in selected_indices:
    ax1.plot(prior_samples[0, i, :, 0], color="green", alpha=0.1)
    ax1.plot(prior_samples[0, i, :, 1], color="gray", alpha=0.1)
    # max_val = max(max_val, prior_samples[0, i, :, 0].max(), prior_samples[0, i, :, 1].max())

max_val = observed_y.max()
ax1.set_ylim([-0.01, max_val * 1.1])

ax2.plot(observed_y[:, 0], label="prey", color="green")
ax2.plot(observed_y[:, 1], label="predator", color="gray")
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
# colors for percentile bands
colors = [
    "#DCBCBC",
    "#C79999",
    "#B97C7C",
    "#A25050",
    "#8F2727",
    "#7C0000"
]

observed_y = idata_prior.observed_data["y"]
prior_samples = idata_prior.prior["y"]

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True)

# observed data
ax1.plot(observed_y[:, 0], label="prey", color="green")
ax1.plot(observed_y[:, 1], label="predator", color="lightgreen")

# compute percentiles
percentiles = np.linspace(0, 100, 8)[1:-1]  # Equally spaced percentiles

# percentile bands
for i, percentile in enumerate(percentiles[::-1]):
    upper = np.percentile(prior_samples[:, :, :, 0], 50 + percentile / 2, axis=1)
    lower = np.percentile(prior_samples[:, :, :, 0], 50 - percentile / 2, axis=1)
    ax1.fill_between(range(prior_samples.shape[2]), lower.mean(axis=0), upper.mean(axis=0), color=colors[i], alpha=0.6)
    
    upper = np.percentile(prior_samples[:, :, :, 1], 50 + percentile / 2, axis=1)
    lower = np.percentile(prior_samples[:, :, :, 1], 50 - percentile / 2, axis=1)
    ax1.fill_between(range(prior_samples.shape[2]), lower.mean(axis=0), upper.mean(axis=0), color=colors[i], alpha=0.6)

# median
median_0 = np.percentile(prior_samples[:, :, :, 0], 50, axis=1).mean(axis=0)
median_1 = np.percentile(prior_samples[:, :, :, 1], 50, axis=1).mean(axis=0)
ax1.plot(median_0, color="black", label="Median prey")
ax1.plot(median_1, color="gray", label="Median predator")

ax1.set_ylabel("Population number (linear)")
ax1.legend()
ax1.set_title("Observed Data with Percentile Bands")

# log scale
ax2.plot(observed_y[:, 0], label="observed prey", color="green")
ax2.plot(observed_y[:, 1], label="observed predator", color="lightgreen")

for i, percentile in enumerate(percentiles[::-1]):
    upper = np.percentile(prior_samples[:, :, :, 0], 50 + percentile / 2, axis=1)
    lower = np.percentile(prior_samples[:, :, :, 0], 50 - percentile / 2, axis=1)
    ax2.fill_between(range(prior_samples.shape[2]), lower.mean(axis=0), upper.mean(axis=0), color=colors[i], alpha=0.6)
    
    upper = np.percentile(prior_samples[:, :, :, 1], 50 + percentile / 2, axis=1)
    lower = np.percentile(prior_samples[:, :, :, 1], 50 - percentile / 2, axis=1)
    ax2.fill_between(range(prior_samples.shape[2]), lower.mean(axis=0), upper.mean(axis=0), color=colors[i], alpha=0.6)

ax2.plot(median_0, color="green", label="median prey")
ax2.plot(median_1, color="gray", label="median predator")

ax2.set_yscale("log")
ax2.set_xlabel("Time (years)")
ax2.set_ylabel("(log)")

plt.tight_layout()
plt.show()
```

```python
def plot_percentile_bands(prior_samples, observed_y, variable_index, ax):
    # Define colors
    colors = [
        "#DCBCBC",
        "#C79999",
        "#B97C7C",
        "#A25050",
        "#8F2727",
        "#7C0000"
    ]
    
    # compute percentiles
    percentiles = np.linspace(0, 100, 8)[1:-1]
    
    # observed data
    ax[0].plot(observed_y[:, variable_index], color="green", label="observed")
    
    # percentile bands
    for i, percentile in enumerate(percentiles[::-1]):
        upper = np.percentile(prior_samples[:, :, :, variable_index], 50 + percentile / 2, axis=1)
        lower = np.percentile(prior_samples[:, :, :, variable_index], 50 - percentile / 2, axis=1)
        ax[0].fill_between(range(prior_samples.shape[2]), lower.mean(axis=0), upper.mean(axis=0), color=colors[i], alpha=0.6)
    
    # median
    median = np.percentile(prior_samples[:, :, :, variable_index], 50, axis=1).mean(axis=0)
    ax[0].plot(median, color="gray", label="median")
    
    # linear scale plot
    ax[0].set_ylabel("Population number (linear)")
    ax[0].legend()
    ax[0].set_title(f"Observed Data with Percentile Bands ({'Prey' if variable_index == 0 else 'Predator'})")
    
    # log scale plot
    ax[1].plot(observed_y[:, variable_index], color="green", label="observed")
    
    for i, percentile in enumerate(percentiles[::-1]):
        upper = np.percentile(prior_samples[:, :, :, variable_index], 50 + percentile / 2, axis=1)
        lower = np.percentile(prior_samples[:, :, :, variable_index], 50 - percentile / 2, axis=1)
        ax[1].fill_between(range(prior_samples.shape[2]), lower.mean(axis=0), upper.mean(axis=0), color=colors[i], alpha=0.6)
    
    ax[1].plot(median, color="gray", label="median")
    
    # Set labels for the log scale plot
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Time (years)")
    ax[1].set_ylabel("(log)")
```

```python
for variable_index, name in enumerate(["Prey", "Predator"]):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True)
    plot_percentile_bands(prior_samples, observed_y, variable_index, ax)
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
