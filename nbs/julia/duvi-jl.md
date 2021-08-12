---
jupyter:
  jupytext:
    formats: ipynb,md,jl:percent
    notebook_metadata_filter: rise,toc-autonumbering,toc-showcode,toc-showmarkdowntxt
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.4
  kernelspec:
    display_name: Julia 1.6.2
    language: julia
    name: julia-1.6
  rise:
    scroll: true
    theme: black
  toc-autonumbering: true
  toc-showcode: false
  toc-showmarkdowntxt: false
---

# Introduction to generative models and variational inference


# Traditional probabilistic programming-based implementation of variational inference

<!-- #region heading_collapsed="true" -->
## Inverse Gamma-Normal conjugate model from Turing.jl
<!-- #endregion -->

This section is slightly edited for clarity from the [Turing.jl variational inference tutorial](https://github.com/TuringLang/TuringTutorials/blob/master/9_VariationalInference.ipynb).

<!-- #region heading_collapsed="true" -->
### Import libraries
<!-- #endregion -->

```julia
# ~1m50s initial
# ~1s subsequent
using Random
using Turing
using Turing: Variational

Random.seed!(42);
```

<!-- #region heading_collapsed="true" -->
### Define model
<!-- #endregion -->

The Normal-(Inverse)Gamma conjugate model is defined by a generative process

\begin{align}
    s &\sim \mathrm{InverseGamma}(2, 3) \\
    m &\sim \mathcal{N}(0, s) \\
    x_i &\overset{\text{i.i.d.}}{=} \mathcal{N}(m, s), \quad i = 1, \dots, n
\end{align}


Generate synthetic data samples. This is the key capability of generative models. This ability derives from the fact that the joint distribution over observed and latent variables is being considered.

```julia
x = randn(2000);
```

```julia
x[1:5]
```

Define the model as an instance of the type `Turing.Model`.

```julia
@model model(x) = begin
    s ~ InverseGamma(2,3)
    m ~ Normal(0.0, sqrt(s))
    for i = 1:length(x)
        x[i] ~ Normal(m, sqrt(s))
    end
end
```

```julia
print(@doc(@model))
```

Construct an instance of the model `m`.

```julia
m = model(x);
```

```julia
typeof(m)
```

```julia
m
```

See [DynamicPPL.jl/src/model.jl](https://github.com/TuringLang/DynamicPPL.jl/blob/master/src/model.jl) for the definition of the `DynamicPPL.Model` type.

<!-- #region heading_collapsed="true" -->
### Sample from the model with MCMC/HMC
<!-- #endregion -->

Here we use the [no U-turn sampler (NUTS)](http://chi-feng.github.io/mcmc-demo/app.html) to generate samples from the model.

```julia
# ?2m12s
samples_nuts = sample(m, NUTS(200, 0.65), 10000);
```

```julia
samples_nuts
```

<!-- #region heading_collapsed="true" -->
### Sample from the model with VI
<!-- #endregion -->

```julia
print(@doc(Variational.vi))
```

`vi` takes 
1. the `Model` you want to approximate and
1. a `VariationalInference` algorithm whose type specifies the method to use and whose fields specify the configuration of the method.

As of 03/2021 the only implementation of `VariationalInference` available in `Turing.jl` is `ADVI`, which is applicable as long as the `Model` is differentiable with respect to the *variational parameters*.

By default, when calling `vi(m, advi)`, Turing uses a *mean-field* approximation with a multivariate normal as the base distribution. [Mean-field](https://en.wikipedia.org/wiki/Mean-field_theory) as borrowed from mean-field theory in statistical physics refers here to the fact that it is assumed all the latent variables are *independent*. This is standard approach in ADVI (see [Automatic Differentiation Variational Inference (2016)](https://arxiv.org/abs/1603.00788)).

In the mean-field approximation, the parameters of the variational distribution are the mean and variance for each latent variable.

```julia
print(@doc(Variational.ADVI))
```

```julia
print(@doc(Variational.meanfield))
```

```julia
# ADVI
advi = ADVI(10, 1000)
q = vi(m, advi);
```

For such a small problem Turing's `NUTS` sampler is nearly as efficient as ADVI.

This is *not* the case in general. For very complex models `ADVI` produces very reasonable results in a much shorter time than `NUTS`.

One significant advantage of using `vi` is that we can sample from the resulting approximate posterior `q` with ease. In fact, the result of the `vi` call is a `TransformedDistribution` from Bijectors.jl, and it implements the Distributions.jl interface for a `Distribution`.


# Deep universal probabilistic programming and inference compilation


# Examples

```julia

```
