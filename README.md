# rate_xvalidation

Cross-validated Hanning-kernel smoothing for estimating point-process firing rates (spike trains, event counts) with data-driven bandwidth selection.

Implements the leave-one-out cross-validation rate estimator from:

> **Prerau M.J., Eden U.T.** "A General Likelihood Framework for Characterizing the Time Course of Neural Activity." *Neural Computation*, 23(10):2537–2566, 2011.

## Motivation — why cross-validation?

You have a binned spike train (or any binary event-count sequence) and you want to recover the underlying time-varying rate `λ(t)` without committing to a parametric model. The choice of kernel bandwidth is the key knob — too narrow and you overfit individual spikes; too wide and you smooth real structure away.

In the electrophysiology literature the bandwidth is almost always chosen ad hoc: the analyst picks a number that "looks right," with no statistical justification. This is a problem because the choice of bandwidth implicitly assumes how fast the neuron's rate can change, and a bad choice can invent, erase, or invert trends in the firing rate — with downstream consequences for any correlation or classification built on top of the estimate.

The obvious fix — maximize the likelihood of the spike train under the rate estimate — **doesn't work** for this problem. As the bandwidth shrinks toward zero the kernel places all its mass at the spike itself, driving the likelihood arbitrarily high. Pure ML bandwidth selection degenerates to "pick the smallest bandwidth you can," which overfits perfectly and predicts nothing.

The fix in Prerau & Eden 2011 is to **score each candidate bandwidth by its ability to predict spikes it hasn't seen**. For every time bin, recompute the rate estimate *excluding that bin's own count*, and score the held-out count under the Poisson likelihood of that estimate. Sum across all bins to get the leave-one-out cross-validated log-likelihood, and choose the bandwidth that maximizes it. Small bandwidths can no longer cheat by placing mass at the held-out spike, because that mass is removed from the kernel. Large bandwidths can no longer cheat by flattening out, because they will underpredict clusters of spikes.

This turns a model-selection problem with no principled solution into a standard optimization problem with a scalar objective.

## The algorithm

### 1. Notation

Let the spike count time series be `s_i = ΔN(t_i, t_{i+1}]` for `i = 1..N_T`, with uniform bin width `dt`. Let `K` be the candidate bandwidth (in number of bins — an odd integer). The rate estimator under a Hanning kernel `w(·, K)` is

```
λ(t_m, K) = (1 / ν) · Σ_i  w(t_m - t_i, K) · (s_i / dt)
```

with normalizer `ν = Σ_i w(t_i, K)` so the kernel integrates to 1.

### 2. The "notch" kernel trick

To compute the leave-one-out rate estimate `λ⁻(t_m, K)` at bin `m`, you want the weighted average of surrounding bins *excluding* the current bin. Naively that costs O(N²). The clever step in the paper: define a **notch kernel**

```
w⁻(t, K)  =  w(t, K)   if t ≠ 0
          =  0         if t = 0
```

— i.e., the Hanning kernel with a single zero punched out at the origin. Now a single full-sequence convolution `s * w⁻` simultaneously produces the leave-one-out estimate at *every* bin. The current bin's contribution is zeroed out by construction. That drops the per-bandwidth cost from O(N²) to O(N log N) and makes the algorithm practical for hour-long recordings.

The Hanning kernel itself is

```
w(t, K)  =  0.5 · (1 + cos(2πt / (K-1)))     for -K/2 < t ≤ K/2
         =  0                                  otherwise
```

Hanning over Gaussian because in discrete time the bandwidth `K` is directly the number of samples in the kernel — easy to interpret, easy to convert to ms via `K · dt · 1000`.

### 3. Cross-validated Poisson log-likelihood

With `λ⁻(t_m, K)` in hand at every bin, the leave-one-out CV log-likelihood under a Poisson point-process assumption is

```
log L_cv(K)  =  Σ_m  [ s_m · log(λ⁻(t_m, K) · dt)  -  λ⁻(t_m, K) · dt  -  log(s_m!) ]
```

This is the standard Poisson log-likelihood where each bin's count is scored against a rate computed *without that bin*.

### 4. Bandwidth selection

Evaluate `log L_cv(K)` across a broad range of candidate bandwidths. Pick

```
K_max  =  argmax_K  log L_cv(K)
```

This is a 1-D optimization over an integer grid — cheap, robust, no gradient required.

### 5. Final rate estimate

Once `K_max` is chosen, compute the final rate with the **full** Hanning kernel (no notch) at bandwidth `K_max`:

```
λ̂(t)  =  (s * w(·, K_max)) / ν
```

### 6. Confidence interval on the bandwidth

The paper derives a Fisher-information-based 95 % CI on `K_max` directly from the curvature of the CV log-likelihood at its peak:

```
CI  =  K_max  ±  2 · [ -∂²log L_cv(K_max) / ∂K² ]^(-1/2)
```

A wide CI tells you the data doesn't strongly distinguish between a range of bandwidths (e.g., a very sparse spike train). A narrow CI tells you the temporal structure of the spikes identifies a specific timescale.

## Entry point

```matlab
[estimate, kmax, loglikelihoods, bandwidths, CI] = cvkernel(spikecounts, dt)
```

| Input | Meaning |
|---|---|
| `spikecounts` | `1×N` vector of binned counts (required) |
| `dt` | bin width in seconds (required) |
| `range` | vector of candidate odd-integer bandwidths in bin units (default: `3:2:3·L` where `L = length(spikecounts)`) |
| `ploton` | if `true`, plots the estimate and the bandwidth likelihood curve (default: `false`) |

| Output | Meaning |
|---|---|
| `estimate` | `1×N` non-parametric rate estimate (Hz), computed at `K_max` |
| `kmax` | ML-chosen bandwidth `K_max` (in bin units) |
| `loglikelihoods` | `log L_cv(K)` evaluated at every candidate bandwidth |
| `bandwidths` | the candidate bandwidths evaluated |
| `CI` | 95 % Fisher-information confidence bounds on `K_max` (in bin units) |

## Implementation notes

- `cvkernel.m` is the entry point. It builds the notch kernel, runs the leave-one-out convolution per candidate bandwidth, computes `log L_cv(K)` per equation above, and returns the argmax plus Fisher CI.
- `kconv.m` is the convolution helper. It runs a `same`-size convolution and then re-weights the boundary samples so they use only the in-range kernel mass — without this, the estimate is biased downward at the start and end of the signal (the standard tapering artifact of finite-length convolutions).
- The range of candidate bandwidths defaults to `3:2:3·N` — odd integers only, because an odd length makes the Hanning kernel symmetric around the central sample. Narrower ranges can be passed for speed on very long signals.

## When to use this vs. alternatives

- **Use this** when you have a single-trial or summed-across-trials spike train and want a nonparametric rate estimate with a principled, reproducible bandwidth choice.
- **Use a state-space or GLM model instead** when you have covariates (history, stimulus, spatial position) and want a parametric fit. The cross-validated kernel smoother has no history dependence and no covariate support — that's the trade-off for its simplicity.
- **Use a PSTH with CV bandwidth** (also described in the paper, but not implemented here) if you specifically need a binned/histogram rate. The kernel smoother is almost always the better default: smooth, no arbitrary bin boundaries, and allows temporal weighting of spikes.

## Example

```matlab
% Simulate a sinusoidal firing rate + Poisson spikes
dt = 5/300;
t = dt:dt:5;
lambda = (sin(2.5*t) - min(sin(2.5*t))) * 150 + 5;   % 5–305 Hz swing
counts = poissrnd(lambda * dt);

% Recover
[est, kmax, ~, ~, CI] = cvkernel(counts, dt, [], true);

disp(['ML bandwidth: ', num2str(kmax), ' bins (', num2str(kmax*dt*1000), ' ms)'])
disp(['95% CI: [', num2str(CI(1)), ', ', num2str(CI(2)), '] bins'])
```

For a full demo with two ground-truth bandwidths plotted side-by-side, run:

```matlab
cvexample
```

## Files in this directory

| File | Role |
|---|---|
| `cvkernel.m` | Main entry point — runs the full CV-bandwidth selection and returns the rate estimate at `K_max` |
| `kconv.m` | Edge-corrected kernel convolution. Re-weights boundary samples so they use only in-range kernel mass, preventing tapering artifacts at the start/end of the signal |
| `cvexample.m` | Runnable demonstration — simulates an inhomogeneous Poisson spike train from a sinusoidal ground-truth rate and recovers it at two bandwidths for comparison |
| `Contents.m` | One-line summaries for MATLAB's `help rate_xvalidation` |
| `help.html` | Rendered documentation (from older publish pipeline) |

## Dependencies

- Base MATLAB only (uses `hanning` from Signal Processing Toolbox, but trivial to replace with `hann` / a hand-rolled window if the toolbox isn't available).
- Statistics Toolbox only needed if you adapt the demo to use `poissrnd` (which `cvexample.m` uses). `cvkernel` itself has no toolbox dependencies.

## Citation

If you use this, please cite:

> Prerau M.J., Eden U.T. "A General Likelihood Framework for Characterizing the Time Course of Neural Activity." *Neural Computation*, 23(10):2537–2566, 2011.
