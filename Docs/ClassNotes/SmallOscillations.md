---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3 (phys-521-2021)
  language: python
  name: phys-521-2021
---

Small Oscillations
==================

```{contents} Contents
:local:
:depth: 3
```

```{code-cell} ipython3
import sys
print(sys.executable)
print(sys.environ)
```

```{code-cell} ipython3
:tags: [hide-cell]

import mmf_setup;mmf_setup.nbinit()
import logging;logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
%pylab inline --no-import-all
```

## Kapitza Oscillator

In class we discussed the Kapitza oscillator, the full analysis of which is presented
in Chapter V, §30 of {cite:p}`LL1:1976`.  We consider a massless rigid pendulum of
length $r$ and angle $\theta$ from the vertical, driven with a vertical oscillation
$h_0\cos(\omega t)$:

\begin{gather*}
    x = r\sin\theta, \qquad z = r\cos\theta + h(t),\\
    \begin{aligned}
      L(\theta, \dot{\theta}, t) 
        &= \frac{m}{2}(\dot{x}^2 + \dot{z}^2) - mgz\\
        &= \frac{m}{2}\bigl(r^2\dot{\theta}^2 + \dot{h}^2(t) - 2rh(t)\dot\theta\sin\theta\bigr)
            -mgr\cos\theta - mgh(t),
    \end{aligned} \\
    p_{\theta} = \pdiff{L}{\dot{\theta}} 
               = mr^2\dot{\theta} - mrh(t)\sin\theta,\\
    \begin{aligned}
      \dot{p}_{\theta} &= \pdiff{L}{\theta}\\
      mr^2\ddot{\theta} - mr\dot{h}(t)\sin\theta - mrh(t)\dot{\theta}\cos\theta 
      &= - mrh(t)\dot\theta\cos\theta + mgr\sin\theta \\
      \ddot{\theta} - \frac{\ddot{h}(t)}{r}\sin\theta
      &= \frac{g}{r}\sin\theta.
    \end{aligned}
\end{gather*}

Thus, we end up with the following equation of motion:

\begin{gather*}
  \ddot{\theta} = \frac{g + \ddot{h}(t)}{r}\sin\theta.
\end{gather*}

Physically, the effect of $h(t)$ is simply to modify the effective acceleration due to
gravity, as expected.  Note that the sign here is correct -- $\theta=0$ corresponds to an
unstable vertically balanced pendulum.

We now specialize to the case $h(t) = h_0\cos\omega t$ where $\omega^2 \gg \omega_0^2 = g/r$ is much
larger than the natural oscillation length of the system.  This corresponds to a driving
force

\begin{gather*}
  f(t) = \frac{m}{r}\ddot{h}(t)\sin\theta  
  = \overbrace{\frac{mh_0\omega^2}{r}\sin\theta}^{f_1}
  \cos\omega t
\end{gather*}

where we have identified the coeffcient $f_1$ from Landau's analysis.

Performing a dimensional analysis we have:

\begin{gather*}
  [h_0] = [r] = L, \qquad
  [m] = M, \qquad
  [\omega] = [\dot{\theta}_0] = \frac{1}{T}, \qquad
  [\theta_0] = 1\\
  1 = \underbrace{m}_{\mathrm{mass}} 
    = \underbrace{r}_{\mathrm{distance}} 
    = \underbrace{\omega_0^{-1}=\sqrt{r/g}}_{\mathrm{distance}},\\
  \tilde{h}_0 = \frac{h_0}{r}, \qquad
  \tilde{\omega} = \frac{\omega}{\omega_0}.
\end{gather*}

Here we use units so that $m=r=g=1$, and dimensionless parameters $\tilde{h}_0$ and
$\tilde{\omega}$.  We will consider the limit $\tilde{\omega} \rightarrow \infty$.  From
the analysis derived in class, and from the book, we expect the effective potential in
this limit to have the form:

\begin{gather*}
  V_{\mathrm{eff}}(\theta) 
  = m\omega_0^2 \cos\theta + \frac{f_1^2}{4m\omega^2}
  = m\omega_0^2 \cos\theta + \frac{mh_0^2 \omega^2}{4r^2}\sin^2\theta.
\end{gather*}

```{code-cell} ipython3
th = np.linspace(-2*np.pi, 2*np.pi, 100)
g = r = 1
w0 = np.sqrt(g/r)
fig, ax = plt.subplots(figsize=(10, 5))
for hw2 in [0, 1, 2, 3, 4]:
    V_eff_m = w0**2*np.cos(th) + hw2/r**2*np.sin(th)**2/4
    ax.plot(th, V_eff_m, label=fr'$h_0\omega/r\omega_0 = \sqrt{{{hw2}}}$')
ax.legend()
ax.set(xlabel=r'$\theta$', ylabel=r'$V_{\mathrm{eff}}/m$');
```

Numerically we will implement these equations in terms of $\vect{y} = (\theta,
\dot{\theta})$, defining a function $\vect{f}(t, \vect{y})$ such that

\begin{gather*}
  \diff{}{t}\vect{y} = \vect{f}(t, \vect{y}) = \begin{pmatrix}
    \dot{\theta}\\
    \frac{g+\ddot{h}(t)}{r}\sin\theta
  \end{pmatrix}.
\end{gather*}

```{margin}
This code generates a figure demonstrating the behaviour of the Kapitza oscillator
starting at rest from $\theta_0=0.1$ for a variety of values $h_0\omega/r\omega_0$
crossing the threashold at $\sqrt{2}$.  The dashed lines show the simple harmonic motion
derived by expanding the effective potential to order $\theta^2$ and are quite accurate
for small initial $\theta_0$, but start to show deviations for $\theta_0\gtrapprox 0.2$.
```

```{code-cell} ipython3
from scipy.integrate import solve_ivp

m = g = r = 1.0
w0 = np.sqrt(g / r)


def h(t, h0, w, d=0):
    """Return the `d`'th derivative of `h(t)`."""
    if d == 0:
        return h0 * np.cos(w * t)
    elif d == 1:
        return -w * h0 * np.sin(w * t)
    else:
        return -w**2 * h(t, h0=h0, w=w, d=d - 2)


def compute_dy_dt(t, y, h0, w):
    theta, dtheta = y
    ddtheta = (g + h(t, h0=h0, w=w, d=2)) / r * np.sin(theta)
    return (dtheta, ddtheta)


h0 = 0.1
y0 = (0.1, 0)
T = 2 * np.pi / w0
t_span = (0, 5 * T)  # 5 oscillations

def fun(t, y):
    return compute_dy_dt(t=t, y=y, **args)

fig, ax = plt.subplots(figsize=(10, 5))

for h0w2 in [0, 1, 2, 3, 4]:
    args = dict(h0=h0, w=np.sqrt(h0w2) / h0)
    res = solve_ivp(fun, t_span=t_span, y0=y0, atol=1e-6, rtol=1e-6)
    ts = res.t
    thetas, dthetas = res.y
    l, = ax.plot(ts / T, thetas, 
                 label=fr'$h_0\omega/r\omega_0 = \sqrt{{{h0w2}}}$')

    # Add harmonic motion predicted by Kapitza formula
    w_h2 = (h0w2/(2*r**2*w0**2) - 1) * w0**2  # See formula below
    if w_h2 >=0:
        ax.plot(ts / T, y0[0]*np.cos(np.sqrt(w_h2) * ts), 
                ls='--', c=l.get_c())
ax.grid(True)
ax.set(xlabel=r'$t/T$', ylabel=r'$\theta$', ylim=(-y0[0], 4 * y0[0]))
ax.legend();
```

This seems to be consistent with the Kapitza result which says the threshold should be
at $h_0\omega/r\omega_0 = \sqrt{2}$.  We have also plotted the predicted harmonic motion
(see below) as dashed lines.  These start to disagree if we take $\theta_0 > 0.2$, so we
have used a fairly small value of $\theta_0=0.1$ here but you can play with the code.

### Checking Our Work

How might we check our numerical results?  The code is pretty simple, but we might have
made a mistake with the derivative computation.  We can use
[`np.gradient`](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html) as
a quick check.

```{margin}
I found a sign error in the derivative by writing this check.  It do not change the
results though.  To write these checks, it takes a bit of interactive playing to get the
correct value of `rtol` since the finite difference operation has intrinsic errors.
```

```{code-cell} ipython3
ts = np.linspace(1, 2)
for d in [0, 1, 2]:
    assert np.allclose(
        np.gradient(h(ts, h0=1.2, w=3.4, d=d), ts, edge_order=2), 
        h(ts, h0=1.2, w=3.4, d=d+1), 
        rtol=0.002)
```

Another possible check is the period of the resultant oscillations.  From the Kapitza
formula above, we have:

\begin{gather*}
  V_{\mathrm{eff}}(\theta) 
  = m\omega_0^2 \cos\theta + \frac{mh_0^2 \omega^2}{4r^2}\sin^2\theta
  = m\omega_0^2 \left(1 - \frac{\theta^2}{2}\right)
  + \frac{mh_0^2 \omega^2}{4r^2}\theta^2 + \order(\theta^4)\\
  = \frac{m\omega_0^2}{2}\left(\frac{h_0^2\omega^2}{2r^2\omega_0^2} - 1\right) \theta^2 
    + \text{const.} + \order(\theta^4)\\
\end{gather*}

Thus, we expect an oscillation frequency of:

\begin{gather*}
  \omega_h = \omega_0\sqrt{\frac{h_0^2\omega^2}{2r^2\omega_0^2} - 1}
\end{gather*}

We include this prediction in our plots above.


