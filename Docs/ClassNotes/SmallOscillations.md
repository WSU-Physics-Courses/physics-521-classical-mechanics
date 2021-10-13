---
jupytext:
  extension: .md
  format_name: myst
  format_version: 0.13
  jupytext_version: 1.11.1
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
:cell_style: center
:hide_input: false

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
  = m\omega_0^2 \cos\theta + \frac{mh_0^2 \omega^2f_1^2}{4r^2}\sin^2\theta.
\end{gather*}

```{code-cell} ipython3
th = np.linspace(-2*np.pi, 2*np.pi, 100)
g = r = 1
hw = 2
fig, ax = plt.subplots()
for hw in [0, 2, 4]:
    V_eff = g/r*np.cos(th) + (hw/r)**2/2*np.sin(th)**2/4
    ax.plot(th, V_eff, label=fr'$h\omega/r\omega_0 = {hw}$')
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

```{code-cell} ipython3
from scipy.integrate import solve_ivp

m = g = r = 1.0
w0 = np.sqrt(g/r)

def h(t, h0, w, d=0):
   """Return the `d`'th derivative of `h(t)`."""
   if d == 0:
       return h0 * np.cos(w*t)
   elif d == 1:
       return w * h0 * np.sin(w*t)
   else:
       return -w**2 * h(t, h0=h0, w=w, d=d-2)

def compute_dy_dt(t, y, h0, w):
    theta, dtheta = y
    ddtheta = (g + h(t, h0=h0, w=w, d=2))/2 * np.sin(theta)
    return (dtheta, ddtheta)


hws = [1, 2, 3, 4]

h0 = 0.1
y0 = (0.5, 0)
T = 2*np.pi / w0
t_span = (0, 5*T)   # 5 oscillations

def fun(t, y):
    return compute_dy_dt(t=t, y=y, **args)

fig, ax = plt.subplots()

for hw in hws:
    args = dict(h0=h0, w=hw/h0)
    res = solve_ivp(fun, t_span=t_span, y0=y0, atol=1e-6, rtol=1e-6)
    ts = res.t
    thetas, dthetas = res.y
    ax.plot(ts/T, thetas, label=fr'$h\omega/r\omega_0 = {hw}$')
ax.set(xlabel=r'$t/T$', ylabel=r'$\theta$', ylim=(-y0[0], 2*y0[0]))
ax.legend()
```
