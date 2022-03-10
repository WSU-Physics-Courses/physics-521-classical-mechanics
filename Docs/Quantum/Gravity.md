---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.5
kernelspec:
  display_name: Python 3 (phys-521-2021)
  language: python
  name: phys-521-2021
---

```{code-cell} ipython3
import mmf_setup;mmf_setup.nbinit()
import logging; logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
```

# Falling Particle

Here we consider the quantum mechanics of a particle falling in a gravitational field:

\begin{gather*}
  \op{H} = \frac{\op{p}^2}{2m} + mg\op{z}.
\end{gather*}

Dimensionally, 

\begin{gather*}
  [\hbar] = \frac{MD^2}{T}, \qquad
  [m] = M, \qquad
  [g] = \frac{D}{T^2},
\end{gather*}

so we can choose units $\hbar = 2m = g/2 = 1$ so that

\begin{gather*}
  1 = \underbrace{2m}_{\text{mass}}
    = \underbrace{\left(\frac{\hbar^2}{2m^2g}\right)^{1/3}}_{\text{length}}
    = \underbrace{\left(\frac{2\hbar}{mg^2}\right)^{1/3}}_{\text{time}}
    = \underbrace{\left(\frac{\hbar^2 mg^2}{2}\right)^{1/3}}_{\text{energy}}
    = \underbrace{\left(\hbar 2m^2g\right)^{1/3}}_{\text{momentum}}.
\end{gather*}

In these units, the time-independent Schrödinger equation has the form:

\begin{gather*}
  \DeclareMathOperator{\Ai}{Ai}
  \DeclareMathOperator{\Bi}{Bi}
  -\psi''(z) + (z-E)\psi(z) = 0, \qquad
  \psi(z) = a\Ai(z-E) + b\Bi(z-E),
\end{gather*}

where the solution is expressed in terms of the Airy functions $y=\Ai(x)$ and $y=\Bi(x)$, which satisfy:

\begin{gather*}
  y'' = xy.
\end{gather*}

We can also find the Green function {cite:p}`Vallee:2010`:

\begin{gather*}
  G(z,z') = -\pi \begin{cases}
    \Ai(z')\bigl(\Bi(z) + \I\Ai(z)\bigr) & z \leq z',\\
    \Ai(z)\bigl(\Bi(z') + \I\Ai(z')\bigr) & z \geq z'
  \end{cases}, \\
  \left(\pdiff[2]{}{z} - z\right)G(z, z') = \delta(z-z')
\end{gather*}

This gives, for example, the steady state solution $G(z,0)$ for a continuous atom laser
with particles continually being injected at $z'=0$.  Note that we can add to $G(z, z')$
any solution of the homogeneous equation, so we can write:

:::{margin}
We only add $\Ai(z)$ to keep the boundary condition $n\rightarrow 0$ for $z \rightarrow \infty$.
:::
\begin{gather*}
  \psi(z) = G(z, 0) + \alpha \Ai(z),\\
  \Bigl(-\pdiff[2]{}{z} + z + \overbrace{\frac{1}{\psi(0)}}^{a}\delta(z)\Bigr)\psi(z).
\end{gather*}

This is the steady state solution with $E=0$ to the time-independent Schrödinger equation with a delta-function potential of strength $a = 1/\psi(0)$.  If this has an imaginary part, then we have a particle source of sink.  Inverting, we have:

\begin{gather*}
  \Bi(0) = \sqrt{3}\Ai(0) = \frac{\sqrt{3}}{3^{2/3}\Gamma(\tfrac{2}{3})},\\
  G(0,0) = -\pi\left(\Ai(0)\Bi(0) + \I \Ai^2(0)\right)
         = -\pi\left(\sqrt{3} + \I\right)\Ai^2(0),\\
  a^{-1} = \psi(0) = -\pi\Bigl(\sqrt{3} + \I\Bigr)\Ai^2(0) + \alpha \Ai(0),\\
  \alpha = \frac{1}{a\Ai(0)} + \pi\Bigl(\sqrt{3} + \I\Bigr)\Ai(0).
\end{gather*}

In the region $z < 0$ the solution will have the form:

\begin{gather*}
  \psi(z<0) = -\pi\Ai(0)\left(\sqrt{3}\Bigr)\Bi(z) + \I \Ai(z)\right) + \alpha \Ai(z),\\
   = -\pi\Ai(0)\sqrt{3}\left(
    \Bi(z) - \Bigl(1 + \frac{1}{a\sqrt{3}\pi\Ai^2(0)}\Bigr) \Ai(z)
  \right)
\end{gather*}

```{code-cell} ipython3
from ipywidgets import interact
from scipy.special import airy

hbar = 1
m = 0.5
g = 2.0

z = np.linspace(-10, 2, 500)
Ai0, dAi0, Bi0, dBi0 = airy(0)
Aiz, dAiz, Biz, dBiz = airy(z)

def G(x):
    Aix, dAix, Bix, dBix = airy(x)
    return (
        - np.pi * np.where(x < 0, Ai0 * (Bix + 1j * Aix), Aix * (Bi0 + 1j * Ai0))
    )

Gz = G(z)

@interact(ar=(-2,2,0.001), ai=(-2,2,0.001))
def go(ar=-np.sqrt(3)/4/np.pi/Ai0**2, ai=1/4/np.pi/Ai0**2):
    a = ar + ai*1j
    alpha = 1/a/Ai0 + np.pi * (np.sqrt(3) + 1j)*Ai0
    psi = Gz + alpha * Aiz
    n = abs(psi)**2
    n_wkb = 1/m/np.sqrt(-2*g*z)
    n_wkb *= n[0]/n_wkb[0]
    j = (-1j * hbar * np.gradient(psi, z) * psi.conj()).real
    fig, ax = plt.subplots()
    ax.plot(z, n, label='$n(z)$')
    ax.plot(z, j, ls='--', label='$j(z)=n(z)v(z)$')
    ylim = ax.get_ylim()
    ax.plot(z, n_wkb, ls=':', label='$n_{WKB}(z)$')
    ax.set(xlabel='$z$', ylabel="$n$, $j$", ylim=ylim)
    ax.grid(True)
    ax.legend();
```

$$
  n(z) \propto \frac{1}{p(z)} = \frac{1}{m\sqrt{2gz}}.
$$

From this it is clear that $a \propto \sqrt{3} - \I \propto e^{-\I\pi/6}$ is required
for a smooth solution (no oscillations).  Why, and why is this natural for an atom
laser?
