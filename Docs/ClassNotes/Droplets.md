---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (phys-521)
  language: python
  name: phys-521
---

```{code-cell}
:tags: [hide-cell]

import mmf_setup;mmf_setup.nbinit()
from pathlib import Path
import os
FIG_DIR = Path(mmf_setup.ROOT) / '../Docs/_build/figures/'
os.makedirs(FIG_DIR, exist_ok=True)
import logging;logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
try: from myst_nb import glue
except: glue = None
```

# Droplets

Here we consider the problem of finding the solution for a spherical drop of some
material with a saturating equation of state $\mathcal{E}(n)$ that has a similar
structure to that in nuclei.  This problem has several interesting features and
challenges we would like to explore:

* Surface tension breaks the traditional convexity of thermodynamic relations for small
  systems. We start by formulating a model inspired by nuclear physics in
  {ref}`sec:droplet-model`, which we then manipulate to simplify the mathematical
  structure.
* Efficient numerical solutions requires solving boundary-value problems, which are
  somewhat finicky.  Here we explore some techniques for taming this problem.

The numerical work solving the BVP starts from {ref}`sec:droplet-asympt`.  If you are
most interested in how to solve this, please skip to that section.

(sec:droplet-model)=
## Model: A Saturating Equation of State

As a physical model, we choose a mean-field model for a quantum system like a
Bose-Einstein condensate, but with a saturating equation of state, whose energy density
(energy per unit volume) is
:::{margin}
Previous incarnations of these notes had $\alpha = 1$:
\begin{gather*}
  \mathcal{E}(n) = \epsilon_0 n_0\left(
    \left(\frac{n}{n_0}\right)^3 
    - 2\left(\frac{n}{n_0}\right)^2
  \right).
\end{gather*}
:::
\begin{align*}
  \mathcal{E}(n) &= \frac{E}{V} = 
  \epsilon_0 n
  \left(
    \kappa\left(\frac{n}{n_0}-1\right)^2 - 1
  \right)\\
  &=
  \epsilon_0 n_0 \left(
    \kappa\left(\frac{n}{n_0}\right)^3 
    - 2\kappa\left(\frac{n}{n_0}\right)^2 
    + (\kappa - 1)\left(\frac{n}{n_0}\right)
  \right),\\
  \mu(n) = \mathcal{E}'(n) &= 
  \epsilon_0 \left(
    \kappa\frac{n}{n_0}\left(3\frac{n}{n_0}-4\right)
    + \kappa - 1
  \right),\\
  &= 
  \epsilon_0 \left(
    \kappa\left(3\frac{n}{n_0} - 1\right)\left(\frac{n}{n_0} - 1\right)
     - 1
  \right),\\
  \frac{P'(n)}{n} = \mu'(n) = \mathcal{E}''(n) &= 
  6\epsilon_0 \kappa\left(\frac{n}{n_0} - \frac{2}{3}\right).
\end{align*}
:::{margin}
  We take our constants $n_0$ and $\epsilon_0$ as positive to make all the signs explicit,
  but note that in nuclear physics it is common to take the saturation energy as negative
  $\epsilon_0 < 0$.  Nuclear physics is more complicated due to the presence of both
  neutrons and charged protons, the latter of which are charged, but the essence is
  similar with $n_0\approx \SI{0.16}{fm^{-3}}$, $\epsilon_0 \approx
  \SI{-16}{MeV}$, and $K_0\approx \SI{250}{MeV}$ (the **isoscalar incompressibility**):
  \begin{gather*}
    \epsilon(n) \approx \epsilon_0 + \tfrac{K_0}{2}\left(\frac{n-n_0}{3n_0}\right)^2
  \end{gather*}
  See e.g. {cite}`Bulgac:2018` for a description.
  In our EoS, we have set $\kappa = -K_0/18\epsilon_0 \approx 0.87$ if one is trying to
  reproduced nuclear matter.
:::
As we shall see shortly, homogeneous matter in such a system will minimize the energy
per particle at the **saturation density** $n_0$ with **saturation energy** $-\epsilon_0$
per particle:
\begin{gather*}
  \epsilon(n) = \frac{\mathcal{E}}{n} = \frac{E}{N}
  =\epsilon_0\left(
    \kappa\left(\frac{n}{n_0}\right)^2 
    - 2\kappa\left(\frac{n}{n_0}\right)
    + (\kappa - 1)
  \right)\\
  =\epsilon_0\left(
    \kappa\left(\frac{n-n_0}{n_0}\right)^2 
    - 1
  \right)
  = -\epsilon_0 + \frac{K_0}{2}\left(\frac{n-n_0}{3n_0}\right)^2.
\end{gather*}

Our model will be to minimize the following energy functional for a "condensate
wavefunction" $\psi(\vect{x})$.  This gives rise to [Gross-Pitaevskii][GPE]--like equations
(GPE) with particle number density $n = \abs{\psi}^2$.
\begin{gather*}
  E[\psi] = \int \d^3{\vect{x}}\left(
    \frac{\hbar^2}{2m}\abs{\vect{\nabla}\psi}^2 + \mathcal{E}(n)
  \right), \qquad 
  N[\psi] = \int \d^3{\vect{x}} \abs{\psi}^2, \qquad
  n = \abs{\psi}^2.
\end{gather*}
Minimizing with respect to fixed particle number, we obtain the following [GPE][]-like
equation
\begin{gather*}
  \frac{\delta}{\delta \psi^{\dagger}(\vect{x})} E[\psi] - \mu N[\psi] = 
  \frac{-\hbar^2\nabla^2}{2m} + \Bigl(
    \mathcal{E}'\bigl(\abs{\psi(\vect{x})}^2\bigr) - \mu
  \Bigr)\psi(\vect{x}) = 0.
\end{gather*}

### Thermodynamic Limit
First we note some thermodynamic properties of this system.  Here is the equation of state:

```{code-cell}
:tags: [hide-input]

class EoSBase:
    def __init__(self, **kw):
        for key in kw:
            if not hasattr(self, key):
                raise ValueError(f"Unknown {key=}")
            setattr(self, key, kw[key])
        self.init()
    
    def init(self):
        pass

    def E(self, n, d=0):
        """Return the dth derivative of the energy density."""
        raise NotImplementedError
    
    def __call__(self, n):
        return self.E(n)
        
    def E_N(self, n):
        """Return the energy per particle."""
        return self.E(n)/n

    def mu(self, n):
        """Return the chemical potential."""
        return self.E(n, d=1)

    def P(self, n):
        """Return the pressure."""
        return self.mu(n) * n - self.E(n)


class EoS(EoSBase):
    e0 = 1.0
    n0 = 1.0
    kappa = 1.0
    
    def init(self):
        k = self.kappa
        self.E_poly = self.e0*self.n0*np.poly1d([k, -2*k, k-1, 0])
        self.n_poly = np.poly1d([self.n0, 0])
        self.mu_poly = self.E_poly.deriv() / self.n0
        self.P_poly = self.mu_poly * self.n_poly - self.E_poly
        self.mu_min = self.mu(self.mu_poly.deriv().roots[-1])
        super().init()
        
    def E(self, n, d=0):
        """Return the dth derivative of the energy density."""
        n_ = n / self.n0
        return self.E_poly.deriv(d)(n_) / self.n0**d
        
        
e0 = n0 = 1
fig, axs = plt.subplots(1, 4, figsize=(12, 2.5))
for i, k in enumerate([0.5, 1, 1.5]):
    E = EoS(kappa=k, e0=e0, n0=n0)
    E_N = E.E_N
    mu = dE = E.mu
    mu0 = mu(n0)
    n_unstable = np.linspace(0, n0)[1:]
    n_stable = np.linspace(n0, 2)
    n = n0*np.poly1d([1, 0])
    ax = axs[0]
    fmt_stable = f'-C{i}'
    fmt_unstable = f':C{i}'
    fmt_mixed = 'k--.'
    ax.plot(n_stable/n0, E(n_stable)/n0/e0, fmt_stable)
    ax.plot([0, 0], [0, 0.5], fmt_stable)
    ax.plot(n_unstable/n0, E(n_unstable)/n0/e0, fmt_unstable)
    ax.plot(n_unstable[[0, -1]]/n0, mu0*n_unstable[[0, -1]]/n0/e0, fmt_mixed)
    ax.set(ylim=(-1.5, 0.5), 
           xlabel="$n$ [$n_0$]", ylabel=r"$\mathcal{E}=E/V$ [$\epsilon_0 n_0$]")

    ax = axs[1]
    ax.plot(n_stable/n0, E(n_stable)/n_stable/e0, fmt_stable)
    ax.plot(n_unstable/n0, E(n_unstable)/n_unstable/e0, fmt_unstable)
    ax.plot([0, 0], [mu0/e0, 0.1], fmt_stable)
    ax.plot([0, 1], [mu0/e0, mu0/e0], fmt_mixed)
    ax.set(ylim=(-1.1, 0.0), 
           xlabel="$n$ [$n_0$]", ylabel=r"$E/N = \mathcal{E}/n$ [$\epsilon_0$]")

    ax = axs[2]
    ax.plot(n_stable/n0, mu(n_stable)/e0, fmt_stable)
    ax.plot([0, 0], [mu0/e0, -1.5], fmt_stable)
    ax.plot([0, 1], [mu0/e0, mu0/e0], fmt_mixed)
    ax.plot(n_unstable/n0, mu(n_unstable)/e0, fmt_unstable)
    assert np.allclose(E.mu_min, -e0*(1+k/3))  # Check mu_min formula
    ax.set(ylim=(-1.5, 0.5), 
           xlabel="$n$ [$n_0$]", ylabel=r"$\mu = \mathcal{E}'(n)$ [$\epsilon_0$]")

    ax = axs[3]
    ax.plot(mu(n_stable)/e0, E.P(n_stable)/n0/e0, fmt_stable, label=f"$\kappa={k:.1f}$")
    ax.plot([-1.5, mu0/e0], [0, 0], fmt_stable)
    ax.plot([mu0/e0], [0], fmt_mixed)
    ax.plot(mu(n_unstable)/e0, E.P(n_unstable)/n0/e0, fmt_unstable)
    ax.set(xlim=(-1.5, 0.5), ylim=(-0.6, 1.1),
           xlabel="$\mu$ [$\epsilon_0$]", ylabel=r"$P=\mu n - \mathcal{E}$ [$\epsilon_0 n_0$]")

axs[3].legend()
plt.tight_layout()
```

This describes a saturating system with an unstable branch between $n_0 \in [0, 1]$.  In
this region, one will have a mixed phase through the Maxwell construction (dashed black
line) between the two phases shown: one at $n=0$ corresponding to the vacuum, and the
other at $n=n_0$.  Note that the Maxwell construction reduces to a point in the $(\mu,
P)$ plot (grand-canonical ensemble) at $\mu_0 = -\epsilon_0$.

In infinite matter, or an infinitely large droplet, the surface will have no curvature, so the
pressures of the two phases must match.  In this case $P=0$ in both the vacuum and the
saturating homogeneous phase with density $n_0$ and chemical potential $\mu_0 < 0$.
*Note: until $\mu_0 < -mc^2$, the vacuum is stable (no anti-particles are produced), and so
can be in chemical equilibrium for any negative $\mu < 0$.*

### Finite Droplets

In a finite droplet, the surface is curved, and the surface tension will increase the
pressure inside the droplet, causing $n>n_0$ and increasing $\mu > 
\mu_0$.  Larger and larger surface tension will correspond to greater curvature --
i.e. smaller droplets.  Thus, as $\mu$ increases towards zero, the size of the droplet
will shrink.  This is slightly paradoxical since one typically associates increasing the
chemical potential with increasing the overall particle number.  In infinite matter,
this arises from the convexity of the energy $E(N)$:
\begin{gather*}
  \mu(N) = E'(N), \qquad \mu'(N) = E''(N) \geq 0.
\end{gather*}
Finite droplets break this convexity.

To get an idea of how this works, consider a spherical droplet with a thin surface of
tension $\sigma$. The surface has an energy
$E_{\sigma}(r) = 4\pi \sigma r^2$ proportional to the area of the sphere.  This leads to
a compressional force $F = E_{\sigma}'(r) = 8 \pi \sigma r$ and a pressure (force per
unit area) $P = F/4\pi r^2 = 2\sigma /r$.  This should equal the pressure in the droplet:
\begin{gather*}
  P = \frac{2\sigma}{r} = \mu n - \mathcal{E} = n \mathcal{E}'(n) - \mathcal{E}(n).
\end{gather*}

This can also be derived using the compressible liquid drop model (CLDM).
Let $n = n_i > n_0$ be the density inside the droplet, and $n_0 = 0$ be the density
outside. The energy $E$ and particle number $N$ of a droplet of radius $r$ are:
\begin{gather*}
  N = nV = n\frac{4}{3}\pi r^3,\\
  E(N, V) = \frac{4}{3}\pi r^3\mathcal{E}(n) + 4\pi r^2 \sigma
          = V \mathcal{E}\left(\frac{N}{V}\right) + \left(36\pi\right)^{1/3} \sigma V^{2/3}.
\end{gather*}
The size of the droplet will minimize the energy for fixed $N$:
\begin{gather*}
  \pdiff{E(N, V)}{V} 
  = \underbrace{\mathcal{E}\left(\frac{N}{V}\right) 
                - \frac{N}{V}\mathcal{E}'\left(\frac{N}{V}\right)}_{-P(n)}
     + 2\sigma\underbrace{\left(\frac{4}{3\pi V}\right)^{1/3}}_{1/r},\\
     P(n) = \frac{2\sigma}{r}.
\end{gather*}
This gives the same relationship between the surface tension and the pressure we found above.

```{code-cell}
:tags: [hide-input]

fig, ax = plt.subplots(figsize=(5,3))

E = EoS(kappa=1)
sigma = 0.2
for i, R in enumerate([1, 2, 3]):
    rs = np.linspace(0, R, 1000)[1:]
    Vs = 4/3*np.pi * rs**3
    ns = np.array([(E.P_poly - _P).roots[0] for _P in 2*sigma / rs]).real
    Es = E(ns)*Vs + sigma*4*np.pi * rs**2
    Ns = ns*Vs

    V = 4/3*np.pi*R**3

    Nmax = Ns[-1]
    ns_ = np.concatenate([Ns/V, [Nmax/V]])
    Es_ = np.concatenate([Es/V, [E(Nmax/V)]])
    ax.plot(ns_, Es_, f"-C{i+1}", label=f"${R=}$")

ns = np.linspace(E.n0, 2.1)
ax.plot(ns, E.E(ns), '-C0')
ns = np.linspace(0, E.n0)
ax.plot(ns, E.mu(E.n0)*ns, '--k')
ax.set(xlabel="n", ylabel="$\mathcal{E}(n)$",
       xlim=(-0.1, 2.1), ylim=(-1.2, 1.0),
       title=rf"Droplet with $\sigma={sigma:.4g}$")
ax.legend()
plt.tight_layout()
```

What about a bubble?  First we consider a fixed total volume of radius $R$.  Now the
volume of matter is that excluded by the bubble:
\begin{gather*}
  N = nV = n\frac{4}{3}\pi (R^3 - r^3),\qquad
  r = R\left(1 - \frac{3V}{4\pi R^3}\right)^{1/3},\\
  E(N, V) = \frac{4}{3}\pi (R^3 - r^3)\mathcal{E}(n) + 4\pi r^2 \sigma
          = V \mathcal{E}\left(\frac{N}{V}\right) 
          + 4\pi \sigma R^2 \left(1 - \frac{3V}{4\pi R^3}\right)^{2/3},\\
  \pdiff{E(N, V)}{V}
  = \underbrace{\mathcal{E}\left(\frac{N}{V}\right) 
    - \frac{N}{V}\mathcal{E}'\left(\frac{N}{V}\right)}_{-P(n)}
    - \frac{2\sigma}{R} \underbrace{\left(1 - \frac{3V}{4\pi R^3}\right)^{-1/3}}_{R/r},\\
  P(n) = - \frac{2\sigma}{r}.
\end{gather*}
Thus, unless our equation of state supports negative pressure OR the surface tension is
negative, then we cannot support bubbles.

:::{note}

We might worry that by restricting our attention to spherical drops, we are missing some
sort of mixed phase.  To generalize the problem, we note that we can perform a
constrained minimization over all possible states $\psi$, which we do with a Legendre transform:
\begin{gather*}
  E(N) = \min_{\psi}E[\psi] \quad \text{where} \quad N[\psi] = N,\\
  \frac{\delta}{\delta\psi}(E[\psi] - \mu N[\psi]) = 0,\qquad
  P(\mu) = \mu N[\psi] - E[\psi].
\end{gather*}
If there are multiple states that satisfy the stationarity condition, the one which
minimize the energy $E[\psi]$ should be chosen.  This guarantees that $P(\mu)$ is
convex, but not that $E(N)$ is convex as the following example demonstrates.

Consider the following artificial problem with $\psi = (n, \theta)$:
\begin{gather*}
  E[n, \theta] = \cos^2 n \cos\theta, \qquad N[n, \theta] = n,\\
  E(N) = -\cos^2 N.
\end{gather*}
The stationarity condition gives
\begin{gather*}
  \mu = \sin 2n \cos\theta, \qquad
  0 = -\cos^2 n \sin\theta.
\end{gather*}
Of the possible solutions $\theta \in [0, \pi]$, only $\theta = 0$ minimizes $E$, so we
have $\mu = - \sin 2 n$
\begin{gather*}
  P(\mu) = E[n, 0] - \mu N[n, 0] = -\cos^2 n - \mu n 
\end{gather*}
:::

```{code-cell}
N = np.linspace(0, 6*np.pi, 1000)
mu = - np.sin(2*N)
E = - np.cos(N)**2
P = mu*N - E
plt.plot(mu, P)
```

### Units
:::{margin} Dimensions
\begin{gather*}
  [\hbar] = \frac{ML^2}{T}\\
  [2m] = M\\
  \left[\frac{\hbar^2}{2m}\right] = \frac{ML^4}{T^2}\\
  [n_0] = \frac{1}{L^3}\\
  [\epsilon_0] = \frac{ML^2}{T^2}\\
  [\kappa] = 1
\end{gather*}
Note that we will miss physics if we try to set $\hbar^2/2m = n_0 = \epsilon_0 = 1$.
:::
To simplify our formula and numerical work, we choose a convenient set of units.  We
simplify the differential equation to the following form
\begin{gather*}
  \frac{\hbar^2}{2m}\nabla^2 \psi 
  = \left(\mathcal{E}'(\psi^2) - \mu\right)\psi,\qquad
  \frac{\mathcal{E}'(n)}{\epsilon_0} = 
    \kappa\frac{n}{n_0}\left(3\frac{n}{n_0} - 4\right)
    + \kappa - 1.
\end{gather*}
The EoS suggests using the following units $\epsilon_0 = n_0 = 2m = 1$.  To restore
results, use the appropriate factors of unity
\begin{gather*}
  1 
  = \underbrace{2m}_{\text{mass}}
  = \underbrace{n_0^{-1/3}}_{\text{length}}
  = \underbrace{\sqrt{\frac{2m}{\epsilon_0n_0^{2/3}}}}_{\text{time}}
  = \underbrace{\epsilon_0}_{\text{energy}}
  = \underbrace{n_0}_{\text{density}} 
\end{gather*}
The ground state can be chosen to be real, so we shall only consider real functions
$\psi$ and write $n = \psi^2$ for simplicity.

### Homogeneous States

First we consider the homogeneous states where $\nabla^2 \psi = 0$.  This will be the
asymptotic form of our solutions far from the center of a droplet or bubble and requires
\begin{gather*}
  \mathcal{E}'(n) = \mu.
\end{gather*}
For our equation of states,
\begin{gather*}
  \mu = \mathcal{E}'(n) \geq \mathcal{E}'(2n_0/3)
  = -\epsilon_0 \left(1+\frac{\kappa}{3}\right) = \mu_{\text{min}}.
\end{gather*}
For a smaller $\mu$, the only asymptotic solution is the vacuum $n=0$.  Conversely,
the vacuum $n=0$ can only exist if $\mu \leq \mathcal{E}'(0) = \epsilon_0(\kappa - 1)$.
Thus, droplets or bubbles require
\begin{gather*}
  -\left(1+\frac{\kappa}{3}\right) < \frac{\mu}{\epsilon_0} < \kappa - 1.
\end{gather*}

### Spherical Solutions

:::{margin}
The Laplacian for spherical solutions (i.e. no angular momentum $l=0$) is
\begin{gather*}
  \nabla^2 \psi = \frac{1}{r^{d-1}}\pdiff{}{r}\left(r^{d-1}\pdiff{\psi}{r}\right)\\
  = \psi''(r) + \frac{d-1}{r}\psi'(r)
\end{gather*}
:::
Switching to spherical coordinates and our dimensionless units, we have the following
equations:
\begin{gather*}
  \frac{\hbar^2}{2m}\left(\psi'' + \frac{2}{r}\psi'\right) 
  = \bigl(\mathcal{E}'(\psi^2) - \mu\bigr)\psi, \qquad
  \mathcal{E}'(n) = \kappa (3n-1)(n-1) - 1,\\
  \psi'(0) = 0, \qquad
  \psi(\infty) = 0.
\end{gather*}
Although there is an apparent singularity at $r=0$, from physical considerations,
nothing special happens at the origin, so we expect $\psi(n)$ to be smooth for small
$n$.

### Spherical Bubbles

:::::{toggle} Bubbles

Another possibility is a spherical bubble with $\psi(\infty) = \sqrt{n_0}$:
\begin{gather*}
  \frac{\hbar^2}{2m}\left(\psi'' + \frac{2}{r}\psi'\right) 
  = \bigl(\mathcal{E}'(\psi^2) - \mu\bigr)\psi, \qquad
  \mathcal{E}'(n) = ???,\\
  \psi'(0) = 0, \qquad
  \psi(\infty) = \sqrt{n_0}.
\end{gather*}

We are looking for solutions that are finite at $r=0$, but which fall to zero as
$r\rightarrow \infty$.


To study spherical droplets, we need $\psi(\infty) \rightarrow 0$ so that $n(\infty) =
0$.  To elucidate the asymptotic   The asymptotic 
, which requires 

It will turn out that droplets require $\mu < \mathcal{E}'(0)$, and so we will rewrite
the equation for numerical work as follows:
\begin{gather*}
  \frac{\hbar^2}{2m(\mathcal{E}'(0) - \mu)}\nabla^2 \psi - \psi
  = V(\psi^2)\psi,\qquad
  V(n) = \frac{\mathcal{E}'(\psi^2) - \mathcal{E}'(0)}{\mathcal{E}'(0) - \mu}.
\end{gather*}
We can then choose the coordinate $x \propto r$ to absorb the constant:
\begin{gather*}
  x = \frac{r}{\hbar}\sqrt{2m(\mathcal{E}'(0) - \mu)}.
\end{gather*}
This gives us our working equation:
\begin{gather*}
  (\nabla_{x}^2 - 1)y = V(y^2)y.
\end{gather*}




The qualitative properties of the system are described by the following dimensionless
constants:
\begin{gather*}
  \kappa, \qquad
  \epsilon = \frac{2m \epsilon_0}{\hbar^2n_0^{2/3}}.
\end{gather*}
Scaling everything out, we have the following differential equation:
\begin{gather*}
  \nabla^2 \psi 
  = \bigl(\mathcal{E}'(\psi^2) - \mu\bigr)\psi, \qquad
  \mathcal{E}'(n) = 
  \epsilon \bigl(\kappa (3n-1)(n-1) - 1\bigr).
\end{gather*}




First we note a couple of trivial solutions: first, we have homogenous solutions
$\psi(r) = 0$ (corresponding to the vaccum), and $\psi = \sqrt{n_0}$ where $\mathcal{E}'(n_0) =
\mu$ is also a solution.  For our form, $n_0 = (1 \pm \sqrt{1+4\mu})/2$

:::::

As discussed above, droplets require $\mu < \mathcal{E}'(0)$, so we can write
\begin{gather*}
  \frac{\hbar^2}{2m}\left(\psi'' + \frac{2}{r}\psi'\right) = 
  \Bigl(\mathcal{E}'(\psi^2)\psi - \mathcal{E}'(0)\Bigr)\psi + ({\mathcal{E}'(0) - \mu})\psi,\\
  \frac{\hbar^2}{2m({\mathcal{E}'(0) - \mu})}\left(\psi'' + \frac{2}{r}\psi'\right) = 
  \underbrace{\frac{\mathcal{E}'(\psi^2) - \mathcal{E}'(0)}
              {\mathcal{E}'(0) - \mu}}_{V_{\mu}(\psi^2)}\psi + \psi
\end{gather*}
:::{margin}
I am somewhat puzzled here by the lack of dependence on the incompressibility $\kappa$?  I
supposed that the compressibility might determine the size of the wall, which is
compensated for by the scaling of the radius?  It is still surprising to me.
:::
:::{margin}
\begin{gather*}
  \mathcal{E}'(n) = \kappa (3n-1)(n-1) - 1\\
  \mathcal{E}'(0) = \kappa - 1\\
  \mu_\min = -\left(1 + \frac{\kappa}{3}\right)\\
  \alpha = \frac{4\kappa/3}
                {\kappa - 1 - \mu}\\
  \mu_\min < \mu < -1\\
\end{gather*}
:::
where the coefficient on the left is positive and can be absorbed into a redefinition of
$x \propto r$, and we can scale out the dependence on $\mu$ into a single parameter $\alpha$:
\begin{gather*}
  x = \frac{r}{\hbar}\sqrt{2m(\mathcal{E}'(0) - \mu)}\\
  V(n) = \frac{\mathcal{E}'(n) - \mathcal{E}'(0)}
              {\mathcal{E}'(0) - \mu_\min}, \qquad
  \alpha = \frac{\mathcal{E}'(0) - \mu_\min}{\mathcal{E}'(0) - \mu}.
\end{gather*}
:::{important}
This gives us our working equation, where we use $y(x) = \psi(r)$ to emphasize this change:
\begin{gather*}
  y'' + \tfrac{2}{x}y' - y = \alpha V(y^2)y, \qquad
  V(n) = \tfrac{3}{4}n(3n-4),\\
  y'(0) = 0, \qquad y(\infty) = 0, \qquad
   \frac{4}{3} < \alpha.
\end{gather*}
where the bounds and form of $V(n)$ are for our EoS.
:::

(sec:droplet-num1)=
## Numerical Attempt 1

We start with a brute-force solution of the problem as written:

```{code-cell}
:tags: [hide-input]

from scipy.integrate import solve_bvp

class BVP1(EoS):
    hbar2_2m = 1.0
    alpha = 2.0
    
    def V(self, n):
        return 3/4*n*(3*n-4)
    
    def rhs(self, x, q):
        """Return dq/dx.  Note q = (y, dy)."""
        y, dy = q
        ddy = (1 + self.alpha * self.V(y**2))*y - 2*dy/x
        dq = np.asarray([dy, ddy])
        return dq
    
    def bc(self, qa, qb):
        """Return the boundary conditions."""
        ya, dya = qa
        yb, dyb = qb
        bc = [dya, yb]
        return np.asarray(bc)
    
    def get_initial_guess(self, sol=None, k=1):
        """Return `(xs, ys, dys)` for an initial guess.
        
        Note: this function needs self.alpha, self._x0 and self._x1.
        """
        alpha = self.alpha
        x0, x1 = self._x0, self._x1
        xs = np.linspace(x0, x1)
        if sol is None:
            ys = 1/np.cosh(k*xs)
            dys = k*np.sinh(k*xs)/np.cosh(k*xs)**2
        else:
            (ys, dys) = sol.sol(xs)

        return map(np.asarray, (xs, ys, dys))
    
    def solve(self, sol0=None, x0=0.01, x1=20.0, alpha=None, k=1, **kw):
        """Return the solution `sol` to the BVP.
        
        Arguments
        ---------
        sol0 : Solution, None
            A previous solution to use as a starting point.
        x0, x1 : float
            Inner and out radii for boundary conditions.
        mu : float
            Chemical potetial.
        """
        self._x0, self._x1 = x0, x1
        if alpha is not None:
            self.alpha = alpha
        
        xs, ys, dys = self.get_initial_guess(sol=sol0, k=k)
        y0s = np.asarray([ys, dys])
        sol = solve_bvp(self.rhs, self.bc, x=xs, y=y0s, **kw)
        
        mu = self.mu(0) - (self.mu(0) - self.mu_min) / self.alpha
        x, (y, dy) = sol.x, sol.y
        sol.alpha = self.alpha
        sol.mu = mu
        sol.r = x/np.sqrt((self.mu(0) - mu)/self.hbar2_2m)
        sol.n = y**2
        
        return sol
    
    def find_sols(self, alpha0=2, dalphas=[0.2, -0.1], n_min=0.1, N=10000):
        """Starting from alpha0, iterate to try to find as many sols as possible."""
        alpha = alpha0
        sol_ = self.solve(alpha=alpha)
        sols = [sol_]
        alphas = [alpha]
        for dalpha in dalphas:
            sol0 = sol_
            alpha = alpha0
            for _N in range(N):
                while abs(dalpha) > 1e-8 and 4/3 < alpha:
                    with np.errstate(all="ignore"):
                        # Suppress floating point warnings.
                        sol = b.solve(sol0=sol0, alpha=alpha+dalpha)
                    if sol.success and sol.n[0] > n_min:
                        break
                    dalpha /= 2
                if not sol.success or sol.n[0] < n_min:
                    break
                sol0 = sol
                alpha = b.alpha
                sols.append(sol)
                alphas.append(alpha)
        inds = np.argsort(alphas)
        alphas = np.array(alphas)[inds]
        sols = np.array(sols)[inds]
        return alphas, sols


b = BVP1()
sol = b.solve()
r, n = sol.r, sol.n
fig, ax = plt.subplots(figsize=(3, 2))
ax.plot(r, n)
ax.set(xlabel="$r$", ylabel="$n$", title=rf"$\alpha={sol.alpha:.2g}$, $\mu={sol.mu:.2g}$");
```

We see that this works, but it is very sensitive to the initial guess.  Here we start
with this solution, and then slowly increase and decrease $\alpha$.  Note:
we must take many small steps, otherwise the solution does not converge, however, the
general procedure is sound.

```{code-cell}
:tags: [hide-input]

b = BVP1()
alphas, sols = b.find_sols()

fig, ax = plt.subplots(figsize=(3, 2))

for alpha, sol in zip(alphas, sols):
    r, n = sol.r, sol.n
    ax.plot(r, n)
ax.set(xlim=(0, 20), 
       xlabel="$r$", ylabel="$n$", 
       title=rf"${alphas[0]:.2f} \leq \alpha \leq {alphas[-1]:.2f}$");
```

From these solutions, we can estimate the surface tension $\sigma$ as follows from the
formula discussed above:
\begin{gather*}
  P = \frac{2\sigma}{r} = \mu n - \mathcal{E} = n \mathcal{E}(n) - \mathcal{E}(n).
\end{gather*}
In the following, we estimate the radius of the surface as the point where $n = n_0/2$,
then plot $\sigma \approx P(n_0) r/2$

```{code-cell}
:tags: [hide-input]

from scipy.interpolate import InterpolatedUnivariateSpline

rs = np.array([sol.r[np.argmin(abs(sol.n - 0.5))] for sol in sols])
mus = np.array([sol.mu for sol in sols])
rs = []
for sol in sols:
    if sol.n.max() > 0.5:
        _r = InterpolatedUnivariateSpline(sol.r, sol.n-0.5).roots()[0]
    else:
        _r = 0
    rs.append(_r)
rs = np.asarray(rs)

fig, ax = plt.subplots(figsize=(3, 2))
n0 = b.n0
E0 = b.E(n0)
ax.plot(rs, (mus*n0 - E0)*rs/2)
#plt.axhline([0.5], ls=":", c="y")
ax.set(xlabel="$r$", ylabel=r"$\sigma \approx P(n_0)r/2$");
```

(sec:droplet-asympt)=
## Asymptotic Behaviour

A good starting point is to consider the asymptotic behavior of droplets.  For large $r\rightarrow
\infty$, we expect that we can neglect the second term, and are looking for solutions
that vanish $\psi \rightarrow 0$.  
Noting that $V(y^2) \propto y^2$, we can probably neglect the right-hand-side as $x \rightarrow
\infty$ since it scales as $y^3$.  If we also neglect the $2y'/x$ term, then we have
simply $y'' = y$, which implies
\begin{gather*}
  y(x) \rightarrow A e^{-x}, \qquad
  y'(x) \rightarrow -y(x).
\end{gather*}
:::{margin}
The [spherical Bessel functions][], i.e. $j_0(x)= \sinc x$,  satisfy
\begin{gather*}
  y'' + \frac{2y'}{x} + \left(1 - \frac{n(n+1)}{x^2}\right)y = 0.
\end{gather*}
Our solution follows from replacing $x\rightarrow \I x$: i.e. $\sinh(x)/x$ and
$\cosh(x)/x$.  Combining these to get the appropriate boundary condition gives our
solution.
:::
We can do a bit better by nothing that
\begin{gather*}
  y(x) \rightarrow A\frac{e^{-x}}{x}, \qquad
  y'(x) \rightarrow -(1+x^{-1})y(x),
\end{gather*}
satisfies $y'' + 2y'/x - y = 0$.  Neglecting the $\alpha V(y^2)y$ thus requires
\begin{gather*}
  \abs{y^2} \ll 1, \qquad A^2 e^{-2x} \ll x.
\end{gather*}
The asymptotic forms give better boundary conditions at large $x$, which will reduce
truncation errors.

At the origin, we must have that $y' \rightarrow 0$ at least as fast a $x$, otherwise
the second term will diverge.  Since we physically expect the solution to be smooth at
the origin, we expect we can expand it as a series in $x$.  Expanding to second lowest
order, we have:
\begin{gather*}
  y(x) = y_0 + \frac{y_0''}{2}x^2 + O(x^3),\qquad
  y''(x) + \frac{2y'(x)}{x} = 3y_0'' + O(x), \qquad
  3y''_0 = y_0\bigl(1+\alpha V(y_0^2)\bigr).
\end{gather*}
This allows us to refine the boundary conditions at finite $x$.
\begin{gather*}
  y'(0) = \frac{xy(0)}{3}\bigl(1 + \alpha V\bigl(y(0)^2\bigr), \qquad
  y'(\infty) + (1+x^{-1})y(\infty) = 0.
\end{gather*}

(sec:droplet-num2)=
## Numerical Attempts 2 and 3

We now include these refined boundary conditions:
* `BVP2`: $y'(x_1) + y(x_1) = 0$.
* `BVP3`: $y'(x_1) + (1+x_1^{-1})y(x_1) = 0$.

```{code-cell}
:tags: [hide-input]

from scipy.integrate import solve_bvp

class BVP2(BVP1):
    def bc(self, qa, qb):
        """Return the boundary conditions."""
        ya, dya = qa
        yb, dyb = qb
        xa, xb = self._x0, self._x1
        bc = [
            3*dya - xa*ya * (1 + self.alpha * self.V(ya**2)),
            yb + dyb
        ]
        return np.asarray(bc)


class BVP3(BVP1):
    def bc(self, qa, qb):
        """Return the boundary conditions."""
        ya, dya = qa
        yb, dyb = qb
        xa, xb = self._x0, self._x1
        bc = [
            3*dya - xa*ya * (1 + self.alpha * self.V(ya**2)),
            dyb + (1+1/xb)*yb
        ]
        return np.asarray(bc)

b = BVP2()
sol = b.solve()
r, n = sol.r, sol.n
fig, ax = plt.subplots(figsize=(3, 2))
ax.plot(r, n)
ax.set(xlabel="$r$", ylabel="$n$", title=rf"$\alpha={sol.alpha:.2g}$, $\mu={sol.mu:.2g}$");
```

```{code-cell}
:tags: [hide-input]

b = BVP3()
alphas, sols = b.find_sols()

fig, ax = plt.subplots(figsize=(3, 2))

for alpha, sol in zip(alphas, sols):
    r, n = sol.r, sol.n
    ax.plot(r, n)
ax.set(xlim=(0, 20), 
       xlabel="$r$", ylabel="$n$", 
       title=rf"${alphas[0]:.2f} \leq \alpha \leq {alphas[-1]:.2f}$");
```

This shows that refining the boundary conditions does not improve the overall
convergence if the guess is not good, however, it improves the accuracy, as we shall now
see.

:::{margin}
Since we do not have an analytic solution, we use the solution with smallest $x_0$ and
largest $x_1$ as the "truth" and compute the errors relative to this.  This will
generate convergence artifacts near these endpoints, but is usually sufficient for
checking convergence in the middle.  Here we max out the tolerance of the solver (which
we increase), so we do not notice these artifacts.
:::
### Error Analysis

The main improvement is with the boundary conditions, which improve the rate of
convergence, allowing us to solve the problem on a restricted domain.x

:::{important}

You should check these relationships **quantitatively**.  The errors should scale as
expected: this provides a very strong and invaluable check on aspects of your code.
:::

In our case, we expect the following three sources of errors due to the boundary
conditions at $x_1$, which we obtain by keeping the first neglected term:
\begin{gather*}
  y'' + \frac{2y'}{x} - y = \alpha V(y^2)y \approx -3\alpha y^3.
\end{gather*}
For estimates, we use the asymptotic solution, though dropping the denominator would
also give approximately correct results:
\begin{gather*}
  y(x) \rightarrow A\frac{e^{-x}}{x}.
\end{gather*}
If the coefficient $A$ cannot be easily estimated (i.e. from a solution), then you can
choose it manually to roughly match the actual errors seen.  The important thing is that
the slope of the scaling is correct:
\begin{align*}
   y_1(x_1) &= 0: 
     & \text{err}_1 
     &\sim y(x_1) 
      \sim A\frac{e^{-x_1}}{x_1}\\
   y_2(x_1) &= -y'_2(x_1): 
     & \text{err}_2 
     &\sim \frac{y'(x_1)}{x_1} 
      \sim A\frac{e^{-x_1}}{x_1}\\
   y_3(x_1) &= -(1+x_1^{-1})y'_3(x_1): 
     & \text{err}_3 
     & \sim V(y^2)y 
       \sim y^3(x_1) \sim A^3\frac{e^{-3x_1}}{x_1^3}.
\end{align*}

:::{margin}
Originally when I tested my code, I found that `BVP2` and `BVP3` had the same errors.
This led me to discover a typo in my coding of the boundary condition.
:::
```{code-cell}
:tags: [hide-input]

from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 2)
fig = plt.figure(figsize=(7, 6))

ax = fig.add_subplot(gs[0, :])
ax0 = fig.add_subplot(gs[1, 0])
ax1 = fig.add_subplot(gs[1, 1])

b0 = BVP2(alpha=1.6)
sol0 = b0.solve(k=0.33)
assert sol0.success

x0s = 10**np.linspace(0.5, -3)
x1s = np.linspace(8, 30)
x_test = np.linspace(0, x1s.min())

ax.plot(sol0.x, sol0.n)
ax.plot(x_test, sol0.sol(x_test)[0]**2, lw=2)
ax.set(xlabel='$x$', ylabel='$n$')

kw = dict(tol=1e-10, max_nodes=10000)

# Estimate coefficient A
x1 = 20.0
A = sol0.sol(x1)[0]/(np.exp(-x1)/x1)
y1 = A*np.exp(-x1s)/x1s
err1_est = [
    abs(y1),
    abs(y1)**2,
    abs(y1)**3
]

for i, _BVP in enumerate([BVP1, BVP2, BVP3]):
    b = _BVP(alpha=1.6)
    sol0 = b.solve(k=0.33)
    assert sol0.success
    sols0 = [b.solve(x0=x0, sol0=sol0, **kw) for x0 in x0s]
    sols1 = [b.solve(x1=x1, sol0=sol0, **kw) for x1 in x1s]
    assert all(sol.success for sol in sols0)
    assert all(sol.success for sol in sols1)
    errs0 = [abs(sol.sol(x_test) - sols0[-1].sol(x_test)).max() for sol in sols0]
    errs1 = [abs(sol.sol(x_test) - sols1[-1].sol(x_test)).max() for sol in sols1]

    ax0.loglog(x0s[:-1], errs0[:-1], f'+-C{i}', label=f"BVP{i+1}")
    ax1.semilogy(x1s[:-1], errs1[:-1], f'+-C{i}')
    ax1.plot(x1s, err1_est[i], f'--C{i}', label=f"$y(x_1)^{{{i+1}}}$")
ax0.set(xlabel="$x_0$", ylabel="max abs err");
ax1.set(ylim=(1e-12, 1), xlabel="$x_1$", ylabel="max abs err");
ax0.legend()
ax1.legend()
plt.tight_layout()
```

From the scaling, we see that our error analysis is not completely correct.  In particular, both
`BPV1` and `BVP2` seem to have the same error scaling, but as $y_1^2$ rather than
linear in $y_1 = y(x_1)$.  I don't fully understand this yet, but has to do with the
fact that the maximum error is incurred somewhere for $x < x_1$ less than $x_1$.
Likely, the error in the boundary condition is enforced by the $e^{x}$ solution which
dampens extremely quickly before the location where the maximum error is incurred.
The scaling for `BVP3` is consistent with, but slightly better than $y_1^3$.

## Additional Ideas (Incomplete)

1. Use a reasonable ansatz to minimize $E[\psi]$ subject to fixed $N[\psi]$. See below.
   This can be used to find a good initial guess.
2. Use the asymptotic form to construct a better solution which factors off the singular
   terms.
3. Use variables like $y = e^{q(x)}$ which will make manifest the positivity of the
   solution.

## Initial Guess

If the problem is sufficiently unstable, then one might be able to tame it by
introducing a constrained form of the solution.  For example, the following functional
form has the correct asymptotic properties for a bubble with radius $X$ and width $\lambda$:
\begin{gather*}
  y(x) = \frac{y_0}{4}\Bigl(1 + \tanh\frac{x+X}{\lambda}\bigr)\Bigr)
                      \Bigl(1 - \tanh\frac{x-X}{\lambda}\Bigr)\\
  = y_0\frac{e^{-2(x-X)/\lambda}}
            {(1 + e^{-2(x+X)/\lambda})(1 + e^{-2(x-X)/\lambda})},\\
  y'(x) = \frac{y_0}{\lambda}
          \frac{-2e^{6 X/\lambda}\sinh \frac{2x}{\lambda}}
               {\left(1+e^{-2(x-X)/\lambda}\right)^2
                \left(1+e^{2(x+X)/\lambda}\right)^2}.
\end{gather*}
One can use these to directly try to minimize the energy functional as a function of
$X$, $y_0$, and $\lambda$.  Note that the asymptotic behaviours constrain, i.e. $\lambda
= 2$, but we are looking for global minimization, so it is best not to try to use the
properties of the "tails" to "wag the dog".

One can proceed analytically, but we will write a simple numerical routine:
\begin{gather*}
  \DeclareMathOperator{\Li}{Li}
  N = \int_0^{\infty}\d{x}\; 4\pi x^2 y^2(x)
    = \lambda^3 y_0^2 \pi \int_0^{\infty}\d{x}\; 
      x^2 \Bigl(1 + \tanh(x+X/\lambda)\Bigr)
          \Bigl(1 - \tanh(x-X/\lambda)\Bigr)\\
    \approx 
      2\lambda^3 y_0^2 \pi \int_0^{\infty}\d{x}\; 
      x^2 \Bigl(1 - \tanh(x-X/\lambda)\Bigr)
    = -\lambda^3 y_0^2 \pi \Li_3(-e^{2X/\lambda}).
\end{gather*}

To find a solution, we return to the original definition.

```{code-cell}
:tags: [hide-input]

class BVPGuess(BVP1):
    def get_initial_guess(self, sol=None):
        x = np.linspace(self._x0, self._x1, 1000)
        
    def y(self, x, p):
        y0, X, lam = p
        return y0/4*(1 + np.tanh((x+X)/lam))*(1 - np.tanh((x-X)/lam))

    def _get_N(self, x, y):
        return np.trapz(x**2 * y**2, x)
    
    def _get_E(self, x, y):
        dy = np.gradient(y, x, edge_order=2)
        n = y**2
        Vint = self.alpha * 3/4*(n**3 - 2*n**2)
        return np.trapz(x**2 * (dy**2 + Vint + n), x)

    def go(self, p, plot=True):
        x = np.linspace(0.001, 20, 1000)
        #X, lam = p
        y = self.y(x, p)
        #N_ = self._get_N(x, y)
        #y0 = np.sqrt(N/N_)
        #y *= y0
        #assert np.allclose(self._get_N(x, y), N)
        E = self._get_E(x, y)
        if plot:
            plt.plot(x, y**2)
            plt.title(f"{N=}, {E=}")
        return E
g = BVPGuess()
g.go((10, 2))
        
```

```{code-cell}
:tags: [hide-input]

from scipy.interpolate import InterpolatedUnivariateSpline

n0s = np.array([sol.sol(0)[0]**2 for sol in sols])
Xs = []
for sol in sols:
    if sol.n.max() > 0.5:
        _X = InterpolatedUnivariateSpline(sol.x, sol.n-0.5).roots()[0]
    else:
        _X = 0
    Xs.append(_X)
Xs = np.asarray(Xs)

fig, ax = plt.subplots(figsize=(3, 2))
plt.plot(Xs, 1+b.alpha*b.V(n0s))
plt.plot(Xs, -6*np.exp(3*Xs)/(1+np.exp(Xs))**4)
```




## Numerical Attempt 4

Now we note that $A(x) \geq 0$ everywhere.  This can be enforced by letting
$A(x) = e^{B(x)}$:
\begin{gather*}
  A(x) = e^{B(x)}, \qquad
  A'(x) = B'(x)A(x), \qquad
  A''(x) = B''(x)A(x) + [B'(x)]^2A(x),\\
  B''(x) = -[B'(x)]^2 + 2(1 - x^{-1})B'(x) + V(n) + 2x^{-1},\\
  B'(0) = 1 + \frac{x}{3}\Bigl(1+V\bigl(e^{2B(0)}\bigr)\Bigr),\qquad
  B'(\infty) = 0.
\end{gather*}

```{code-cell}
b1 = BVP1()
b = BVP2()
sol1 = b1.solve(R=5)
sol = b.solve(R=5)
x1, (y, dy) = sol1.x, sol1.y
x, (A, dA) = sol.x, sol.y
dA, ddA = b.rhs(x, sol.y)
B = np.log(A)
n = np.exp(B-x)**2
dB = dA/A
ddB = ddA/A - dB**2
assert np.allclose(dB, np.gradient(B, x, edge_order=2), rtol=0.02, atol=0.01)
assert np.allclose(ddB, np.gradient(dB, x, edge_order=2), rtol=0.05, atol=0.05)
assert np.allclose(ddB, -dB**2 + 2*(1-1/x)*dB + b.V(n) + 2/x, rtol=0.05, atol=0.05)
```

```{code-cell}
class BVP4(BVP1):
    def rhs(self, x, q):
        """Return dq/dr.  Note q = (B, dB)."""
        B, dB = q
        n = np.exp(B-x)**2
        ddB = -dB**2 + 2*(1-1/x)*dB + self.V(n) + 2/x
        dq = np.asarray([dB, ddB])
        return dq
    
    def bc(self, qa, qb):
        """Return the boundary conditions."""
        B0, dB0 = qa
        Bb, dBb = qb
        x0 = self._r0
        bc = [
            dB0 - (1+x0/3*(1 + self.V(np.exp(2*B0)))),
            dBb
        ]
        return np.asarray(bc)
    
    def solve(self, sol0=None, r0=0.01, R=20.0, mu=-0.3, k=1, **kw):
        """Return the solution `sol` to the BVP.
        
        Arguments
        ---------
        sol0 : Solution, None
            A previous solution to use as a starting point.
        r0, R : float
            Inner and out radii for boundary conditions.
        mu : float
            Chemical potetial.
        """
        self._r0 = r0
        self._mu = mu
        if sol0 is None:
            xs = np.linspace(r0, R)
            ys = 1/np.cosh(k*xs)
            dys = k*np.sinh(k*xs)/np.cosh(k*xs)**2
            As = ys*np.exp(xs)
            dAs = As + dys*np.exp(xs)
            Bs = np.log(As)
            dBs = dAs/As
        else:
            xs, (Bs, dBs) = sol0.x, sol0.y
        y0s = np.asarray([Bs, dBs])
        sol = solve_bvp(self.rhs, self.bc, x=xs, y=y0s, **kw)
        x, (B, dB) = sol.x, sol.y
        A = np.exp(B)
        y = A*np.exp(-x)
        sol.r = x/np.sqrt(-mu)
        sol.n = y**2
        return sol

b = BVP4()
sol = b.solve(mu=-0.3, R=5)
r, n = sol.r, sol.n
fig, ax = plt.subplots()
ax.plot(r, n)
ax.set(xlabel="$r$", ylabel="$n$", title=f"$\mu={b._mu:.2f}$");
```

## Bessel Functions

One approach is to try to use some known functions to construct approximate solutions,
then numerically find corrections.  A similar equation is solved by the [spherical
Bessel functions][], specifically for $n=0$
\begin{gather*}
  j_n'' + \frac{2}{x}j_n' + \left(1 - \frac{n(n+1)}{x^2}\right)j_n = 0, \qquad
  j_0(x) = \frac{\sin x}{x}.
\end{gather*}
Thus, we can consider a solution of the form $\psi(r) = Aj_0(r\sqrt{\mu})$:
\begin{gather*}
  \psi  
\end{gather*}

#a = 1/\sqrt{\mu}

```{code-cell}
from sympy import var, sinh, sqrt
r, A, a, mu = var('r, A, a, mu')
a = 1/sqrt(-mu)
psi = A*sinh(r/a)/(r/a)
display((psi.diff(r, r) + 2/r*psi.diff(r) + mu*psi).simplify())
```

```{code-cell}
r = np.linspace(0, 10)[1:]
plt.plot(r, np.sinh(r)/r)
```


## Two Components

\begin{gather*}
  \mathcal{E}(n_a, n_b) = \frac{1}{2}
  \begin{pmatrix}
    n_a & n_b
  \end{pmatrix}
  \begin{pmatrix}
    g_{a} & g_{ab}\\
    g_{ab} & g_{b}
  \end{pmatrix}
  \begin{pmatrix}
    n_a\\
    n_b
  \end{pmatrix}\\
  \begin{pmatrix}
    \mu_a\\
    \mu_b
  \end{pmatrix}
  =
  \begin{pmatrix}
    g_{a} & g_{ab}\\
    g_{ab} & g_{b}
  \end{pmatrix}
  \begin{pmatrix}
    n_a\\
    n_b
  \end{pmatrix}.
\end{gather*}

\begin{gather*}
  g_ag_b \geq g_{ab}^2  
\end{gather*}


\begin{gather*}
  n_b = 0, \qquad
  \mu_a = g_an_a, \qquad
  \mu_b \leq g_{ab} n_a,\\
  n_a = 0, \qquad
  \mu_b = g_bn_b, \qquad
  \mu_a \leq g_{ab} n_b,\\
  \mu_b = g_bn_b\leq g_{ab} n_a, \qquad
  \mu_a = g_an_a \leq g_{ab} n_b,\\
\end{gather*}



[Bessel functions]: <https://en.wikipedia.org/wiki/Bessel_function>
[spherical Bessel functions]: <https://en.wikipedia.org/wiki/Bessel_function#Spherical_Bessel_functions:_jn,_yn>
[GPE]: <https://en.wikipedia.org/wiki/Gross%E2%80%93Pitaevskii_equation>
