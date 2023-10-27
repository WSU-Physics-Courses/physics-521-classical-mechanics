---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
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
FIG_DIR = Path(mmf_setup.ROOT) / 'Docs/_build/figures/'
os.makedirs(FIG_DIR, exist_ok=True)
import logging;logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
try: from myst_nb import glue
except: glue = None
```

# Boundary Value Problems

Here we consider the problem of finding the solution for a spherical drop of some
material with a saturating equation of state $\mathcal{E}(n)$.  Choosing units $\hbar^2/2m =
1$, we have the following equation for the wavefunction as a function of radius $r$:

:::{margin}
The Laplacian for spherical solutions (i.e. no angular momentum $l=0$) is
\begin{gather*}
  \nabla^2 \psi = \frac{1}{r^{d-1}}\pdiff{}{r}\left(r^{d-1}\pdiff{\psi}{r}\right)\\
  = \psi''(r) + \frac{d-1}{r}\psi'(r)
\end{gather*}
:::
\begin{gather*}
  \psi'' + \frac{2}{r}\psi' = \bigl(\mathcal{E}'(\psi^2) - \mu\bigr)\psi, \qquad
  \mathcal{E}'(n) = n(3n-4),\\
  \psi(0) = \psi_0, \qquad
  \psi(\infty) = 0.
\end{gather*}
We are looking for solutions that are finite at $r=0$, but which fall to zero as
$r\rightarrow \infty$.


First we note some properties of this system.  Here is the equation of state:

```{code-cell}
:tags: [hide-input]

a, b = 1, 2
n0 = b/2/a
mu0 = 3*a*n0**2 - 2*b*n0
n_unstable = np.linspace(0, n0)[1:]
n_stable = np.linspace(n0, 2)
n = np.poly1d([1, 0])
E = np.poly1d([a, -b, 0, 0])
mu = dE = E.deriv()
fig, axs = plt.subplots(1, 4, figsize=(9, 2))
ax = axs[0]
fmt_stable = '-C0'
fmt_unstable = ':C0'
fmt_mixed = 'k--.'
ax.plot(n_stable, E(n_stable), fmt_stable)
ax.plot([0, 0], [0, 0.25], fmt_stable)
ax.plot(n_unstable, E(n_unstable), fmt_unstable)
ax.plot(n_unstable[[0, -1]], mu0*n_unstable[[0, -1]], fmt_mixed)
ax.set(xlabel="$n$", ylabel=r"$\mathcal{E}=E/V$")

ax = axs[1]
ax.plot(n_stable, E(n_stable)/n_stable, fmt_stable)
ax.plot(n_unstable, E(n_unstable)/n_unstable, fmt_unstable)
ax.plot([0, 0], [mu0, 0.1], fmt_stable)
ax.plot([0, n0], [mu0, mu0], fmt_mixed)
ax.set(#xlim=(-0.1, 2.1), #ylim=(-1.5, 1.5), 
       xlabel="$n$", ylabel=r"$E/N = \mathcal{E}/n$")

ax = axs[2]
ax.plot(n_stable, mu(n_stable), fmt_stable)
ax.plot([0, 0], [mu0, -1.5, ], fmt_stable)
ax.plot([0, n0], [mu0, mu0], fmt_mixed)
ax.plot(n_unstable,mu(n_unstable), fmt_unstable)
ax.set(xlim=(-0.1, 2.1), ylim=(-1.5, 1.5), xlabel="$n$", ylabel=r"$\mu = \mathcal{E}'(n)$")

P = mu*n - E
ax = axs[3]
ax.plot(mu(n_stable), P(n_stable), fmt_stable)
ax.plot([-1.5, mu0], [0, 0], fmt_stable)
ax.plot([mu0], [0], fmt_mixed)
ax.plot(mu(n_unstable), P(n_unstable), fmt_unstable)
ax.set(xlim=(-1.5, 0.5), ylim=(-0.4, 1.0), xlabel="$\mu$", ylabel=r"$P=\mu n - \mathcal{E}$")

plt.tight_layout()
```

```{code-cell}
:tags: [hide-input]

a, b = 1, 2
n0 = b/2/a
mu0 = 3*a*n0**2 - 2*b*n0

n = np.poly1d([1, 0])
E = np.poly1d([a, -b, 0, 0])
mu = dE = E.deriv()
P = mu*n - E

sigma = 0.2
R = 1
rs = np.linspace(0, R)[1:]
Vs = 4/3*np.pi * rs**3
ns = np.array([(P-_P).roots[0] for _P in 2*sigma / rs]).real
Es = E(ns)*Vs + sigma*4*np.pi * rs**2
Ns = ns*Vs

fig, ax = plt.subplots()
ax.plot(Ns, Es)

V = 4/3*np.pi*R**3
Ns = np.linspace(Ns.max(), 2*Ns.max())
ax.plot(Ns, E(Ns/V)*V)
ax.set(xlabel="N", ylabel="E", title= f"$\sigma={sigma:.4g}$, $R={R:.4g}$")
plt.tight_layout()
```

This describes a saturating system with an unstable branch between $n_0 \in [0, 1]$.  In
this region, one will have a mixed phase through the Maxwell construction (dashed black
line) between the two phases shown: one at $n=0$ corresponding to the vacuum, and the
other at $n=n_0$.  Note that the Maxwell construction reduces to a point in the $(\mu,
P)$ plot (grand-canonical ensemble) at $\mu_0 = -1$.

In infinite matter, or an infinitely large droplet, the surface will have no curvature, so the
pressures of the two phases must match.  In this case $P=0$ in both the vacuum and the
saturating homogeneous phase with density $n_0$ and chemical potential $\mu_0 < 0$.
Note: until $\mu_0 < -mc^2$, the vacuum is stable (no anti-particles are produced), and so
can be in chemical equilibrium for any negative $\mu < 0$.

In a finite droplet, the surface is curved, and the surface tension will increase the
pressure inside the droplet, causing $n>n_0$ and increasing $\mu > 
\mu_0$.  Larger and larger surface tension will correspond to greater curvature --
i.e. smaller droplets.  Thus, as $\mu$ increases towards zero, the size of the droplet
will shrink.

This is slightly paradoxical since increasing the chemical potential *decreases* the
overall particle number, although it increases the central density for a while.  This is
because we have neglected to account for the surface tension.  The "chemical potential"
$\mu$ here is really only a property of homogeneous matter.  Think of it this way: in
the mixed phase, the particle number can increase without a change in chemical
potential.

To get an idea of how this works, consider the compressible liquid drop model (CLDM)
where we have an inner density $n = n_i > n_0$, a surface tension $\sigma$ describing a thin
surface, and an outer density $n_o=0$.  The energy $E$ and particle number $N$ of a
droplet of radius $r$ is:
\begin{gather*}
  N = nV = n\frac{4}{3}\pi r^3,\\
  E(N, V) = \frac{4}{3}\pi r^3\mathcal{E}(n) + 4\pi r^2 \sigma
          = V \mathcal{E}\left(\frac{N}{V}\right) + \left(36\pi\right)^{1/3} \sigma V^{2/3}.
\end{gather*}
The size of the droplet will minimize the energy for fixed $N$:
\begin{align*}
  \d{N} &= 4\pi r^2 n\d{r} + \frac{4}{3}\pi r^3\d{n} = 0,\\
  \d{E} &= \Bigl(4\pi r^2\mathcal{E}(n) + 8\pi r \sigma)\d{r}
           + \frac{4}{3}\pi r^3\mathcal{E}'(n)\d{n}\\
        &= \Bigl(4\pi r^2\mathcal{E}(n) + 8\pi r \sigma
           - 4\pi r^2 n\mathcal{E}'(n)\Bigr)\d{r}\\
        &= \Bigl(8\pi r \sigma - 4\pi r^2P(n)\Bigr)\d{r} = 0.
\end{align*}
Solving, we find that the surface tension causes the pressure:
\begin{gather*}
  \frac{2 \sigma}{r} = P(n) = n\underbrace{\mathcal{E}'(n)}_{\mu(n)} - \mathcal{E}(n),\\
  -\frac{2 \sigma}{r^2}\d{r} = P'(n)\d{n} = n\mathcal{E}''(n)\d{n}.
\end{gather*}
From these, we can compute $\d{E}/\d{N}$:
\begin{gather*}
  \diff{E}{N} = \mathcal{E}'(n), \qquad
  \diff{r}{n} = -\frac{r^2P'(n)}{2\sigma}.
\end{gather*}


Hence,
\begin{gather*}
  \mu = \frac{2 \sigma/r + \mathcal{E}(n)}{n}
      = \frac{2 \sigma V}{rN} + \frac{V\mathcal{E}(N/V)}{N}.
\end{gather*}




First we note a couple of trivial solutions: first, we have homogenous solutions
$\psi(r) = 0$ (corresponding to the vaccum), and $\psi = \sqrt{n_0}$ where $\mathcal{E}'(n_0) =
\mu$ is also a solution.  For our form, $n_0 = (1 \pm \sqrt{1+4\mu})/2$

## Asymptotic Behaviour

A good starting point is to consider the asymptotic behavior.  For large $r\rightarrow
\infty$, we expect that we can neglect the second term, and are looking for solutions
that vanish $\psi \rightarrow 0$, so if we choose $\mathcal{E}(0) = 0$, we should have
the asymptotic behaviour
\begin{gather*}
  \psi'' \approx -\mu \psi.  
\end{gather*}
This will be oscillatory unless $\mu < 0$.  We thus recast the problem as
\begin{gather*}
  x = \sqrt{-\mu} r, \qquad
  V(n) = \frac{1}{-\mu}\mathcal{E}'(n) = \frac{n(n-1)}{-\mu},\\
  y'' + \frac{2}{x}y' = \bigl(1+V(y^2)\bigr)y, \qquad
  y(0) = y_0, \qquad
  y(\infty) = 0.
\end{gather*}
Now the asymptotic behaviour is given by $\psi'' \approx \psi$, and so the decaying
solution has the asymptotic form
\begin{gather*}
  y \rightarrow Ae^{-r}.
\end{gather*}
As a check, the neglected terms must satisfy
\begin{gather*}
  \frac{2y'}{r}, \qquad V(y^2)y
\end{gather*}
must be asymptotically smaller than $y'' \sim y$.  This gives the conditions:
\begin{gather*}
  2\ll r, \qquad
  \abs{y(r)^2}\ll \abs{\mu}.
\end{gather*}

At the origin, we must have that $y' \rightarrow 0$ at least as fast a $r$, otherwise
the second term will diverge.  Since we physically expect the solution to be smooth at
the origin, we expect
\begin{gather*}
  y(r) = y_0 + \frac{y_0''}{2}r^2 + O(r^3),\qquad
  y''(r) + \frac{2y'(r)}{r} = 3y_0'' + O(r), \qquad
  3y''_0 = y_0\bigl(1+V(y_0^2)\bigr).
\end{gather*}

Apart from refining the formulation of the problem, this also allows us to refine the
boundary conditions:
\begin{gather*}
  y'' + \frac{2}{r}y' - y = V(y^2)y,\\
  3y''(0) = y(0)\Bigl(1 + V\bigl(y(0)^2\bigr)\Bigr) \quad \text{or}\quad
  y'(0) = \frac{ry(0)}{3}\Bigl(1 +V\bigl(y(0)^2\bigr)\\
  y'(\infty) + y(\infty) = 0.
\end{gather*}

## Numerical Attempt 1

We start with a brute-force solution:

```{code-cell}
:tags: [hide-input]

from scipy.integrate import solve_bvp

class BVP1:
    def V(self, n):
        return n*(3*n-4)/(-self._mu)
    
    def rhs(self, r, q):
        """Return dq/dr.  Note q = (y, dy)."""
        y, dy = q
        ddy = (1+self.V(y**2))*y - 2*dy/r
        dq = np.asarray([dy, ddy])
        return dq
    
    def bc(self, qa, qb):
        """Return the boundary conditions."""
        y0, dy0 = qa
        yb, dyb = qb
        bc = [
            3*dy0 - self._r0*y0 * (1+self.V(y0**2)),
            yb + dyb
        ]
        return np.asarray(bc)
    
    def solve(self, sol0=None, r0=0.01, R=20.0, mu=-0.1, k=1, **kw):
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
            rs = np.linspace(r0, R)
            ys = 1/np.cosh(k*rs)
            dys = k*np.sinh(k*rs)/np.cosh(k*rs)**2
        else:
            rs, (ys, dys) = sol0.x, sol0.y
        y0s = np.asarray([ys, dys])
        sol = solve_bvp(self.rhs, self.bc, x=rs, y=y0s, **kw)
        return sol
        
b = BVP1()
sol = b.solve(mu=-0.3)
r = sol.x
y, dy = sol.y
fig, ax = plt.subplots()
ax.plot(r, y)
ax.set(xlabel="$r$", ylabel="$y$", title=f"$\mu={b._mu:.2f}$");
```

We see that this works, but it is very sensitive to the initial guess.  Here we start
with this solution, and then slowly increase and decrease the magnitude of $\mu$.  Note:
we must take thousands of steps, otherwise the solution does not converge, however, the
general procedure is sound.

```{code-cell}
:tags: [hide-input]

b = BVP1()
mu_ = -0.3
sol_ = b.solve(mu=mu_)
sols = [sol_]
mus = [mu_]
for dmu in [0.1, -0.02]:
    sol0 = sol_
    mu = mu_
    for N in range(1000):
        while abs(dmu) > 1e-8 and mu < -0.001:
            with np.errstate(all="ignore"):
                # Suppress floating point warnings.
                sol = b.solve(sol0=sol0, mu=mu+dmu)
            if sol.success:
                break
            dmu /= 2
        if not sol.success:
            break
        sol0 = sol
        mu = b._mu
        sols.append(sol)
        mus.append(mu)

fig, ax = plt.subplots()
mus = np.asarray(mus)
inds = np.argsort(mus)
mus = mus[inds]
sols = np.array(sols)[inds]

for mu, sol in zip(mus, sols):
    r = sol.x
    y, dy = sol.y
    ax.plot(r, y)
ax.set(xlabel="$r$", ylabel="$y$", title=f"${mus[0]:.2f} \leq \mu \leq {mus[-1]:.2f}$");
```

Now we estimate the surface tension $\sigma$ as follows.  The surface has an energy
$E_{\sigma}(r) = 4\pi \sigma r^2$ proportional to the area of the sphere.  This leads to
a compressional force $F = E_{\sigma}'(r) = 8 \pi \sigma r$ and a pressure (force per
unit area) $P = F/4\pi r^2 = 2\sigma /r$.  This should equal the pressure in the droplet:
\begin{gather*}
  P = \frac{2\sigma}{r} = \mu n - \mathcal{E} = n \mathcal{E}(n) - \mathcal{E}(n).
\end{gather*}
In the following, we estimate the radius of the surface as the point where $n = n_0/2$,
then plot $\sigma \approx P(n_0) r/2$


```{code-cell}
:tags: [hide-input]

from scipy.interpolate import InterpolatedUnivariateSpline

rs = np.array([sol.x[np.argmin(abs(sol.y[0] - 0.5))] for sol in sols])
rs = []
for sol in sols:
    if sol.y[0].max() > 0.5:
        _r = InterpolatedUnivariateSpline(sol.x, sol.y[0]-0.5).roots()[0]
    else:
        _r = 0
    rs.append(_r)
rs = np.asarray(rs)

fig, ax = plt.subplots()
a, b = 1, 2
n0 = 1.0
E0 = a*n0**3 - b*n0**2
mu0 = 3*a*n0**2 - 2*b*n0
ax.plot(rs, (mus*n0 - E0)*rs/2)
#plt.axhline([0.5], ls=":", c="y")
ax.set(xlabel="$r$", ylabel="$\sigma \approx P(n_0)r/2$");
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




[Bessel functions]: <https://en.wikipedia.org/wiki/Bessel_function>
[spherical Bessel functions]: <https://en.wikipedia.org/wiki/Bessel_function#Spherical_Bessel_functions:_jn,_yn>






