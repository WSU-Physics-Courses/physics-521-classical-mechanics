---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (phys-521)
  language: python
  name: phys-521
---

```{code-cell} ipython3
:tags: [hide-cell]

import mmf_setup;mmf_setup.nbinit()
import logging; logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
```

(sec:atom-laser)=
# Falling Atom Laser

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
    = \overbrace{\underbrace{\left(\frac{\hbar^2}{2m^2g}\right)^{1/3}}_{\text{length}}}^{\xi}
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

where the solution is expressed in terms of the Airy functions $y=\Ai(x)$ and
$y=\Bi(x)$, which satisfy:

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
with particles continually being injected at $z'=0$.

:::{margin}
Here we have used the natural length scale:
\begin{gather*}
  \xi = \sqrt[3]{\frac{\hbar^2}{2m^2g}}.
\end{gather*}
:::
For $z<0$, the qualitative form can be deduced from the WKB approximation:

\begin{gather*}
  \psi_{\mathrm{WKB}}(z) \propto \frac{1}{\sqrt{p(z)}}e^{S(z)/\I\hbar}, \\
  z(t) = - \frac{gt^2}{2}, \qquad
  p(z) = -mgt = -m\sqrt{-2gz}\\
  S(z) = \int_0^{t}\left(\frac{p^2}{2m} - mgz(t)\right)\d{t}
       = \frac{-mg^2t^3}{3} 
       = -\hbar \frac{2}{3}\sqrt{\frac{-z^3}{\xi^3}},\\
  \psi_{\mathrm{WKB}}(z) \propto \frac{1}{\abs{z}^{1/4}}
  \exp\left(\frac{2\I}{3}\sqrt{\frac{-z^3}{\xi^3}}\right)
\end{gather*}

```{code-cell} ipython3
:tags: [hide-input]

from scipy.special import airy

z = np.linspace(-10, 4, 500)

Ai0, dAi0, Bi0, dBi0 = airy(0)
Aiz, dAiz, Biz, dBiz = airy(z)

def G(x, y=0):
    Aix, dAix, Bix, dBix = airy(x)
    Aiy, dAiy, Biy, dBiy = airy(y)
    return (
        - np.pi * np.where(x < y, Aiy * (Bix + 1j * Aix), Aix * (Biy + 1j * Aiy))
    )

Gz = G(z)
Gz_WKB = np.where(z<0, np.exp(2j/3*abs(z)**(3/2)), np.nan) / abs(z)**(1/4)
Gz_WKB *= Gz[0]/Gz_WKB[0]
fig, ax = plt.subplots()
ax.plot(z, abs(Gz), '-C0', label=r'$|G(z, 0)|$')
ax.plot(z, Gz.real, '--C0', label='Real part')
ax.plot(z, Gz.imag, ':C0', label='Imaginary part')
ylim = ax.get_ylim()
ax.plot(z, abs(Gz_WKB), '-C1', label=r'$|G_{\rm WKB}(z, 0)|$')
ax.plot(z, Gz_WKB.real, '--C1', alpha=0.5)
ax.plot(z, Gz_WKB.imag, ':C1', alpha=0.5)
ax.legend()
ax.set(xlabel="$z$", ylabel="$G(z, 0)$", ylim=ylim);
```

:::{margin}
Of course, one should use the matching conditions at the turning point, but this uses
the Airy functions, and so would give the exact result.
:::
Once properly normalized, this gives a very good approximation except very close to the
injection site which is a classical turning point and the WKB approximation breaks
down.

:::{admonition} Atom Laser
:class: dropdown

An atom laser can be produced by "out-coupling" a trapped state (the source) to the
falling state with the following two-component Hamiltonian 

\begin{gather*}
  \op{H} = \begin{pmatrix}
    \frac{\op{p}_z^2}{2m} + mg\op{z} & \Omega e^{\I\omega t}\\
    \Omega e^{-\I\omega t} & \frac{\op{p}_z^2}{2m} + V(\op{z})
  \end{pmatrix}.
\end{gather*}

The idea of an atom laser is that a large reservoir of the lower component is held in
the trapping potential $V(z)$.  The off-diagonal coupling converts this lower component
to the upper component, which falls in the gravitational field. If the off-diagonal
coupling is small, the one can treat the trapped component as a constant, and we have
the following coupled equation (essentially neglecting the lower-left block):

\begin{align*}
  \I\hbar \ket{\dot{\psi}_a(t)} 
    &= \left(\frac{\op{p}_z^2}{2m} + mg\op{z}\right)\ket{\psi_a(t)}
       + \Omega e^{\I\omega t}\ket{\psi_b(t)}\\
   \I\hbar \ket{\dot{\psi}_b(t)} 
   &= \left(\frac{\op{p}_z^2}{2m} + V(\op{z}) - E_b\right)\ket{\psi_b(t)}.
\end{align*}

After some time, the states become quasi-stationary, and we can write:

\begin{gather*}
  \ket{\psi_b(t)} = \ket{\psi_b}e^{-\I E_b t/\hbar}, \qquad
  \ket{\psi_a(t)} = \ket{\psi_a}e^{\I (\hbar \omega - E_b) t /\hbar},\\
  \begin{aligned}
    \left(\frac{\op{p}_z^2}{2m} +  mg\op{z} - \I\hbar \partial_t\right)\ket{\psi_a(t)}
    &= - \Omega e^{\I(\hbar \omega - E_b)t/\hbar}\ket{\psi_b},\\
   \left(\frac{\op{p}_z^2}{2m} + V(\op{z}) - E_b\right)\ket{\psi_b} &= 0,\\
    \left(\frac{\op{p}_z^2}{2m} + mg\op{z} + \hbar \omega - E_b\right)\ket{\psi_a}
    &= -\Omega \ket{\psi_b},\\
    \left(\frac{\op{p}_z^2}{2m} + mg(\op{z} + z_0)\right)\ket{\psi_a}
    &= -\Omega \ket{\psi_b},\\
  \end{aligned}
\end{gather*}

The bottom equation was simplified by setting $mgz_0 = \hbar \omega - E_b$, which we can
redefine as the zero of our coordinate system $z \rightarrow z - z_0$.  Doing this, and
switching to our natural units, we have:

\begin{gather*}
    \left(\pdiff[2]{}{z} - z\right)\psi_a(z-z_0) = \Omega \psi_b(z-z_0) =\\
    = \Omega\int\d{z'}\;\psi_b(z'-z_0)\delta(z-z').
\end{gather*}

Hence, our solution can be expressed in terms of the Green's function:

\begin{gather*}
  \psi_a(z-z_0) = \Omega \int\d{z'}\; G(z,z') \psi_b(z' - z_0)\\
  \psi_a(z) = \Omega \int\d{z'}\; G(z+z_0,z'+z_0)\psi_b(z').
\end{gather*}

The Green's function is quite sharply peaked about $\abs{z-z'} \lesssim 2$, so the laser
out-couples atoms from a fairly narrow region about

\begin{gather*}
  \abs{z - z_0} \lesssim \sqrt[3]{\frac{4\hbar^2}{m^2g}},
\end{gather*}

the location of which can be tuned by adjust the drive frequency $\omega$.
:::

## Atom Interferometry

An application of the atom laser is to image differential potentials (see
e.g. {cite-p}`Mossman:2022`.)  In this application, we have two streams of particles,
which experience slightly different potentials
\begin{gather*}
  V_{a,b}(z) = V_0(z) \pm V(z).
\end{gather*}

The corresponding actions are (see {ref}`eg:FallingParticles`):
\begin{gather*}
  S_{a,b}(z;z_0; E)
  = -Et \pm \int_{z_0}^{z} p_{a,b}(z)\d{z}, \\
  p_{a,b}(z) = \sqrt{2m(E-V_{a,b}(z))}.
\end{gather*}
For a falling particle with, we can set $z_0 = 0$, $E=0$, and $V_0(z) = mgz$.  An
interferometer allows us to recover the relative phase difference, so by
differentiating, we can directly extract $V(z)$:
\begin{align*}
  S_a'(z) - S_b'(z) &= -p_a(z) + p_b(z) \\
  &= \sqrt{-2m^2 gz + 2mV(z)} -  \sqrt{-2m^2 gz - 2mV(z)}.
\end{align*}
If $V(z)$ is small, we can expand:
\begin{gather*}
  V(z) \approx
  \sqrt{\frac{-gz}{2}}\Bigl(S_a'(z) - S_b'(z)\Bigr)
\end{gather*}

:::{margin}
The function $\mathbf{1}_{A}(t)$ is called an [indicator function]:
\begin{gather*}
  \mathbf{1}_{A}(t) = \begin{cases}
    1 & x\in A,\\
    0 & x\notin A.
  \end{cases}
\end{gather*}
:::
The experiment is slightly more complicated in that the differential potential is pulsed
on for a short period of time:
\begin{gather*}
  V_{a,b}(z, t) = V_0(z) \pm V(z)\mathbf{1}_{[t_1, t_2]}(t).
\end{gather*}
To deal with this, we define the trajectory $z(t, z_f)$ for the particle ending up at
$z(t_f, z_f) = z_f$ at imaging time $t_f$.  We can then use the same formalism with the
potential, but now with a potential that depends on $z_f$.  Assuming again that $z_0=0$,
$E=0$, and that the particle is falling, we have
\begin{gather*}
  V_{a, b}(z; z_f) = V_0(z) \pm V(z)\mathbf{1}_{[z(t_1;z_f),z(t_2;z_f)]}(z),\\
  S_{a,b}(z_f) = -\int_{z_0}^{z_f} p_{a,b}(z;z_f)\d{z}, \\
  \begin{aligned}
    p_{a, b}(z;z_f) &= \sqrt{-2mV_{a,b}(z;z_f)}\\ 
    &\approx \sqrt{-2m V_0(z)} \pm
    \frac{mV(z)}{\sqrt{-2mV_0(z)}}\mathbf{1}_{[z(t_1;z_f),z(t_2;z_f)]}(z)
  \end{aligned}
\end{gather*}
The derivative is slightly more complicated now because of the addition $z_f$ dependence
in the momentum:
\begin{gather*}
  \pdiff{V_{a,b}(z;z_f)}{z_f} =   
  \pm V(z)[\delta\bigl(z-z(t_2, z_f)\bigr) - \delta\bigl(z-z(t_1;z_f)\bigr)],\\
  \begin{aligned}
  \pdiff{p_{a,b}(z;z_f)}{z_f} &= \sqrt{\frac{-m}{2V_{a,b}(z;z_f)}}\pdiff{V_{a,b}(z;z_f)}{z_f}\\
                              &= \frac{m}{p_{a,b}(z;z_f)}\pdiff{V_{a,b}(z;z_f)}{z_f},
  \end{aligned}
\end{gather*}
This gives:
\begin{gather*}
  S_{a, b}'(z_f) = -p_{a,b}(z_f;z_f)
  -\int_{z_0}^{z_f} \pdiff{p_{a,b}(z;z_f)}{z_f}\d{z},\\
    = -p_{a,b}(z_f;z_f) 
    \mp\left.
    \frac{m V(z)}{p_{a,b}(z;z_f)}\right|_{z=z_2=z(t_2;z_f)}^{z=z_1=z(t_1;z_f)}.
\end{gather*}

In a typical experiment, the differential potential is turned off at the imaging time,
so $p_{a}(z_f;z_f) = p_{b}(z_f;z_f)$, so the first term -- which gives the dominant
contribution above -- vanishes, leaving the following difference in the action:
\begin{align*}
  S_a'(z_f) - S_b'(z_f) &= 
  \left.
    -m V(z)\left(
      \frac{1}{p_{b}(z;z_f)} + \frac{1}{p_{a}(z;z_f)}
    \right)
  \right|_{z=z_2=z(t_2;z_f)}^{z=z_1=z(t_1;z_f)}\\
  &\approx
  \left.
    \frac{-2mV(z)}{\sqrt{-2mV_0(z)}}
    \right|_{z=z_2=z(t_2;z_f)}^{z=z_1=z(t_1;z_f)}.
\end{align*}
    

```{code-cell} ipython3
t_1 = 1.0
t_2 = 2.0
t_f = 3.0
g = 1.0
m = 1.0
z_f = -g*t_f**2/2

z_V = -1.0
sigma = 0.1
dV = 1.0

@np.vectorize
def V(z, t):
    return m*g*z + dV * np.exp(-(z-z_V)**2/2/sigma**2)*np.where(t_1 < t < t_2, 1, 0)

@np.vectorize
def Vzf(z, z_f):
    # How long the particle takes to get to z
    dt = np.sqrt(-2*z/g)
    # How long the particle takes to get to z_f dtf = (t_f-t_0)
    dtf = np.sqrt(-2*z_f/g)
    t_0 = t_f - dtf
    t = t_0 + dt
    return V(z=z, t=t)

t, z = np.meshgrid(np.linspace(0, t_f, 200), 
                   np.linspace(z_f, 0, 201), 
                   indexing='ij', sparse=True)
fig, axs = plt.subplots(1, 2)
ax = axs[0]
ax.pcolormesh(t.ravel(), z.ravel(), V(z, t).T, shading='auto')

ax = axs[1]
zf = z.T
ax.pcolormesh(zf.ravel(), z.ravel(), Vzf(z, zf).T, shading='auto')
ax.set(xlabel="z_f", ylabel="z", aspect=1);
```




[indicator function]: <https://en.wikipedia.org/wiki/Indicator_function>

# (Old Stuff)











```{code-cell} ipython3
from ipywidgets import interact
from scipy.special import airy

hbar = 1
m = 0.5
g = 2.0
Omega = 0.1

sigma = 1.0

z = np.linspace(-100, 2, 500)
Ai0, dAi0, Bi0, dBi0 = airy(0)
Aiz, dAiz, Biz, dBiz = airy(z)

def G(x, xx=0):
    Aix, dAix, Bix, dBix = airy(x)
    Ai0, dAi0, Bi0, dBi0 = airy(xx)
    return (
        - np.pi * np.where(x < xx, Ai0 * (Bix + 1j * Aix), Aix * (Bi0 + 1j * Ai0))
    )

@interact(sigma=(0.01, 10.0))
def go(sigma=1.0):
    def psi_b(z):
        return np.exp(-(z/sigma)**2/2)
    
    zz = np.linspace(-4*sigma, 4*sigma, 200)
    psi_a = Omega*G(z[:, None], zz[None, :]) @ psi_b(zz)
    n_a = abs(psi_a)**2
    n_b = abs(psi_b(z))**2
    fig, ax = plt.subplots()
    ax.plot(z, n_a, 'C0', label='$n_a(z)$')
    ax.twinx()plot(z, n_b, 'C1', label='$n_b(z)$')
    ax.set(xlabel='$z$', ylabel="$n_a$")
    ax.grid(True)
    ax.legend();
```

```{code-cell} ipython3

```

```{code-cell} ipython3
def psi_b(z):
    return np.exp(-(z/sigma)**2/2)

zz = np.linspace(-2*sigma, 2*sigma, 100)
psi_a = G(z[:, None], zz[None, :]) @ psi_b(zz[None, :])

plt.plot(z, abs(G(z, 2.0))**2)
```

```{code-cell} ipython3
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

:::::{admonition} Old Discussion: Potential Source
:class: dropdown

Note that we can add to $G(z, z')$ any solution of the homogeneous equation, so we can write:

:::{margin}
We only add $\Ai(z)$ to keep the boundary condition $n\rightarrow 0$ for $z \rightarrow \infty$.
:::
\begin{gather*}
  \psi(z) = G(z, 0) + \alpha \Ai(z),\\
  \Bigl(-\pdiff[2]{}{z} + z + \overbrace{\frac{1}{\psi(0)}}^{a}\delta(z)\Bigr)\psi(z) = 0.
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
  \psi(z<0) = -\pi\Ai(0)\left(\Bi(z) + \I \Ai(z)\right) + \alpha \Ai(z),\\
   = -\pi\Ai(0)\left(
    \Bi(z) - \Bigl(\sqrt{3} + \frac{1}{a\pi\Ai^2(0)}\Bigr) \Ai(z)
  \right)
\end{gather*}

```{code-cell} ipython3
:tags: [hide-input]

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

From this it is clear that $a \propto -\sqrt{3} + \I \propto e^{5\I\pi/6}$ is required
for a smooth solution (no oscillations).  I.e. in addition to the source term, we need
an attractive $\delta(z)$ potential to ensure that the phase remains coherent without
any interference effects.
:::::
