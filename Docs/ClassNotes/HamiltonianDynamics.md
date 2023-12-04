---
jupytext:
  formats: ipynb,md:myst,py:light
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

```{code-cell}
:tags: [hide-cell]

import mmf_setup; mmf_setup.nbinit()
import os
from pathlib import Path
FIG_DIR = Path(mmf_setup.ROOT) / '../Docs/_build/figures/'
os.makedirs(FIG_DIR, exist_ok=True)
import logging; logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
try: from myst_nb import glue
except: glue = None
import manim.utils.ipython_magic
!manim --version
```

(sec:HamiltonianMechanics)=
Hamiltonian Mechanics
=====================

Recall that with [Lagrangian mechanics][], one recovers Newton's laws as a principle of
extremal action:

:::{margin}
The vector notation here means that the equations hold for each component:
\begin{gather*}
  \left.\diff{}{t}\pdiff{L}{\dot{q}_{i}} = \pdiff{L}{q_i}\right|_{i=1}^{N}
\end{gather*}
where there are $N$ degrees of freedom. 
:::

\begin{gather*}
  S[\vect{q}] = \int_{t_0}^{t_1}\!\!\d{t}\; L(\vect{q}, \dot{\vect{q}}, t), \qquad
  \delta S = 0, \qquad
  \diff{}{t}\pdiff{L}{\dot{\vect{q}}} = \pdiff{L}{\vect{q}}.
\end{gather*}


The quantities $p_i = \partial L/\partial\dot{q}_i$ are called the *conjugate momenta*.
Expressed in this form, we have:

\begin{gather*}
  \vect{p} = \pdiff{L}{\dot{\vect{q}}}, \qquad
  \diff{\vect{p}}{t} = \pdiff{L}{\vect{q}}.
\end{gather*}

The idea of [Hamiltonian mechanics][] is to effect a [Legendre transformation][], replacing
the coordinates $\dot{\vect{q}}$ with the conjugate momenta $\vect{p}$:
:::{margin}
Note: one must invert the first set of equations to express $\dot{\vect{q}}(\vect{q},
\vect{p}, t)$ as a function of the coordinate and generalized momenta.
:::
\begin{gather*}
  \vect{p} = \pdiff{L}{\dot{\vect{q}}}, \qquad
  H(\vect{p}, \vect{q}, t) =
  \underbrace{\vect{p}\cdot\dot{\vect{q}}}_{
    \mathclap{\vect{p}\cdot\dot{\vect{q}}(\vect{q}, \vect{p}, t)}
  } - \overbrace{L}^{\mathclap{
    L\bigl(\vect{q}, \dot{\vect{q}}(\vect{q}, \vect{p}, t), t\bigr)
  }}.
\end{gather*}

Now we have [Hamilton's equations][] of motion:

\begin{gather*}
  \dot{\vect{q}} = \pdiff{H}{\vect{p}}, \qquad
  \dot{\vect{p}} = -\pdiff{H}{\vect{q}}.
\end{gather*}

::::{admonition} Do it!  Prove that these are correct.
:class: dropdown

We first solve the first equation for $\dot{\vect{q}} = \dot{\vect{q}}(\vect{q},
\vect{p}, t)$, then substituting:

\begin{gather*}
  H(\vect{p}, \vect{q}, t) 
  = \vect{p}\cdot\dot{\vect{q}}(\vect{q}, \vect{p}, t) - L\Bigl(\vect{q}, \dot{\vect{q}}(\vect{q}, \vect{p}, t), t\Bigr).
\end{gather*}

Now we simply differentiate, using the chain rule, and cancel $\vect{p} = \pdiff
L/\partial\dot{\vect{q}}$:

\begin{gather*}
  \pdiff{H(\vect{p}, \vect{q}, t)}{\vect{p}} 
  = \dot{\vect{q}}(\vect{q}, \vect{p}, t) + \cancel{\left(
    \vect{p} - \overbrace{\pdiff{}{\dot{\vect{q}}} L\Bigl(\vect{q},
    \dot{\vect{q}}(\vect{q}, \vect{p}, t), t\Bigr)}^{\vect{p}}
  \right)}\cdot\pdiff{\dot{\vect{q}}}{\vect{p}},
  \\
  \pdiff{H(\vect{p}, \vect{q}, t)}{\vect{q}} = 
  \cancel{\left(
    \vect{p} - \underbrace{\pdiff{}{\dot{\vect{q}}}L\Bigl(\vect{q},
  \dot{\vect{q}}(\vect{q}, \vect{p}, t), t\Bigr)}_{\vect{p}}\right)}
  \cdot\pdiff{\dot{\vect{q}}}{\vect{q}}
  - \underbrace{\pdiff{L}{\vect{q}}}_{\dot{\vect{q}}}.
\end{gather*}

::::

::::{admonition} Do it! Invert the process: find $L(\vect{q}, \dot{\vect{q}}, t)$ from $H(\vect{q}, \vect{p}, t)$.
:class: dropdown

From $H(\vect{q}, \vect{p}, t)$, use [Hamilton's equation][] of motion to find the velocity,
then reverse the transformation:
\begin{gather*}
  \dot{\vect{q}} = \pdiff{H}{\vect{p}}, \qquad
  L(\vect{q}, \dot{\vect{q}}, t) = 
  \vect{p}\cdot\dot{\vect{q}} - H
\end{gather*}

::::

## Phase Flow

Although one can study dynamics by plotting the flow of trajectories in configuration
space by plotting the velocity verses position $(\vect{q}, \dot{\vect{q}})$, there are
advantages to considering dynamics in [phase space][] $(\vect{q}, \vect{p})$, replacing
the velocity with the generalized momentum.  Specifically, Hamiltonian dynamics
preserves area in [phase space][], which is not necessarily true if one uses the velocity.

[Hamilton's equations][] define **phase flow**.  Let $\vect{y}(0) = (\vect{q}_0, \vect{p}_0)$ be a
point in phase space.  The phase flow is defined by the trajectory $\vect{y}(t)$ through
phase space which satisfies [Hamilton's equations][]:

\begin{gather*}
  \diff{}{t}\vect{y}(t) = \begin{pmatrix}
    \mat{0} & \mat{1}\\
    -\mat{1} & \mat{0}
  \end{pmatrix}
  \cdot
  \pdiff{H(\vect{y})}{\vect{y}}.
\end{gather*}

### Example
Here we implement a simple example of a pendulum of mass $m$ hanging down distance $r$
from a pivot point in a gravitational field $g>0$ with coordinate $\theta$ so that the
mass is at $(x, z) = (r\sin\theta, -r\cos\theta)$ and $\theta=0$ is the downward
equilibrium position: 

\begin{gather*}
  L(\theta, \dot{\theta}, t) = \frac{m}{2}r^2\dot{\theta}^2 + mgr\cos\theta\\
  p_{\theta} = \pdiff{L}{\dot{\theta}} = mr^2 \dot{\theta}, \qquad
  \dot{\theta}(\theta, p_{\theta}, t) = \frac{p_{\theta}}{mr^2},\\
  H(\theta, p_{\theta}) = p_{\theta}\dot{\theta} - L = \frac{p_{\theta}^2}{2mr^2} - mgr\cos\theta,\\
  \vect{y} = \begin{pmatrix}
    \theta\\
    p_{\theta}
  \end{pmatrix},\qquad
  \dot{\vect{y}} = 
  \begin{pmatrix}
    0 & 1\\
    -1 & 0
  \end{pmatrix}
  \cdot
  \begin{pmatrix}
    \pdiff{H}{\theta}\\
    \pdiff{H}{p_{\theta}}
  \end{pmatrix}
  =
  \begin{pmatrix}
    p_{\theta}/mr^2\\
    -mgr\sin\theta
  \end{pmatrix}.
\end{gather*}

```{code-cell}
:tags: [hide-input, full-width]

plt.rcParams['figure.dpi'] = 300
from scipy.integrate import solve_ivp

alpha = 0.0
m = r = g = 1.0
w = g/r
T = 2*np.pi / w   # Period of small oscillations.


def f(t, y):
    # We will simultaneously evolve N points.
    N = len(y)//2
    theta, p_theta = np.reshape(y, (2, N))
    dy = np.array([p_theta/m/r**2*np.exp(-alpha*t),
                   -m*g*r*np.sin(theta)*np.exp(alpha*t)])
    return dy.ravel()


# Start with a circle of points in phase space centered here
def plot_set(y, dy=0.1, T=T, phase_space=True, N=10, Nt=5, c='C0', 
             Ncirc=1000, max_step=0.01, _alpha=0.7, 
             fig=None, ax=None):
    """Plot the phase flow of a circle centered about y0.
    
    Arguments
    ---------
    y : (theta0, ptheta_0)
        Center of initial circle.
    dy : float
        Radius of initial circle.
    T : float
        Time to evolve to.
    phase_space : bool
        If `True`, plot in phase space $(q, p)$, otherwise plot in
        the "physical" phase space $(q, P)$ where $P = pe^{-\alpha t}$.
    N : int
        Number of points to show on circle and along path.
    Nt : int
        Number of images along trajectory to show.
    c : color
        Color.
    _alpha : float
        Transparency of regions.
    max_step : float
        Maximum spacing for times dt.
    Ncirc : int
        Minimum number of points in circle.
    """
    global alpha
    skip = int(np.ceil(Ncirc // N))
    N_ = N * skip
    th = np.linspace(0, 2*np.pi, N_ + 1)[:-1]
    z = dy * np.exp(1j*th) + np.asarray(y).view(dtype=complex)
    y0 = np.ravel([z.real, z.imag])

    skip_t = int(np.ceil(T / max_step / Nt))
    Nt_ = Nt * skip_t + 1
    t_eval = np.linspace(0, T, Nt_)
    res = solve_ivp(f, t_span=(0, T), y0=y0, t_eval=t_eval)
    assert Nt_ == len(res.t)
    thetas, p_thetas = res.y.reshape(2, N_, Nt_)
    ylabel = r"$p_{\theta}$"
    if phase_space:
        ylabel = r"$P_{\theta}=p_{\theta}e^{-\alpha t}$"
        p_thetas *= np.exp(-alpha * res.t)
    
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(thetas[::skip].T, p_thetas[::skip].T, "-k", lw=0.1)
    for n in range(Nt+1):
        tind = n*skip_t
        ax.plot(thetas[::skip, tind], p_thetas[::skip, tind], '.', ms=0.5, c=c)
        ax.fill(thetas[:, tind], p_thetas[:, tind], c=c, alpha=_alpha)
    ax.set(xlabel=r"$\theta$", ylabel=ylabel, aspect=1)
    return fig, ax

# Version without damping for now
alpha = 0.0
xlim = (-3, 14)
ylim = (-2, 3)

fig, ax = plt.subplots(figsize=(10, 5))

ys = np.linspace(0.25, 3.5, 13)

for n, y in enumerate(ys):
    plot_set(y=(y, y), c=f"C{n}", phase_space=False, ax=ax)

ax.set(xlim=xlim, ylim=ylim)
plt.tight_layout()
fig.savefig(FIG_DIR / "phase_space_pendulum_no_damping.svg")
display(fig)
plt.close(fig)

# Version with damping for use elsewhere.
xlim = (-1.5, 14)
fig, ax = plt.subplots(figsize=(10, 10 * abs(np.diff(ylim)/np.diff(xlim))[0]))

alpha = 0.3

for n, y in enumerate(ys):
    plot_set(y=(y, y), c=f"C{n}", ax=ax, T=1.3*T, phase_space=True)
ax.set(ylabel=r"$p_{\theta}$", xlim=xlim, ylim=ylim, aspect='auto')

plt.tight_layout()
fig.savefig(FIG_DIR / "phase_space_pendulum_damping.svg")
plt.close(fig)
```

:::{sidebar} Phase flow for a pendulum.

The small colored circular region is evolved forward in
time according to the Hamiltonian equations.  The trajectories of 10 equally spaced
points are shown at 6 different times equally spaced from $t=0$ to $T =
2\pi\sqrt{r/g}$, the period of oscillation for small amplitude modes.  At small energies
(blue circle), the period is almost independent of the amplitude, and the circle stays
together.  As the energy increases (orange, green, and red), the period starts
to depend more sensitively on the amplitude, and the initial circular region starts to
shear.  The purple region divides the two qualitatively different regions of rotation
and libration, and gets stretched as some points oscillate and others orbit. 
The areas remain constant, despite this stretching, as a demonstration of
Liouville's theorem.
:::

## Liouville's Theorem

An interesting property of phase flow is that it conserves areas in phase-space.  To see
this, recall that the a mapping $\vect{y} \mapsto \vect{f}(\vect{y})$ changes areas by a factor
of the determinant of the 
[Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant#Jacobian_determinant)
of the transformation:

\begin{gather*}
  \det \mat{J} = \begin{vmatrix}
    \pdiff{f_0}{y_0} & \pdiff{f_0}{y_1} & \cdots\\
    \pdiff{f_1}{y_0} & \pdiff{f_1}{y_1} & \cdots\\
    \vdots & \vdots
  \end{vmatrix}.
\end{gather*}

Liouville's theorem states that if the divergence of $f(\vect{y})$ vanishes
$\vect{\nabla}_{\vect{y}}\cdot \vect{f}(\vect{y}) = 0$, then the mapping preserves
volumes.

::::{admonition} Sketch of Proof
  :class: dropdown

See {cite:p}`Arnold:1989` Chapter 3 section 16 for a complete proof.

Let $v(t)$ be the volume of a region $D(t)$ at time $t$:
\begin{gather*}
  v(t) = \int_{D(t)}\d{\vect{y}} = \int_{D(0)}\det\mat{J}(t) \d{\vect{y}(0)}.
\end{gather*}

Show that $\dot{v}|_{t=0} = \int_{D(0)} \vect{\nabla}_{\vect{y}}\cdot \vect{f} \d{y} = 0$ by
expanding the Jacobian at time $t=0$:

\begin{gather*}
  \det \mat{J} = \mat{1} + \pdiff{\vect{f}}{\vect{y}} t + \order(t^2),\\
  \det(\mat{1} + \mat{A} t) = 1 + t \Tr A + \order(t^2),\\
  \Tr\pdiff{\vect{f}}{\vect{y}} = \vect{\nabla}_{\vect{y}}\cdot \vect{f} = 0.
\end{gather*}

This is valid at any time, so $\dot{v}(t) = 0$ and the area is unchanging.  That the
divergence of $\vect{f}$ is zero follows from the equality of mixed partial derivatives:

\begin{gather*}
  \vect{f}(\vect{y}) = \begin{pmatrix}
    \pdiff{H}{\vect{p}} \\
    -\pdiff{H}{\vect{q}}\end{pmatrix}, \\\
  \vect{\nabla}_{\vect{y}}\cdot \vect{f} = 
  \pdiff{}{\vect{q}} \cdot \pdiff{H}{\vect{p}}
  \pdiff{}{\vect{p}} \cdot \pdiff{H}{\vect{q}} = 0 .
\end{gather*}

::::

## A Hamiltonian with Damping

:::{margin}
Note that although the coordinate $q$ describes damped motion, the momentum $p$ is not
the usual momentum.  More conventionally we would have:

\begin{align*}
  \dot{P} &= -V'(Q) - m \alpha \dot{Q}\\
  \dot{Q} &= \frac{P}{m}.
\end{align*}

This comes from the transformation $Q = q$, $P = pe^{-\alpha t}$, which is **not
canonical**:

\begin{gather*}
  \{Q, P\}_{qp} = e^{-\alpha t} \neq 1.
\end{gather*}

In particular, Liouville's theorem holds for $\vect{y} = (q, p)$, but not for $\vect{Y}
= (Q, P)$.  In the later case, phase-space volumes shrink as the particles lose energy.

:::
A common misconception is that damping or dissipation cannot be included in a
Hamiltonian or Lagrangian framework.  Consider the following:

\begin{gather*}
  H(q, p) = \frac{p^2}{2m}e^{-\alpha t} + V(q)e^{\alpha t}\\
  \dot{p} = - V'(q) e^{\alpha t}, \qquad
  \dot{q} = \frac{p}{m}e^{-\alpha t}, \qquad
  p = m\dot{q} e^{\alpha t},\\
  \begin{aligned}
  m\ddot{q} &= \dot{p}e^{-\alpha t} - m\alpha \dot{q}\\
           &= - V'(q) e^{\alpha t}e^{-\alpha t} - m\alpha \dot{q}\\
           &= - V'(q) - m\alpha \dot{q}.\end{aligned}
\end{gather*}

This is just Newtonian dynamics for a particle of mass $m$ in a potential $V(q)$ with a
drag term $-m\alpha v$ such as might be seen for a particle falling through the
atmosphere.

It is instructive to consider this in the Lagrangian framework:

\begin{gather*}
  L(q, \dot{q}, t) = p\dot{q} - H  = \overbrace{\left(\frac{m \dot{q}^2}{2} -
  V(q)\right)}^{L_{0}(q, \dot{q})}e^{\alpha t}.
\end{gather*}

Here we see that the factor $\lambda(t) = e^{\alpha t}$ simplify multiplies the
Lagrangian.  In the action, this can be interpreted as a rescaling of time $\tau =
e^{\alpha t}\d{t}$:

\begin{gather*}
  S[q] = \int L_0(q, \dot{q})\; \underbrace{e^{\alpha t} \d{t}}_{\d{\tau}}.
\end{gather*}

This presents an analogy of cosmology where the universe is decelerating as it
expands, leading to an effective cooling from the dissipative term introduced by the
scaling of time.

We now demonstrate Liouville's theorem.  Note that this applies only to he **canonically
conjugate** variables $(q, p)$ that enter the Hamiltonian (top plot below), but not to
the conventional coordinates $(q, m\dot{q})$ for which the damping results in
contraction.

The key is that the canonical momentum is not $m\dot{q}$, but instead
\begin{gather*}
  p = \pdiff{L}{\dot{q}} = e^{\alpha t} m \dot{q}.  
\end{gather*}
The extra time-dependent factor $e^{\alpha t}$ exactly counters this contraction to
ensure Liouville's theorem.  However, now the phase space dynamics are not independent
of time.

:::{margin}
This plot shows the phase-flow for the time-dependent Hamiltionian for the pendulum with
damping.  The upper plot shows the flow in the original phase space $\vect{y} = (q, p)$
whereas the bottom figure shows the flow in the "physical" phase space $\vect{Y} = (q,
P)$.  The flow in the upper plot obey's Liouville's theorem, preserving areas, but is
time dependent.  This is why the trajectories intersect.

The bottom pane is independent of time, but does not follow from [Hamilton's equations][],
and hence, does not obey Liouville.  In the bottom figure, we see the expected
contraction of areas as the trajectories coalesce to the equilibrium points with $\theta
= 2\pi n$.

Note that the plots share the same abscissa $q=\theta$, but that the vertical axis is not just
a simple rescaling.  In the upper plot, the momenta are scaled differently at different
times.

We have evolved slightly longer than before ($1.3$ small-oscillation periods) to better
demonstrate the structure of the flow. 
:::

```{code-cell}
:tags: [hide-input]

fig, axs = plt.subplots(
    2, 1, figsize=(20, 12), sharex=True, 
    gridspec_kw=dict(height_ratios=(3.5, 1)))

# Saved version with damping.
fig0, ax0 = plt.subplots(figsize=(10,5))

alpha = 0.3

for n, y in enumerate(np.linspace(0.25, 1.75, 6)):
    plot_set(y=(y, y), c=f"C{n}", ax=axs[0], T=1.3*T, phase_space=False)
    [plot_set(y=(y, y), c=f"C{n}", ax=ax, T=1.3*T)
     for ax in [ax0, axs[1]]]
    
axs[0].set(ylim=(-6, 7))
axs[0].set(title=fr"$\alpha = {alpha}$, $T=1.3\times 2\pi \sqrt{{r/g}}$");

#fig0.savefig(FIG_DIR / "phase_space_pendulum_damping.svg")
plt.close(fig0)
```


# Dispersion and Relativity

A useful application of the Hamiltonian framework is if you know the **dispersion
relation** $E(p)$ -- the kinetic energy as a function of the momentum.  Simply apply
[Hamilton's equations][] of motion:
\begin{gather*}
  H(q, p) = E(p) + V(q), \\
  \dot{q} = \pdiff{H}{p} = E'(p), \\
  \dot{p} = -\pdiff{H}{q} = -V'(q) = F(q).
\end{gather*}

## Group Velocity

If you have studied waves, you may recognize the second equation $\dot{q} = E'(p)$ as
the [**group velocity**](https://en.wikipedia.org/wiki/Group_velocity) of a wave, which
is the derivative of the dispersion relationship.  The last equation $\dot{q} = F =
-V'(q)$ is just Newton's law.

## Effective Mass

We can get the more conventional form of Newton's law by taking the time derivative of
the second equation and using the chain rule:

:::{margin}
In higher dimensions, the inverse effective mass becomes a matrix:
\begin{gather*}
  [\mat{M}^{-1}]_{ij} = \frac{\partial^2 E(\vect{p})}{\partial p_i\partial p_j}\\
  \vect{a} = \mat{M}^{-1}\cdot\vect{F}.
\end{gather*}
:::
\begin{gather*}
  \ddot{q} = \diff{}{t}E'(p) = E''(p)\dot{p} = \overbrace{E''(p)}^{m^{-1}}F(q).
\end{gather*}

This shows us that the effective mass is the inverse of the second derivative of the
dispersion $m = 1/E''(q)$.  I.e. the curvature of the dispersion defines the mass.

## Constant Force

There is a peculiar special case which we can solve easily: that of a particle under a
constant force $F$:
\begin{gather*}
  V'(q) = -F, \qquad
  V(q) = -Fq.
\end{gather*}
The solution has two arbitrary constants $a$ (position) and $b$ (speed):
\begin{gather*}
  q(t) = \frac{E(p_0 + Ft)}{F} + a + bt.
\end{gather*}

::::{admonition} Do It!  Derive this solution.
:class: dropdown

In this case, we have the following equations:
\begin{gather*}
  \ddot{q} = E''(p)F, \qquad \dot{p} = F.
\end{gather*}
Hence, the momentum is linear in time:
\begin{gather*}
  p(t) = p_0 + Ft.  
\end{gather*}
Using the usual chain-rule trick twice:
\begin{gather*}
  \ddot{q} = \diff{\dot{q}}{p}\diff{p}{t} = \diff{\dot{q}}{p}\overbrace{\dot{p}}^{F} = FE''(p),\\
  \int \d{\dot{q}} = \int E''(p)\d{p}.\\
  \dot{q} = F\diff{q}{p} = E'(p) + C,\\
  \int F \d{q} = \left(E'(p) + C\right)\d{p},\\
  F q = E(p) + Cp + D.
\end{gather*}
Finally, substituting $p(t)$ and redefining the constants, have
\begin{gather*}
  q(t) = \frac{E(p_0+Ft)}{F} + a + bt
\end{gather*}
where $a$ is a constant position, and $b$ is a constant velocity.  Evaluating this at
time $t=0$ allows us to replace $a$ and $b$ with the initial position $q_0$ and velocity
$v_0$:
\begin{gather*}
  q(t) = \frac{E(p_0 + Ft) - E(p_0) - FE'(p_0)t}{F} + q_0 + v_0t.
\end{gather*}
::::

Here the trajectory of the particle has the same shape as the dispersion.

## Special Relativity

:::{margin}
This comes from the Lorentz-invariant magnitude of the energy-momentum four-vector
$p^\mu = (E/c, \vect{p})$, which defines the rest mass $m$:
\begin{gather*}
  p^\mu p_\mu = \frac{E^2}{c^2} - p^2 \\
  = m^2c^2.
\end{gather*}
:::
For a specific example, recall that in special relativity, the kinetic energy of a
particle of rest-mass $m$ and momentum $p$ is
\begin{gather*}
  E(p) = \sqrt{p^2c^2 + m^2c^4}
\end{gather*}
where $c$ is the speed of light, which should be constant for all observers.

## [Rindler Coordinates][]

:::{margin}
To achieve a truly constant acceleration, a rocket would need to adjust slowly reduce its power to
compensate for the loss.
:::
Combining the relativistic dispersion and constant force result gives us the
relativistic motion of a constantly accelerating object.
Choosing the constants so that we start at time $t=0$ at rest $v_0 = p_0 = 0$ and at position
$q_0$, we have
\begin{gather*}
  q(t) = \sqrt{t^2c^2 + h^2} + (q_0 - h), \qquad
  h = \frac{mc^2}{F},
\end{gather*}
where $h$ is the natural length scale.  Choosing units so that $h = c = 1$:
\begin{gather*}
  q(t) = q_0 + \underbrace{\sqrt{1 + t^2}}_{\gamma} - 1.
\end{gather*}
::::{admonition} Do It! Find this solution.
:class: dropdown

Using $E(p) = \sqrt{p^2c^2 + m^2c^4}$ with $p_0 = v_0 = 0$ and taking $a=0$, we have
\begin{gather*}
  q(t) = \frac{\sqrt{F^2 c^2 t^2 + m^2c^4}}{F}
   = \sqrt{c^2 t^2 + \frac{m^2c^4}{F^2}}\\
   = \sqrt{c^2 t^2 + h^2},
\end{gather*}
where $h = mc^2/F$.
::::
:::{margin}
Of course, this picture is only relevant for the accelerated observer.  To a
non-accelerating observer, the watch will move along a line with a constant velocity,
and will cross the horizon at a specified time.  This is what a free-falling observer
would see at the event horizon of a black hole: i.e. nothing special.
:::
With some work, this allows one to consider physics from the perspective of a constantly
accelerating observer using [Rindler coordinates][]: the relativistic equivalent of a
constant gravitational field.
Even though this is purely special-relativity, the
Rindler frame has some very interesting properties, including the existence of an event
horizon distance $h$ below the observer.  If you drop a watch, the watch will approach
this horizon, slowing both the rate at which it falls, and the rate at which the clock
runs, never actually falling across the horizon: just like a black hole.

## [Relativistic Lagrangian][]


One might like to try to figure out what Lagrangian gives rise to an equation with
dispersion $E(p)$ by effecting the [Legendre transformation][]:
\begin{gather*}
  L(q, \dot{q}, t) = \dot{q}p - H(q, p, t).
\end{gather*}
We must now solve for $p(\dot{q})$ by inverting $\dot{q} = E'(p)$:
\begin{gather*}
  L(q, \dot{q}, t) = \dot{q}p(\dot{q}) - E\Bigl(p(\dot{q})\Bigr) - V(q).
\end{gather*}
Note that, in general, this does **not have the form** of $L=T-V$, kinetic minus
potential unless $E(p) \propto p^2$, i.e. as for Newtonian mechanics.

::::{admonition} Do It!  Find the conditions for $L = T-V$ to work.
:class: dropdown

Noting that $\dot{q} = E'(p)$ we are looking for the forms of $E(p)$ such that
\begin{gather*}
  pE'(p) - E(p) = E(p), \qquad
  pE'(p) = 2E(p), \\
  \frac{\d{E}}{E} = 2\frac{\d{p}}{p}\\
  \ln(E) = 2\ln(p) + \ln C\\
  E(p) = C p^2.
\end{gather*}
Thus, the form $L=T(\dot{q})-V(q)$ only works for Newtonian mechanics where $E(p)$ is
purely quadratic in $p$.
::::

:::{margin}
Be careful with the signs of the roots when squaring: here
everything works out, even if $\dot{q} < 0$.
:::
Simplifying the relationship for the group velocity $\dot{q} = E'(p)$ we have
\begin{gather*}
  \dot{q} = E'(p) = \frac{pc^2}{\sqrt{p^2c^2 + m^2c^4}}, \qquad
  p = m\dot{q}\frac{1}{\sqrt{1 - \frac{\dot{q}^2}{c^2}}}.
\end{gather*}
Thus, we recover the well-known relativistic relationship:
:::{margin}
This form has led some people to interpret $\gamma m$ as the "mass of a moving particle".
The idea is that a moving particle has extra kinetic energy $E_k$, which then has a mass
equivalence via $\delta m = E_k/c^2$.  Problems appear in higher dimensions where this
interpretation requires introducing different masses in different directions, which does
not really make much sense.  It is best to stick with only one definition of mass, the
rest mass $m$, which is the mass one measures in the co-moving "rest" frame.
:::
\begin{gather*}
  p = \gamma m \dot{q}, \qquad
  \gamma = \frac{1}{\sqrt{1-\beta^2}}, \qquad
  \beta = \frac{\dot{q}}{c},
\end{gather*}
in terms of the [Lorentz
factor](https://en.wikipedia.org/wiki/Lorentz_factor) $\gamma(\beta)$ that relates the
rate of change of time $t$ in the inertial frame to proper time $\tau$ in the co-moving
frame:
\begin{gather*}
  \gamma = \diff{t}{\tau}.
\end{gather*}

Collecting and simplifying we find the correct Lagrangian:
\begin{gather*}
  L(q, \dot{q}, t) = -\frac{m c^2}{\gamma(\dot{q})} - V(q).
\end{gather*}
:::{admonition} Do It!  Derive $L(q, \dot{q}, t)$.
:class: dropdown

We start by noting that $\dot{q} = pc^2/E(p)$ so that:
\begin{gather*}
  p = \gamma m \dot{q}, \qquad
  E'(p) = \dot{q}, \qquad
  E(p) = pc^2/\dot{q} = \gamma m c^2.
\end{gather*}
Then, we use the [Legendre transformation][]
\begin{gather*}
  L = pE' - E - V = \gamma m \dot{q}^2 - \gamma m c^2 - V(q) = \\ 
  \gamma m c^2 \underbrace{\left(\frac{\dot{q}^2}{c^2} - 1\right)}_{-\gamma^{-2}} - V(q)
  =
  -\frac{m c^2}{\gamma} - V(q).
\end{gather*}
:::
This makes more sense when we consider the action and introduce proper time $\tau$ such
that $1/\gamma = \d{\tau}/\d{t}$:
\begin{gather*}
  I = \int_{t_0}^{t_1} L \d{t} 
  = \int_{t_0}^{t_1} \left(-mc^2\diff{\tau}{t} - V(q)\right) \d{t}\\
  = -mc^2\int_{\tau_0}^{\tau_1}\d{\tau} - \int_{t_0}^{t_1}V(q)\d{t}.
\end{gather*}
Note that the [relativistic Lagrangian][] does **not** have the form $T - V$.  Instead, the action
corresponding to the kinetic piece becomes manifestly Lorentz invariant: it is simply
the proper time of a clock moving with the observer, multiplied by appropriate
dimensional factors.

[Rindler Coordinates]: <https://en.wikipedia.org/wiki/Rindler_coordinates>
[relativistic Lagrangian]: <https://en.wikipedia.org/wiki/Relativistic_Lagrangian_mechanics>

# Canonical Transformations

The power of the Hamiltonian framework is that it allows one to express the most general
possible coordinate transformations that preserve the structure of [Hamilton's
equations][].  These are called a [Canonical transformation][]s, and can be checked
using the [Poisson bracket][] which we now define.

## Poisson Bracket

Given a set of canonical variables: coordinates $\vect{q}$ and associated conjugate
momenta $\vect{p}$, one can define the [Poisson bracket][] $\{\cdot, \cdot\}$:
\begin{gather*}
  \{f, g\}_{qp} = \sum_{i}\left(\pdiff{f}{q_i}\pdiff{g}{p_i} - \pdiff{g}{q_i}\pdiff{f}{p_i}\right)
\end{gather*}
:::::{admonition} Do It!  Prove the following properties of the [Poisson bracket][]:

\begin{align*}
  \{f, g\} &= -\{g, f\}, \tag{Anticommutativity}\\
  \{af + bg, h\} &= a\{f, h\} + b\{g, h\}, \tag{Bilinearlity}\\
  \{fg, h\} &= \{f, h\}g + f\{g, h\}, \tag{Leibniz' rule}\\
  0 &= \underbrace{\{f, \{g, h\}\} + \{g, \{h, f\}\} + \{h, \{f, g\}\}}_{
    \{f, \{g, h\}\} + \text{cyclic permutations}}. \tag{Jacobi identity}
\end{align*}
Note that these are the same properties shared by the matrix commutator $[\mat{A},
  \mat{B}] = \mat{A}\mat{B} - \mat{B}\mat{A}$.
:::::

Note that
\begin{gather*}
  \{q_i, p_j\}_{qp} = \delta_{ij}, \qquad
  \{q_i, q_j\}_{qp} = \{p_i, p_j\}_{qp} = 0.
\end{gather*}

:::::{admonition} Canonical Quantization

Once canonically conjugate variables $q$ and $p$ have been identified, one can seek
linear operators $\op{q}$ and $\op{p}$ whose commutator satisfy an analogous commutation
relation:
\begin{gather*}
  \{q, p\}_{qp} \rightarrow \frac{[\op{q}, \op{p}]}{\I\hbar}, \qquad
  \{q_i, p_j\}_{qp} = \delta_{ij} \rightarrow [\op{q}_{i}, \op{p}_{j}] = \I\hbar\delta_{ij}.
\end{gather*}
Once such operators are found, the theory can be [canonically quantized][canonical
quantization] by replacing classical observables the Hamiltonian $H(q, p)$ with the
quantum operators:
\begin{gather*}
  \op{H} = H(\op{q}, \op{p}).  
\end{gather*}
This works for operators that are quadratic in the coordinates and momenta, but can fail
in general, especially if there are higher order terms in both $p$ and $q$ where the
order of operators becomes ambiguous.  For more information, see the associated
discussion with the [Moyal bracket][], which generalizes the [Poisson bracket][].
:::::

## Canonically Conjugate Variables

Starting from a Lagrangian framework, we can identify canonically conjugate variables
$q_i$ and $p_i = \partial L/\partial \dot{q}_i$.  The [Poisson bracket][] allows us to
check whether or not new coordinates $\vect{Q}(\vect{q}, \vect{p}, t)$ and
$\vect{P}(\vect{q}, \vect{p}, t)$ are canonically conjugate: They are if the they satisfy
\begin{gather*}
  \{Q_i, P_j\}_{qp} = \delta_{ij}, \qquad
  \{Q_i, Q_j\}_{qp} = \{P_i, P_j\}_{qp} = 0.
\end{gather*}
Note that these relations hold for any set of conjugate pairs $(\vect{q}, \vect{p})$ and
$(\vect{Q}, \vect{P})$, so we can suppress the subscript $\{\cdot, \cdot\}_{qp} \equiv
\{\cdot, \cdot\}_{QP} \equiv \{\cdot, \cdot\}$: the [Poisson bracket][] expresses an
operation that does not fundamentally depend on the coordinates.  (See [the Poisson bracket in coordinate-free language](https://en.wikipedia.org/wiki/Poisson_bracket#The_Poisson_bracket_in_coordinate-free_language))

:::::{admonition} Do It!  Prove that $\{f, g\}_{qp} = \{f, g\}_{QP}$.
:class: dropdown

We limit our proof to a single variable and suppress the dependence on $t$ -- the
generalization is straightforward.  They key is to note that
\begin{gather*}
  \{f, g\}_{qp} = \{f, g\}_{QP}\underbrace{\{Q, P\}_{qp}}_{1} = \{f, g\}_{QP}.
\end{gather*}
Introducing the comma notation for partial derivatives, $f_{,Q} = \partial f/\partial
Q$ etc., we have 
\begin{gather*}
  \{f, g\}_{qp}
  =
  (f_{,Q}Q_{,q} + f_{,P}P_{,q})(g_{,Q}Q_{,p} + g_{,P}P_{,p})
  -
  (g_{,Q}Q_{,q} + g_{,P}P_{,q})(f_{,Q}Q_{,p} + f_{,P}P_{,p})\\
  =
  \underbrace{(f_{,Q}g_{,P} - g_{,Q}f_{,P})}_{\{f, g\}_{QP}}
  \underbrace{(Q_{,q}P_{,p} - P_{,q}Q_{,p})}_{\{Q, P\}_{qp}}.
\end{gather*}
:::::

## Evolution as a Canonical Transformation

One very important set of canonically conjugate variables is defined by evolution.  Note
that evolution defines an map from [phase space][] into itself:
\begin{gather*}
  \begin{pmatrix}
    q_0 \\
    p_0
  \end{pmatrix}
  \rightarrow
  \begin{pmatrix}
    q(q_0, p_0, t) \\
    p(q_0, p_0, t)
  \end{pmatrix}.
\end{gather*}
I.e., if we take our initial conditions $(q_0, p_0)$ at time $t=t_0$ as a point in
[phase space][], then evolving these for time $t$ gives a new point in [phase space][]
$(Q, P)$ which where the initial point ends up at time $t_0 + t$.  This evolution
defines a **canonical transformation** from the old coordinates $(q, p) \equiv (q_0, p_0)$:
\begin{gather*}
  \begin{pmatrix}
    Q(q_0, p_0, t) = q(q_0, p_0, t_0+t)\\
    P(q_0, p_0, t) = p(q_0, p_0, t_0+t).
  \end{pmatrix}
\end{gather*}
:::::{admonition} Do It!  Show that evolution is indeed a canonical transformation.
:class: dropdown

Hint: show that the [Poisson bracket][] $\{Q, P\}_{q_0p_0} = 1$.  This requires relating
changes at the end of a trajectory to changes in the initial condition.  This can be
done by using the fact that physical trajectories extremize the action.  Recall that
physical trajectories extremize the action
\begin{gather*}
  S(q_0, t_0; q_1, t_1) = \int_{t_0}^{t_1} \d{t}\; L(q, \dot{q}, t), \qquad
  q(t_0) = q_0, \qquad q(t_1) = q_1.
\end{gather*}
These trajectories satisfy the Euler-Lagrange equation:
\begin{gather*}
  \delta S = 0 \qquad \implies\qquad
  \diff{}{t}\pdiff{L}{\dot{q}} = \pdiff{L}{q}.
\end{gather*}
Now consider physical trajectories (i.e. satisfying the Euler-Lagrange equations) that
start and end at slightly **different** positions $q_0 + \d{q_0}$ and $q_1 + \d{q_1}$.
The variation in the action is now
\begin{gather*}
  \d{S} = S_{,t_0}\d{t_0} + S_{,t_1}\d{t_1} + S_{,q_0}\d{q_0} + S_{,q_1}\d{q_1}.
\end{gather*}
The first two terms are related to varying the endpoints of the integral, but note that
they are not simply the Lagrangian because we must change $t_1$ **while readjusting the
trajectory so that $q(t_1+\delta t_1) = q_1$ remains fixed**.  We will return to this in
a later problem, but for now, we fixed $t_0$ and $t_1$ so $\d{t_0} = \d{t_1} = 0$.
To compute the second contribution, we use the equations of motion
\begin{gather*}
  \delta S = \int_{t_0}^{t_1}\d{t}\Biggl(
    \pdiff{L}{\dot{q}}\underbrace{\delta \dot{q}}_{\diff{\delta q}{t}} 
    + 
    \underbrace{\pdiff{L}{q}}_{\diff{}{t}\pdiff{L}{\dot{q}}}\delta q
  \Biggr)
  = \int_{t_0}^{t_1}\d{t}\diff{}{t}\left(
    \pdiff{L}{\dot{q}}\delta q
  \right)
  = p_1\delta{q_1} - p_0\delta{q_0}.
\end{gather*}
Hence, $p_1 = S_{,q_1}$ and $p_0 = -S_{,q_0}$.

Now let $(Q, P) = (q_1, p_1)$ be the new coordinates, and $q=(q_0, p_0)$ be the old
coordinates, i.e., looking at evolution as a phase-space homomorphism:
\begin{gather*}
  \d{P} = \d{S_{,Q}} = S_{,Qq}\d{q} + S_{,QQ}\d{Q},\\
  \d{p} = -\d{S_{,q}} = -S_{,qq}\d{q} - S_{,qQ}\d{Q},\\
  \begin{pmatrix}
    \d{Q}\\
    \d{P}
  \end{pmatrix}
  =
  \begin{pmatrix}
    S_{,QQ} & - 1\\
    -S_{,qQ} & 0
  \end{pmatrix}^{-1}
  \begin{pmatrix}
    S_{,Qq} & 0\\
    S_{,qq} & 1
  \end{pmatrix}
  \begin{pmatrix}
    \d{q}\\
    \d{p}
  \end{pmatrix},\\
  =
  \frac{-1}{S_{,qQ}}
  \begin{pmatrix}
    0 & 1\\
    S_{,qQ} & S_{,QQ}
  \end{pmatrix}
  \begin{pmatrix}
    S_{,Qq} & 0\\
    S_{,qq} & 1
  \end{pmatrix}
  \begin{pmatrix}
    \d{q}\\
    \d{p}
  \end{pmatrix}
  =
  -
  \begin{pmatrix}
    \frac{S_{,qq}}{S_{,qQ}} & \frac{1}{S_{,qQ}}\\
    S_{,Qq} + \frac{S_{QQ}S_{qq}}{S_{,qQ}} & \frac{S_{QQ}}{S_{,qQ}}
  \end{pmatrix}
  \begin{pmatrix}
    \d{q}\\
    \d{p}
  \end{pmatrix}.  
\end{gather*}
Therefore
\begin{gather*}
  \pdiff{Q}{q} = -\frac{S_{,qq}}{S_{,qQ}}, \qquad
  \pdiff{Q}{p} = -\frac{1}{S_{,qQ}}, \qquad
  \pdiff{P}{q} = -S_{,Qq} - \frac{S_{QQ}S_{qq}}{S_{,qQ}}, \qquad
  \pdiff{P}{p} = -\frac{S_{QQ}}{S_{,qQ}},\\
  \{Q,P\}_{qp} = 
  \frac{S_{,qq}}{S_{,qQ}}\frac{S_{QQ}}{S_{,qQ}} 
  - \left(\frac{S_{,Qq}}{S_{,qQ}} + \frac{S_{QQ}S_{qq}}{S_{,qQ}S_{,qQ}}\right)
  = 1.
\end{gather*}
:::::

## Generating Functions

Canonical transformations can be generated by specifying an arbitrary [generating
function][] function of one old and one new coordinate.  There are thus four types of
such functions:
\begin{gather*}
  F_1(q, Q, t), \qquad
  F_2(q, P, t), \qquad
  F_3(p, Q, t), \qquad
  F_4(p, P, t).
\end{gather*}
From these, one can **generate** the omitted coordinates and the new Hamiltonian as
follows:
:::{margin}
To remember these, start from $F_2(q, P, t)$ for which both equations are positive, and
then change sign as needed when you change the variables.
:::
\begin{align*}
  p &= +\pdiff{}{q}F_1(q, Q, t), &  P &= -\pdiff{}{Q}F_1(q, Q, t),\\
  p &= +\pdiff{}{q}F_2(q, P, t), &  Q &= +\pdiff{}{P}F_2(q, P, t),\\
  q &= -\pdiff{}{p}F_3(p, Q, t), &  P &= -\pdiff{}{Q}F_3(p, Q, t),\\
  q &= -\pdiff{}{p}F_4(p, P, t), &  Q &= +\pdiff{}{P}F_4(p, P, t).
\end{align*}
Or, more compactly:
\begin{align*}
  p &= +\pdiff{}{q}F, & P &= -\pdiff{}{Q}F,\\
  q &= -\pdiff{}{q}F, & Q &= \pdiff{}{P}F.
\end{align*}
Note that these equations give implicit relationships between the old and new
coordinates which must be solved to complete the transformation.  Once this is done, the
new Hamiltonian in all cases is:
\begin{gather*}
  H'(Q, P, t) = H(q, p, t) + \pdiff{F}{t},
\end{gather*}
which will give Hamilton's equations
\begin{gather*}
  \dot{Q} = \pdiff{H'(Q, P, t)}{P}, \qquad
  \dot{P} = -\pdiff{H'(Q, P, t)}{Q}.
\end{gather*}

The general approach for deriving these is to note that the most general transformation
from $(q, p)$ to $(Q, P)$ preserves the equations of motion after minimizing the action
consists of three things: 1) scaling the Lagrangian, (just changes the time unit), 2)
effecting a coordinate change, and 3) adding a total derivative.
\begin{gather*}
  L(q, \dot{q}, t\bigr) = \alpha L'(Q, \dot{Q}, t) + \diff{F_1(q, Q, t)}{t}.
\end{gather*}
The scaling just changes the units of time, so we neglect this ($\alpha = 1$).  It is
the addition of a total derivative that provides additional freedom over the standard
Lagrangian approach of writing the new coordinates as a simple function of the old
$Q=Q(q, t)$.

Expressing this relationship in terms of differentials, and introducing the
Hamiltonians, we have
\begin{align*}
  \d{F_1} &= (L - L')\d{t} = (p\dot{q} - H)\d{t} - (P\dot{Q} - H')\d{t}\\
          &= p\d{q} - P\d{Q} + (H' - H)\d{t} ,\\
  \d{F_1(q, Q, t)} &= \pdiff{F_1}{q}\d{q} + \pdiff{F_1}{Q}\d{Q} + \pdiff{F_1}{t}\d{t}.
\end{align*}
:::{margin}
For completeness
\begin{align*}
  F_1(q, Q, t) &= F_1\\
  F_2(q, P, t) &= F_1 + PQ\\
  F_3(p, Q, t) &= F_1 - pq\\
  F_4(p, P, t) &= F_1 + PQ - pq\\
\end{align*}
:::
Equating these, and collecting the differentials gives our formula above for the
canonical transformation generated by $F_1(q, Q, t)$.  To obtain the relationship for
another function, say $F_2(q, P, t)$ we need to switch $-P\d{Q}$ to $Q\d{P}$, which can
be done by adding $\d{QP} = Q\d{P} + P\d{Q}$.  I.e., $F_2 = F_1 + QP$:
\begin{align*}
  \d{F_2} = \d{(F_1 + QP)} 
  &= p\d{q} + Q\d{P} + (H' - H)\d{t},\\
  \d{F_2(q, P, t)} &= \pdiff{F_2}{q}\d{q} + \pdiff{F_2}{P}\d{P} + \pdiff{F_2}{t}\d{t}.
\end{align*}
This accounts for the sign change.

:::{admonition} Do It!  Show explicitly that this procedure works.
:class: dropdown

Our task is to show that if
\begin{gather*}
  F = F_2\bigl(q, P(q, p, t), t\bigr), \qquad
  H' = H + F_{,t},\\
  p = F_{,q} \equiv \pdiff{F}{q}, \qquad
  Q = F_{,P} \equiv \pdiff{F}{P},
\end{gather*}
then
\begin{gather*}
  \begin{pmatrix}
    \dot{q}\\
    \dot{p}
  \end{pmatrix}
  =
  \begin{pmatrix}
    H_{,p}\\
    -H_{,q}
  \end{pmatrix}
  \implies
  \begin{pmatrix}
    \dot{Q}\\
    \dot{P}
  \end{pmatrix}
  =
  \begin{pmatrix}
    H'_{,P}\\
    -H'_{,Q}
  \end{pmatrix}.
\end{gather*}
One can proceed by brute force.  The idea here is to choose three independent variables,
and then equate partials.  We choose $(q, P, t)$ since these are the natural variables
for $F$:
\begin{align*}
  \d{Q} &= \d{F_{,P}} = F_{,Pq}\d{q} + F_{,PP}\d{P} + F_{,Pt}\d{t},\\
  \d{p} &= \d{F_{,q}} = F_{,qq}\d{q} + F_{,qP}\d{P} + F_{,qt}\d{t},\\
  \d{H'} &= H'_{,Q}\d{Q} + H'_{,P}\d{P} + H'_{,t}\d{t},\\
         &= H'_{,Q}F_{,Pq}\d{q} + (H'_{,Q}F_{,PP} + H'_{,P})\d{P} 
            + (H'_{,Q}F_{,Pt} + H'_{,t})\d{t},\\
  \d{H} &= H_{,q}\d{q} + H_{,p}\d{p} + H_{,t}\d{t},\\
        & (H_{,q} + H_{,p}F_{,qq})\d{q} + H_{,p}F_{,qP}\d{P} + (H_{,p}F_{,qt} + H_{,t})\d{t},\\
   \d{F_{,t}} &= F_{,tq}\d{q} + F_{,tP}\d{P} + F_{,tt}\d{t}.
\end{align*}
Equating $H' = H + F_{,t}$ gives the following conditions (by varying only one of
  $\d{q}$, $\d{P}$, and $\d{t}$ at a time respectively):
\begin{align*}
    H'_{,Q}F_{,Pq} &= H_{,q} + H_{,p}F_{,qq} + F_{,tq},\\
    H'_{,Q}F_{,PP} + H'_{,P} &= H_{,p}F_{,qP} + F_{,tP},\\
    H'_{,Q}F_{,Pt} + H'_{,t} &= H_{,p}F_{,qt} + H_{,t} + F_{,tt}.
\end{align*}
Solving, we have
\begin{align*}
    H'_{,Q} &= \frac{H_{,q} + H_{,p}F_{,qq} + F_{,tq}}{F_{,Pq}},\\
    H'_{,P} &= H_{,p}F_{,qP} + F_{,tP} - F_{,PP}H'_{,Q}.
\end{align*}
We can then solve for $\dot{Q}$ and $\dot{P}$:
\begin{align*}
  \dot{P} &= -\frac{H_{,q} + F_{,qq}H_{,p} + F_{,qt}}{F_{,qP}} = -H'_{,Q},\\
  \dot{Q} &= H_{,p}F_{,Pq} + F_{,Pt} + F_{,PP}\dot{P},\\
          &= H_{,p}F_{,Pq} + F_{,Pt} - F_{,PP}H'_{,Q} = H'_{,P}.
\end{align*}
Q.E.D.
:::

## Hamilton-Jacobi Equation

One important transformation is to find a generating function $F_{1}(q, Q, t) = S(q, Q,
t)$ such that $H' = 0$.  This transformation defines coordinates which are completely
independent of time $\dot{Q} = \dot{P} = 0$.  The conditions $H'(Q, P, t) = H(q, p, t) +
S_{,t}$ and $p = S_{,q}$ define the [Hamilton-Jacobi equation][]:
:::{margin}
Since the coordinate $Q$ plays no role in this equation, one could equivalently use a type-2
generating function $S(q, t) = F_2(q, P, t)$.  The important point is to choose
constants of motion for either $P$ or $Q$.
:::
\begin{gather*}
  H\left(q, \pdiff{S(q, Q, t)}{q}, t\right) + \pdiff{S(q, Q, t)}{t} = 0.
\end{gather*}

:::::{admonition} Do It!  Show that the classical action $S(q, Q, t)$ is a solution.

Show that classical action for a physical trajectory $q(t)$ with $q(t_0) = q_0$ and
$q(t_1) = q_1$,
\begin{gather*}
  S(q_0, t_0; q_1, t_1) = \int_{t_0}^{t_1} L(q, \dot{q}, t)\d{t}, \qquad
  \diff{}{t}\pdiff{L(q, \dot{q}, t)}{\dot{q}} = \pdiff{L(q, \dot{q}, t)}{q},
\end{gather*}
gives a solution $S(q, Q, t)$ to the Hamilton-Jacobi equation with $Q=q_0$, and $t=t_1$.

::::{solution}
Computing the total differential, we have
\begin{gather*}
  \d{S} = S_{,t_0}\d{t_0} + S_{,t_1}\d{t_1}
        + \underbrace{S_{,q_0}}_{-p_0}\d{q_0} 
        + \underbrace{S_{,q_1}}_{p_1=p}\d{q_1}
\end{gather*}
as we did above when showing that evolution is a canonical transformation.  We now focus
on the first partials.

:::{warning}
There is a subtly here: $S_{,t_1} \neq L_1$ as we might expect from varying the upper
endpoint.  This is because the action is defined over physical trajectories.  Computing
the partial $S_{,t_1}$ **requires readjusting** $q(t)$ so that $q(t_1+\delta t_1) = q_1$
remains fixed!
:::

Suppose that $q(t)$ is a physical trajectory satisfying some
initial conditions $q(t_0) = q_0$ and $\dot{q}(t_0) = \dot{q}_0$.  In this case, the
functions $q(t)$, $\dot{q}(t)$, and thus $L(q, \dot{q}, t)$ are simply functions of
time.  **For this trajectory only**, we can say that $\delta S = L_1\delta t_1$, or
similar by extending backwards in time.  What this really means is:
\begin{align*}
  \overbrace{S(q_0, t_0; q_1 + \delta q_1, t_1 + \delta t_1) - S(q_0, t_0; q_1, t_1)}
  ^{S_{,q_1}\delta q_1 + S_{,t_1}\delta t_1}
  &= \overbrace{L(q_1, \dot{q}_1, t_1)}^{L_1} \delta t_1,\\
  \underbrace{S(q_0+\delta q_0, t_0 + \delta t_0; q_1, t_1) - S(q_0, t_0; q_1, t_1)}
  _{S_{,q_0}\delta q_0 + S_{,t_0}\delta t_0}
  &= -\underbrace{L(q_0, \dot{q}_0, t_0)}_{L_0} \delta t_0.
\end{align*}
Hence, the relationships we need follow from $\delta{q}_i / \delta t_i = \dot{q}_i$:
\begin{gather*}
  S_{,t_1} = \underbrace{L_1 - \overbrace{S_{,q_1}}^{p_1}\dot{q}_1}_{-H_1}, \qquad
  S_{,t_0} = \underbrace{-L_0 - \overbrace{S_{,q_0}}^{-p_0}\dot{q}_0}_{H_0}.
\end{gather*}
The total differential is thus:
\begin{gather*}
  \d{S} = \underbrace{S_{,t_0}}_{H_0}\d{t_0} 
        + \underbrace{S_{,t_1}}_{-H_1}\d{t_1}
        + \underbrace{S_{,q_0}}_{-p_0}\d{q_0} 
        + \underbrace{S_{,q_1}}_{p_1=p}\d{q_1}.
\end{gather*}
Thus, setting $Q=q_0$, $q=q_1$, $p=p_1$, and $t=t_1$ we have
\begin{gather*}
  H\Bigl(q, \overbrace{\pdiff{S(q, Q, t)}{q}}^{S_{,q_1}=p_1=p}, t\Bigr) = 
  H(q, p, t) = - \pdiff{S(q, Q, t)}{t}.
\end{gather*}
Q.E.D.
::::
:::::

## Hamilton's Principal and Characteristic Functions

:::{margin}
The Hamilton-Jacobi equation generalizes to many degrees of freedom, so we introduce
vectors $\vect{q}$ and $\vect{Q}$ here.
:::
The typical approach for solving the Hamilton-Jacobi equation is to look for
separability.  For example, if the Hamiltonian is time-independent, then we can write
\begin{gather*}
  S(\vect{q}, \vect{Q}, t) = W(\vect{q}, \vect{Q}) + f(t), \qquad
  H\left(\vect{q}, \pdiff{W(\vect{q}, \vect{Q})}{\vect{q}}\right) = -\pdiff{f(t)}{t} = \text{const.} = E.
\end{gather*}
The left-hand-side contains no time, while the right-hand-side contains no $\vect{q}$ and $\vect{Q}$.
Thus, the two sides must both be constant.  That constant is the conserved Hamiltonian
implied by Noether's theorem when one has time-translation invariance.  In most cases,
that constant is the energy $E$ is the energy of the system, so we use this notation,
and solve for $f(t) = -Et$:
\begin{gather*}
  S(\vect{q}, \vect{Q}, t) = W(\vect{q}, \vect{Q}) - Et.
\end{gather*}
This differentiates between the time-dependent action $S(\vect{q}, \vect{Q}, t)$, which
is called [Hamilton's principal function][], and the time-independent action
$W(\vect{q}, \vect{Q}) = S + Et$, which is called [Hamilton's characteristic function][].


# Action-Angle Coordinates

Another related set of coordinates called [action-angle coordinates][]. We will give the
general formulation below, but the basic idea applies to time-independent Hamiltonians
where the Hamilton-Jacobi equation is separable so that the motion can be considered
quasi-periodic.  I.e., each of the coordinates executes either a libration $q_k(t+T_k) =
q_k(t)$ or a rotation (where the coordinate is periodic like an angle) $q_k + Q_k \equiv
q_k$.  In both cases, we can define the **action** over a complete period/cycle:
:::{margin}
The factor of $(2\pi)^{-1}$ here is not conventional, but ensures that the resulting
frequencies $\omega_k$ are angular frequencies.  This is conventional in many fields of
physics like particle theory where one keeps the factors of $(2\pi)^{-1}$ with the
momentum integrals when doing Fourier transforms.  We use the notation $\vect{I} =
\vect{J}/2\pi$ for our "angular" action variables to avoid confusion with other literature.
:::
\begin{gather*}
  I_k(E) = \oint q_k\,\frac{\d{p_k}}{2\pi},
\end{gather*}
where the integral is over the path defined by the condition of constant energy
$E=E(q_k, p_k)$.
The energy of such a system is a function only of these actions
\begin{gather*}
  H'(\vect{I}) = E(\vect{I}),
\end{gather*}
hence the corresponding canonical coordinates (the **angle coordinates**) have constant
velocity:
\begin{gather*}
  \dot{\theta}_{k} = \pdiff{H'(\vect{I})}{I_k} = \text{const.},\qquad
  \theta_k(t) = \theta_k(0) + \dot{\theta}_{k} t.
\end{gather*}

The geometric structure is one of an $n$-torus in phase space defined by the constant
energy manifold $E = E(\vect{q}, \vect{p}) = E(\vect{I})$.  The periodic trajectories
are orbits around these tori, and the angle variables specify where on the orbits the
system is at a given time, parameterized in such a way as to advance at a constant rate
in time.

The process of finding these is straightforward
1. Calculate the new generalized momenta $\vect{I}$ -- the **action variables**:
   \begin{gather*}
     I_k = \oint q_k\,\frac{\d{p_k}}{2\pi} = \tfrac{1}{2\pi}\oint q_{k}\dot{p}_k \, \d{t}.
   \end{gather*}
2. Express the original Hamiltonian in terms of these $H'(\vect{I}) = E(\vect{I})$.
3. Differentiate to get the frequencies:
   \begin{gather*}
     \omega_k = \pdiff{H'}{I_k}, \qquad
     \theta_k = \theta_k(0) + \omega_k t.
   \end{gather*}

:::{margin}
***Incomplete.** This section gives the spirit, and works for single degree of freedom,
but I need to generalize it properly.*
:::
## Action-Angle Generating Function

These coordinates are generated by the type-2 generating function $W(\vect{q}, \vect{I})$ called
[Hamilton's characteristic function][]
\begin{gather*}
  W(\vect{q}, \vect{I}) = S(q, \vect{I}, t) + E(\vect{I}) t.
\end{gather*}

For a single coordinate, the action-angle coordinates can be generated from the
following type-2 function with $P = I$ being the area in phase space (divided by $2\pi$
for our units):
\begin{gather*}
  W(q, I) = \int_0^{q}p(q, I)\d{q},\\
  Q = \theta = \pdiff{W}{I} = \int_0^{q}\pdiff{p(q, I)}{I}\d{q},\qquad
  p = \pdiff{W}{q} = p(q, I),\qquad
  H' = H.
\end{gather*}

:::{margin}
The time-independent function $W = S + Et$ here is called [Hamilton's characteristic
function][], in contrast with [Hamilton's principal function][] $S$ which solves the
Hamilton-Jacobi Equation.
:::
For applications to {ref}`sec:CanonicalPerturbationTheory`, we would like to have a form
of action-angle variables where the transformed Hamiltonian $H' = 0$.  To do this, we
simply note that the Hamiltonian is a constant function of the action variables
$H'(\vect{I}) = E$.  Thus, we simply use the type-2 generating functional
\begin{gather*}
  S(\vect{q}, \vect{I}, t) = W(\vect{q}, \vect{I}) - H'(\vect{I})t,
\end{gather*}
which is [Hamilton's principal function][] as before.  This defines the following
coordinates
:::{margin}
The old momenta are the same:
\begin{gather*}
  \vect{p} = \pdiff{S}{\vect{q}} = \pdiff{W}{\vect{q}}.
\end{gather*}
:::
\begin{gather*}
  \vect{Q} = \pdiff{S}{\vect{I}} 
  = \overbrace{
    \pdiff{W}{\vect{I}} - \pdiff{H'}{\vect{I}}
  }^{\vect{\theta}(t) - \vect{\omega} t}
  = \vect{\theta}_0.
\end{gather*}
I.e., Recall that the angle variables march forward with constant angular velocity
$\vect{\omega}$:
\begin{gather*}
  \vect{\theta}(t) = \vect{\theta}_0  + \vect{\omega} t.
\end{gather*}
The new coordinates are just the (constant) starting angles $\vect{\theta}_0$.

In summary, we have two transformation:
\begin{align*}
  F_2(\vect{q}, \vect{I}) &= W(\vect{q}, \vect{I}): &
  (\vect{q}, \vect{p}) &\rightarrow (\vect{\theta}, \vect{I}),
  \tag{Action-Angle}\\
  F_2(\vect{q}, \vect{I}, t) &= S(\vect{q}, \vect{I}, t): &
  (\vect{q}, \vect{p}) &\rightarrow (\vect{\theta}_0, \vect{I}), \tag{Hamilton-Jacobi}\\
\end{align*}
where
\begin{gather*}
  S(\vect{q}, \vect{I}, t) = W(\vect{q}, \vect{I}) - E(\vect{I})t,
  \qquad
  \vect{\theta}(t) = \vect{\theta}_0 + \vect{\omega}t,
\end{gather*}
and $\vect{I}$, $\vect{\theta}_0$, and $\vect{\omega}$ are constants.

## Formalism

:::{margin}
This discussion follows {cite}`Arnold:1989` §10: "Introduction to perturbation theory".
It is somewhat incomplete, but gives you a flavour. See {cite}`Arnold:1989` for details.
:::
The formalism for a system of $n$ particles starts with identifying $n$ integrals of
motion $F_{k}(\vect{q}, \vect{p})$ with $F_1(\vect{q}, \vect{p}) = H(\vect{q},
\vect{p})$ which are **in involution** with each other, meaning that their Poisson
brackets are zero:
\begin{gather*}
  \{F_i, F_j\}_{pq} = 0.
\end{gather*}
Liouville proved the following about the level set 
\begin{gather*}
  M_{\vect{f}} = \{(\vect{q}, \vect{p}) \quad |\quad \vect{F}(\vect{q}, \vect{p}) = \vect{f}\}.
\end{gather*}
:::{margin}
From {cite}`Arnold:1989` §10: "Introduction to perturbation theory".
:::
> 1. $M_{\vect{f}}$ is a smooth manifold, invariant under phase flow with the Hamiltonian
>    $H = F_1$.
> 2. If the manifold $M_{\vect{f}}$ is compact and connected, then it is diffeomorphic
>    to the $n$-dimensional torus
>    \begin{gather*}
       T^{n} = \{\vect{\phi} \mod 2\pi\}
     \end{gather*}
> 3. The phase flow with Hamiltonian $H = F_1$ determines conditionally periodic motion
>    on $M_{\vect{f}}$.  I.e.: 
>    \begin{gather*}
       \dot{\vect{\phi}} = \vect{\omega}(\vect{f}).
     \end{gather*}
> 4. The canonical equations can be integrated by quadratures.  (I.e., although we only
>    have $n$ integrals, we can get all $2n$ integrals of motion.

The angle coordinates will be
\begin{gather*}
  \vect{\phi}(t) = \vect{\phi}(0) + \vect{\omega}(\vect{f}) t
\end{gather*}
but we must still identify the conjugate momenta $\vect{I}$, which will not in general
be the functions $\vect{F}$.  They are given as discussed above, by computing the action
about the periodic orbit *(though, it should not be obvious from this discussion that
these do indeed do the job)*.

:::::{admonition} Example: Harmonic Oscillator
:class: dropdown

The harmonic oscillator has the following solution:

\begin{gather*}
  \newcommand{\t}{\tau}
  H(q, p) = \frac{p^2}{2m} + \frac{m\omega^2 q^2}{2}, \qquad 
  \t= t-t_0,\\
  q(t) = A\cos \omega \t,\qquad
  p(t) = -m \omega A\sin\omega \t, \qquad
  E = \frac{m\omega^2A^2}{2}.
\end{gather*}
1. From this, we can compute the action variable
   \begin{gather*}
     I = \oint p\, \frac{\d{q}}{2\pi} = \frac{1}{2\pi}\int p\dot{q}\,\d{t}
       = \frac{1}{2\pi}\int_0^{2\pi/\omega} m \omega^2 A^2\sin^2\omega \t \,\d{\t}
       = \frac{m \omega A^2}{2} = \frac{E}{\omega}.
   \end{gather*}
2. We then invert this to express $E(I)$:
   \begin{gather*}
      E(I) = I\omega.
   \end{gather*}
3. Now we differentiate to obtain the angular velocity and angle variable:
   \begin{gather*}
     \omega = \pdiff{E}{I}, \qquad \phi(\t) = \omega \t.
   \end{gather*}
   Thus, we see that we recover the angular frequency.

:::{note}
In this case it was trivial to invert the first expression to obtain $E(I)$, but this
might be complicated in general.  Since one really only needs the partials to
identify the angle variables, it is easiest to express the differentials from which the
angular velocities can be obtained implicitly:
\begin{gather*}
  \d{E} = \sum_k \omega_k \d{I}_k = \vect{\omega}\cdot\d\vect{I}.
\end{gather*}
:::

We now have the coordinate transformation:
\begin{gather*}
  q = \sqrt{\frac{2I}{m\omega}} \cos \phi, \qquad
  p = -\sqrt{2m\omega I}\sin\phi.
\end{gather*}
The generating function is
\begin{gather*}
  S(q, I) = \int_{0}^{q} p(q, I)\d{q}.
\end{gather*}
This is most easily evaluated by considering $\phi(q, I)$:
\begin{gather*}
  \phi(q, I) = \cos^{-1}\left(\sqrt{\frac{m\omega}{2I}}q\right), \qquad
  \phi_{,q} = \sqrt{\frac{m\omega}{2I}}\frac{-1}{\sin\phi},
  \qquad \phi_{,I} = \frac{\cos\phi}{2I\sin\phi}\\
  S(q, I) = I\int_{0}^{\phi}\sin^2\phi \;\d{\phi}
          = I\Bigl(\phi(q,I) - \tfrac{1}{2}\sin 2\phi(q,I)\Bigr).
\end{gather*}
As a check:
\begin{gather*}
  \pdiff{S}{q} = I\overbrace{(1 - \cos 2\phi)}^{2\sin^2\phi}\phi_{,q} = p,\\
  \pdiff{S}{I} = \phi - \tfrac{1}{2}\sin 2\phi + 2I\sin^2\phi\; \phi_{,I} = \phi.
\end{gather*}

Note that this is [Hamilton's characteristic function][] $W = S + Et$ expressed in
the appropriate coordinates as a type-2 generating function.  **Needs checking.***

*Note: When trying to derive these types of relationships, it is much easier to work in
natural units where $m\omega = 2I = 1$.  The latter throws some of the baby out with the
bathwater (you lose the dependence on $I$ which is needed for derivatives) but
simplifies algebra.*
:::::


(sec:CanonicalPerturbationTheory)=
## Canonical Perturbation Theory

To organize the perturbative calculation, one can turn to **canonical perturbation
theory** which maintains the canonical relationship between variables as we add
perturbative corrections.

We consider a Hamiltonian system with conjugate variables $(\vect{q}, \vect{p})$:
\begin{gather*}
  H(\vect{q}, \vect{p}, t) = H_0(\vect{q}, \vect{p}, t) 
  + \epsilon H_1(\vect{q}, \vect{p}, t).
\end{gather*}
We now assume that a solution to the Hamiltonian problem $H_0(\vect{q}, \vect{p}, t)$ is
known such that we can effect a canonical transform via the generating function
$F_2(\vect{q}, \vect{P}, t)$ to a new set of conjugate variables $(\vect{Q}, \vect{P})$
with Hamiltonian $K_0(\vect{Q}, \vect{P}, t)$:
\begin{gather*}
  p_i = \pdiff{F_2}{q_i}, \\
  Q_i = \pdiff{F_2}{P_i}, \\
  K_0 = H_0 + \pdiff{F_2}{t} = 0.
\end{gather*}
:::{margin}
Recall that a canonical transformation is independent of the actual equations of motion:
it can be applied to any system.
:::
Effecting the same transform on the full Hamiltonian $\vect{H}$ gives:
\begin{gather*}
  K = \epsilon H_1, \\
  \dot{\vect{Q}} = \pdiff{K}{\vect{P}} = \epsilon \pdiff{H_1}{\vect{P}}, \\
  \dot{\vect{P}} = -\pdiff{K}{\vect{Q}} = -\epsilon \pdiff{H_1}{\vect{Q}}.
\end{gather*}
This is still exact, but note that the motion is now of order $\epsilon$.

The first approach, explained in {cite:p}`Goldstein:2000`, is to use these equations to
recursively build successive approximations.  Thus, substituting the order $\epsilon^0$
solution on the right-hand-side of these equations gives a linear system that can be
integrated to obtain the order $\epsilon^1$ approximation.  The order $\epsilon^1$ solution
can then be inserted to compute the order $\epsilon^2$ approximation, etc.

Explicitly, let $\vect{y} = (\vect{Q}, \vect{P})$. The full solution will have the
form:
\begin{gather*}
  \vect{y} = \vect{y}_0 + \epsilon \vect{y}_1 + \epsilon^2 \vect{y}_2 + \cdots,\\
  \dot{\vect{y}}_{n} = \epsilon
  \begin{pmatrix}
    \mat{0}& \mat{1}\\
    -\mat{1} & \mat{0} 
  \end{pmatrix}
  \cdot
  \left.\pdiff{H_1(\vect{y}, t)}{\vect{y}}\right|_{\vect{y} = \vect{y}_{n-1}}.
\end{gather*}
In the case of a single degree of freedom $\vect{y} = (Q, P)$ we have
\begin{gather*}
  \begin{pmatrix}
    Q(t)\\
    P(t)
  \end{pmatrix}
  =
  \begin{pmatrix}
    Q_0\\
    P_0
  \end{pmatrix}
  +
  \epsilon
  \begin{pmatrix}
    Q_1(t)\\
    P_1(t)
  \end{pmatrix}
  +
  \epsilon^2
  \begin{pmatrix}
    Q_2(t)\\
    P_2(t)
  \end{pmatrix}
  +\cdots,\\
  \begin{pmatrix}
    \dot{Q}_{n+1}(t)\\
    \dot{P}_{n+1}(t)
  \end{pmatrix}
  =
  \left.
  \begin{pmatrix}
    \pdiff{H_1(Q, P, t)}{P}\\
    -\pdiff{H_1(Q, P, t)}{Q}
  \end{pmatrix}
  \right|_{Q=Q_n(t), P=P_n(t)}.
\end{gather*}
Note that if $H_1(Q,P)$ is independent of time, then, since $Q_0$ and $P_0$ are
constant, $(Q_1, P_1)$ is linear in time, $(Q_2, P_2)$ is quadratic, etc., and all
integrals are trivial.

:::{important}

Note: By itself, applying this formalism for canonical perturbation theory does not
ensure that you won't have secular terms.  I.e. if you try to apply these equations to
the transformation to constant coordinates $(\vect{q}_0, \vect{p}_0)$ generated by
$F_2(\vect{q}, \vect{q}_0, t) = S(\vect{q}_0, t_0; \vect{q}, t)$, you will generally end
up with linearly increasing perturbations.

You must also use action-angle coordinates as generated by the $F_2(\vect{q}, \vect{I},
t) = S(\vect{q}, \vect{I}, t)$ described above.  In this case, the secular terms give
rise to changes in the angular frequency $\vect{\omega}$, preserving the periodic nature
of the solutions.

For details, see {ref}`sec:PendulumExample`.
:::


## Adiabatic Invariants

:::{margin}
This could easily be implemented by varying the length, e.g. by putting a mass on a
string that passes through a hole.  Pulling slowly on the string will vary the length.
Alternatively, the pendulum could be place in an elevator, or perhaps heat could be used
to change the spring constant of a spring.
:::
For 1D systems, the action variable $I$ is an **adabatic invariant**, which we explain
here by way of an example.  Consider a harmonic oscillator with a time-dependent
frequency that varies slowly
\begin{gather*}
  H(q, p, t) = \frac{p^2}{2m} + \frac{m}{2}\omega^2(t) q^2, \qquad
  \ddot{q} = -\omega^2(t) q.
\end{gather*}
Since the Hamiltonian now depends on time, the energy will not be conserved as we change
the frequency, i.e., work may be done on the system.  However, if we vary $\omega(t)$
slowly enough, then we might expect that the energy will return if we take the system
slowly (adiabatically) away from $\omega_0$ but then slowly bring it back.  The question
is then, how does the energy depend on $\omega(t)$ in the adiabatic limit?

For constant $\omega$ we can find the action-angle variables (see the example above):
\begin{gather*}
  I = \frac{E}{\omega}, \qquad
  \phi = \phi_0 + \omega t,
\end{gather*}
and we shall now show that $I$ is an **adiabatic invariant**, meaning that it is
quadratic $I = I_0 + O(\dot{\omega}^2)$ for small $\dot{\omega}$.  Thus, the answer to
the question posed above is:
\begin{gather*}
  E(t) = I\omega(t) + O(\dot{\omega}^2).
\end{gather*}

:::::{admonition} Do It! Find a $g(t)$ so that $2J = q^2/g^2 + (g\dot{q} - q\dot{g})^2$ is constant.
:class: dropdown

**Incomplete: See Tong**

Constants of motion should have zero Poisson bracket with the Hamiltonian,
\begin{gather*}
  \{J, H\}_{pq} = 0 
  = J_{,q}\underbrace{H_{,p}}_{\dot{q}} - J_{,p}\underbrace{H_{,q}}_{-\dot{p}} = 0,\\
  J_{,q}\frac{p}{m} + J_{,p}m \omega^2(t)q = 0,
\end{gather*}
so we start by expressing this constant in terms of $p = m \dot{q}$:
\begin{gather*}
  2J(q, p, t) = \frac{q^2}{g^2} + \left(g\frac{p}{m} - q\dot{g}\right)^2,\\
  J_{,q} = \frac{q}{g^2} - \dot{g}\left(g\frac{p}{m} - q\dot{g}\right), \qquad
  J_{,p} = \frac{g}{m}\left(g\frac{p}{m} - q\dot{g}\right).
\end{gather*}
Thus, we are looking for a function $g(t)$ which satisfies:
\begin{gather*}
  -\frac{qp}{mg^2} + \dot{g}\frac{p}{m}\left(g\frac{p}{m} - q\dot{g}\right)
  = qg\left(\frac{qg}{m} - q\dot{g}\right) \omega^2(t),\\
  -\frac{qp}{mg^2} + \left(\dot{g}\frac{p}{m} - qg\omega^2(t)\right)\left(g\frac{p}{m} -
  q\dot{g}\right) = 0
\end{gather*}


:::::

:::{warning}
Exactly what slowly or adiabatic means is quite subtle.  See {cite}`Wells:2006` for a
discussion.  Averaging over cycles place a key role that we have not discussed here.
:::











## WKB Approximation

The traditional approach for the [WKB approximation][] of quantum mechanics is to
express the wavefunction as follows, then insert it into the Schrödinger equation:

```{margin}
To simplify the notation here, we use:
\begin{gather*}
  S' = \pdiff{S(x, t)}{x},\\
  \dot{S} = \pdiff{S(x,t)}{t}.
\end{gather*}
```

\begin{align*}
  \psi(x, t) &= \exp\left\{\frac{i}{\hbar}W(x,t)\right\},\\
  \psi'(x, t) &= \frac{i}{\hbar}W'(x, t)\psi(x, t),\\ 
  \psi''(x, t) &= \left(\frac{i}{\hbar}W''(x, t) - \frac{W''(x,
  t)}{\hbar^2}\right)\psi(x, t),\\
  0 &= \left(\frac{(W')^2 - \I\hbar W''}{2m} + V + \dot{W}\right)\psi.
\end{align*}

Expanding $W$ in powers of $\hbar$, we have the following lowest two orders:

\begin{align*}
  &W(x,t) = S(x,t) - \I\hbar \log A + \order(\hbar^2),\\
  &W'(x,t) = S'(x,t) - \I\hbar \frac{A'}{A} + \order(\hbar^2),\\
  &W''(x,t) = S''(x,t) - \I\hbar \frac{A A'' - (A')^2}{A^2} + \order(\hbar^2),\\
  \text{Order $\hbar^0$:}\quad &\frac{(S')^2}{2m} + V(x,t) + \dot{S} 
  = H(x, S', t) + \dot{S} = 0,\\
  \text{Order $\hbar^1$:}\quad &\frac{S''}{2m} + \frac{A'S'}{mA} + \frac{\dot{A}}{A} = 0,\\
  &\psi_{WKB}(x, t) = A(x,t)\exp\left\{\frac{i}{\hbar}S(x,t)\right\},\\
\end{align*}

### $\order(\hbar^0)$: Hamilton-Jacobi Equation

The order $\hbar^0$ equation is the well-known Hamilton-Jacobi equation, which is
satisfied by the classical action as a function of initial and final states.

### $\order(\hbar^1)$: Continuity Equation

The order $\hbar^1$ equation is the well-known continuity equation, expressing the
conservation of particle number or probability.  To see this, multiply through by
$2A^2$:

\begin{gather*}
  \frac{A^2 S'' + 2S'A'A}{m} + 2\dot{A}A = 0\\
  \left(\frac{S'}{m}A^2\right)' + \pdiff{A^2}{t} = 0.
\end{gather*}

This is the 1-dimensional form of the familiar continuity equation, once we identify
$S'$ as the momentum, and $A^2$ as the probability density:

\begin{gather*}
  \vect{\nabla}\cdot\vect{j} + \dot{n} = 0,\\
  n \equiv A^2 = \abs{\psi}^2, \qquad
  \vect{v} \equiv \frac{\vect{p}}{m} = \frac{\vect{\nabla} S}{m}, \qquad
  \vect{j} = n\vect{v}.
\end{gather*}

````{margin}
Hint: Use the form above with $A^2 = \partial S'/\partial x_0$ (the constants do not
matter), and the properties $S' = p$, $\dot{S} = -H$ to express the result as:

\begin{gather*}
  \pdiff{}{x_0}Q(x) = 0.
\end{gather*}

where you can argue that the quantity $Q(x)$ is independent of $x_0$.
````
```{admonition} Exercise

Show that the order $\hbar^1$ equation is satisfied by

\begin{gather*}
  A(x, t) = \sqrt{-\frac{\partial^2 S(x, t; x_0, t_0) / (2\pi \I\hbar)}
                        {\partial x\partial x_0}}
\end{gather*}

as given by the path integral formulation.  *Note: this is only valid up to an overall
constant giving the dependence of $A(x, t)$ on $x$ and $t$.  One must adjust this overall
constant to normalize the wavefunction.*
```
```{admonition} Solution
:class: dropdown

Following the hint, we express $A^2 = \partial S'/\partial x_0 \equiv S'_{,x_0}$ (using
Einstein's notation).  We are trying to show that:

\begin{gather*}
  \left(\frac{S' S'_{,x_0}}{m}\right)' + \pdiff{S'_{,x_0}}{t} = 0.
\end{gather*}

We proceed using the linearity of the derivatives to rearrange the left-hand-side as:

\begin{gather*}
  \frac{1}{m}\pdiff{}{x}\Bigl(
    \overbrace{\pdiff{S}{x}\frac{\partial^2 S}{\partial x\partial x_0}}
             ^{\tfrac{1}{2}\partial (S')^2/\partial x_0}
  \Bigr)
  + \frac{\partial^2 \overbrace{\dot{S}}^{-H}}{\partial x\partial x_0} = \\
  \frac{\partial^2}{\partial x_0 \partial x}\left(
    \frac{(S')^2}{2m} - H\right) = -\pdiff{V'(x, t)}{x_0} = 0,
\end{gather*}

because the potential does not depend on the initial position $x_0$.
```

```{admonition} Exercise: Geometry
Note that since $S' = p$, the factor $A$ can be expressed as
\begin{gather*}
  n(x, t) \propto A^2(x, t) \propto \pdiff{p(x,  t;x_0, t_0)}{x_0}.
\end{gather*}
Explain this geometrically using Liouville's theorem and the requirement that classical
evolution conserves particle number
\begin{gather*}
  n_0(x_0, t_0)\d{x_0} = n(x, t)\d{x}
\end{gather*}
for fixed initial condition $p_0(x_0)$.
```
```{admonition} Solution (Incomplete)
:class: dropdown

Liouville's theorem tells us that evolution conserves phase-space volumes when
considered as functions of $t$ and initial conditions $x_0$, and $p_0$ at time $t_0$:
\begin{gather*}
  x(t) = x(t; x_0, p_0, t_0), \qquad
  p(t) = p(t; x_0, p_0, t_0),\\
  \frac{\partial (x, p)}{\partial (x_0, p_0)} =
  \pdiff{x}{x_0}\pdiff{p}{p_0} - \pdiff{x}{p_0}\pdiff{p}{x_0}
  = 1.
\end{gather*}

\begin{gather*}
\d{p} = \pdiff{p}{t}\d{t} + \pdiff{p}{x_0}\d{x_0}
        + \pdiff{p}{p_0}\d{p_0} + \pdiff{p}{t_0}\d{t_0},\\
\d{x} = \pdiff{x}{t}\d{t} + \pdiff{x}{x_0}\d{x_0}
        + \pdiff{x}{p_0}\d{p_0} + \pdiff{x}{t_0}\d{t_0}.
\end{gather*}

To get $n(x, t)$ from the previous relationship, we hold $\d{x} = \d{t} = \d{t_0} = 0$.

\begin{gather*}
  \d{p} = \pdiff{p}{x_0}\d{x_0} + \pdiff{p}{p_0}\d{p_0},\\
  \d{x} = 0 = \pdiff{x}{x_0}\d{x_0} + \pdiff{x}{p_0}\d{p_0},\\
  \frac{\d{p_0}}{\d{x_0}} = -\frac{\pdiff{x}{x_0}}{\pdiff{x}{p_0}},\\
  \pdiff{p(x, t;x_0, t_0)}{x_0} 
  = \pdiff{p}{x_0} + \pdiff{p}{p_0}\frac{\d{p_0}}{\d{x_0}}\\
  = \pdiff{p}{x_0} - \frac{\pdiff{x}{x_0}\pdiff{p}{p_0}}{\pdiff{x}{p_0}}\\
  = \pdiff{p}{x_0} - \frac{1 + \pdiff{x}{p_0}\pdiff{p}{x_0}}{\pdiff{x}{p_0}}\\
  = - \frac{1}{\pdiff{x}{p_0}}.
\end{gather*}
```

## WKB: Path Integral Formulation

In quantum mechanics, one can use the Feynman path-integral approach to construct the
propagator (here expressed in terms of position-to-position
transitions):

```{margin}
See {cite:p}`DeWitt-Morette:1976` and {cite:p}`Cartier:2006` for extensive details and
rigorous definitions of what the path integral means.  *Note: the expressions in
{cite:p}`Cartier:2006` are missing factors of $2\pi \I$ -- see
{cite:p}`DeWitt-Morette:1976` for the correct expressions.*
```
\begin{gather*}
  \newcommand{\S}{\mathcal{S}}
  U(q, t; q_0, t_0) = \int \mathcal{D}[q]\; \exp\left\{\frac{\I}{\hbar}S[q]\right\},\\
  \psi(q, t) = \int U(q, t;q_0, t_0)\psi(q_0, t_0)\d{q_0}.
\end{gather*}

where the integral is over all paths that start at $q(t_0) = q_0$ and end at $q(t) =
q$, and $S[q]$ is the classical action

\begin{gather*}
  S[q] = \int_{t_0}^{t}\d{t}\; L(q, \dot{q}, t).  
\end{gather*}

Given an initial wavefunction $\psi(q_0, t_0)$, the wavefunction at time $t$ is:

\begin{gather*}
  \psi(q, t) = \int \d{q_0}\; U(q, t;q_0, t_0)\psi(q_0, t_0).
\end{gather*}

the [WKB approximation] relies on the idea that classical trajectories where
$S'[q_{\mathrm{cl}}] = 0$ -- the famous principle of extremal action -- dominate the
propagator, and use the expansion of the action

\begin{gather*}
  S[q+\xi] = S[q] + S'[q]\cdot \xi + \frac{1}{2!}S''[q]\cdot\xi\xi 
  + \frac{1}{3!}S'''[q]\cdot\xi\xi\xi  + \cdots.
\end{gather*}

```{margin}
The path integral over $\xi$ here can be computed analytically as it is simply a
[gaussian integral].  This result is just the multi-dimensional generalization of the
elementary result that

\begin{gather*}
  \int \d^n{\vect{x}} e^{-\tfrac{1}{2}\vect{x}^T\mat{A}\vect{x}} =\\
  = \sqrt{\det(2\pi \mat{A}^{-1})}.
\end{gather*}
```
The [WKB approximation] amount to considering all classical trajectories with
appropriate boundary conditions, performing the path integral over $\xi$, and dropping
terms of order $\order(\xi^3)$ and higher to obtain:
\begin{gather*}
  U_{WKB}(q, t; q_0, t_0) = \int \mathcal{D}[\xi]\; \exp\left\{\frac{\I}{\hbar}\left(
  S[q_{\mathrm{cl}}] + \frac{1}{2}S''[q_{\mathrm{cl}}]\cdot\xi\xi\right)\right\}\\
  = \sqrt{\frac{-\partial^2 S / (2\pi \I \hbar)}
          {\partial q_{\mathrm{cl}}(t)\partial q_{\mathrm{cl}}(t_0)}}
    \exp\left\{\frac{\I}{\hbar}\S(q_{\mathrm{cl}}(t),t;q_{\mathrm{cl}}(t_0),t_0)\right\},
\end{gather*}
where $\S = \S(q,t;q_0,t_0)$ is the classical action with $q=q_{\mathrm{cl}}(t)$ and $q_0
= q_{\mathrm{cl}}(t_0)$ are the final and initial points of the classical trajectory.
The key point here is that all of the information about the propagator in this
approximation is contained in the classical action $\S(q,t;q_0,t_0)$, sometimes called
[Hamilton's principal function][].

Once the path integrals over $\xi$ have been done, everything is expressed in terms of
the classical trajectory $q_{\mathrm{cl}}(t)$ and we shall drop the "cl" subscript in
what follows.

*(Note: if there are multiple trajectories that satisfy the boundary conditions, then
they should be added, giving rise to quantum interference patterns.)*

````{admonition} Example: Free Particle
:class: dropdown

As an example, consider a free particle with Hamiltonian $H = p^2/2m$.  This has
the following solution $x(t)$ and [Hamilton's principal function] $\S$:

\begin{gather*}
  x(t) = x_0 + \frac{p_0}{m}(t-t_0), \qquad
  p_0 = m\frac{x-x_0}{t-t_0},\\
  \S(x,t;x_0,t_0) = \frac{p_0^2}{2m}(t-t_0) = \frac{m(x-x_0)^2}{2(t-t_0)},\\
  \frac{\partial^2 \S}{\partial x\partial x_0} = -\frac{m}{(t-t_0)}.
\end{gather*}

Hence, the WKB propagator is:

\begin{gather*}
  U_{WKB}(x,t;x_0,t_0) = \sqrt{\frac{m}{2\pi \I \hbar (t-t_0)}}\exp\left\{
    \frac{i}{\hbar}\frac{m(x-x_0)^2}{2(t-t_0)}
  \right\}.
\end{gather*}

A quick check shows that this is exact:

\begin{gather*}
  U(x,t;x_0,t_0) = \braket{x|e^{\op{H}(t-t_0)/\I\hbar}|x_0}
  = \int \frac{\d{k}}{2\pi}\braket{x|k}e^{\hbar^2k^2(t-t_0)/(2m\I\hbar)}\braket{k|x_0}\\
  = \int \frac{\d{k}}{2\pi}e^{\hbar^2k^2(t-t_0)/(2m\I\hbar) + \I k (x-x_0)}\\
  = \sqrt{\frac{m}{2\pi \I\hbar (t-t_0)}}\exp\left\{
    \frac{\I m(x-x_0)^2}{2\hbar (t-t_0)}
  \right\}.
\end{gather*}

Extending this to higher dimensions, we have:

\begin{gather*}
  \vect{x}(t) = \vect{x}_0 + \frac{\vect{p}_0}{m}(t-t_0), \qquad
  \vect{p}_0 = m\frac{\vect{x}-\vect{x}_0}{t-t_0},\\
  \S(\vect{x},t;\vect{x}_0,t_0) = \frac{m\norm{\vect{x}-\vect{x}_0}^2}{2(t-t_0)},\\
  \frac{\partial^2 \S}{\partial x_{i}\partial [x_0]_{j}} = -\frac{m\delta_{ij}}{(t-t_0)}.
\end{gather*}
````

Similar results can be obtained from the momentum-to-position transitions if the initial
state is expressed in terms of momentum, however, in this case, since the boundary
conditions are no longer the same, we must use a different form of $\S(x,t;p_0,t_0)$:

\begin{gather*}
  \S(q,t;p_0,t_0) = p_0q_0(q,t;p_0,t_0) + \S\Bigl(q,t;q_0(q,t;p_0,t_0),t_0\Bigr),\\
  U_{WKB}(q, t; p_0, t_0) 
  = \sqrt{\frac{\partial^2 \S}{\partial q_{\mathrm{cl}}(t)\partial p_{\mathrm{cl}}(t_0)}}
    \exp\left\{\frac{\I}{\hbar}\S(q_{\mathrm{cl}}(t),t;p_{\mathrm{cl}}(t_0);t_0)\right\}.
\end{gather*}

````{admonition} Example: Free Particle continued
:class: dropdown

Changing variables, we now have:

\begin{gather*}
  \S(x,t;p_0,t_0) = p_0\left(x - \frac{p_0}{m}(t-t_0)\right) 
                    + \overbrace{\frac{p_0^2}{2m}(t-t_0)}^{S}\\
                  = p_0 x - \frac{p_0^2}{2m}(t-t_0), \qquad
  \frac{\partial^2 \S}{\partial x\partial p_0} = 1
\end{gather*}.

Hence, the WKB propagator is:

\begin{gather*}
  U_{WKB}(x,t;p_0,t_0) = \exp\left\{
    \frac{i}{\hbar}\left(xp_0 - \frac{p_0^2 (t-t_0)}{2m}\right)\right\}.
\end{gather*}

A quick check shows that this is also exact, and confirms the need for the extra piece
$q_0 p_0$ in $\S$:

\begin{gather*}
  U(x,t;p_0,t_0) = \braket{x|e^{\op{H}(t-t_0)/\I\hbar}|p_0}
  = \braket{x|p_0}e^{p_0^2(t-t_0)/(2m\I\hbar)}\\
  = e^{\tfrac{\I}{\hbar} x p_0}e^{p_0^2(t-t_0)/(2m\I\hbar)}\\
  = \exp\left\{\frac{\I}{\hbar}\left(
      x p_0 - \frac{p_0^2}{2m}(t-t_0)
    \right)\right\}.
\end{gather*}
````

(eg:FallingParticles)=
## Examples

````{admonition} Example: Falling Particle 1D
:class: dropdown

As a second example, consider a particle in free-fall with Hamiltonian $H = p^2/2m +
mgz$.  The classical problem is most easily solved with conservation of energy $E$:

\begin{align*}
  E &= \frac{p^2}{2m} + mgz = \frac{p_{0}^2}{2m} + mgz_0, \\
  p &= \pm\sqrt{2mE - 2m^2gz},\\ 
    &= \pm\sqrt{p_{0}^2 - 2m^2g(z-z_0)},\\
  L(z) &= E - 2mgz.
\end{align*}

<!-- For simplicity, we assume that $z<z_0$ and that the particle is falling throughout the -->
<!-- region of interest so that $p \leq 0$. -->
Since our Hamiltonian is time-independent, the results will depend only on $t-t_0$ and
we can choose $t_0=0$ without loss of generality.  This has the following solution
$z(t)$ and [Hamilton's principal function] $\S$:

\begin{gather*}
  \newcommand{\t}{t}
  z(t) = z_0 + \frac{p_0}{m}\t - \frac{g}{2}\t^2,\\
  \S(z,\t;z_0,t_0=0) = \int_{0}^{\t}\left(E - 2mgz_0 - 2gp_0\t + mg^2\t^2\right)\d{\t},\\
                     = E(z,\t;z_0)\t - 2mgz_0\t - gp_0(z,\t;z_0)\t^2 + \frac{mg^2\t^3}{3}.
\end{gather*}

The appropriate functional dependence can be deduced by solving for $p_0(z,\t;z_0)$ and
$E(z,\t;z_0)$:

\begin{align*}
  p_0(z,\t;z_0) &= m\frac{z-z_0}{\t} + \frac{mg\t}{2}, &
  \pdiff{p_0}{z_0} &= -\frac{m}{\t}, \\
  E(z,\t;z_0) &= \frac{p_0^2(z,\t;z_0)}{2m} + mgz_0, &
  \pdiff{E}{z_0} &= -\frac{p_0}{\t} + mg.
\end{align*}

From these, we can compute the action and various partials:

\begin{gather*}
\S(z,\t;z_0,t_0=0) = m\left(
  \frac{(z-z_0)^2}{2\t} - \frac{g}{2}(z+z_0)\t - \frac{g^2\t^3}{24}\right),\\
  \frac{\partial\S(z,\t;z_0,t_0=0)}{\partial z} = p = p_0(z, \t;z_0) - mg\t,\\
  \frac{\partial\S(z,\t;z_0,t_0=0)}{\partial z_0} = -p_0\\
  \frac{\partial^2\S(z,\t;z_0,t_0=0)}{\partial z\partial z_0} = -\frac{m}{\t}.
\end{gather*}

Hence, the WKB propagator is:

\begin{gather*}
  U_{WKB}(z,t;z_0,t_0) = \sqrt{\frac{m}{2\pi \I \hbar (t-t_0)}}\\
  \exp\left\{
    \frac{\I m}{\hbar}\left(
  \frac{(z-z_0)^2}{2(t-t_0)} - \frac{g}{2}(z+z_0)(t-t_0) - \frac{g^2(t-t_0)^3}{24}
  \right)
  \right\}.
\end{gather*}

To obtain the approximate Airy function, note that $U_{WKB}\bigr(z,z_0,-(t-t_0)\bigl) =
U_{WKB}^*\bigl(z,z_0,(t-t_0)\bigr)$: i.e. for a particle traveling up, then falling back down,
there will be an interference between the up-going and down-going wavefunctions that
results in twice the real part of $U_{WKB}$.  Suitably normalized, this provides an
extremely good approximation of the [Airy function][] which exactly solves the
corresponding quantum mechanics problem:
\begin{gather*}
  \DeclareMathOperator{\Ai}{Ai}
  \Ai''(x) = x\Ai(x).
\end{gather*}
The quantum wavefunction for a particle in a gravitational field is solved by this if we
shift $z-z_0$ by the maximum height $z_0 = E/mg$, and scale by the natural scale $\xi$
so that $x = (z-z_0)/\xi$ is dimensionless:
\begin{gather*}
  \frac{-\hbar^2}{2m}\psi''(z) + mgz\psi(z) = E\psi(z),\\
  \psi(z) = \Ai(x) = \Ai\left(\frac{z-z_0}{\xi}\right), \qquad
  \psi''(z) = \xi^{-2}\Ai''(x),\\
  \Ai''(x) = \underbrace{\xi^2\frac{2m^2g(z - z_0)}{\hbar^2}}_{x}\Ai(x),\\
  x = \frac{z-z_0}{\xi} = \xi^2\frac{2m^2g(z - z_0)}{\hbar^2}, \qquad
  \xi = \sqrt[3]{\frac{\hbar^2}{2m^2g}}.
\end{gather*}

Thus, the WKB approximation gives 
\begin{gather*}
  \Ai(x) \propto \Re U_{WKB}\Bigl(z-z_0=\xi x, t-t_0=\sqrt{2\xi z/g}\Bigr)\\
  \approx \frac{1}{\sqrt{\pi}\abs{x}^{1/4}}\cos\left(\tfrac{2}{3}(-x)^{3/2} + \tfrac{\pi}{4}\right)
\end{gather*}
where the normalization needs to be fixed for negative $x$, but is given here
$1/\sqrt{\pi}$ for the standard [Airy function][] normalization.

```{glue:figure} fig:airy

WKB approximation to the standard [Airy function][] $\Ai(x)$.
```
````

```{code-cell}
:tags: [hide-cell]
from scipy.special import airy

m = hbar = g = 1
t0 = 0
z0 = 0
t = t0 + np.linspace(0, 6, 1000)[1:]
z = z0 - g*t**2/2
xi = (hbar**2/m**2/2/g)**(1/3)
x = (z-z0) / xi

phi = m/hbar*(
    (z-z0)**2/2/(t-t0) - g/2*(z+z0)*(t-t0) - g**2*(t-t0)**3/24
)
psi = np.sqrt(m/2/np.pi/hbar/1j/(t-t0))*np.exp(1j*phi)
Ai, Aip, Bi, Bip = airy(z/xi)
fig, ax = plt.subplots(figsize=(5, 2.5))
ax.plot(x, Ai, label=r"$\mathrm{Ai}(z/\xi)$")
ax.plot(x, psi.real * Ai[-1] /np.real(psi)[-1], label=r"$\Re \psi_{WKB}$")
y = np.cos((-x)**(3/2)*2/3 - np.pi/4)/abs(x)**(1/4)/np.sqrt(np.pi)
ax.plot(x, y, ":", 
        label=r"$\cos(\frac{2}{3}(-x)^{3/2} + \frac{\pi}{4})/\sqrt{\pi}|x|^{1/4}$")
ax.set(ylim=(-0.5, 1), xlabel=r"$z/\xi$", ylabel=r"$\psi$",
       title=r"$\xi=\sqrt[3]{\hbar^2/(2m^2g)}$")
ax.legend();
if glue: glue("fig:airy", fig, display=False);
```



````{admonition} Example: General Particle 1D
:class: dropdown

Slightly more general, we now consider a particle falling in an arbitrary
time-independent potential $V(z) = mgz + \delta(z)$ where we will ultimately consider
$\delta(z)$ to be small.  Here, since energy is conserved, we immediately have:
\begin{gather*}
  E = \frac{p^2}{2m} + V(z) = \frac{p_{0}^2}{2m} + V(z_0), \\
  p = \pm\sqrt{2mE - 2mV(z)},\\
  L(z) = E - 2V(z).
\end{gather*}
The trajectory $z(t)$ no longer has a closed form, but we can still express the action
by changing variables $\d{z} = \dot{z}\d{t} = p\d{t}/m$:
\begin{gather*}
  \S(z;z_0;E) = m\int_{z_0}^{z}\frac{\Bigl(E - 2V(z)\Bigr)}{p(z)}\d{z}\\
  = \sqrt{\frac{m}{2}}\int_{z_0}^{z}\frac{\Bigl(E - 2V(z)\Bigr)}
                                         {\pm\sqrt{E - V(z)}}\d{z}.
\end{gather*}
This form has two complications.  First, the sign of the denominator must be chosen
appropriately to match the direction of motion.  This is often clear from the physics,
and so does not pose a fundamental problem.  Second, this form of the action as an
explicit function of either $S(z, p; z_0)$ or $S(z;z_0, p_0)$ since $E = E(z_0, p_0) = E(z,
p)$ is conserved and a function of the initial or final coordinates.

A comment about the role of the conserved energy $E$ here.  Note that if $E=0$, then the
numerator and denominator both contain factors of $\sqrt{-V(z)}$ and can be combined.
The presence of $E$ seems to spoil this, but, as is generally well known, in classical
mechanics, only the relative value of the energy is physically significant.  To make
this explicit, we note that
\begin{gather*}
  V(z) = E - \frac{p^2}{2m}, \quad
  L(z) = E - 2V(z) = \frac{p^2}{m} - E,\\
  \begin{aligned}
    S(z;z_0;E) &= -\int_{0}^{t}E\d{t} + \int_{z_0}^{z}\bigl(\pm p(z)\bigr)\d{z}\\
    &= -Et \pm \int_{z_0}^{z}p(z)\d{z}.
  \end{aligned}
\end{gather*}
The first term clearly does not affect the physics, and in quantum mechanics,
corresponds to an overall global phase.  This is exactly the effect of shifting the
zero-energy level.  The second term is a common form of the action, as an integral of a
generalized momentum with respect to the corresponding coordinate.  This form appears in
the action-angle variable formulation for example.

To compute the normalization factor, we must perform the appropriate change of variables
using the analogy of the Maxwell relations in thermodynamics using:

\begin{gather*}
  f(z, z_0, E) = t - t_0 = \int_{t_0}^{t}\d{t} = \int_{z_0}^{z}\frac{m}{p}\d{z}\\
  = \sqrt{\frac{m}{2}}\int_{z_0}^{z}\frac{1}{\pm\sqrt{E - V(z)}}\d{z},\\
  \pdiff{f(z, z_0, E)}{z} = \pm\frac{\sqrt{m/2}}{\sqrt{E-V(z)}}\\
  \pdiff{f(z, z_0, E)}{z_0} = \mp\frac{\sqrt{m/2}}{\sqrt{E-V(z_0)}}\\
  \pdiff{f(z, z_0, E)}{E} = \mp\int_{z_0}^{z}\frac{\sqrt{m/8}}{\sqrt{E - V(z)}^3}\d{z}.
\end{gather*}

We must compute the partials holding this constant, so we have:

\begin{gather*}
  \pdiff{f(z, z_0, E)}{z} \d{z} 
  + \pdiff{f(z, z_0, E)}{z_0} \d{z_0} 
  + \pdiff{f(z, z_0, E)}{E} \d{E} = 0,\\
  \d{E} = \frac{\pdiff{f(z, z_0, E)}{z}\d{z} + \pdiff{f(z, z_0, E)}{z_0}\d{z_0}}
               {\pdiff{f(z, z_0, E)}{E}}\\
        = \frac{-2}{\int_{z_0}^{z}\frac{1}{\sqrt{E - V(z)}^3}\d{z}}
          \left(\frac{\d{z}}{\sqrt{E-V(z)}} - \frac{\d{z_0}}{\sqrt{E-V(z_0)}}\right).
\end{gather*}

We can now compute the normalization factor.  We take the first derivative using the
well-known properties of Hamilton's principle function

\begin{gather*}
  \frac{\partial^2 S(z,t;z_0,t_0)}{\partial z\partial z_0} =
  \frac{\partial p(z,t;z_0,t_0)}{\partial z_0} =
  -\frac{\partial p_0(z,t;z_0,t_0)}{\partial z}
\end{gather*}

and then use the expression above for $p(z, E) = \pm \sqrt{2m\bigl(E-V(z)\bigr)}$ to compute:

\begin{gather*}
  \d{p} = \frac{\pm \sqrt{2m}}{\sqrt{E-V(z)}}\Bigl(\d{E} - V'(z) \d{z}\Bigr)\\
  \pdiff{p(z, t;z_0, t_0)}{z_0} = 
  \frac{\pm \sqrt{8m}}{\sqrt{\bigl(E-V(z)\bigr)\bigl(E-V(z_0)\bigr)}}
  \frac{1}{\int_{z_0}^{z}\frac{1}{\sqrt{E - V(z)}^3}\d{z}}.
\end{gather*}
````

````{admonition} Example: Harmonic Oscillator
:class: dropdown

The harmonic oscillator has the following solution:

\begin{gather*}
  \newcommand{\t}{\tau}
  H(x, p) = \frac{p^2}{2m} + \frac{m\omega^2 x^2}{2}, \qquad 
  \t= t-t_0\\
  S(x, t; x_0, t_0) = \frac{\omega(xp - x_0p_0)}{2}\\
  = \frac{m\omega}{2\sin\omega\t}\Bigl(
  (x^2+x_0^2)\cos\omega\t - 2xx_0\Bigr),\\
  \frac{\partial^2 S}{\partial x \partial x_0} = -\frac{m\omega}{\sin\omega\tau},\qquad
  A = \sqrt{\frac{m\omega}{\sin\omega\tau}}\\
  \begin{aligned}
    S' &\equiv \pdiff{S}{x} = p = \frac{m\omega}{\sin\omega\t}\Bigl(x\cos\omega\t - x_0\Bigr),\\
    S'' &\equiv \pdiff[2]{S}{x} = m\omega\cot\omega\t,\\
    \dot{A} &\equiv \pdiff{A}{t} 
    %= \frac{-1}{2}\sqrt{\frac{m\omega^3\cos^2\omega \tau}{\sin^3\omega \tau}}
    = \frac{-\omega A \cot \omega \tau}{2}
  \end{aligned}\\
  \frac{S''}{2m} + \frac{\dot{A}}{A} + \frac{A'S'}{mA} = 
  \frac{\omega}{2}\cot\omega\t - \frac{\omega}{2}\cot\omega\t + 0 = 0.
\end{gather*}
````




[Action-angle coordinates]: <https://en.wikipedia.org/wiki/Action-angle_coordinates>

[Lagrangian mechanics]: <https://en.wikipedia.org/wiki/Legendre_transformation>
[Hamiltonian mechanics]: <https://en.wikipedia.org/wiki/Hamiltonian_mechanics>
[Legendre transformation]: <https://en.wikipedia.org/wiki/Legendre_transformation>
[WKB approximation]: <https://en.wikipedia.org/wiki/WKB_approximation>
[gaussian integral]: <https://en.wikipedia.org/wiki/Gaussian_integral>
[Maxwell relations]: <https://en.wikipedia.org/wiki/Maxwell_relations>
[Hamilton's principal function]: <https://en.wikipedia.org/wiki/Hamilton%E2%80%93Jacobi_equation#Hamilton's_principal_function>
[phase space]: <https://en.wikipedia.org/wiki/Phase_space>
[Hamilton's equations]: <https://en.wikipedia.org/wiki/Hamilton%27s_equations>
[Canonical transformation]: <https://en.wikipedia.org/wiki/Canonical_transformation>
[Poisson bracket]: <https://en.wikipedia.org/wiki/Poisson_bracket>
[Canonical quantization]: <https://en.wikipedia.org/wiki/Canonical_quantization>
[Moyal bracket]: <https://en.wikipedia.org/wiki/Moyal_bracket>
[generating function]: <https://en.wikipedia.org/wiki/Canonical_transformation#Generating_function_approach>
[Hamilton-Jacobi equation]: <https://en.wikipedia.org/wiki/Hamilton%E2%80%93Jacobi_equation>
[Hamilton's characteristic function]: <https://en.wikipedia.org/wiki/Hamilton%E2%80%93Jacobi_equation#Separation_of_variables>
[Airy function]: <https://en.wikipedia.org/wiki/Airy_function>
[Hamilton's principal function]: <https://en.wikipedia.org/wiki/Hamilton%E2%80%93Jacobi_equation#Hamilton's_principal_function>
