---
jupytext:
  formats: ipynb,md:myst,py:light
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.0
kernelspec:
  display_name: Python 3 (phys-521-2021)
  language: python
  name: phys-521-2021
---

```{code-cell} ipython3
:tags: [hide-cell]

import mmf_setup; mmf_setup.nbinit()
import logging; logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
%pylab inline --no-import-all
import manim.utils.ipython_magic
!manim --version
```


Hamiltonian Mechanics
=====================

Recall that with [Lagrangian mechanics], one recovers Newton's laws as a principle of
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

The idea of [Hamiltonian mechanics] is to effect a [Legendre transformation], replacing
the coordinates $\dot{\vect{q}}$ with the conjugate momenta $\vect{p}$:

\begin{gather*}
  \vect{p} = \pdiff{L}{\dot{\vect{q}}}, \qquad
  H(\vect{p}, \vect{q}, t) = \vect{p}\cdot\dot{\vect{q}} - L.
\end{gather*}

How we have Hamilton's equations of motion:

\begin{gather*}
  \dot{\vect{q}} = \pdiff{H}{\vect{p}}, \qquad
  \dot{\vect{p}} = -\pdiff{H}{\vect{q}}.
\end{gather*}

::::{admonition} Proof
:class: dropdown

We first solve the first equation for $\dot{\vect{q}} = \dot{\vect{q}}(\vect{q},
\vect{p}, t)$, then substituting:

\begin{gather*}
  H(\vect{p}, \vect{q}, t) 
  = \vect{p}\cdot\dot{\vect{q}}(\vect{q}, \vect{p}, t) - L\Bigl(\vect{q}, \dot{\vect{q}}(\vect{q}, \vect{p}, t), t\Bigr).
\end{gather*}

Now we simply differentiate, using the chain rule, and cancel $\vect{p} = \pdiff L/\partial\dot{\vect{q}}$:

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

## Phase Flow

Hamilton's equations define **phase flow**.  Let $\vect{y}(0) = (\vect{q}_0, \vect{p}_0)$ be a
point in phase space.  The phase flow is defined by the trajectory $\vect{y}(t)$ through
phase space which satisfies Hamilton's equations:

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

```{code-cell} ipython3
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
    alpha : float
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


fig, ax = plt.subplots(figsize=(10,5))


for n, y in enumerate(np.linspace(0.25, 1.75, 6)):
    plot_set(y=(y, y), c=f"C{n}", ax=ax)
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
  m\ddot{q} &= \dot{p}e^{-\alpha t} - m\alpha \dot{q}
           = - V'(q) e^{\alpha t}e^{-\alpha t} - m\alpha \dot{q}\\
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
  S[q] = \int L_0(q, \dot{q}) \underbrace{e^{\alpha t} \d{t}}_{\d{\tau}}.
\end{gather*}

This is presents an analogy of cosmology where the universe is decelerating as it
expands, leading to an effective cooling from the dissipative term introduced by the
scaling of time.


:::{margin}
This plot shows the phase-flow for the time-dependent Hamiltionian for the pendulum with
damping.  The upper plot shows the flow in the original phase space $\vect{y} = (q, p)$
whereas the bottom figure shows the flow in the "physical" phase space $\vect{Y} = (q,
P)$.  The flow in the upper plot obey's Liouville's theorem, preserving areas, but is
time dependent.  This is why the trajectories intersect.

The bottom pane is independent of time, but does not follow from Hamilton's equations,
and hence, does not obey Liouville.  In the bottom figure, we see the expected
contraction of areas as the trajectories coalesce to the equilibrium points with $\theta
= 2\pi n$.

Note that the plots share the same abscissa $q=\theta$, but that the vertical axis is not just
a simple rescaling.  In the upper plot, the momenta are scaled differently at different
times.

We have evolved slightly longer than before ($1.3$ small-oscillation periods) to better
demonstrate the structure of the flow. 
:::
```{code-cell} ipython3
:tags: [hide-input]

fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True, 
                        gridspec_kw=dict(height_ratios=(3.5, 1)))

alpha = 0.3

for n, y in enumerate(np.linspace(0.25, 1.75, 6)):
    plot_set(y=(y, y), c=f"C{n}", ax=axs[0], T=1.3*T, phase_space=False)
    plot_set(y=(y, y), c=f"C{n}", ax=axs[1], T=1.3*T)

axs[0].set(ylim=(-6, 7))
axs[0].set(title=fr"$\alpha = {alpha}$, $T=1.3\times 2\pi \sqrt{{r/g}}$");
```





[Lagrangian mechanics]: <https://en.wikipedia.org/wiki/Legendre_transformation>
[Hamiltonian mechanics]: <https://en.wikipedia.org/wiki/Hamiltonian_mechanics>
[Legendre transformation]: <https://en.wikipedia.org/wiki/Legendre_transformation>
