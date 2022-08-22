---
jupytext:
  formats: ipynb,md:myst,py:light
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.0
kernelspec:
  display_name: Python 3 (phys-521-2022)
  language: python
  name: phys-521-2022
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

# WKB Approximation

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
  S[q_{\mathrm{cl}}] + \frac{1}{2}S''[q_{\mathrm{cl}}]\cdot\xi\xi\right)\right\} =\\
  = \sqrt{\frac{-\partial^2 S / (2\pi \I \hbar)}
          {\partial q_{\mathrm{cl}}(t)\partial q_{\mathrm{cl}}(t_0)}}
    \exp\left\{\frac{\I}{\hbar}\S(q_{\mathrm{cl}}(t),t;q_{\mathrm{cl}}(t_0),t_0)\right\},
\end{gather*}

where $\S = \S(q,t;q_0,t_0)$ is the classical action with $q=q_{\mathrm{cl}}(t)$ and $q_0
= q_{\mathrm{cl}}(t_0)$ being the final and initial points of the classical trajectory.
The key point here is that all of the information about the propagator in this
approximation is contained in the classical action $\S(q,t;q_0,t_0)$, sometimes called
[Hamilton's principal function].

Once the path integrals over $\xi$ have been done, everything is expressed in terms of
the classical trajectory $q_{\mathrm{cl}}(t)$ and we shall drop the $\mathcal{cl}$
subscript in what follows.

*(Note: if there are multiple trajectories that satisfy the boundary conditions, then
they should be added, giving rise to quantum interference patterns.)*


````{admonition} Example: Free Particle
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

## Examples

````{admonition} Example: Falling Particle

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

\begin{align*}
  \newcommand{\t}{t}
  z(t) &= z_0 + \frac{p_0}{m}\t - \frac{g}{2}\t^2,\\
  \S(z,\t;z_0,t_0=0) &= \int_{0}^{\t}\left(E - 2mgz_0 - 2gp_0\t + mg^2\t^2\right)\d{\t},\\
                     &= E(z,\t;z_0)\t - 2mgz_0\t - gp_0(z,\t;z_0)\t^2 + \frac{mg^2\t^3}{3}.
\end{align*}

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
````

````{admonition} Example: Falling Particle

As a second example, consider a particle in free-fall with Hamiltonian $H = (p_x^2 +
p_z^2)/2m + mgz$.  The classical problem is most easily solved with conservation of
energy $E$ and momentum $p_x$:

\begin{gather*}
  E = \frac{p_x^2+p_z^2}{2m} + mgz = \frac{p_{x0}^2 + p_{z0}^2}{2m} + mgz_0, \\
  p_z = \pm\sqrt{2mE - 2m^2gz - p_x^2} 
      = \pm\sqrt{p_{z0}^2 - 2m^2g(z-z_0)},\\
  L(\vect{r},t) = E - 2mgz
\end{gather*}





This has the following solution $\vect{r}(t)$ and [Hamilton's
principal function] $\S$:

\begin{gather*}
  \vect{r}(t) = \vect{r}_0 + \frac{\vect{p}_0}{m}(t - t_0) - \frac{g}{2}(t-t_0)^2\uvect{z},\\
  p_0 = m\frac{x-x_0}{t-t_0},\\
  \S(\vect{r},t;\vect{r}_0,t_0) = \frac{p_0^2}{2m}(t-t_0) = \frac{m(x-x_0)^2}{2(t-t_0)},\\
  \frac{\partial^2 \S}{\partial x\partial x_0} = -\frac{m}{(t-t_0)}.
\end{gather*}

Hence, the WKB propagator is:

\begin{gather*}
  U_{WKB}(x,t;x_0,t_0) = \sqrt{\frac{m}{2\pi \I \hbar (t-t_0)}}\exp\left\{
    \frac{i}{\hbar}\frac{m(x-x_0)^2}{2(t-t_0)}
  \right\}.
\end{gather*}
Changing variables, we now have:

\begin{gather*}
  \S(x,t;p_0,t_0) = xp_0 - \frac{p_0^2}{2m}(t-t_0),\qquad
  \frac{\partial^2 \S}{\partial x\partial p_0} = 1
\end{gather*}.

Hence, the WKB propagator is:

\begin{gather*}
  U_{WKB}(x,t;p_0,t_0) = \exp\left\{
    \frac{i}{\hbar}\left(xp_0 - \frac{p_0^2 (t-t_0)}{2m}\right)\right\}.
\end{gather*}

A quick check shows that this is also exact, and confirms the need for the extra piece
$q p_0$ in $\S$:

\begin{gather*}
  U(x,t;p_0,t_0) = \braket{x|e^{\op{H}(t-t_0)/\I\hbar}|p_0}
  = \braket{x|p_0}e^{p_0^2(t-t_0)/(2m\I\hbar)}\\
  = e^{\tfrac{\I}{\hbar} x p_0}e^{p_0^2(t-t_0)/(2m\I\hbar)}\\
  = \exp\left\{\frac{\I}{\hbar}\left(
      x p_0 - \frac{p_0^2}{2m}(t-t_0)
    \right)\right\}.
\end{gather*}
````

````{admonition} Example: Harmonic Oscillator

The harmonic oscillator has the following solution:

\begin{gather*}
  \newcommand{\t}{\tau}
  H(x, p) = \frac{p^2}{2m} + \frac{m\omega^2 x^2}{2}, \qquad 
  \t= t-t_0\\
  S(x, t; x_0, t_0) = \frac{\omega(xp - x_0p_0)}{2}\\
  = \frac{m\omega}{2\sin\omega\t}\Bigl(
  (x^2+x_0^2)\cos\omega\t - 2xx_0\Bigr),\\
  S_{,xx_0} = -\frac{m\omega}{\sin\omega\tau},\\
  A = \sqrt{\frac{m\omega}{\sin\omega\tau}}\\
  S' = \frac{m\omega}{\sin\omega\t}\Bigl(x\cos\omega\t - x_0\Bigr),\\
  S'' = m\omega\cot\omega\t,\\
  \frac{S''}{2m} + \frac{\dot{A}}{A} + \frac{A'S'}{mA} = 
  \frac{\omega}{2}\cot\omega\t - \frac{\omega}{2}\cot\omega\t + 0 = 0.
\end{gather*}

````

### Traditional WKB

The more traditional approach is to express the wavefunction as follows, then insert it
into the Schrödinger equation:

```{margin}
To simplify the notation here, we use:
\begin{gather*}
  S' = \pdiff{S(x, t)}{x},\\
  \dot{S} = \pdiff{S(x,t)}{t}.
\end{gather*}
```

\begin{gather*}
  \psi(x, t) = \exp\left\{\frac{i}{\hbar}W(x,t)\right\}\psi_0(x),\\
  \left(\frac{(S')^2}{2m} - \frac{\I\hbar}{2m}W'' + V + \dot{W}\right)\psi = 0.
\end{gather*}

Expanding $W$ in powers of $\hbar$, we have the following lowest two orders:

\begin{align*}
  &W(x,t) = S(x,t) - \I\hbar \log A + \order(\hbar^2),\\
  \text{Order $\hbar^0$:}\quad &\frac{(S')^2}{2m} + V(x,t) + \dot{S} 
  = H(x, S', t) + \dot{S} = 0\\
  \text{Order $\hbar^1$:}\quad &\frac{S''}{2m} + \frac{A'S'}{mA} + \frac{\dot{A}}{A} = 0\\
  &\psi_{WKB}(x, t) = A(x,t)\exp\left\{\frac{i}{\hbar}S(x,t)\right\}\psi_0(x).
\end{align*}

#### $\order(\hbar^0)$: Hamilton-Jacobi Equation

The order $\hbar^0$ equation is the well-known Hamilton-Jacobi equation, which is
satisfied by the classical action as a function of initial and final states.

```{admonition} Exercise
Prove that the classical action $S(x, t)$ satisfies the Hamilton-Jacobi equation:

\begin{gather*}
  S(x, t) = S(x,t;x_0,t_0) = \int_{t_0}^{t} L\Bigl(x(t), \dot{x}(t), t\Bigr) \d{t},  
\end{gather*}

where $x(t)$ is a solution to the classical equations of motion with boundary conditions
$x(t_0) = x_0$ and $x(t) = x$, as discussed above.  I.e., show that

\begin{gather*}
  S' \equiv \pdiff{S(x, t; x_0, t_0}{x} = p, \qquad
  \dot{S} = \pdiff{S(x, t;x_0, t_0)}{t} = -H(x, S', t).
\end{gather*}
```

#### $\order(\hbar^1)$: Continuity Equation

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

where you can argue that the quantity $Q(x)$ is independet of $x_0$.
````
```{admonition} Exercise

Show that the order $\hbar^1$ equation is satisfied by

\begin{gather*}
  A(z, t) = \sqrt{-\frac{\partial^2 S(x, t; x_0, t_0) / (2\pi \I\hbar)}
                        {\partial x\partial x_0}}
\end{gather*}

as given by the path integral formulation.
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



## Maxwell Relations

To work with these expressions, we must compute the classical action for a particle
which follows the classical trajectory, for example $S(q,t;q_0,t_0)$ for the
position-to-position transitions, or $S(q,t;p_0,t_0)$ for the momentum-to-position
transitions.  We also need the various partial derivatives for the normalization factor
and for additional analysis.  Relating the partials of $S(q,t;q_0,t_0)$ to the partials
of $S(q,t;p_0,t_0)$ follows the same process of deriving thermodynamic relationships
such as the [Maxwell relations].

```{admonition} Warm-up Exercise
Consider a function $f(q, q_0)$ and another variable $p_0(q, q_0)$.  Show that the following
hold:

\begin{align*}
  \newcommand{\mypd}[3]{\left.\pdiff{#1}{#2}\right|_{#3}}
  \mypd{f(q, p_0)}{p_0}{q} &= \frac{\mypd{f(q,q_0)}{q_0}{q}}{\mypd{p_0(q,q_0)}{q_0}{q}},\\
  \mypd{f(q, p_0)}{q}{p_0} &= 
  \mypd{f(q,q_0)}{q}{q_0} - \mypd{f(q,q_0)}{q_0}{q}\frac{\mypd{p_0(q,q_0)}{q}{q_0}}{\mypd{p_0(q,q_0)}{q_0}{q}}.
\end{align*}

Check this by providing explicit forms for $f(q, q_0)$ and $p_0(q, q_0)$, then
explicitly changing variables and differentiating.
```

```{margin}
Some further relationships with this notation are:

\begin{align*}
  S_{,t} &= \mypd{\!S(q,\!t;q_0,\!t_0)}{t}{q,q_0,t_0}\\
  S^{P}_{,p_0} &= \mypd{\!S^{P}(q,\!t;p_0,\!t_0)}{p_0}{q,t,t_0}\\
  S^{G}_{,p_0} &= \mypd{\!S^{G}(q,\!t;q_0,\!p_0)}{p_0}{q,t,q_0}\\
  {p_0}_{,t} &= \mypd{\!p_0(q,\!t;q_0,\!t_0)}{t}{q,q_0,t_0}
\end{align*}
```
Here we consider three different sets of variables $S(q,t;q_0,t_0)$, $S^{P}(q,t;p_0,t_0)$, and
$S^{G}(q,t;q_0,p_0)$.  To simplify the equations, we use the following notation for
partials:

\begin{gather*}
  S_{,q} = \mypd{S(q,t;q_0,t_0)}{q}{t,q_0,t_0}, \qquad
  S^{P}_{,q} = \mypd{S^{P}(q,t;p_0,t_0)}{q}{t,p_0,t_0}, \\
  S^{G}_{,q} = \mypd{S^{G}(q,t;q_0,p_0)}{t}{q,q_0,p_0}, \qquad
  p_{0,q} = \mypd{p_0(q,t;q_0,t_0)}{t}{t,q_0,t_0}, \qquad \text{etc.}
\end{gather*}

I.e., the superscript denotes which set of variables is held fixed, and the subscript
denotes which variable we differentiate.  If there is no subscript  Prove the following:

\begin{align*}
  S^{P}_{,p_0} &= \frac{S_{,q_0}}{p_{0,q_0}}, &
  S^{G}_{,p_0} &= \frac{S_{,t_0}}{p_{0,t_0}},
  \\
  S^{P}_{,q} &= S_{,q} - S_{,q_0}\frac{p_{0,q}}{p_{0,q_0}}, &
  S^{G}_{,q} &= S_{,q} - S_{,t_0}\frac{p_{0,q}}{p_{0,t_0}},
  \\
  S^{P}_{,t} &= S_{,t} - S_{,q_0}\frac{p_{0,t}}{p_{0,q_0}}, &
  S^{G}_{,t} &= S_{,t} - S_{,t_0}\frac{p_{0,t}}{p_{0,t_0}},
  \\
  S^{P}_{,t_0} &= S_{,t_0} - S_{,q_0}\frac{p_{0,t_0}}{p_{0,q_0}}, &
  S^{G}_{,q_0} &= S_{,p_0} - S_{,t_0}\frac{p_{0,q_0}}{p_{0,t_0}}.
\end{align*}
  
To simplify these further, we need some mechanics.  We shall work with $S(q,t;q_0,t_0)$
explicitly:

```{admonition} Exercise
Show that:
\begin{align*}
  S_{,q} &= p &
  S_{,t} &= -H &
  S_{,q_0} &= -p_0 &
  S_{,t_0} &= H_0.
\end{align*}

Use the explicit form for the action, and then integrate by parts using the equations of
motion to obtain simpler results.  For some details, see {cite:p}`Houchmandzadeh:2020`.
```

These relationships allow us to derive formula similar to the [Maxwell relations] in
thermodynamics.  For example, using the fact that $p_0 = -S_{,q_0}$, $H_0 = S_{,t_0}$,
and $\dot{p}_0 = -\partial H_0/\partial q_0$ from the Hamilton equations of motion, we have:

\begin{align*}
  p_{0,t_0} &= -\frac{\partial^2 S}{\partial q_0\partial t_0}
             = -\frac{\partial^2 S}{\partial t_0\partial q_0}
             = -\frac{\partial H_0}{\partial q_0}
             = \dot{p}_0.
\end{align*}

```{margin}
We can use the Hamilton equations to simplify $H_{,q_0}$ since $H$ is the Hamiltonian at
time $t$, while $q_0$ is the coordinate at time $t_0$.
```
Likewise, though less useful:

\begin{align*}
  p_{0,q} &= - p_{,q_0}, &
  p_{0,t} &= - H_{,q_0}, &
  p_{0,q_0} &= S_{,q_0,q_0}.
\end{align*}

## Falling Particle

To check these relationships, we consider a particle falling in a gravitational field:

\begin{gather*}
  \newcommand{\t}{\tau}
  H(p, q) = \frac{p^2}{2m} + mgq, \qquad \t = (t-t_0),\\
  \dot{p} = -\pdiff{H}{q} = -mg \quad \implies \quad
  p = p_0 - mg\t,\\
  \dot{q} = \pdiff{H}{p} = \frac{p}{m} = \frac{p_0}{m} - g\t \quad \implies \quad
  q = q_0 + \frac{p_0}{m}\t - \frac{g}{2}\t^2.
\end{gather*}

From this solution, we can construct the action $S(q_0, p_0, \t=t-t_0)$:

\begin{gather*}
  L = p\dot{q} - H = \frac{p^2}{2m} - mgq 
  = \frac{p_0^2}{2m} - mgq_0 - 2gp_0\t + mg^2\t^2,\\
  S(q_0, p_0, \t) = \left(\frac{p_0^2}{2m} - mgq_0\right)\t - gp_0\t^2 + \frac{mg^2}{3}\t^3.
\end{gather*}

As a function of $S^{0}(t;p_0,q_0,t_0)$, this is not in any of the forms we considered
above.  To obtain those, we must eliminate one of the variables:

| Independent Variables | Replacement           | Name                 |
|-----------------------|-----------------------|----------------------|
| $(q, t;q_0, t_0)$     | $p_0(q, t;q_0, t_0)$ | position-to-position |
| $(q, t;p_0, t_0)$     | $q_0(q, t; p_0, t_0)$ | momentum-to-position |
| $(q, t;q_0, p_0)$     | $t_0(q, t; p_0, q_0)$ | geometric optics     |

This requires inverting the equations to solve for $p_0$, $q_0$, or $t_0$ respectively.
For the first two, we make use of the fact that the Hamiltonian (energy) is conserved
$H(p_0, q_0) = H(p, q)$:

\begin{gather*}
  \frac{p^2}{2m} + mgq = \frac{p_0^2}{2m} mgq_0.
\end{gather*}

For the last one, we need to solve the solution $q(\tau)$ for $\tau$.  These give the
following transformations:

\begin{align*}
  p_0(q, t, q_0, t_0) &= m\left(\frac{q-q_0}{\t} + \frac{g\t}{2}\right),\\
  q_0(q, t, p_0, t_0) &= q + \frac{g\t^2}{2} - \frac{p_0\t}{m},\\
  t_0(q, t, p_0, q_0) &= t - \frac{p_0}{mg} \mp \sqrt{\frac{p_0^2}{m^2g^2} -2 \frac{q - q_0}{g}}
\end{align*}

In the last case, the appropriate branch must be chosen to meet the physical boundary
conditions.  Using these, we can express the action as:

\begin{align*}
  S(q,t;q_0,t_0) &= m\left(\frac{(q-q_0)^2}{2\t} - \frac{g(q+q_0)\t}{2} - \frac{g^2\t^3}{24} \right),\\
  S^{P}(q,t;p_0,t_0) &= \left(\frac{p_0^2}{2m} - mgq\right)\t - \frac{mg^2\t^3}{6},\\
  S^{G}(q,t;p_0,q_0) &= \frac{-p_0^3}{6m^2g} - p_0q_0 \mp \frac{\frac{p_0^2}{2m} +
  mg(2q+q_0)}{3g\sqrt{m/2}}\sqrt{\frac{p_0^2}{2m} - mg(q - q_0)},\\
                     &= \frac{2p_0(q-q_0) \mp \left(\frac{p_0^2}{2m} + mg(2q+q_0)\right)\t}{3}.
\end{align*}

The expression for $S^{G}(q;p_0,q_0)$ is a bit messy, but notably does not depend on
$t_0$ due to the time-invariance of the problem.  However, to simplify expressions
later, we present it in the second form where $\t$ should be replaced by
\begin{gather*}
  \t(q;p_0,q_0) = \frac{p_0}{mg} \pm \sqrt{\frac{p_0^2}{m^2g^2} -2 \frac{q - q_0}{g}}.
\end{gather*}


```{margin}
The partials on the rhs are wrt $(q, t;q_0, t_0)$:

\begin{align*}
  p_{0,q} &= -p_{0,q_0} = \frac{m}{\t}\\
  p_{0,t_0} &= -p_{0,t}\\
  &=\frac{p_0}{\t}-mg\\
  &=\frac{m(q-q_0)}{\t^2} - \frac{mg}{2}
\end{align*}
```

````{admonition} Exercise

Check the relationships between the various partial derivatives of the actions.  The
following may be helpful:

\begin{gather*}
  p(q, t, q_0, t_0) = m\left(\frac{q-q_0}{\t} - \frac{g\t}{2}\right),\\
  \begin{aligned}
    H = H_0 &= m\left(\frac{(q-q_0)^2}{2\t^2} + \frac{g^2\t^2}{8} + \frac{g(q+q_0)}{2}\right)\\
      &= \frac{p_0^2}{2m} + mgq - gp_0\t + \frac{mg^2\t^2}{2}.
  \end{aligned}
\end{gather*}

First with respect to the variables $(q, t;q_0, t_0)$:

\begin{align*}
  \pdiff{S}{q} &= p = m\left(\frac{q-q_0}{\t} - \frac{g\t}{2}\right),\\
  \pdiff{S}{q_0} &= -p_0 = -m\left(\frac{q-q_0}{2\t} + \frac{g\t}{2}\right),\\
  \pdiff{S}{t} &= -H = -m\left(\frac{(q-q_0)^2}{2\t^2} + \frac{g(q+q_0)}{2} + \frac{g^2\t^2}{8}\right),\\
  \pdiff{S}{t_0} &= H_0 = m\left(\frac{(q-q_0)^2}{2\t^2} + \frac{g(q+q_0)}{2} + \frac{g^2\t^3}{8} \right).
\end{align*}

Next, with respect to the variables $(q, t;p_0, t_0)$:

\begin{align*}
  \pdiff{S^{P}}{q} &= p - p_0 = -mg\t,\\
  \pdiff{S^{P}}{t} &= -H + p_0\left(\frac{p_0}{m} -g\t\right)
    = \frac{p_0^2}{2m} - mgq - \frac{mg^2\t^2}{2} ,\\
  \pdiff{S^{P}}{t_0} &= H_0 - p_0\left(\frac{p_0}{m} -g\t\right)
     =-\frac{p_0^2}{2m} + mgq + \frac{mg^2\t^2}{2},\\
  \pdiff{S^{P}}{p_0} &= p_0\frac{\t}{m}
                      =\frac{p_0}{m}\t.
\end{align*}

Finally, with respect to the variables $(q, t;p_0, q_0)$:

\begin{align*}
  \pdiff{S^{G}}{q} &= p - \frac{H_0}{\frac{p_0}{m} - g\t}
                    = ???,\\
  \pdiff{S^{G}}{t} &= -H + H_0 = 0,\\
  \pdiff{S^{G}}{q_0} &= -p_0 + \frac{H_0}{\frac{p_0}{m} - g\t}
                      =???,\\
  \pdiff{S^{G}}{p_0} &= H_0\left(\frac{p_0}{\t} - mg\right) 
                      = ???.
\end{align*}
````

# Atom Laser

This section contains some notes about a continuous atom laser related to a research
project I am working on.  It might not make much sense if you are not familiar with the
project.  (Feel free to ask.)

## Experimental Setup

A continuous atom laser is formed by resonantly out-coupling atoms in a trapped BEC to a
state that is not trapped, which falls under the influence of gravity and some
additional potentials.  (For more details, see {ref}`sec:atom-laser`.)

:::{margin}
The out-coupled atoms are at a sufficiently low density that the non-linear interaction
can be neglected.  If the out-coupling $\Omega$ is weak, then the depletion of the
condensate $\ket{\psi_0}$ can be neglected, leading to this formulation.
:::
The system can be modeled quite well by the following Schrödinger equation:

\begin{gather*}
  \I\hbar \partial_t \ket{\psi_a(t)} 
  = \left(\frac{\op{p}_z^2}{2m} + V_a(\op{z}) - E_0\right)\ket{\psi_a(t)}
  + \Omega\ket{\psi_0}
\end{gather*}

where $\ket{\psi_0}$ is the condensate wavefunction, and $E_0 = \mu - \hbar\omega$ is the
chemical-potential of the condensate minus an energy shift due the frequency of the
out-coupling, which can be used to shift where the atoms out-couple.  (As discussed in
{ref}`sec:atom-laser`, efficient out-coupling happens within a small region around where
$V_a(z) = E_0$.)

The out-coupled atoms fall under the influence of potential $V_a(z) \approx mgz$ where
deviations occure due to small effects related to magnetic field gradients and optical
"poky" potentials.  As part of the experimental procedure, the atoms are subjected to
$\pi$ or $\pi/2$ pulses, which create mixtures with another state $b$ which experiences
a slightly different potential $V_b(z) = V_a(z) + \delta V(z) \approx V_a(z)$.  These
transitions occur about an unknown axis which we take to be $\hat{y}$ and have the
following form:

\begin{gather*}
  \op{U}_{\theta} = e^{\I \theta \mat{\sigma}_y/2} 
  = \cos\frac{\theta}{2}\mat{1} + \I\sin\frac{\theta}{2}\mat{\sigma}_y
  =
  \begin{pmatrix}
    \cos\frac{\theta}{2} & \sin\frac{\theta}{2}\\
    -\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
  \end{pmatrix}, \\
  \op{U}_{\pi/2}
  =
  \frac{1}{\sqrt{2}}
  \begin{pmatrix}
    1 & 1\\
    -1 & 1
  \end{pmatrix}, \qquad
  \op{U}_{\pi} = 
  \begin{pmatrix}
    0 & 1\\
    -1 & 0
  \end{pmatrix}.
\end{gather*}

Two procedures are used.  Both start with a well-established atom laser in the
quasi-stationary state $\ket{\psi_a}$ which we can, by shifting coordinates, take to satisfy:

\begin{gather*}
  \left(\frac{\op{p}_z^2}{2m} + V_a(\op{z})\right)\ket{\psi_a} =
  - \Omega e^{\I z_0 \op{p}_z/\hbar}\ket{\psi_0}
\end{gather*}

1. The first interferometer applies a $\op{U}_{\pi/2}$ pulse at time $t_1$, then a second
   $\op{U}_{\pi/2}$ pulse at time $t_2 = t_1 + t_{\mathrm{wait}}$.
2. The second "spin-echo" interferometer first applies a $\op{U}_{\pi/2}$ pulse at time
   $t_1$, then a $\op{U}_{\pi}$ pulse at time $t_2$, and finally a $\op{U}_{\pi/2}$
   pulse at time $t_3  = t_1 + t_{\mathrm{wait}}$.

### Simplified WKB Analysis

:::{margin}
This assumes that we can use a single trajectory for both states, which is approximately
correct, but in the code, we properly compute the full action, using only the
approximation that the $\theta$-pulses are instantaneous at some point in the appropriate interval.
:::
Heuristically, the two procedures give rise to the following propagation:

\begin{gather*}
  \mat{U}_{t_f, t_2}
  \underbrace{
    \frac{1}{\sqrt{2}}
    \begin{pmatrix}
      1 & 1\\
      -1 & 1\\
    \end{pmatrix}
  }_{\mat{U}_{\pi/2}}
  \mat{U}_{t_2, t_1}
  \underbrace{
    \frac{1}{\sqrt{2}}
    \begin{pmatrix}
      1 & 1\\
      -1 & 1\\
    \end{pmatrix}
  }_{\mat{U}_{\pi/2}}
  \mat{U}_{t_1, t_0}
  \begin{pmatrix}
    1\\
    0
  \end{pmatrix}\\
  \mat{U}_{t_f, t_3}
  \underbrace{
    \frac{1}{\sqrt{2}}
    \begin{pmatrix}
      1 & 1\\
      -1 & 1\\
    \end{pmatrix}
  }_{\mat{U}_{\pi/2}}
  \mat{U}_{t_3, t_2}
  \underbrace{
    \begin{pmatrix}
     0 & 1\\
     -1 & 0\\
    \end{pmatrix}
  }_{\mat{U}_{\pi}}
  \mat{U}_{t_2, t_1}
  \underbrace{
    \frac{1}{\sqrt{2}}
    \begin{pmatrix}
      1 & 1\\
      -1 & 1\\
    \end{pmatrix}
  }_{\mat{U}_{\pi/2}}
  \mat{U}_{t_1, t_0}
  \begin{pmatrix}
    1\\
    0
  \end{pmatrix},\\
  \mat{U}_{t_2, t_1}
   = 
   \begin{pmatrix}
      e^{\I S^{a}_{12}/\hbar}\\
      & e^{\I S^{b}_{12}/\hbar}
    \end{pmatrix}.
\end{gather*}

Using the notation $A_{ij} = \exp(\I S^{a}_{ij}/\hbar)$ and $B_{ij} = \exp(\I
S^{b}_{ij}/\hbar)$, we have:

* Simple interferometer:

  \begin{align*}
    \Psi(t_f)
    &\approx 
    \frac{1}{2}
    \begin{pmatrix}
      A_{01}A_{12}A_{2f} - A_{01}B_{12}A_{2f}\\
      -A_{01}A_{12}B_{2f} - A_{01}B_{12}B_{2f}
    \end{pmatrix}, \\
    \begin{pmatrix}
      n_a(t_f)\\
      n_b(t_f)
    \end{pmatrix} 
    &\propto
    \begin{pmatrix}
      \abs{A_{12} - B_{12}}^2\\
      \abs{A_{12} + B_{12}}^2
    \end{pmatrix}.
  \end{align*}

* Spin-echo interferometer:

  \begin{align*}
    \Psi(t_f) 
    &\approx 
    \frac{1}{2}
    \begin{pmatrix}
      -A_{01}B_{12}A_{23}A_{3f} - A_{01}A_{12}B_{23}A_{3f}\\
      A_{01}B_{12}A_{23}B_{3f} - A_{01}A_{12}B_{23}B_{3f}
    \end{pmatrix},\\
    \begin{pmatrix}
      n_a(t_f)\\
      n_b(t_f)
    \end{pmatrix} 
    &\propto
    \begin{pmatrix}
      \abs{B_{12}A_{23} + A_{12}B_{23}}^2\\
      \abs{B_{12}A_{23} - A_{12}B_{23}}^2
    \end{pmatrix}.
  \end{align*}

To obtain an estimate for what we see, we now make some additional approximations:

:::{margin}
The essential complication is that the action $\mat{S}$ must now be regarded as a
matrix, and since this generally will not commute at different times, expanding the
wavefunction does not yield a simple Hamilton-Jacobi equation.
:::
1. We assume that the dominant contribution to the potentials $V_a(z) \approx
   V_b(z) \approx mgz$ is gravity, and that, consequently, the motion can be simply
   describe by free-fall.  Under this approximation, we may view the particle as a
   two-component $SU(2)$-valued particle, whose components evolve on the Bloch sphere as
   the particle falls.  To go beyond this, we must use some sort of multi-component WKB
   formalism which is quite complicated.
2. We assume that the pulses $\op{U}_{\theta}$ are essentially instantaneous.
3. To obtain simple expressions, we shall also consider the limit where
   $t_{\mathrm{wait}} \rightarrow 0$, which we call the "impulse approximation".

With these assumptions, we may consider a single particle falling from $z=0$ at some
initial time $t_0$.  The position and momentum are approximately:

\begin{gather*}
  q(t) \approx -\frac{g(t-t_0)^2}{2}, \qquad
  p(t) \approx -mg(t-t_0) = -m\sqrt{-2gq(t)}.
\end{gather*}

The phase accumulates through the integral of the action

\begin{gather*}
  \delta S(z_1, z_2) = 
  \int_{t_1}^{t_2}\mathcal{L}\d{t} = \int_{z_1}^{z_2}\frac{\mathcal{mL}}{p}\d{z},\qquad
  \mathcal{L} = E - 2 V(z),\\
  V(z) = V_0(z)\mat{1} + \delta V(z) \mat{\sigma}_z = \begin{pmatrix}
    V_a(z)\\
    & V_b(z)
  \end{pmatrix}.
\end{gather*}

Under our approximations, we can take $E\approx 0$, so that

\begin{gather*}
  A_{ij}, B_{ij} \approx \exp\left(
    \frac{\I}{\hbar}
    \int_{z_i}^{z_j} \frac{2V_{a,b}(z)}{\sqrt{-2gz}}
  \right).
\end{gather*}

### Impulse approximation

If the intervals $\delta t_{ij} = t_i - t_j$, then we can take the integrand to be
constant over this interval, and use $z_j \approx z_i + \delta t_{ij} p_i/m$, which
cancels the denominator:

\begin{gather*}
  A_{ij}, B_{ij} \approx \exp\left(
    \frac{\I}{\hbar}
    (z_i - z_j)\frac{2V_{a,b}(z_{ij})}{\sqrt{-2gz_{ij}}}
  \right)
  \approx
  \exp\left(
    -\frac{\I}{\hbar}\delta t_{ij}2V_{a,b}(z_{ij})
  \right)
\end{gather*}

Looking at the densities $n_{a,b}$ above, the simple interferometer measures the phase
difference between $A_{12}$ and $B_{12}$, while the spin-echo interferometer measures
the phase difference between $B_{12}A_{23}$ and $A_{12}B_{23}$:

* Simple interferometer -- measures contours of $\delta V(z)$:

  \begin{gather*}
    \hbar\delta\phi \approx  2\delta t_{12}\bigl(V_{a}(z_{12})-V_{b}(z_{12})\bigr)\\
    = 2 t_{w}\bigl(V_{a}(z_{12})-V_{b}(z_{12})\bigr)
  \end{gather*}

* Spin-echo interferometer -- measures contours of $\delta V'(z)$:

  \begin{gather*}
    \hbar\delta\phi \approx  2\Bigl(
      \delta t_{12}\bigl(V_{b}(z_{12}) - V_{a}(z_{12})\bigr)
      -
      \delta t_{23}\bigl(V_{b}(z_{23}) - V_{a}(z_{23})\bigr)
      \Bigr),\\
    \approx
    \frac{2t_{w}^2 p_{13}}{m} \Bigl(V_{a}'(z_{13}) - V_{b}'(z_{13})\Bigr)
  \end{gather*}
  
In the second expressions, we have taken $\delta t_{12} = \delta t_{23} = t_{w}$ and
$z_{23} \approx z_{12} + t_w p_{13}/m$.  This




## Falling Gaussian

Consider an initial state $\psi_0(z)$.  The WKB approximation for the time evolution is:

\begin{gather*}
  \psi(z, t) = \int \d{z_0} I(z, z_0;t)\psi_0(z), \qquad
  I(z, z_0; t) = \sqrt{S_{,z,z_0}}e^{\tfrac{i}{\hbar}S(z, z_0;t)},
\end{gather*}

where the action $S(q,t;q_0,t_0)$ is computed over classical trajectories $q(0) = z_0$
and $q(t) = z$ -- the first case above.  We immediately have

\begin{gather*}
  S = m\left(\frac{(z-z_0)^2}{2 t} - \frac{g(z+z_0)t}{2} - \frac{g^2t^3}{24}\right),\\
  S_{,z,z_0} = -\frac{m}{t}, \\
  I(z, z_0; t) = \sqrt{-\frac{m}{t}}e^{\tfrac{i}{\hbar}S(z, z_0;t)},\\
  \psi(z, t) = \int \d{z_0} I(z, z_0;t)\psi_0(z), \qquad
\end{gather*}


\begin{gather*}
  S_{,z,z_0} = -\frac{m}{t}, \\
  S^P_{,z,p_0} = 0, \\
  S^{G}_{,z,z_0} = -\frac{m}{t}, \\  
\end{gather*}

$$
  \mp \frac{1}{\sqrt{2m}} S^{G}_{,z} = 
    \frac{\frac{p_0^2}{2m} - mg(2q - q_0)}{2\sqrt{\frac{p_0^2}{2m} - mg(q - q_0)}},\\
  \mp \frac{1}{mg\sqrt{2m}} S^{G}_{,z,z_0} = 
    \frac{\frac{p_0^2}{2m} + mgq_0}{4\sqrt{\frac{p_0^2}{2m} - mg(q - q_0)}^3}\\
  S^{G}_{,z,z_0} = \mp \frac{m^2E V'(q)}{p^3}
$$

p^2 = 2m(E-V(q))

```{code-cell} ipython3
:tags: [hide-input]

plt.rcParams['figure.dpi'] = 300
from scipy.integrate import solve_ivp

z = np.linspace(-10, 3, 300)[:, None]
z0 = np.linspace(-3, 3, 200)[None, :]
dz0 = np.diff(z0.ravel()).mean()
m = 1.0
hbar = 1.0
sigma = 1.0
g = 10.0
t = 1.0

psi0 = np.exp(-(z0.ravel()/sigma)**2/2)
n0 = abs(psi0**2)
N = np.trapz(n0, z0.ravel())
psi0 /= np.sqrt(N)
n0 = abs(psi0**2)

fig, ax = plt.subplots()
ax.plot(z0.ravel(), n0, label=r"$\psi_0$")

zcs = np.array([-2, -4, -6, -8])
for zc in zcs:
    t = np.sqrt(-2*zc/g)
    S = m * ((z-z0)**2/2/t - g*(z+z0)*t/2 - g**2*t**3/24)
    I = np.sqrt(m/t)*np.exp(1j/hbar*S)
    psi = I.dot(psi0) * dz0
    n = abs(psi.ravel())**2
    N = np.trapz(n, z.ravel())
    print(N/2/np.pi)   ### Where does this 2\pi come from?
    l, = ax.plot(z.ravel(), n/N, label=fr"$z_c(t)={zc}$")
    ax.axvline([zc], ls=":", c=l.get_c(), alpha=0.5)

ax.legend()
ax.set(xlabel="$z$", ylabel="$n$");

```



### Interference 1

Now consider two streams of particles continuously injected at $z = 0$.  The first steam
falls without any external potential other than gravity, while the second experiences an additional
potential $\lambda V(z)$.  We choose our reference frame so that $H_0 = H = 0$.  Then, for the
first set of particles, we have:

\begin{gather*}
  z(t) = -g \frac{t^2}{2}, \qquad
  \dot{z} = -gt = -g\sqrt{\frac{-2z}{g}}, \qquad
  t = \sqrt{\frac{-2z}{g}},\\
  S = -mgzt - \frac{mg^2t^3}{6} = \frac{mg^2t^3}{3}
    = \frac{mg^2}{3}\left(\frac{-2z}{g}\right)^{3/2}.
\end{gather*}

If the potential $\lambda V(z)$ for the second species is small, we may use the Born
approximation, under which the leading order correction to the action is:

\begin{gather*}
  S_a - S = - \lambda \int_{t_0}^{t}V\bigl(z(t)\bigr)\d{t} + \order(\lambda^2)
          = - \lambda \int_{0}^{z}\frac{V(z)}{\dot{z}}\d{z} + \order(\lambda^2)\\
          = - \frac{\lambda}{\sqrt{2g}}\int_{0}^{z}\frac{V(z)}{\sqrt{-z}}\d{z} + \order(\lambda^2)
\end{gather*}


```{code-cell} ipython3
from scipy.integrate import cumtrapz
micron = 1.0
mm = 1000*micron
meter = 1000*mm
sec = 1.0
amu = 1e-3
V0 = 85340

m = 87.0 * amu
g1 = 9.81 * meter/sec**2
g2 = 9.80 * meter/sec**2

x = np.linspace(-200, 200, 500)[:, None]
z = np.linspace(-400, 0, 502)[None, :]
z0 = -100

sigma = 20.0

def V(x, z):
    return V0 * np.exp(-((z-z0)**2+x**2)/2/sigma**2)
    
V1 = m * g1 * z + 0*x
V2 = m * g2 * z + 10*V(x, z)

dS1_dz = -np.sqrt(abs(-2*m*V1))
dS2_dz = -np.sqrt(abs(-2*m*V2))

S = cumtrapz((dS2_dz - dS1_dz)[:, ::-1], axis=1, initial=0)[:, ::-1]

```

### Interference 2

```{margin}
We assume here that the particles continually fall: $z<0$ and $p<0$.
```
Now we consider a slightly different interference phenomenon, again with two streams
injected with $p=0$ at $z=0$.  The first falls in the potential $V(z)$ for all time,
while the second experiences a different potential $V(z) + \Delta(z)$ for a short time
interval between $t_1$ and $t_2 = t_1 + \delta_t$.

For the first particle, we have the usual:

$$
  t_i - t_0^{a} = \int_0^{z_i} \frac{-m}{\sqrt{-2mV(Z)}}\d{Z}.
$$

For the second particle, we must consider the three different time intervals:

\begin{gather*}
  t_i - t_0^{b} = 
  \overbrace{
    \int_0^{z_1} \frac{-m}{\sqrt{-2mV(Z)}}\d{Z}
  }^{t_1 - t_0^b}\\
  +
  \overbrace{
    \int_{z_1}^{z_2} \frac{-m}{\sqrt{-2m\Bigl(V(Z) + \Delta(Z) - \Delta(z_1)\Bigr)}}\d{Z}
  }^{t_2 - t_1}\\
  +
  \overbrace{
    \int_{z_2}^{z_i} \frac{-m}{\sqrt{-2m\Bigl(V(Z) - \Delta(z_1) + \Delta(z_2)\Bigr)}}\d{Z}
  }^{t_i - t_2}.
\end{gather*}

If we assume that the motion is monotonic $p<0$, then we can use the geometric optics
picture to re-express this in terms of view both particles falling through a
conservative potential:

\begin{align*}
  V^a(z) &= V(z),\\
  V^b(z) &= \begin{cases}
    V(z) & z_1 < z\\
    V(z) + \Delta(z) - \Delta(z_1) & z_2 < z < z_1\\
    V(z) + \Delta(z_2) - \Delta(z_1)  & z < z_2.
  \end{cases}
\end{align*}

```{margin}
The simplification here comes from

\begin{gather*}
  L = K-V \\
    = H-2V
\end{gather*}

where $H = p^2/2m + V$ is
conserved, hence 

\begin{gather*}
  L\d{t} = (-2V/v) \d{z} \\
         = -2mV/p \d{z} = p\d{z}.
\end{gather*}
```
Here we have exchanged $t_1$ and $t_2$ for $z_1$ and $z_2$.  Of course, this must be
done for each trajectory, but for a given trajectory, we can solve for the final
momentum and action geometrically:

\begin{gather*}
  p^{a,b}(z) = -\sqrt{-2mV^{a,b}(z)},\qquad
  S^{a,b}(z) = \int_{0}^{z}p^{a,b}(z) \d{z}.
\end{gather*}

The impulse approximation considers the limit $z_2 \rightarrow z_1$ so that

<!--
\begin{align*}
  V^b(z) &= V^a(z) + \delta_z \Delta'(z_1) \Theta(z_1 - z) + \order(\delta_z^2)\\
  p^b(z) &= p^{a}(z) + \delta_z \Delta'(z_1) \Theta(z_1 - z) \frac{m}{\sqrt{-2mV(z)}} 
  + \order(\delta_z^2)\\
  S^b(z) &= S^{a}(z) 
  +
  \delta_z \Delta'(z_1)\Theta(z_1 - z)
  \int_{z_1}^{z}\frac{m}{\sqrt{-2mV(Z)}}\d{Z} 
  +\order(\delta_z^2).
\end{align*}
-->

**Something is wrong with $S$ here... should have an integral so $\Delta(z_1)$ not $\Delta'(z_1)$.**
\begin{align*}
  V^b(z) &= V^a(z) + \delta_z \Delta'(z_1) \Theta(z_1 - z) + \order(\delta_z^2)\\
  p^b(z) &= p^{a}(z) - \delta_z \Delta'(z_1) \Theta(z_1 - z) \frac{m}{p^{a}(z)} 
  + \order(\delta_z^2)\\
  S^b(z) &= S^{a}(z) 
  -
  \delta_z \Delta'(z_1)\Theta(z_1 - z)
  \int_{z_1}^{z}\frac{m}{p^{a}(z)}\d{Z} 
  +\order(\delta_z^2).
\end{align*}

The only task now is to relate $z_1$ to 

\begin{gather*}
  \delta_t = \delta_z\frac{m}{p^{a}(z_1)} + \order(\delta_z^2)\\
  t_i - t_1 = \int_{z_1}^{z_i} \frac{m}{p^b(z)}\d{Z}\\
  =
  \int_{z_1}^{z_i} \frac{m}{p^a(z)}\left(
    1 + \delta_z \Delta'(z_1)\frac{m}{[p^{a}(Z)]^2} 
  \right)\d{Z}+\order(\delta_z^2).
\end{gather*}

To leading order, we can just solve for the reference particle to determine $z_1$:

\begin{gather*}
  t_i - t_1 = \int_{z_1}^{z_i} \frac{m}{p^a(Z)}\d{Z}.
\end{gather*}

Then, the phase shift at $z_i$ will be:

\begin{gather*}
  S^b(z_i) - S^{a}(z_i) = -(t_i-t_1)\delta_z \Delta'(z_1)\Theta(z_1 - z)
  +\order(\delta_z^2).
\end{gather*}

Hence, the phase shift directly maps the gradient of the potential at the position $z_1$.

From a geometric perspective, the motion of the second particle can be through of as
described by a spatially dependent potential.


[Lagrangian mechanics]: <https://en.wikipedia.org/wiki/Legendre_transformation>
[Hamiltonian mechanics]: <https://en.wikipedia.org/wiki/Hamiltonian_mechanics>
[Legendre transformation]: <https://en.wikipedia.org/wiki/Legendre_transformation>
[WKB approximation]: <https://en.wikipedia.org/wiki/WKB_approximation>
[gaussian integral]: <https://en.wikipedia.org/wiki/Gaussian_integral>
[Maxwell relations]: <https://en.wikipedia.org/wiki/Maxwell_relations>
[Hamilton's principal function]: <https://en.wikipedia.org/wiki/Hamilton%E2%80%93Jacobi_equation#Hamilton's_principal_function>
