---
jupytext:
  formats: ipynb,md:myst
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

Strings
=======

```{contents} Contents
:local:
:depth: 3
```

```{code-cell} ipython3
:tags: [hide-cell]

import mmf_setup;mmf_setup.nbinit()
import logging;logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
%pylab inline --no-import-all
import manim.utils.ipython_magic
```

## Balls and Springs

:::{margin}
The idea of a system on a **lattice** with spacing $a$ is very common in quantum
mechanics, for example, when studying electrons in a solid, or quantum spin chains.  We
will say a bit more about this later.
:::

Here we carefully review the balls-and-springs model for a string.  Let the total length
be $l$ with $N$ balls and $N+1$ springs so that the equilibrium position of the $n$th
ball is $x_n = an$ where $a = l/(N+1)$ is the **lattice spacing**.  We let $n=0$ denote
the fixed left endpoint $x_0 = 0$ and $x_{N+1} = l$ denote the fixed right end-point.
We will include these "ghost" points in our numerical arrays to simplify the coding, but
they will not be allowed to evolve.

:::{margin}
Note that the equilibrium length $l_0$ drops out here as discussed in class.  This is
true for longitudinal modes but not transverse modes, so don't just assume it to be the case.
:::

The acceleration of each point is then given by Hook's law summing the
extension/compression of each neighboring spring:

\begin{gather*}
  m\ddot{x}_{n} = k(x_{n+1} - x_{n} - l_0) - k(x_{n} - x_{n-1} - l_0)
                = k(x_{n+1} - 2x_{n} + x_{n-1}).
\end{gather*}

::::{note}

In code, this can be simply computed using arrays with the following:

```python
ddx[1:-1] = k * (x[2:] - 2 * x[1:-1] + x[:-2]) / m
```

::::

which makes use of the fixed ghost points `x[0]`$=x_0$ and `x[N+1]`$=x_{N+1}$.

```{code-cell} ipython3
:cell_style: center

from scipy.integrate import solve_ivp


class String:
    """Representation of balls and springs model."""

    def __init__(self, L, N, m=1.0, k=1.0, l0=1.0):
        self.L, self.N, self.m, self.k, l0 = L, N, m, k, l0
        self.dx = self.L / (self.N+1)
        # Equilibrium positions.  Includes fixed end-points (ghosts)
        self.xs = np.arange(self.N+2) * self.dx - L/2
        self.dxs = np.zeros(N+2)

    def compute_dy_dt(self, t, y):
        x, dx = y.reshape((2, self.N))
        ddx = np.zeros_like(x)
        ddx[1:-1] = self.k*(x[2:] - 2*x[1:-1] + x[:-2])/self.m
        return np.ravel([dx, ddx])
        
    def evolve(self, t=1.0, **kw):
        """Evolve by time `t` brute force solution with `solve_ivp`."""
        y0 = np.ravel([self.xs, self.dxs])
        res = solve_ivp(self.compute_dy_dt, t_span=(0, t), y0=y0, **kw)
        self.xs, self.dxs = res.y
        return res        
```

```{code-cell} ipython3
:tags: [hide-input]

%%manim -v WARNING -qm String1
from manim import *


class Spring(ParametricFunction):
    def __init__(self, x0, x1, y=0.1, N=3, **kw):
        super().__init__(
            lambda t: [x0 + t * (x1 - x0), y * np.sin(2 * np.pi * N * t), 0],
            t_range=[0, 1],
            **kw
        )


class String1(Scene):

    model = String(L=10.0, N=5)

    def construct(self):
        xs, L = self.model.xs, self.model.L

        balls = Group(*[LabeledDot("m", point=(_x, 0, 0)) for _x in xs[1:-1]])
        rs = [0] + [_d.radius for _d in balls] + [0]
        springs = Group(
            *[
                Spring(x0=_x0 + _r0, x1=_x1 - _r1)
                for _x0, _x1, _r0, _r1 in zip(xs[:-1], xs[1:], rs[:-1], rs[1:])
            ]
        ).set_color(BLUE)
        N = 6
        dx = 0.4
        dy = 0.2
        height = 1
        left_wall = Group(
            Line([0, height, 0], [0, -height, 0]),
            *[
                Line([0, y, 0], [-dx, y - dy, 0])
                for y in np.linspace(-height, height, N)
            ]
        )

        right_wall = left_wall.copy().rotate_about_origin(np.pi)
        left_wall.shift(LEFT * self.model.L / 2)
        right_wall.shift(RIGHT * self.model.L / 2)

        br_L = BraceBetweenPoints([L / 2, height, 0], [-L / 2, height, 0])
        br_dx = BraceBetweenPoints(*[_b.get_bottom() for _b in balls[:2]])
        annotations = Group(
            *[br_L, br_dx, br_L.get_tex("l = (N+1)a"), br_dx.get_tex(r"a")]
        )
        # dot2 = Dot([2, 1, 0])
        # line = Line(dot.get_center(), dot2.get_center()).set_color(ORANGE)
        # b1 = Brace(line)
        # b1text = b1.get_text("Horizontal distance")
        # b2 = Brace(line, direction=line.copy().rotate(PI / 2).get_unit_vector())
        # b2text = b2.get_tex("x-x_1")
        # self.add(line, dots, dot2, b1, b2, b1text, b2text)
        self.add(balls, springs, left_wall, right_wall, annotations)
```

### Normal Modes

To solve the normal-mode problem, we consider deviations $\eta_n$ of each spring from
the equilibrium positions.  This gives the following equations, including the ghost
points $\eta_0 = \eta_{N+1} = 0$:

\begin{gather*}
  \left.\ddot{\eta}_{n} = 
    \frac{k}{m}(\eta_{n+1} - 2\eta_{n} + \eta_{n-1})
  \right|_{n=1}^{N},\qquad
  \ddot{\vect{\eta}} = 
  \frac{k}{m}
  \begin{pmatrix}
    -2 & 1\\
     1 & -2 & 1 \\
       & 1 & \ddots & \ddots\\
       &   & \ddots & -2 & 1\\
       &   &        &  1 & -2
  \end{pmatrix}
  \cdot
  \vect{\eta}.
\end{gather*}

When trying to work out matrices like this, it can be helpful to include the "ghost"
points $\eta_{0}$ and $\eta_{N+1}$ so that you can be explicit about the boundary
conditions.  For example:

\begin{gather*}
  \newcommand{\ghost}[1]{{\color{red}#1}}
  \ddot{\vect{\eta}} = 
  \frac{k}{m}
  \begin{pmatrix}
    \ghost{1} & -2 & 1\\
              &  1 & -2 & 1 \\
              &    & 1 & \ddots & \ddots\\
              &    &   & \ddots & -2 & 1\\
              &    &   &        &  1 & -2 & \ghost{1}
  \end{pmatrix}
  \cdot
  \begin{pmatrix}
    \ghost{\eta_{0}} \\
    \vect{\eta} \\
    \ghost{\eta_{N+1}}
  \end{pmatrix}
\end{gather*}

This makes it clear that our previous formulation is assuming [**Dirichlet boundary
conditions**](https://en.wikipedia.org/wiki/Dirichlet_boundary_condition) where
$\eta_{0} = \eta_{N+1} = 0$.  Other options are possible, for example, [periodic
boundary conditions](https://en.wikipedia.org/wiki/Periodic_boundary_conditions) where
$\eta_{0} = \eta_{N}$ and $\eta_{N+1} = \eta_{1}$.  This could be realized as a
loop of string (see for example Fig. 24.4 in the book). This leads to:

:::{math}
:class: full-width
\begin{gather*}
  \newcommand{\ghost}[1]{{\color{red}#1}}
  \ddot{\vect{\eta}} = 
  \frac{k}{m}
  \begin{pmatrix}
    \ghost{1} & -2 & 1\\
              &  1 & -2 & 1 \\
              &    & 1 & \ddots & \ddots\\
              &    &   & \ddots & -2 & 1\\
              &    &   &        &  1 & -2 & \ghost{1}
  \end{pmatrix}
  \cdot
  \begin{pmatrix}
    \ghost{\eta_{N}} \\
    \vect{\eta} \\
    \ghost{\eta_{1}}
  \end{pmatrix}
  =
  \frac{k}{m}
  \begin{pmatrix}
    -2 & 1  &        &        &    & 1\\
     1 & -2 & 1                       \\
       &  1 & \ddots & \ddots         \\
       &    &        & \ddots & -2 & 1\\
     1 &    &        &        &  1 & -2
  \end{pmatrix}
  \cdot
   \vect{\eta}
\end{gather*}
:::

While there are formal ways of solving this equation in terms of the eigenvectors of
[Toeplitz matrices](https://en.wikipedia.org/wiki/Toeplitz_matrix) (diagonal-constant
matrices), we can easily guess the form of the solution to be sine waves with nodes at
$n=0$ and $n=N+1$:

\begin{gather*}
  \DeclareMathOperator{\Re}{Re}
  \eta_{n}(t) = \Re\left( Ae^{\I \omega t}\sin(\alpha_{j} n)\right), \qquad
  \left.\alpha_{j} = \frac{\pi j}{N+1}\right|_{j=1}^{N}.
\end{gather*}

This is equation (24.55) in {cite:p}`Fetter:2003`.  Substituting, we find:

\begin{align*}
  -\omega^2 \eta_{n}(t) &= 
    \frac{k}{m}
    \frac{\overbrace{\sin(\alpha_{j} n+\alpha_{j})}
          ^{\llap{\sin(\alpha_{j} n)\cos(\alpha_{j}) 
                  + \cos(\alpha_{j} n)\sin(\alpha_{j})\hspace{-0.8in}}}
          - 2\sin(\alpha_{j} n)
          + \overbrace{\sin(\alpha_{j} n-\alpha_{j})}
            ^{\rlap{\hspace{-0.8in}\sin(\alpha_{j} n)\cos(\alpha_{j}) 
                    - \cos(\alpha_{j}n)\sin(\alpha_{j})}}}{\sin(\alpha_{j} n)}
    \eta_n\\
    &=
    \frac{k}{m}\Bigl(2\cos (\alpha_{j}) - 2\Bigr)\eta_n.
\end{align*}

Hence, we have the following relationship between the frequency of the
mode $\omega$ to the mode number $l$:

\begin{gather*}
  \omega_{j}^2 = \frac{k}{m}2(1-\cos\alpha_{j}) 
               = \frac{k}{m}2\left(1-\cos\frac{\pi j}{N+1}\right).
\end{gather*}

Since this problem is exactly quadratic, this solution is actually an exact solution for
the original problem.

:::{note}
To compare, periodic boundary conditions have the same form of solution, but with a
slightly different "quantization" condition $\alpha_{j} (N+1) = \alpha_{j} + 2\pi j$ which gives
$\alpha_{j} = 2\pi j/N$.  Note that only $N$ of these modes are distinct since $j
\rightarrow j + N$ gives the same mode $\sin(\alpha_{j+N}n) = \sin(\alpha_{j} + 2\pi) =
\sin(\alpha_{j})$ as discussing in Eq. (24.46) of {cite:p}`Fetter:2003`.
:::

### Lagrangian

In the case described above without any dissipation, we can express the problem in the
Lagrangian framework in terms of a matrix equation:

\begin{gather*}
  L(\vect{\eta}, \dot{\vect{\eta}}) = K - V = 
  \frac{m}{2} \dot{\vect{\eta}}^T\cdot\dot{\vect{\eta}}
  -
  \frac{k}{2}\vect{\eta}^T\cdot
  \overbrace{
  \begin{pmatrix}
     2 & -1\\
    -1 &  2 & -1 \\
       & -1 & \ddots & \ddots\\
       &    & \ddots &  2 & -1\\
       &    &        & -1 &  2
  \end{pmatrix}
  }^{-\mat{D_2}}
  \cdot\vect{\eta}.
\end{gather*}

Note that this matrix, which we call $\mat{D}_2$ looks like a finite-difference
approximation for the second derivative with unit spacing.

### Continuum Limit

:::{margin}
Recall that [springs combine like
capacitors](https://en.wikipedia.org/wiki/Series_and_parallel_springs): for spring in
parallel, the spring constants directly add, but for springs in series, their inverses
add.
:::
We now consider taking the what is called the **continuum limit** whereby the
ball-and-spring model approaches a continuum model for a string.  In this limit, we take
$N \rightarrow \infty$ while holding various combinations of the parameters constant to
represent the physical properties of the string, such as the total mass $M = Nm$, length
$l = (N+1)a$, and overall string elasticity $k_{\mathrm{string}} = k/(N+1)$:

:::{margin}
The second row of constants, the mass-density (mass-per-unit-length) $\sigma$ and the
tension $\tau$ will appear in the final Lagrangian density below.
:::

\begin{gather*}
  m = \frac{M}{N}, \qquad
  a = \frac{l}{N+1}, \qquad
  k = (N+1)k_{\mathrm{string}},\\
  \frac{m}{a} = \frac{M(N+1)}{Nl} \rightarrow \frac{M}{l} = \sigma, \qquad
  ka = lk_{\mathrm{string}} = \tau.
\end{gather*}

We also note the following mapping of sums to integrals, and $\mat{D_2}$ to derivatives,
after identifying $\eta_n$ as the displacement of the original string at position $x =
x_n$:

\begin{gather*}
  a\sum_{n} \rightarrow \int_{0}^{l} \d{x}, \qquad
  \frac{\mat{D_2}}{a^2}\cdot\vect{\eta} \rightarrow \pdiff[2]{}{x}\eta(x), \qquad
  \eta_n = \eta(x_n)
\end{gather*}

The Lagrangian becomes

\begin{gather*}
  L(\vect{\eta}, \dot{\vect{\eta}}) \rightarrow
  \frac{m}{2} \overbrace{\frac{1}{a}\int_{0}^{l} \dot{\eta}^2(x) \d{x}}
                        ^{\sum_{n} \dot{\eta}_{n}^2}
  +
  \frac{k}{2}
  \overbrace{
  \frac{1}{a}
  \int_{0}^{l}
  \eta(x)
  \underbrace{a^2 \pdiff[2]{}{x}\eta(x)}_{\mat{D_2}\cdot\vect{\eta}}
  \d{x}}
  ^{\sum_{n}\eta_n[\mat{D_2}\cdot\vect{\eta}]_{n}},\\
  L = \int_{0}^{l} \mathcal{L}(\eta, \dot{\eta})\d{x}, \qquad
  \mathcal{L}(\eta, \dot{\eta}) = 
  \frac{m}{2a}\dot{\eta}^2 + \frac{ka}{2}\eta\nabla^2\eta
  \equiv
  \frac{\sigma}{2}\dot{\eta}^2 - \frac{\tau}{2}\norm{\vect{\nabla}\eta}^2,
\end{gather*}

where the last equivalence is valid after integrating by parts with the fixed boundary
conditions.  The quantity $\mathcal{L}(\eta, \dot{\eta})$ is called the **Lagrangian
density**, and plays the role of the Lagrangian in [classical field
theory](https://en.wikipedia.org/wiki/Classical_field_theory).  The "field equations"
follow by extremizing the continuum limit of the action:

::::{margin}
Here we must be a little more careful with what the partials mean, so we have used
$\dot{\eta}$ and $\vect{\nabla}\eta$ as the notation for the arguments of
$\mathcal{L}(\eta, \dot{\eta}, \vect{\nabla}\eta)$ while after these partials are
complete, the arguments are those of $x$ and $t$ as in $\eta(x,t)$.  One of the goals of
{cite:p}`Sussman:2015` is to develop a vary careful notation for this.

::::

\begin{gather*}
  S[\eta] = \int_{t_0}^{t_1} \d{t}\int_{0}^{L} \mathcal{L}(\eta, \dot{\eta}, \vect{\nabla}\eta),\\
  \delta S = 0 \quad \implies\quad
  \pdiff{}{t}\pdiff{\mathcal{L}}{\dot{\eta}} 
  + \vect{\nabla}\cdot \pdiff{\mathcal{L}}{\vect{\nabla}\eta} = \pdiff{\mathcal{L}}{\eta}.
\end{gather*}

Thus, for our string, we have the [wave equation](https://en.wikipedia.org/wiki/Wave_equation):

\begin{gather*}
  0 = \pdiff{}{t} \left(\sigma \pdiff{\eta}{t}\right)
  - \pdiff{}{x} \left(\tau \pdiff{\eta}{x}\right)
  =  \sigma\pdiff[2]{\eta}{t} + \tau \pdiff[2]{\eta}{x},\\
    \pdiff[2]{\eta}{t} = c^2\pdiff[2]{\eta}{x}, \qquad
    c = \sqrt{\frac{\tau}{\sigma}}.
\end{gather*}

The constant $c$ here is the longitudinal speed of sound along the string.  Note that if
we divide the Lagrangian density by $\tau/2$, and integrating by parts,
we can obtain the following forms:

\begin{gather*}
  \frac{-2}{\tau}\mathcal{L}(\eta) \equiv  
    \eta(x, t)\left(
      \overbrace{\frac{1}{c^2}\pdiff[2]{}{t} - \pdiff[2]{}{x}}^{\Box}
    \right)\eta(x, t)
  \equiv
  \partial^{\mu}\eta \partial_{\mu}\eta
\end{gather*}


where $\Box$ is the [d'Alembert
operator](https://en.wikipedia.org/wiki/D%27Alembert_operator), or the relativistic
covariant form if we consider $c$ as the speed of light.

Finally, we consider the normal modes in the continuum limit by relating $x_n = na
\rightarrow x$:

\begin{gather*}
  \eta(x, t) = \Re A e^{\I\omega t} \sin(k_j x) \leftarrow \Re A e^{\I\omega t} \sin(\alpha_j n)\\
  k_j = \frac{\pi j}{l}.
\end{gather*}

:::{margin}

Note that in quantum mechanics, if we multiply both $\omega$ and $k$ by $\hbar$, the
dispersion relationship becomes the kinetic energy as a function of momentum:

\begin{gather*}
  \hbar \omega = E(p) = E(\hbar k).
\end{gather*}
:::

We can then express the frequency $\omega(k_j)$ as a function of the **wave-vector**
$k_j$, which gives us the **dispersion relationship**:

\begin{gather*}
  \omega^2(k_{j}) = \frac{k}{m}2\left(1 - \cos\frac{\pi j}{N+1}\right)
                  = c^2\frac{2}{a^2}\Bigl(1 - \cos(k_j a)\Bigr)\\
  \omega(k_{j}) = \pm ck_j \left(1 - \frac{k_j^2 a^2}{24} + \order\bigl((k_j a)^4\bigr)\right).
\end{gather*}

This is the usual linear dispersion relationship associated with the wave equation,
which says that all waves travel at the same speed $c$.  The corrections to this, which
vanish in the continuum limit, describe the effects of the lattice.  Once can look for
these effects as deviations in high-energy physics, and provide Constraints on the
universe as a numerical simulation {cite:p}`Beane:2014`, for example.


```{code-cell} ipython3
import numpy as np, matplotlib.pyplot as plt

L = 10.0
c = 1.0
a = 1.0
k = np.linspace(0, 60, 100)

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(k*L, c*k, 'k-', label=r'$N=\infty$')

for N, fmt in [(10, 'o'), 
               (100,'+'),
               (200, '.')]:
    j = np.arange(1, N+1)
    kj = np.pi * j / L
    a = L/(N+1)
    w = c * np.sqrt(2 / a**2*(1 - np.cos(kj*a)))
    ax.plot(kj*L, w, fmt, label=fr'$N={N}$')
    

ax.grid(True)
ax.set(xlabel=r'$kL$', ylabel=r'$\omega(k)$', )
ax.legend();
```
