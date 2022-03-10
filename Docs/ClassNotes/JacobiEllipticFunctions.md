---
jupytext:
  formats: ipynb,md:myst,py:light
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
:tags: [hide-cell]

from myst_nb import glue

import mmf_setup

mmf_setup.nbinit()
import logging

logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
import manim.utils.ipython_magic
!manim --version
```

Jacobi Elliptic Functions
=========================

The exact solution to several classic problems can be expressed in terms of the [Jacobi
elliptic functions].  We present some of the key properties here and demonstrate these
solutions.

## Definition and Properties
```{margin}
One often sees the notation $m = k^2$ for *the parameter* in terms of the *elliptic
modulus* or *eccentricity* $k$.  Check your numerical libraries carefully to see the
meaning of the parameter.  Scipy's {py:data}`scipy.special.ellipj` function expects
$m=k^2$ but uses the notation $\cn(u|m) = \cn(u,k)$.
```
The [Jacobi elliptic functions] are generalizations of $\sin \phi$ and $\cos \phi$
obtained by replacing the angle $\phi$ with the parameter $u$ which is the incomplete
[elliptic integral] of the first kind:

\begin{gather*}
  u = F(\phi, k) 
    = \int_0^{\phi}\frac{\d\theta}{\sqrt{1-k^2\sin^2\theta}},\\
  \begin{aligned}
    \cn u \equiv \cn(u, k) &= \cos\phi, \\
    \sn u \equiv \sn(u, k) &= \sin\phi, \qquad  (\text{hence } \cn^2 u + \sn^2 u = 1)\\
    \dn u \equiv \dn(u, k) &= \sqrt{1 - \smash{k^2\sin^2 \phi}} 
                            = \sqrt{1 - \smash{k^2\sn^2 u}},\\
    \phi(u, k) &= \sin^{-1}(\sn u)\\
    %\dn^2 u &= 1 - k^2\sn^2 u = (1 - k^2) + k^2\cn^2 u,\\
    %\tan\phi & =\frac{\sn u}{\cn u},\\
  \end{aligned}
\end{gather*}

If $k=0$, then $u = \phi$, and these reduce to the standard trigonometric
functions $\cn(u,0) = \cos(u)$, $\sn(u,0) = \sin(u)$, and $\dn(u,0) = 1$.  Note that
$\sn$, $\cn$, and $\dn$ are periodic, but that the period is in general not $2\pi$, but
instead given by the [complete elliptic integral of the first kind] $K(u) =
F(\tfrac{\pi}{2}, k)$, which is a quarter period:

```{glue:figure} period_fig 
:figclass: margin

Period of the [Jacobi elliptic functions] in units of $2\pi$.  Note that for small $k$
the familiar $2\pi$ periodicity is restored, but that the period diverges as the
eccentricity $k \rightarrow 1$.
```
\begin{gather*}
  u \equiv u + F(2\pi, k) = u + 4K(k).
\end{gather*}

*If you ever need to compute the inverse $K^{-1}$, see {cite:p}`Boyd:2015` which
includes a never-failing Newton's method.*

```{code-cell} ipython3
:tags: [hide-cell]

from scipy.special import ellipj, ellipk

sn = lambda u, k: ellipj(u, k**2)[0]
cn = lambda u, k: ellipj(u, k**2)[1]
dn = lambda u, k: ellipj(u, k**2)[2]
varphi = lambda u, k: ellipj(u, k**2)[3]
K = lambda k: ellipk(k**2)

# Fudge here to get more points near k=1
k = 1-np.linspace(0, 1)**4

fig, ax = plt.subplots(figsize=(2, 1.5))
ax.plot(k, 4*K(k)/2/np.pi)
ax.axvline([1], c='y', ls=':')
ax.set(xlabel="$k$", ylabel="$4K(k)/2\pi$", 
       ylim=(0, 3), yticks=[0, 1, 2, 3])
fig.tight_layout()
glue("period_fig", fig, display=False)
```

### $k$ Inversion Formulae

In this formulation, $k^2 \in [0, 1]$, but one can consider $k^2 > 1$ as long as
$\abs{\phi} < \sin^{-1}(k^{-1})$.  To evaluate these for $k > 1$, we can use the
following inversion relationships which 

\begin{align*}
  \sn(u, k) &= k^{-1}\sn(ku, k^{-1}), \\
  \cn(u, k) &= \dn(ku, k^{-1}), \\
  \dn(u, k) &= \cn(ku, k^{-1}), \\
  \phi(u, k) &= \sin^{-1}\bigl(k^{-1}\sin\phi(ku, k^{-1})\bigr),\\
  F(\phi, k) &= kF\bigl(\sin^{-1}(k\sin\phi), k^{-1}\bigr).
\end{align*}

```{admonition} Proof of $k$ Inversion Formulae
:class: dropdown

Let $t = \sin \theta$, $\d{t} = \cos \theta\; \d\theta$ to obtain an alternate
representation:

\begin{gather*}
  u = F(\phi, k)
    = \int_0^{\sin \phi} \frac{\d{t}}{\cos\theta\sqrt{1 - k^2\sin^2\theta}}
    = \int_0^{\sin \phi} \frac{\d{t}}{\sqrt{1-t^2}\sqrt{1-k^2t^2}}.
\end{gather*}

Now multiply through by $k$ and change integration variables to $\tilde{t} = kt$, with
$\tilde{u} = ku$ and $\tilde{k} = k^{-1}$:

\begin{align*}
  \tilde{u} &= ku = kF(\phi, k)\\
    &= \int_0^{k\sin \phi} 
           \frac{\d{(kt)}}
                {\sqrt{1-\frac{(kt)^2}{k^2}}\sqrt{1-(kt)^2}},\\
    &= \int_0^{\overbrace{\sin \tilde{\phi}}^{k\sin \phi}}
          \frac{\d{\tilde{t}}}
               {\sqrt{1-\tilde{k}^2\tilde{t}^2}\sqrt{1-\tilde{t}^2}}.
\end{align*}

Now, by analogy,

\begin{align*}
  u = F(\phi, k) 
    &\implies \sn(u, k) = \sin \phi,\\
  ku = \tilde{u} = F(\tilde{\phi}, \tilde{k}) 
    &\implies 
    \sn(\tilde{u}, \tilde{k}) = \sin\tilde{\phi} = \sn(\tilde{u}, \tilde{k}),\\
    &\implies 
    \underbrace{\sn(ku, k^{-1})}_{\sn(\tilde{u}, \tilde{k})} 
     = \underbrace{k \sn(u, k)}_{\sin\tilde{\phi}},
\end{align*}

The other relationships follow simply from the original relationships:

\begin{align*}
  \dn(u,k) &= \sqrt{1 - k^2\sn^2(u, k)}
            = \sqrt{1 - \sn^2(ku, k^{-1})}\\
           &= \cn(ku, k^{-1}),\\
  \dn(ku,k^{-1}) &= \cn(u, k),
\end{align*}

where we obtain the last relationship by letting $k \rightarrow k^{-1}$ and $u \rightarrow ku$.
```

### Derivatives

The derivatives satisfy the following:

\begin{gather*}
  \diff{\phi}{u} = \dn u, \\
  \begin{aligned}
  \sn' u &= \cn u\dn u, \\
  \cn' u &= -\sn u\dn u, \\
  \dn' u &= -k^2\sn u\cn u,\\
    (\sn u\dn u)' &= \cn u (\dn^2 u - k^2\sn^2 u)\\
                  &= \cn u (1 - 2k^2\sn^2 u)\\
                  &= (1-2k^2)\cn u + 2k^2\cn^3 u.
  \end{aligned}
\end{gather*}


### Unit Ellipse

The [Jacobi elliptic functions] are related to a unit [ellipse] in the same way that the
trigonometric functions are related to the unit circle, with $k$ being the
[eccentricity] of the [ellipse]:

\begin{gather*}
    x^2 + (1-k^2)y^2 = 1, \qquad
    (x, y) = (r\cos\phi, r\sin\phi).
\end{gather*}

Unlike with a circle, the radius cannot have a constant magnitude:

\begin{gather*}
  x^2+(1-k^2)y^2 = r^2\overbrace{(1-k^2\sin^2\phi)}^{\dn^2(u,k)} = 1, \\
  r(\phi) = \frac{1}{1-k^2\sin^2\phi} = \frac{1}{\dn(u,k)},\\
  x = \frac{\cn(u, k)}{\dn(u, k)} = \frac{\cos\phi}{\sqrt{1-k^2\sin^2\phi}},\\
  y = \frac{\sn(u, k)}{\dn(u, k)} = \frac{\sin\phi}{\sqrt{1-k^2\sin^2\phi}}.
\end{gather*}

Thus, we have the familiar relationship:

\begin{gather*}
  \cn(u, k) = \frac{x}{r}, \qquad
  \sn(u, k) = \frac{y}{r}, \qquad
  \dn(u, k) = \frac{1}{r}, \\
  r = \frac{1}{\sqrt{1-k^2\sin^2\phi}}.
\end{gather*}





The arc-length of the ellipse is

\begin{gather*}
  \d{x} = \\
  \d{y} = \\
  \d c/d  = s/d(k^2 c^2/d^2 - 1) \d{\phi}
  \d s/d  = c/d(k^2 s^2/d^2 - 1)
\end{gather*}


\begin{gather*}
  \int \sqrt{\d{x}^2 + \d{y}^2} = 
  \int_0^{\phi} \sqrt{\d{x}^2 + \d{y}^2}\d\phi = 
\end{gather*}

```{code-cell} ipython3
from scipy.special import ellipj, ellipkinc

def get_scd(phi, k):
    m = k**2
    u = ellipkinc(phi, m)
    s, c, d, phi_ = ellipj(u, m)
    return s, c, d

phi = np.linspace(0, 2*np.pi)
k = 0.8
s, c, d = get_scd(phi, k)

fig, axs = plt.subplots(1, 2, figsize=(10,3))
ax = axs[1]
ax.plot(phi, s, phi, c, phi, d)

ax = fig.add_subplot(axs[0].get_subplotspec(), projection="polar")
axs[0].remove()
ax.plot(phi, 1/d)
```

```{code-cell} ipython3
:tags: [hide-input]

%%manim -v WARNING --progress_bar None -qm JacobiEllipse
from scipy.special import ellipj, ellipkinc
from manim import *

config.media_width = "100%"
config.media_embed = True

my_tex_template = TexTemplate()
with open("../_static/math_defs.tex") as f:
    my_tex_template.add_to_preamble(f.read())

def get_scd(phi, k):
    m = k**2
    u = ellipkinc(phi, m)
    s, c, d, phi_ = ellipj(u, m)
    return s, c, d

def get_xyr(phi, k):
    s, c, d = get_scd(phi, k)
    r = 1/d
    x, y = r*c, r*s
    return x, y, r

class JacobiEllipse(Scene):
    def construct(self):
        config["tex_template"] = my_tex_template
        config["media_width"] = "100%"
        class colors:
            ellipse = YELLOW
            x = BLUE
            y = GREEN
            r = RED
            
        phi = ValueTracker(0.01)
        k = 0.8
        axes = Axes(x_range=[0, 2*np.pi, 1], 
                    y_range=[-2, 2, 1],
                    x_length=6, 
                    y_length=4,
                    axis_config=dict(include_tip=False),
                    x_axis_config=dict(numbers_to_exclude=[0]),
                   ).add_coordinates()
        
        
        plane = PolarPlane(radius_max=2).add_coordinates()

        x = lambda phi: get_xyr(phi, k)[0]
        y = lambda phi: get_xyr(phi, k)[1]
        r = lambda phi: get_xyr(phi, k)[2]
        
        g_ellipse = plane.plot_polar_graph(r, [0, 2*np.pi], color=colors.ellipse)
        
        points_colors = [(x, colors.x), (y, colors.y), (r, colors.r)]
        x_graph, y_graph, r_graph = [axes.plot(_x, color=_c) for _x, _c in points_colors]
        graphs = VGroup(x_graph, y_graph, r_graph)
        
        dot = always_redraw(lambda:
            Dot(plane.polar_to_point(r(phi.get_value()), phi.get_value()), 
                fill_color=colors.ellipse, 
                fill_opacity=0.8))

        @always_redraw
        def lines():
            c2p = plane.coords_to_point
            phi_ = phi.get_value()
            x_, y_ = x(phi_), y(phi_)
            return VGroup(
                Line(c2p(0, 0), c2p(x_, 0), color=colors.x),
                Line(c2p(0, y_), c2p(x_, y_), color=colors.x),
                Line(c2p(0, 0), c2p(0, y_), color=colors.y),
                Line(c2p(x_, 0), c2p(x_, y_), color=colors.y),
                Line(axes.c2p(phi_, y_), c2p(x_, y_), color=colors.y),
                Line(c2p(0, 0), c2p(x_, y_), color=colors.r),
            ).set_opacity(0.8)

        dots = always_redraw(lambda:
            VGroup(*(Dot(axes.c2p(phi.get_value(), _x(phi.get_value())), fill_color=_c, fill_opacity=1)
                     for _x, _c in points_colors)))

        a_group = VGroup(axes, dots, graphs)
        p_group = VGroup(plane, g_ellipse, dot, lines)
        a_group.shift(RIGHT*2)
        p_group.shift(LEFT*4)
    
        labels = VGroup(
            axes.get_graph_label(
                x_graph, label=r"x=\cn(u, k)", color=colors.x,
                x_val=2.5, direction=DOWN).shift(0.2*LEFT),
            axes.get_graph_label(
                y_graph, label=r"y=\sn(u, k)", color=colors.y,
                x_val=4.5, direction=DR),
            axes.get_graph_label(
                r_graph, label=r"r=1/\dn(u, k)", color=colors.r,
                x_val=2, direction=UR),
        )
        #labels.next_to(a_group, RIGHT)
        self.add(p_group, a_group, labels)
        self.play(phi.animate.set_value(2*np.pi), run_time=5, rate_func=linear)
```


## Anharmonic Oscillator

```{sidebar} Proof

Let $u = \Omega (t - t_0)$ so that $\dot{u} = \Omega$ and $q(t) = q_0\cn u$:

\begin{align*}
  q(t) &= q_0 \cn u\\
  \dot{q}(t) &= -q_0 \Omega \sn u \dn u\\
  \ddot{q}(t) &= -q_0 \Omega^2 (\cn u \dn^2 u - k^2\sn^2 u \cn u)\\
              &= - \Omega^2 \left((1 - 2k^2) q(t) + 2k^2\frac{q^3(t)}{q_0^2}\right).
\end{align*}

which solves our equation if

\begin{gather*}
  \Omega^2(1-2k^2) = \omega^2, \qquad
  \frac{2k^2\Omega^2}{q_0^2} = \lambda.
\end{gather*}
```

Consider the anharmonic oscillator

\begin{align*}
  V(q) &= \frac{m\omega^2}{2}q^2 + \frac{m\lambda}{4}q^4,\\
  \ddot{q} &= -\omega^2 q - \lambda q^3.
\end{align*}

The general solution is:

\begin{align*}
  q(t) &= q_0\cn\bigl(\Omega (t - t_0), k\bigr),\\
  \Omega^2 &= \omega^2 + \lambda q_0^2, \\
  k^2 &= \frac{\lambda}{2}\left(\frac{q_0}{\Omega}\right)^2 
      = \frac{1}{2}\frac{1}{1 + \omega^2/(\lambda q_0^2)}.
\end{align*}

### Some Interesting Properties (Incomplete)

In {cite:p}`Cartier:2006`, the discuss the WKB solution to the anharmonic oscillator for
$\lambda > 0$.  To formulate the WKB evolution of a wavefunction $\psi(x, t_0)$ to
$\psi(x, t)$, one needs the classical action $S(x, x_0)$ for particles moving from
$x=x_0$ at time $t=t_0$ to position $x$ at time $t$.  The WKB propagator is then (up to
a phase) (see Eq. (4.45) of {cite:p}`Cartier:2006`, the so-called position-to-position
transition)

\begin{gather*}
  U_{WKB}(x,t;x_0,t_0) = \sqrt{\frac{1}{\hbar}
                                \frac{\partial^2 S(x, x_0)}
                                     {\partial x \partial x_0}}
                          e^{\frac{\I}{\hbar}S(x,x_0)}.
\end{gather*}

```{margin}
*Hint: Thinking of this as an initial value problem, we must find different initial
velocities $v_0$ such that the particle returns in time $T$.  As we increse $v_0$, the
particle travels further and takes longer to return.  This argument should convince you
that there are at most a finite number of solutions.*
```
For many problems, there is a unique trajectory from $x_0(t_0)$ to $x(t)$, but the
anharmonic oscillator admits a countably infinite number which must be summed over.
Consider, for example, a return $x(t) = x_0(t_0)$ where the particle starts at $x_0$ and
returns at time $T = t-t_0$ later.  Convince yourself that, for a Harmonic oscillator,
assuming $omega T \neq n\pi$, there are only a finite number of solutions.

With the anharmonic oscillator, however, increasing the velocity, can *decrease* the
return time.

```{code-cell} ipython3
:tags: [hide-input]

from functools import partial

from scipy.integrate import solve_ivp
m = w = 1
lam = 1
T = 2*np.pi / w   # Upper bound on period

def compute_dy_dt(t, y):
    q, dq = y
    ddq = -q*(w**2 + lam*q**2)
    return (dq, ddq)

def return_event(t, y, q0):
    """Event signaling the return of the particle to q0 from the right."""
    q, dq = y
    return q - q0

def solve(v0, q0=1):
    y0 = (q0, v0)
    event = partial(return_event, q0=q0)
    event.direction = -1
    event.terminal = True
    
    res = solve_ivp(
        compute_dy_dt,
        t_span=(0, T),
        y0=y0,
        max_step=0.01,
        events=[event])
    return res

fig, ax = plt.subplots()
v0s = np.linspace(1, 10.0, 20)
for v0 in v0s:
    q0 = 1.0
    res = solve(v0=v0, q0=q0)
    ts = res.t
    ys = qs, dqs = res.y
    ax.plot(ts, qs)
ax.set(xlabel="t", ylabel="q");
```

Here we start with the particle at $q_0=1$ and throw it to the right with increasing
initial velocities.  We see that, at first, as we increase the velocity, the return time
increases, but once the initial velocity is high enough, the particle starts to see the
steeper $q^4$ potential, and returns more quickly.  Given an integer $n$, one can find a
large-enough initial velocity that the particle will oscillate $n$ times in any given
period $T$, resulting in countably infinitely many solutions to the boundary value
problem.

## Pendulum (Incomplete)

```{margin}
For a complete solution to the bead on a rotating circular wire, which gives this as a
limiting case, see {cite:p}`Baker:2012`.
```
Let the coordinate $q$ be the angle from vertical so that $q = 0$ is the ground state:

\begin{gather*}
  V(q) = mgl(1 - \cos q) = ml^2\omega^2(1 - \cos q),\\
  \ddot{q} = - \omega^2 \sin q, \\
  \omega^2 = \sqrt{\frac{g}{l}}.
\end{gather*}

Conservation of energy $E$ gives immediately

\begin{gather*}
  \frac{E}{ml^2} = \frac{\dot{q}^2}{2} + \omega^2(1-\cos q), \qquad
  \dot{q} = \sqrt{\frac{2E}{ml^2} - 2\omega^2(1-\cos q)},\\
  t-t_0 = \int_{q_0}^{q} \frac{\d q}{\sqrt{\frac{2E}{ml^2} - 2\omega^2(1-\cos q)}}.
\end{gather*}

With the exception of the stationary solution $q = \pi$, all solutions will pass through
$q=0$, so we can choose our time $t_0$ to be the time when $q_0 = q(t_0) = 0$.  Now it
remains to change variables so that the integral has the form in the incomplete [elliptic
integral] of the first kind.  This can be done with the half-angle identity $1-\cos q = 2\sin^2 \tfrac{q}{2}$:

\begin{gather*}
  t-t_0 = \int_{0}^{q/2} \frac{2\d \tfrac{q}{2}}
                              {\sqrt{\frac{2E}{ml^2} - 4\omega^2\sin^2\tfrac{q}{2}}}
    = \underbrace{\sqrt{\frac{2ml^2}{E}}}_{\tau}
      \int_{0}^{q/2} 
      \frac{\d\tfrac{q}{2}}
           {\vphantom{\underbrace{\frac{2ml^2\omega^2}{E}}_{k^2}}
           \sqrt{1 - \smash{\underbrace{\frac{2ml^2\omega^2}{E}}_{k^2=\tau^2\omega^2}}
                     \sin^2\tfrac{q}{2}}},\\
    t - t_0 = \tau F(\tfrac{q}{2}, k), \qquad
    \tau = \sqrt{\frac{2ml^2}{E}}, \qquad
    k = \tau\omega = \sqrt{\frac{2mgl}{E}}
\end{gather*}

Hence, we have $\phi = q/2$ and $u = (t-t_0)/\tau$, so the solution can be expressed
as $\cn u = \cos \phi$ or $\sn u = \sin \phi$:

\begin{gather*}
  q(t) = 2 \cos^{-1}\left(\cn\Bigl(\frac{t - t_0}{\tau}, k=\tau\omega \Bigr)\right)
       = 2 \sin^{-1}\left(\sn\Bigl(\frac{t - t_0}{\tau}, k=\tau\omega \Bigr)\right).
\end{gather*}

Note that with the standard implementation, $k\in [0, 1)$ means that $E > 2mgl$,
corresponding to rotations.  For [libration]s, $E \in [0, 2mgl)$ corresponding to $k >
1$, meaning that there is a finite range $\abs{q} \leq \cos^{-1}(1 - E/mgl)$.  To
compute these, we use the $k^{-1}$ relationships above to obtain:

\begin{gather*}
  q(t) = 2 \cos^{-1}\left(
    \dn\Bigl(\omega(t - t_0), \frac{1}{\tau\omega}\Bigr)
  \right)
       = 2 \sin^{-1}\left(
    \frac{1}{\tau\omega}\sn\Bigl(\omega(t - t_0), \frac{1}{\tau\omega}\Bigr)
  \right).
\end{gather*}

```{code-cell} ipython3
#:tags: [hide-cell]

from scipy.special import ellipj, ellipk

def sn(u, k):
    return np.where(
        k < 1, 
        ellipj(u, k**2)[0],
        ellipj(k*u, 1/k**2)[0]/k)

def cn(u, k):
    return np.where(
        k < 1, 
        ellipj(u, k**2)[1],
        ellipj(k*u, 1/k**2)[2])

def dn(u, k):
    return np.where(
        k < 1, 
        ellipj(u, k**2)[2],
        ellipj(k*u, 1/k**2)[1])

ks = np.linspace(0.8, 1.2, 5)
tau = 1.0
t0 = 0

fig, ax = plt.subplots()

ts = tau * np.linspace(-3*np.pi, 3*np.pi, 500)

for k in ks:
    w = k/tau
    u = (ts-t0)/tau
    qs = 2*np.arctan2(sn(u, k=k), cn(u, k=k))
    ax.plot(ts * w / 2 / np.pi, qs, label=f"$k={k:.2f}$")
    ax.set(xlabel=r"$t\omega/2\pi$", ylabel="$q$", 
       #ylim=(0, 3), yticks=[0, 1, 2, 3]
       );
ax.legend()
#glue("period_fig", fig, display=False)
```

## References

* {cite:p}`Baker:2012`: Full solution to the rotating bead on a circular wire problem.
* [Intro to Jacobi Elliptic Functions](https://www.youtube.com/watch?v=CmAaC2Z4FAI): A
  short YouTube video showing how the [Jacobi elliptic functions] are to ellipses as the
  trigonometric functions are to circles.

[elliptic integral]: <https://en.wikipedia.org/wiki/Elliptic_integral>
[complete elliptic integral of the first kind]: <https://en.wikipedia.org/wiki/Elliptic_integral#Complete_elliptic_integral_of_the_first_kind>
[Jacobi elliptic functions]: <https://en.wikipedia.org/wiki/Jacobi_elliptic_functions>
[libration]: <https://en.wikipedia.org/wiki/Libration_(molecule)>
[eccentricity]: <https://en.wikipedia.org/wiki/Eccentricity_(mathematics)>
[ellipse]: <https://en.wikipedia.org/wiki/Ellipse>
