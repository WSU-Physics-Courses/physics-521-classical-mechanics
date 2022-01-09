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

from myst_nb import glue

import mmf_setup

mmf_setup.nbinit()
import logging

logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
%pylab inline --no-import-all
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
The [Jacobi elliptic functions] are generalizations of $\sin \varphi$ and $\cos \varphi$
obtained by replacing the angle $\varphi$ with the parameter $u$ which is the incomplete
[elliptic integral] of the first kind:

\begin{gather*}
  u = F(\varphi, k) 
    = \int_0^{\varphi}\frac{\d\theta}{\sqrt{1-k^2\sin^2\theta}},\\
  \begin{aligned}
    \cn u \equiv \cn(u, k) &= \cos\varphi, \\
    \sn u \equiv \sn(u, k) &= \sin\varphi, \qquad  (\text{hence } \cn^2 u + \sn^2 u = 1)\\
    \dn u \equiv \dn(u, k) &= \sqrt{1 - \smash{k^2\sin^2 \varphi}} 
                            = \sqrt{1 - \smash{k^2\sn^2 u}},\\
    \varphi(u, k) &= \sin^{-1}(\sn u)\\
    %\dn^2 u &= 1 - k^2\sn^2 u = (1 - k^2) + k^2\cn^2 u,\\
    %\tan\varphi & =\frac{\sn u}{\cn u},\\
  \end{aligned}
\end{gather*}

If $k=0$, then $u = \varphi$, and these reduce to the standard trigonometric
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
$\abs{\varphi} < \sin^{-1}(k^{-1})$.  To evaluate these for $k > 1$, we can use the
following inversion relationships which 

\begin{align*}
  \sn(u, k) &= k^{-1}\sn(ku, k^{-1}), \\
  \cn(u, k) &= \dn(ku, k^{-1}), \\
  \dn(u, k) &= \cn(ku, k^{-1}), \\
  \varphi(u, k) &= \sin^{-1}\bigl(k^{-1}\sin\varphi(ku, k^{-1})\bigr),\\
  F(\varphi, k) &= kF\bigl(\sin^{-1}(k\sin\varphi), k^{-1}\bigr).
\end{align*}

```{admonition} Proof of $k$ Inversion Formulae
:class: dropdown

Let $t = \sin \theta$, $\d{t} = \cos \theta\; \d\theta$ to obtain an alternate
representation:

\begin{gather*}
  u = F(\varphi, k)
    = \int_0^{\sin \varphi} \frac{\d{t}}{\cos\theta\sqrt{1 - k^2\sin^2\theta}}
    = \int_0^{\sin \varphi} \frac{\d{t}}{\sqrt{1-t^2}\sqrt{1-k^2t^2}}.
\end{gather*}

Now multiply through by $k$ and change integration variables to $\tilde{t} = kt$, with
$\tilde{u} = ku$ and $\tilde{k} = k^{-1}$:

\begin{align*}
  \tilde{u} &= ku = kF(\varphi, k)\\
    &= \int_0^{k\sin \varphi} 
           \frac{\d{(kt)}}
                {\sqrt{1-\frac{(kt)^2}{k^2}}\sqrt{1-(kt)^2}},\\
    &= \int_0^{\overbrace{\sin \tilde{\varphi}}^{k\sin \varphi}}
          \frac{\d{\tilde{t}}}
               {\sqrt{1-\tilde{k}^2\tilde{t}^2}\sqrt{1-\tilde{t}^2}}.
\end{align*}

Now, by analogy,

\begin{align*}
  u = F(\varphi, k) 
    &\implies \sn(u, k) = \sin \varphi,\\
  ku = \tilde{u} = F(\tilde{\varphi}, \tilde{k}) 
    &\implies 
    \sn(\tilde{u}, \tilde{k}) = \sin\tilde{\varphi} = \sn(\tilde{u}, \tilde{k}),\\
    &\implies 
    \underbrace{\sn(ku, k^{-1})}_{\sn(\tilde{u}, \tilde{k})} 
     = \underbrace{k \sn(u, k)}_{\sin\tilde{\varphi}},
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
  \diff{\varphi}{u} = \dn u, \\
  \begin{aligned}
  \sn' u &= \cn u\dn u, \\
  \cn' u &= -\sn u\dn u, \\
  \dn' u &= -k^2\sn u\cn u,\\
    (\sn u\dn u)' &= \cn u (\dn^2 u - k^2\sn^2 u)\\
                  &= \cn u (1 - 2k^2\sn^2 u)\\
                  &= (1-2k^2)\cn u + 2k^2\cn^3 u.
  \end{aligned}
\end{gather*}


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

Hence, we have $\varphi = q/2$ and $u = (t-t_0)/\tau$, so the solution can be expressed
as $\cn u = \cos \varphi$ or $\sn u = \sin \varphi$:

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
