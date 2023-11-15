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
```

(sec:PendulumExample)=
# Worked Example: The Pendulum

Here we will develop a rather complete worked example: the motion of a pendulum.  You
should try to work along with this, both analytically, and checking your results
numerically.  I *highly* recommend that you try to anticipate what I am going to do by
scanning through the figures, equations, etc., and try to solve the problem yourself
before reading the details.  To this end, I have "hidden" most of the work in expandable
sections.

## Overview

We start with an overview demonstrating some interesting properties of this system.  I
find a good way of trying to understand a system like this is to try to understand and
then reproduce these figures, diving into the details when I get stuck.

:::{margin}
Physically, we imagine a pendulum with a massless rod of length $l$ with a point mass
$m$ at the end, in a gravitational field of strength $g$, some damping $\lambda$ and a
driving torque $\Gamma(t)$.  After deriving the equations of motion, we replace these with the
more convenient parameters $\omega_0 = \sqrt{l/g}$ and $f(t) = \Gamma(t)/m$.
:::
We will consider the following general model for a rigid pendulum.  We will describe the
motion of the pendulum with the angle $\theta(t)$ from equilibrium:
\begin{gather*}
  \ddot{\theta} + 2\lambda \dot{\theta} + \omega_0^2 \sin\theta = f(t).
\end{gather*}
We will consider various generalizations, such as expanding $\sin\theta$ (harmonic and
anharmonic terms), and consider a time-dependent frequency $\omega_0(t)$ (parametric
resonance).

If we consider constant $\omega_0$ without any driving term $f(t)=0$, then we have an
**autonomous** second-order equation, and we can plot the following phase diagram.  (For
details, see {ref}`sec:HamiltonianMechanics`.)

:::{figure} /_images/phase_space_pendulum_damping.svg
:alt: Phase space flow for a pendulum with damping.

Phase-space evolution of a damped pendulum.  Several closed regions are shown, evolved
for time $1.3T_0$ where $T_0 = 2\pi/\omega_0$ is the natural period of the
oscillator.  Due to the damping, this evolution is not Hamiltonian and Liouville's
theorem does not hold -- the areas of the regions get smaller.  Properly scaled,
however, Hamiltonian dynamics can be restored.  See {ref}`sec:HamiltonianMechanics` for
details.
:::

Another interesting property is the **linear response** of the pendulum to small
perturbations:

:::{figure} /_images/LinearResponse.svg
:figclass: margin-caption

Linear response of a damped harmonic oscillator for various amounts of damping.  The
dotted line shows $\abs{\chi_\max}^2$ vs $\omega_{\max}$.
:::

I recommend you try to make this figure two ways:

1. Directly calculate the response function $\chi$ analytically.  *(This is how I
   generated this plot.)*
2. Setup an "experiment" where you simulate a driven pendulum and then "measure" the
   amplification.
   
The response function has a very interesting property when one includes non-linear terms.



## Description

:::{margin} A less-ideal pendulum.

Consider instead a real pendulum where the rod is not massless and the bob is extended.
How does the analysis change?  Is the problem fundamentally different, or can it be
reduced to the same form?
:::
We start by considering how to describe potential solutions to the motion of an ideal
pendulum of length $l$ from the fulcrum with a point mass $m$ bob.  A natural
description is in terms of the angle $\theta(t)$ from the equilibrium position.  Since
$\theta(t) = \tan^{-1}\tfrac{x}{-z}$ is a non-linear function of the Cartesian
coordinates of the mass, the Lagrangian framework provides a natural approach for
expressing the equations of motion with $L=T-V$ where $T=mv^2/2$ is the kinetic energy
and $V=mgz$ is the potential (we take $g>0$ here with gravity pointing down):
:::{margin}
Economy of notation helps me both work faster and to understand the structure of
equations.  Here I use dots to denote time derivatives and Einstein's notation for
partial derivatives as a subscript preceded by a comma for partial derivatives:
\begin{gather*}
  \dot{\theta} = \diff{\theta(t)}{t},\\
  L_{,\theta} = \pdiff{L}{\theta},\\
  L_{,\dot{\theta}} = \pdiff{L}{\dot{\theta}}.
\end{gather*}
I also often suppress arguments $\dot{\theta} = \dot{\theta}(t)$ where I think it is
obvious.  You might want to keep these for a while in your work until it becomes
intuitive and clear for you which variables depend on time etc.
:::
\begin{gather*}
  L(\theta, \dot{\theta}) = \frac{ml^2\dot{\theta}^2}{2} + mgl(\cos\theta - 1)\\
  p_\theta = L_{,\dot{\theta}} = ml^2\dot{\theta},\qquad
  \dot{p}_\theta = L_{,\theta} = -mgl\sin\theta\\
  \ddot{\theta} = -\frac{g}{l}\sin\theta.
\end{gather*}
From these equations we recover the well-known result that the motion depends only on
the ratio $g/l$ which has dimensions of $1/T^2$ and hence must be related to the square
of the frequency $\nu = \omega / 2\pi$.  *(Note: we subtract $1$ from the potential so
that $\cos\theta - 1 \sim \theta^2/2 + O(\theta^4)$.  This makes the energy of the
ground state $\theta = 0$ zero.)*

To simplify further discussions, we shall introduce the **resonant frequency**
$\omega_0$ so that our equation of motion becomes:
\begin{gather*}
  \ddot{\theta} = - \omega_0^2\sin \theta, \qquad \omega_0 = \sqrt{\frac{g}{l}}.
\end{gather*}

:::{doit} Do It! Dimensional Analysis.

We found by formulating the equations of motion that the physics of an ideal pendulum
depends only on the ratio $g/l = \sqrt{\omega_0}$.  Perform a complete dimensional
analysis and show that you could have deduced this without the equations of motion.
:::
:::{solution}

We have the following physical constants and their associated dimensions:
\begin{gather*}
  [m] = M, \qquad
  [l] = D, \qquad
  [g] = \frac{D}{T^2}.
\end{gather*}
I refer to these as **intrinsic**.  Note that one cannot form any dimensionless
combinations.  Instead, we can use these to set our units, measuring masses in units of
$m$, distances in units of $l$ and times in units of $\sqrt{l/g}$.  This is often stated
as **choosing units such that $m=l=g=1$**.  Once this is done, to recover the full
solution, simply multiply the desired result by the appropriate factor of $1$ to recover
the required units:
\begin{gather*}
  1 = \underbrace{m}_{\text{mass}} 
    = \underbrace{l}_{\text{distance}}
    = \underbrace{\sqrt{\frac{l}{g}}}_{\text{time}}
    = \underbrace{\sqrt{lg}}_{\text{speed}}
    = \underbrace{mlg}_{\text{energy}}
    = \cdots.
\end{gather*}
I strongly recommend that you write down such a conversion table whenever you choose a
set of units.

In addition, we have the following constants and associated dimensions corresponding to
initial conditions $\theta_0 = \theta(0)$ and $\dot{\theta}_0 = \dot{\theta}(0)$.
*(Note: we consider only the initial-value problem (IVP) here.  One could also specify a
boundary-value problem (BVP) with e.g. $\theta_0 = \theta(0)$ and $\theta_T = \theta(T)$
at some time $T$, but this is not guaranteed to have a unique solution.)*

\begin{gather*}
  [\theta_0)] = 1, \qquad
  [\dot{\theta}_0] = \frac{1}{T}.
\end{gather*}

Now we can form two dimensionless quantities:
\begin{gather*}
  \alpha = \theta_0, \qquad
  \beta = \sqrt{\frac{l\dot{\theta}^2}{g}} = \frac{\dot{\theta}}{\omega_0}.
\end{gather*}
The qualitative nature of the solutions for the ideal pendulum depend on the initial
conditions and will be completely characterized by these dimensionless combinations.
From this we deduce that only the ratio $g/l$ is of physical significance, QED.

Comments/Caveats:
* Choosing appropriate units like $m=l=g=1$ can make the math much cleaner, however,
  this is at the expense of being able to use dimensions as a check of your work.  You
  must know at the end what the dimension should be of the quantity you are calculating
  so you can restore the constants.  Personally I tend to play around with such units to
  understand the forms of the equations (especially for identifying common forms of
  e.g. integrals or differential equations), but then redo the final calculation with
  the parameters explicitly factoring them as appropriate into dimensionless constants
  so I can perform a final check of the dimensions and make sure I made no mistakes.
* There is an art to choosing the best units and dimensionless parameters.  I often need
  to revise my original choice two or three times as I learn more about the nature of
  the problem.  The correct choice should emphasize the qualitative physics, clearly
  separating different physical regimes.  An incorrect choice will leave your equations
  cluttered with factors of $2$, $\pi$, additive constants, etc.
  
  Given your knowledge of the motion of a pendulum, can you think of a better set of
  dimensionless parameters than $\alpha$ and $\beta$ defined above?  We will revisit
  this below.
:::

:::{doit}

Express the problem directly in terms of the Cartesian coordinates $x(t) = l\sin\theta$ and $z(t)=-l\cos\theta$ so
that you can write the equations of motion as $\vect{F} = m\vect{a}$.  Show that you
obtain the same equations of motion once you impose the constraint that the length of
the pendulum ls fixed to $l$.  Describe the physics responsible for this constraint.
:::

## Generalizations

In the following we shall consider the following generalizations of the pendulum
problem:

:::{margin}
  I say "almost" here because this breaks down for e.g. $V(x) \propto x^4$.  Such
  behaviour is uncommon in nature, and generally requires precisely controlled
  experiments.  Despite being uncommon, studying such systems may provide interesting
  qualitative effects that can be harnessed for e.g. new technologies.
:::
* **Harmonic Oscillator (HO)**: By restricting our attention to small angles $\abs{\theta}\ll
  1$ we recover the well-known harmonic oscillator.
  \begin{gather*}
    \ddot{\theta} = -\omega_0^2\theta.
  \end{gather*}
  This problem is key across physics because: 1) it is completely solvable, and 2) it
  describes the motion of almost all physical systems close to stationary or equilibrium
  solutions.  I.e. almost any potential looks quadratic near a local minimum (or
  maximum) $x_0$:
  \begin{gather*}
    V(x) \approx V(x_0) + \frac{(x-x_0)^2}{2}V''(x_0) + O(x-x_0)^3.
  \end{gather*}
  The full generalization of this is the theory of **normal modes**.
* **Perturbation Theory**: Since the HO is exactly solvable in both classical and
  quantum mechanics, it forms the starting point for perturbative analyses.  We can try
  to use perturbative techniques to look at the behaviour of an ideal pendulum:
  \begin{gather*}
    V(\theta) = mgl(1-\cos\theta) = \frac{mgl}{2}\Biggl(\theta^2 -
    \underbrace{\frac{1}{12}}_{\epsilon}\theta^4 + O(\theta^6)\Biggr)
  \end{gather*}
  treating $\epsilon = 1/12$ as a small parameter.  *(How well do you think such a
  perturbative treatment will work?  Do you expect convergence keeping only the
  $\theta^4$ term?)*
  
  A naïve approach to this will converge very poorly, but a technique called
  **canonical perturbation theory** can work well, providing a strong motivation for
  understanding **Hamilton-Jacobi** theory, **cannonical transformations**, and
  **action-angle variables**. 
:::{margin}
The factor of $2$ in $2\lambda$ will simplify expressions later.
:::
* **Resonance**: One can easily add both damping $\lambda$ and driving $f(t)$ to the equations
  of motion for the harmonic oscillator:
  \begin{gather*}
    \ddot{\theta} = -\omega_0^2\theta - 2\lambda \dot{\theta} + f(t).
  \end{gather*}
  This is the canonical second-order linear inhomogeneous differential equation which is
  easily solvable and describes the phenomena of a driven resonance.  It is a starting
  point for a discussion of **Fourier analysis** (consider $f(t) = A\sin(\omega_0 t)$),
  **linear response theory**, and **Green's functions** (consider $f(t) = A\delta(t)$).
  These systems exhibit resonant behaviour when driven near the single resonant
  frequency $\omega_0$.
* **Parametric Resonance**: This is a different type of resonance that occurs when we
  modulate parameters, e.g. $\omega_0(t)$.  *(Physically, one could change the length of
  the pendulum $l(t)$, or perform the experiment in an elevator, which would modulate $g(t)$.)*:
  \begin{gather*}
    \ddot{\theta}(t) = -\omega_0^2(t)\theta(t) - 2\lambda \dot{\theta}(t).
  \end{gather*}
  The structure of parametric resonances is very different than that of driven
  resonances. For example, there are many resonances $\omega_n \approx 2\omega_0/n$ as
  opposed to the single resonance near $\omega_0$ for driven systems.
  * **Adiabatic Invariance**: If we change $\omega_0(t)$ very slowly, then certain qualitative
    aspects of the behaviour of the system can be understood in terms of what are called
    adiabatic invariances.  For example, one can understand the amplitude of the
    oscillation in terms of the length.  This is related to an interesting area of
    physics known as **geometric phases**.
  * **Floquet Analysis/Kaptiza's Pendulum**: In the other limit, if $\omega_0(t)$ is rapidly
    oscillating, we can use techniques related to Floquet analysis to separate out the
    slow and fast degrees of freedom.  By appropriately averaging one obtains an
    effective theory for the pendulum that stabilize the unstable equilibrium point
    $\theta \approx \pi$.  This is known as **Kaptiza's** pendulum.

The motion of a pendulum is familiar, and admits straight-forward numerical and analytic
solutions, making it a great problem for careful study.

## Numerical Solution

:::{margin}
We largely use the notation of {cite:p}`LL1:1976` with the modifications: we use $\gamma
\equiv \omega_d$ for the driving frequency and $\gamma \equiv \omega_p$ for the parametric driving
frequency, and we set $m=1$ so $f/m \equiv f$ for the driving potential.
:::
We start with a simple numerical solution as an initial-value problem.  We write the
problem in the following form:

\begin{gather*}
  \ddot{\theta} = -\omega^2(t)\sin\theta - 2\lambda\dot{\theta} + f(t), \\
  f(t) = f \cos\omega_d t, \qquad
  \omega^2(t) = \omega_0^2(1 + h\cos\omega_p t).
\end{gather*}
In the code, we spell these: $\lambda=$`damping`, $\omega_0=$`w0`, $\omega_d=$`w_d`,
$\omega_p=$`w_p`, $\theta_0=$`theta0`, $\dot{\theta}_0=$`dtheta0`, $\omega(t)=$`w_t(t)`,
$f(t)=$`f_t(t)`.

  
SciPy's {func}`scipy.integrate.solve_ivp` requires a first-order differential equation,
so we use the method of order reduction to solve for the two-component vector $\vect{y}$:
\begin{gather*}
  \vect{y} = \begin{pmatrix}
    \theta\\
    \dot{\theta}
  \end{pmatrix}, \qquad
  \dot{\vect{y}} =
  \begin{pmatrix}
    \dot{\theta}\\
    -\omega^2(t)\sin\theta - 2\lambda\dot{\theta} + f(t).
  \end{pmatrix}
\end{gather*}



```{code-cell}
from scipy.integrate import solve_ivp

class Pendulum:
    w0 = 1.0       # Natural resonant angular frequency [w0] = 1/T
    damping = 0.1  # Damping rate [damping] = 1/T 
    w_d = 1.2      # Driving frequency [w_d] = 1/T
    w_p = 2/3      # Parametric driving frequency [w_d] = 1/T
    f = 0.1        # Driving amplitude [f] = 1/T^2
    h = 0.12       # Parameteric driving amplitude [h] = 1
    
    theta0 = 0     # Initial position
    dtheta0 = 1.0  # Initial velocity
    
    def __init__(self, **kw):
        for key in kw:
            if not hasattr(self, key):
                raise ValueError(f"Unknown {key=}")
            setattr(self, key, kw[key])
    
    def w_t(self, t):
        """Return the natural frequency at time `t`."""
        return self.w0*np.sqrt(1 + self.h*np.cos(self.w_p*t))

    def f_t(self, t):
        """Return the driving force at time `t`."""
        return self.f * np.cos(self.w_d*t)
        
    def V_w2(self, theta, d=0):
        """Return the specific potential `V(theta)/m/l/w0**2` and its derivatives."""
        if d == 0:
            return 1 - np.cos(theta)
        elif d == 1:
            return np.sin(theta)
        else:
            raise NotImplementedError(f"{d=}")

    def compute_dy_dt(self, t, y):
        """Return dy_dt."""
        theta, dtheta = y
        w_t = self.w_t(t)
        f_t = self.f_t(t)
        V_w2 = self.V_w2(theta, d=1)
        ddtheta = (-w_t**2*V_w2 - 2*self.damping * dtheta + f_t)
        return (dtheta, ddtheta)
    
    def solve(self, T, method="BDF", **kw):
        """Return the solution `(t, theta, dtheta)`."""
        y0 = (self.theta0, self.dtheta0)
        sol = solve_ivp(
            self.compute_dy_dt, y0=y0, t_span=(0, T), method=method, **kw)
        t = sol.t
        theta, dtheta = sol.y
        return (t, theta, dtheta)
        
    def solve_and_plot(self, T_T0=10, ax=None, label=""):
        """Solve and plot.  Return `ax`.
        
        Arguments
        ---------
        T_T0 : float
            Number of natural periods to plot.
        """
        T0 = 2*np.pi / self.w0     # Natural period
        t, theta, dtheta = self.solve(T=T_T0*T0)
        if ax is None:
            fig, ax = plt.subplots()
        w0 = self.w0
        ax.plot(w0*t/2/np.pi, theta, label=label)
        ax.set(xlabel=r"$t/(2\pi/\omega_0)$", 
              ylabel=r"$\theta(t)$",
              title=", ".join([
                  fr"$f/\omega_0^2={self.f/w0**2:.2g}\cos({self.w_d/w0:.2g}\omega_0 t)$",
                  fr"$\omega^2/\omega_0^2=1+{self.h:.2g}\cos({self.w_p/w0:.2g}\omega_0t)$",
                  fr"$\lambda/\omega_0={self.damping:.2g}$",
              ]))
        return ax
        

p = Pendulum()
ax = p.solve_and_plot(T_T0=10)  # 10 natural periods
```

*(Note, we have taken care to plot dimensionless quantities.  One should check that the
plot does not change if we change the numerical values appropriately.)*

Here are some examples.  *Note: It takes quite a bit of playing around to come up with
good parameters to demonstrate these features.  Generating such plots from scratch is a
good way to make sure you understand the physics.*

### Resonance

```{code-cell}
ax = None
w0 = 1.0
for w_d_w0 in [0.5, 1.0, 1.5]:
    p = Pendulum(f=0.2, h=0, w_d=w_d_w0*w0)
    label = f"$\omega_d={p.w_d/p.w0:.2f}\omega_0$"
    ax = p.solve_and_plot(T_T0=10, ax=ax, label=label)
plt.legend()
ax.set(title="Driven resonance");
```

Here we drive at three difference frequencies.  After a transient period of about 5
periods, the oscillations settle down with $\omega = \omega_d$ having maximum
amplitude.


### Parametric Resonance

```{code-cell}
ax = None
w0 = 1.0
h = 0.5
lam = 0.003

for n in [1, 2, 3]:
    w_p = 2*w0/n
    dw_p = 0
    if n == 1:
        dw_p = h**2*w0/32
    if n == 2:
        dw_p = h**2*w0/24
    if n == 3:
        dw_p = h**2*w0/20  # Emperical...
    w_p = w_p - dw_p
    p = Pendulum(f=0, h=h, dtheta0=0.01, damping=lam, w0=w0, w_p=w_p)
    label = f"$\omega_p/\omega_0=2/{n}-{dw_p:.2f}$"
    ax = p.solve_and_plot(T_T0=100, ax=ax, label=label)
plt.legend()
ax.set(title="Parametric resonance");
```

# Parametric Resonance

Please review §27 of {cite:p}`LL1:1976` for a complete derivation and details.  To
better understand the results in {cite:p}`LL1:1976` (I found this section quite
difficult to read initially), consider {cite:p}`Arnold:1989` §25,
starting with the **Hill's equation** (Eq.(27.2) in {cite:p}`LL1:1976`):
\begin{gather*}
  \ddot{\theta} = -\omega^2(t)\theta, \qquad
  \omega(t + T) = \omega(t).
\end{gather*}
Express this as a first-order system:
\begin{gather*}
  \dot{\vect{y}} = \vect{f}(t, \vect{y}), \qquad
  \vect{f}(t+T, \vect{y}) = \vect{f}(t, \vect{y}), \qquad
  \vect{y} \in \mathbb{R}^{2}.
\end{gather*}
Now define the "propagator" $\op{U}_{t}: \mathbb{R}^2\mapsto\mathbb{R}^2$ which maps an
initial state $\vect{y}$ at time $t=0$ to the state $\vect{y}(t)$ satisfying the
equations. Arnol'd asks you to show that if $\vect{f}$ has period $T$, then the
operators $\op{U}_{nT}$ form a group
\begin{gather*}
  \op{U}_{nT} = (\op{U}_{T})^n.
\end{gather*}

Call $\op{A} = \op{U}_T$, the "mapping at a period" $T$.  Arnol'd proves the following
properties:
> 1. The point $\vect{y}_0$ is a fixedpoint of mapping $\op{A}$ iff the solution with
>    initial condition $\vect{y}(0) = \vect{y}_0$ is periodic with period $T$.
> 2. The periodic solution $\vect{y}(t)$ is Lyapunov stable (asymptotically stable) iff
>    the fixed point $\vect{y}_0$ of $\op{A}$ is similarly stable.
> 3. If $\vect{f}(t, \vect{x}) = f(t)\vect{y}$ is a linear function of $\vect{y}$, then
>    $\op{A}$ is linear.
> 4. If the system is Hamiltonian, then $\op{A}$ preserves volume: $\det \op{A} = 1$.

He then defines the following:

> The trivial solution of a Hamiltonian linear system is **strong stable** if it is
> stable, and if the trivial solution of every sufficiently close linear Hamiltonian
> system is also stable.

and proves the corollary

> If $\abs{\Tr \op{A}} < 2$, then the trivial solution is strongly stable.

For the parametric oscillator, strong stability implies that small perturbations to the
equilibrium state will not grow.  Thus, we are looking for regions that are **not**
strongly stable to find the parametric resonances.

The manipulations in {cite:p}`LL1:1976` amount to finding the eigenvalues $\mu_1$ and
$\mu_2$ of $\op{A}$, with condition (27.5) being $\mu_1\mu_2 = \det{\op{A}} = 1$.  The
case of real $\mu_1 = \mu_2^*$ which implies that $\abs{\mu_1} = \abs{\mu_2} = 1$
corresponds to Arnol'd's condition $\abs{\Tr\op{A}} \leq 2$: i.e. no parametric
resonance.  The other case of real $\mu_{i}$ means $\mu_2 = 1/\mu_1$, hence $\abs{\Tr\op{A}} =
\abs{\mu_1 + 1/\mu_1} \geq 2$ implies a parametric resonances everywhere except
$\mu_1=1$:

```{code-cell}
mu = np.linspace(0.1, 2)
fig, ax = plt.subplots()
ax.plot(mu, mu + 1/mu)
ax.set(xlabel="$\mu_1$", ylabel=r"$\mu_1 + 1/\mu_1$", ylim=(2, 5));
```

The method of solution is to solve the following equation for small $h$:
\begin{gather*}
  \ddot{\theta} = -\omega_0^2(1 + h\cos\omega_p t)\theta - 2\lambda \dot{\theta}.
\end{gather*}
We start by setting $\omega_p=1$ to set our time scale:
\begin{gather*}
  \ddot{\theta} + 2\lambda \dot{\theta} + \omega^2(1 + h\cos t)\theta = 0.
\end{gather*}
Now we look for steady state periodic solutions where $\theta(t+T) = \theta(T)$ with
$T=2\pi$:
:::{margin}
Here we use the following:
\begin{align*}
  \cos a \cos b &= \frac{\cos(a-b) + \cos(a+b)}{2},\\
  \sin a \sin b &= \frac{\cos(a-b) - \cos(a+b)}{2},\\
  \sin a \cos b &= \frac{\sin(a-b) + \sin(a+b)}{2}.
\end{align*}
:::
\begin{gather*}
  \theta = \sum_{n=1}^{\infty}\Bigl(a_n\cos(nt) + b_n\sin(nt)\Bigr),\\
  \dot{\theta} = -\sum_{n=1}^{\infty}n\Bigl(a_n\sin(nt) - b_n\cos(nt)\Bigr),\\
  \ddot{\theta} = -\sum_{n=1}^{\infty}n^2\Bigl(a_n\cos(nt) + b_n\sin(nt)\Bigr),\\
  \cos(t)\theta = \frac{1}{2}\sum_{n=1}^{\infty}\Biggl(
    a_n\Bigl(\cos(n-1)t+\cos(n+1)t\Bigr) 
    + b_n\Bigl(\sin(n-1)t + \sin(n+1)t\Bigr)
  \Biggr).
\end{gather*}
Collecting terms $\cos(nt)$ and $\sin(nt)$, which all must vanish, we get the following
system of equations:

\begin{gather*}
  -n^2a_n + 2\lambda n b_n + \omega^2 \Bigl(a_n + \frac{h}{2} (a_{n+1} + a_{n-1}\Bigr) = 0,\\
  -n^2b_n - 2\lambda n a_n + \omega^2 \Bigl(b_n + \frac{h}{2} (b_{n+1} + b_{n-1}\Bigr) = 0.
\end{gather*}

If we set $h=0$, the equations decouple pairwise and we get
\begin{gather*}
  \begin{pmatrix}
    \omega^2 - n^2 & 2\lambda n\\
    -2\lambda n & \omega^2 - n^2
  \end{pmatrix}
  \begin{pmatrix}
    a_n\\
    b_n
  \end{pmatrix}
  = 0, \qquad
  \omega = \sqrt{n(n \pm \I 2\lambda)}
\end{gather*}

\begin{gather*}
  \frac{h}{2}
  \begin{pmatrix}
    a_{2}\\
    b_{2}
  \end{pmatrix}
  =
  \begin{pmatrix}
    1 - \omega^2 & -2\lambda\\
    2\lambda & 1 - \omega^2\\
  \end{pmatrix}
  \begin{pmatrix}
    a_{1}\\
    b_{1}
  \end{pmatrix},\\
  \frac{h}{2}
  \begin{pmatrix}
    a_{3}\\
    b_{3}
  \end{pmatrix}
  =
  \begin{pmatrix}
    2^2 - \omega^2 & -4\lambda\\
    4\lambda & 2^2 - \omega^2\\
  \end{pmatrix}
  \begin{pmatrix}
    a_{2}\\
    b_{2}
  \end{pmatrix}
  -
  \frac{h}{2}
  \begin{pmatrix}
    a_{1}\\
    b_{1}
  \end{pmatrix},\\
  \frac{h}{2}
  \begin{pmatrix}
    a_{4}\\
    b_{4}
  \end{pmatrix}
  =
  \begin{pmatrix}
    3^2 - \omega^2 & -6\lambda\\
    6\lambda & 3^2 - \omega^2\\
  \end{pmatrix}
  \begin{pmatrix}
    a_{3}\\
    b_{3}
  \end{pmatrix}
  -
  \frac{h}{2}
  \begin{pmatrix}
    a_{2}\\
    b_{2}
  \end{pmatrix}  
\end{gather*}


Noting that 

.  In the limit $\lambda = h = 0$, the solutions are $\theta(t) =
A\cos(n t)$, so the equation is periodic iff 

\begin{gather*}
  2\omega_0 = n - \epsilon
\end{gather*}

```{code-cell}
def get_A(dx=1e-8, tol=1e-12, **kw):
    p1 = Pendulum(f=0, theta0=dx, dtheta0=0, **kw)
    p2 = Pendulum(f=0, theta0=0, dtheta0=dx, **kw)
    w_p = p1.w_p
    T = 2*np.pi / w_p
    
    t1, theta1, dtheta1 = p1.solve(T=T, atol=tol, rtol=tol)
    t2, theta2, dtheta2 = p2.solve(T=T, atol=tol, rtol=tol)
    A = np.array([[theta1[-1], theta2[-1]],
                 [dtheta1[-1], dtheta2[-1]]])/dx
    return A

w0 = 1.0
h = 0.2
lam = 0.003

for n in [1, 2, 3]:
    w_p = 2*w0/n
    dw_p = 0
    if n == 1:
        dw_p = h**2*w0/32
    elif n == 2:
        dw_p = 2*h**2*w0/24
    elif n == 3:
        dw_p = 2*h**2*w0/24
    kw = dict(h=0.2, damping=lam, w_p=w_p-dw_p)
    A = get_A(**kw)
    print(np.trace(A), np.linalg.det(A))
```

```{code-cell}
lam = 0.003
w_p = 1
w0s = np.linspace(0.25, 2.25, 100)
#w0s = np.linspace(0.4, 0.6, 50)
#w0s = np.linspace(0.9, 1.1, 50)
#w0s = np.linspace(1.4, 1.6, 50)
hs = np.linspace(0, 0.8, 50)
@np.vectorize
def stable(w0, h):
    return abs(np.trace(get_A(w_p=w_p, w0=w0, h=h, damping=lam))) < 2

res = stable(w0s[:, None], hs[None, :])
fig, ax = plt.subplots()
ax.pcolormesh(w0s, hs, res.T, shading='nearest')
ax.set(xlabel=r"$\omega_0/\omega_p$", ylabel="$h$");
```

# Harmonic Oscillator

We start with the complete solution for a Harmonic Oscillator: i.e. the limit of a
pendulum for small amplitude motion:
\begin{gather*}
  \ddot{\theta} + 2\lambda \dot{\theta} + \omega_0^2 \theta = 0, \qquad
  \theta(0) = \theta_0, \qquad
  \dot{\theta}_0 = \dot{\theta}_0.
\end{gather*}
This is a homogeneous second-order differential equation.  Substituting $\theta =
ae^{\I\omega t}$, we find the characteristic polynomial
\begin{gather*}
  -\omega^2 + 2\lambda \I \omega + \omega_0^2 = 0, \qquad
  \omega = \lambda \I \pm \underbrace{\sqrt{\omega_0^2 - \lambda^2}}_{\bar{\omega}},\\
  \theta(t) = e^{-\lambda t}\Bigl(a\cos\bar{\omega} t + b \sin\bar{\omega} t\Bigr)
\end{gather*}
Solving for the initial conditions, we have
\begin{gather*}
  \theta_0 = a, \qquad
  \dot{\theta}_0 = -\lambda a + b\omega,\\
  a = \theta_0, \qquad
  b = \frac{\dot{\theta}_0 + \lambda \theta_0}{\bar{\omega}},\\
  \theta(t) = e^{-\lambda t}\Bigl(\theta_0\cos\bar{\omega} t 
  + \frac{\dot{\theta}_0 + \lambda \theta_0}{\bar{\omega}} \sin\bar{\omega} t\Bigr).
\end{gather*}

From this solution, we can compute the solution to the Hamilton-Jacobi equation, which
is the classical action $S(q, P, t)$
\begin{gather*}
  
\end{gather*}










# Naïve Perturbation Theory

Consider the parametric resonance problem:

\begin{gather*}
  \ddot{\theta} = - \omega^2(1 + h \cos \omega_p t)\theta, \qquad
  \theta(0) = 0, \qquad \dot{\theta}(0) = \dot{\theta}_0,
\end{gather*}
where $h$ is small.  A naïve attempt to apply perturbation theory might be to express
\begin{gather*}
  \theta(t) = q_0(t) + hq_1(t) + h^2 q_2(t) + \cdots.
\end{gather*}
Plugging this in, we find:
\begin{align*}
  \ddot{q}_0 + \omega^2 q_0 &= 0\\
  \ddot{q}_1 + \omega^2 q_1 &= -\omega^2q_0\cos\omega_p t\\
  &\vdots\\
  \ddot{q}_{n} + \omega^2 q_{n} &= -\omega^2q_{n-1}\cos\omega_p t.
\end{align*}
There is still the question of the initial conditions, but supposed we take $q_0(0) =
0$, $\dot{q}_0(0) = \dot{\theta}_0$, then $q_{n>0}(0) = \dot{q}_{n>0}(0) = 0$ and we can
solve:
\begin{align*}
  q_0(t) &= \frac{\dot{\theta}_0}{\omega_0}\sin\omega t,\\
  \ddot{q}_1 + \omega^2 q_1 &= \omega\dot{\theta}_0\cos(\omega_p t)\sin(\omega t)
  = \omega\dot{\theta}_0\frac{\sin(\omega-\omega_p)t + \sin(\omega+\omega_p)t}{2}
\end{align*}

We could solve these equations analytically, but it is a mess.  Instead, we will
numerically integrate the set of them along with the full solution for comparison,
packing and indexing them as
\begin{gather*}
  \vect{y} = \begin{pmatrix}
    \begin{pmatrix}
      \theta\\
      \dot{\theta}
    \end{pmatrix}\\
    \begin{pmatrix}
      q_0\\
      \dot{q}_0
    \end{pmatrix}\\
    \begin{pmatrix}
      q_1\\
      \dot{q}_1
    \end{pmatrix}\\
    \vdots\\
    \begin{pmatrix}
      q_{N}\\
      \dot{q}_{N}
    \end{pmatrix}
  \end{pmatrix}, \qquad
  \texttt{y[2::2]} = \begin{pmatrix}
    q_0\\
    q_1\\
    \vdots\\
    q_{N}
  \end{pmatrix}, \qquad
  \texttt{y[3::2]} = \begin{pmatrix}
    \dot{q}_0\\
    \dot{q}_1\\
    \vdots\\
    \dot{q}_{N}
  \end{pmatrix}
\end{gather*}

```{code-cell}
N = 30   # Include 30 terms
theta0 = 0.0
dtheta0 = 0.1
w = 1.0
w_p = 2.1
h = 0.07

y0 = [theta0, dtheta0, theta0, dtheta0] + [0]*(2*N)

def compute_dy_dt(t, y):
    y = np.asarray(y)
    dy = np.zeros_like(y)

    # Just shift over for dtheta_dt and dq/dt 
    dy[0::2] = y[1::2]
    
    # Full equation for ddtheta
    dy[1] = -w**2*(1+h*np.cos(w_p*t))*y[0]
    
    q = y[2::2]
    ddq = dy[3::2]
    
    ddq[0] = -w**2*q[0]
    ddq[1:] = -w**2*(q[1:] + q[0:-1]*np.cos(w_p*t))
    return dy

Np = 200  # Number of periods 
T = 2*np.pi / w * Np
res = solve_ivp(compute_dy_dt, y0=y0, t_span=(0, T), method="BDF")
t = res.t
theta, dtheta = res.y[:2]
q = res.y[2::2]
dq = res.y[3::2]

ax.plot(t, theta)
ax.set(xlabel="$t$", ylabel="$\theta(t)$");
```

```{code-cell}
n = np.arange(N+1)
approx = np.cumsum((h**n)[:, None]*q, axis=0)
errs = abs(approx-theta)
fig, axs = plt.subplots(1, 2, figsize=(10,5), sharey=True)
ax = axs[0]
for Np in [20, 40, 60, 80, 100]:
    inds = np.where(t<=2*np.pi / w * Np)[0]
    ax.semilogy(n, errs[:, inds].max(axis=1), 
                label=fr"$T={Np}(2\pi/\omega)$")
ax.legend()
ax.set(xlabel="$N$ (terms)", ylabel="Max err");

ax = axs[1]
for N in [10, 20, 30]:
    ax.semilogy(t, errs[N], label=fr"${N=}$ terms")
ax.set(xlabel="$t$");
ax.legend();
```

It seems like we have convergence in $h$, but we see a common problem with naïve
perturbation theory that the error increases with longer time -- in this case, almost
exponentially.  Often, the crux of the problem is that the frequency of the full
solution ($\omega_p$) and the frequency of the perturbations ($\omega$) are mismatched.
This leads to the "beating" seen in the error plot.

Another example worked out in {cite:p}`Fetter:2006` is the anharmonic oscillator.  Try
it now.

::::{margin}
:::{glue:figure} fig:naivePT

Naïve perturbation theory applied to the anharmonic oscillator gives rise to an
unphysical linearly increasing amplitude.  It does start to correct the frequency in the
correct direction, but even this fails once the growing term starts to dominate.
:::
::::
:::{doit} Do It!

Try using naïve to solve for the motion of a particle in an anharmonic potential:
\begin{gather*}
  V(1) = \frac{m\omega_0^2}{2}x^2 + \epsilon\omega_0^2 \frac{m}{4} x^4, \qquad
  x(0) = a, \qquad \dot{x}(0) = 0.
\end{gather*}
Solve the equations to obtain the first-order correction and show that the solution has
an amplitude the grows linearly with time.
:::

:::{solution}
:show:

Let $x(t) = x_0(t) + \epsilon x_1(t) + \epsilon^2 x_2(t) + \cdots$.  Then we have
\begin{multline*}
  \sum_{n}\epsilon^n(\ddot{x}_{n} + \omega_0^2 x_n) 
  = -\epsilon \left(\sum_{n}\epsilon^n x_{n}\right)^3\\
  = -\epsilon x_0^3 - 3\epsilon^2 x_0^2x_1 - 3\epsilon^3(x_0^2x_2 + x_0x_1^2) \\
    -\epsilon^4(3x_0^2x_3 + 6x_0x_1x_2 + x_1^3) + O(\epsilon^5).
\end{multline*}
This gives the following system, order by order:
\begin{align*}
  \ddot{x}_0 + \omega_0^2x_0 &= 0\\
  \ddot{x}_1 + \omega_0^2x_1 &= -x_0^3\\
  \ddot{x}_2 + \omega_0^2x_2 &= -3x_0^2x_1\\
  \ddot{x}_3 + \omega_0^2x_3 &= -3(x_0^2x_2 + x_0x_1^2)\\
  & \vdots
\end{align*}
Let's take the solution with $x_0(0) = a$ and $\dot{x}_0(0)=0$: $x_0(t) = a\cos\omega_0
t$.  Then to 1st order we have:
\begin{gather*}
  \ddot{x}_1 + \omega_0^2x_1 = -\omega_0^2a^3\cos^3\omega_0 t 
  = -\omega_0^2 a^3\frac{\cos 3\omega_0 t + 3\cos\omega_0 t}{4}.
\end{gather*}
This is a linear inhomogeneous equation with two inhomogeneous terms which can be
treated independently.  Each is a driven harmonic oscillator.  The term driving at
$3\omega_0$ is not a problem, but the term driving on resonance is as it will lead to
linear growth.  Here are two solutions:
\begin{gather*}
  q_3(t) = c_3\cos 3\omega_0 t, \qquad
  \ddot{q}_3 +\omega_0^2 q_3 = -8\omega_0^2 c_3\cos 3\omega_0 t 
  = \frac{-\omega_0^2 a^3}{4}\cos 3\omega_0 t,\qquad
  c_3 = \frac{\omega_0^2 a^3}{32},\\
  q_1(t) = c_1 t \sin \omega_0 t, \qquad
  \dot{q}_1 = c_1 \sin \omega_0 t + \omega_0 c_1 t \cos \omega_0 t, \qquad
  \ddot{q}_1 = 2 c_1 \cos \omega_0 t - \omega_0^2 c_1 t \sin \omega_0 t, \\
  \ddot{q}_1 + \omega_0^2q_1 = 2 c_1 \cos \omega_0 t = \frac{-3\omega_0^2a^3}{4}\cos\omega_0 t, \qquad
  c_1 = \frac{-3\omega_0^2a^3}{8},\\
  x_1 = A\cos\omega_0 t + B \sin\omega_0 t 
      + \frac{\omega_0^2a^3(\cos 3\omega_0 t - 12 \omega_0 t \sin \omega_0 t)}{32}.
\end{gather*}
Solving for the initial conditions $x_1(0) = \dot{x}_1(0) = 0$ we have
$A = - \omega_0^2a^3/32$ and $B=0$ (cf. {cite:p}`Fetter:2006` (7.29)):
\begin{gather*}
  x_1 = \frac{\omega_0^2a^3(\cos 3\omega_0 t - \cos\omega_0 t - 12 \omega_0 t \sin \omega_0 t)}{32},\\
  x(t) = a\cos\omega_0 t + \epsilon\frac{\omega_0^2a^3}{32}(\cos 3\omega_0 t - \cos\omega_0 t 
  - 12 \omega_0 t \sin \omega_0 t) + O(\epsilon^2).
\end{gather*}
:::

```{code-cell}
#:tags: [hide-cell]

try: from myst_nb import glue
except: glue = None

a = 1.0
w0 = 1.0
lam = 0.0
w = np.sqrt(w0**2-lam**2)
eps = 0.6/a**2/w0**2

def f(t, y):
    x, dx = y
    ddx = -w0**2*x - 2*lam*dx - eps*w0**2*x**3
    return dx, ddx

y0 = (a, 0.0)
T = 2*np.pi / w0
res = solve_ivp(f, y0=y0, t_span=(0, 3*T), max_step=T/20)
t = res.t
x, dx = res.y
x0 = a*np.cos(w0*t)
x1 = w0**2*a**3/32*(np.cos(3*w0*t) - np.cos(w0*t) - 12*w0*t*np.sin(w0*t))
x1_cpt = w0**2*a**3/w**4/np.exp(4*lam*t)*np.sin(w*t)*(w*np.cos(w*t)+lam*np.sin(w*t))

fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(t/T, x/a, label="Exact answer (numerical)")
ax.plot(t/T, x0/a, ':', label="Unperturbed (0th order)")
ax.plot(t/T, (x0 + eps*x1)/a, '--', label="Naive perturbation (1st order)")
ax.plot(t/T, (x0 + eps*x1_cpt)/a, '-.', label="Canonical perturbation (1st order)")
ax.legend()
ax.set(xlabel=r"$t/(2\pi/\omega_0)$", 
       ylabel=r"$x/x(0)$",
       title=fr"$\epsilon = {eps*a**2:.2g}/x^2(0)$")
plt.tight_layout()
if glue: glue("fig:naivePT", fig);
```

:::{doit} Do It!

How could you improve your result?  *Hint: try also expanding the frequency $\omega_0$ to
get rid of the linearly increasing term. Cf. {cite:p}`Fetter:2006`.*


:::


# Canonical Perturbation Theory

A much better approach is provided by **canonical perturbation theory**.  This can be
expressed in several equivalent ways.

We consider a Hamiltonian system with conjugate variables $(\vect{q}, \vect{p})$:
\begin{gather*}
  H(\vect{q}, \vect{p}, t) = H_0(\vect{q}, \vect{p}, t) 
  + \epsilon H_1(\vect{q}, \vect{p}, t).
\end{gather*}
We now assume that a solution to the Hamiltonian problem $H_0(\vect{q}, \vect{p})$ is
known such that we can effect a canonical transform via the generating function
$F_2(\vect{q}, \vect{P}, t)$ to a new set of conjugate variables $(\vect{Q}, \vect{P})$
with Hamiltonian $K_0(\vect{Q}, \vect{P}, t) = 0$:
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


## Example: Parametric Resonance

Let's apply this to study parametric resonance with $\theta=q$, $h=\epsilon$ from before:
\begin{gather*}
  \ddot{q} + 2\lambda \dot{q} + \omega^2(t)q = 0, \qquad
  \omega^2(t) = \omega_0^2(1+\epsilon \cos\omega_pt).
\end{gather*}
This follows from the Hamiltonian
\begin{gather*}
  H(q, p, t) = e^{-2\lambda t}\frac{p^2}{2} + e^{2\lambda t}\frac{\omega^2(t)q^2}{2}.
\end{gather*}
:::{solution} Do it!  Show that this gives the correct equation of motion.

\begin{gather*}
  \dot{q} = \pdiff{H}{p} = e^{-2\lambda t}p, \qquad
  \dot{p} = -\pdiff{H}{q} = -e^{2\lambda t}\omega^2(t)q, \\
  p = e^{2\lambda t} \dot{q},\\
  \ddot{q} = e^{-2\lambda t}(\dot{p} -2\lambda p)
  = e^{-2\lambda t}(-e^{2\lambda t}\omega^2(t)q  - 2\lambda e^{2\lambda t} \dot{q})
  = -\omega^2(t)q - 2\lambda \dot{q}.
\end{gather*}
:::

Thus, we have the following perturbative problem:
\begin{gather*}
  H = H_0 + \epsilon H_1, \qquad
  H_0 = e^{-2\lambda t}\frac{p^2}{2} + e^{2\lambda t}\frac{\omega_0^2q^2}{2}, \qquad
  H_1 = e^{2\lambda t}\frac{\omega_0^2\cos\omega_p t}{2}q^2.
\end{gather*}

We must now solve the unperturbed problem:
\begin{gather*}
   H_0 = e^{-2\lambda t}\frac{p^2}{2} + e^{2\lambda t}\frac{\omega_0^2q^2}{2},\\
  \dot{q} = \pdiff{H_0}{p} = e^{-2\lambda t}p, \qquad
  \dot{p} = -\pdiff{H_0}{q} = -e^{2\lambda t}\omega_0^2q,\\
  L_0(q, \dot{q}, t) = p\dot{q} - H_0 = e^{-2\lambda t}p^2 - H_0
  = e^{-2\lambda t}\frac{p^2}{2} - e^{2\lambda t}\frac{\omega_0^2q^2}{2}\\
  = e^{2\lambda t}\left(
    \frac{\dot{q}^2}{2} - \frac{\omega_0^2q^2}{2}\right),\qquad
  S_0 = \int_0^{t}L_0\d{t}.
\end{gather*}

:::{margin}
We will restrict our discussion to under-damped cases where $\omega_0^2 \geq \lambda^2$.
:::
The general solution $q(t) = \sum_{\pm}a_{\pm}e^{\I\omega_{\pm}t}$ to the unperturbed
problem can be expressed in terms of the complex
frequencies $\omega_{\pm}$ which satisfy
\begin{gather*}
  -\omega_{\pm}^2 + 2\I\lambda \omega_{\pm} + \omega_0^2 = 0, \qquad
  \omega_{\pm} = \I\lambda \pm \underbrace{\sqrt{\omega_0^2 -\lambda^2}}_{\bar{\omega}}.
\end{gather*}
Thus, the general solution is:
\begin{gather*}
  q(t) = e^{-\lambda t}(a\cos\bar{\omega}t + b\sin\bar{\omega}t),\\
  \dot{q}(t) = e^{-\lambda t}\Bigl(
    (-a\bar{\omega} - b\lambda)\sin\bar{\omega}t + (b\bar{\omega}-\lambda a)\cos\bar{\omega}t)
    \Bigr),\\
  L_0 = \frac{1}{2}\Bigl(
    (-a\bar{\omega} - b\lambda)\sin\bar{\omega}t + (b\bar{\omega}-\lambda a)\cos\bar{\omega}t)
    \Bigr)^2 
    - \frac{\omega_0^2}{2}(a\cos\bar{\omega}t + b\sin\bar{\omega}t)^2.
\end{gather*}
Recall that the generating function $F_2(q, P, t) = S_0(q, P, t)$ of the canonical
transformation satisfies the Hamilton-Jacobi equation
\begin{gather*}
  K_0 = H_0(q, p) + \pdiff{S_0}{t} = H_0\left(q, \pdiff{S_0}{q}\right) + \pdiff{S_0}{t} = 0,
\end{gather*}
which can be expressed in terms of the classical action integrated along these solutions
with an appropriately chosen constant of the motion $P$:
\begin{gather*}
  S_0(q, P, t) = \int_{t_0}^{t_1}L_0(q, \dot{q}, t)\d{t}.
\end{gather*}
A reasonable choice with $t_0=0$, $t_1=t$, and constant is $P=q(0)$ with $q=q(t)$:
\begin{gather*}
  P = a, \qquad
  q = e^{-\lambda t}(a\cos\bar{\omega}t + b\sin\bar{\omega}t),\\
  a = P, \qquad
  b = e^{\lambda t}q - P\cot\bar{\omega}t.
\end{gather*}
*Note: we must include the time-dependence in the coefficient $b(t)$ here to enforce the
appropriate boundary conditions.  For any given solution it is constant.*

:::{margin}
If you want to compute this yourself, the following might be helpful:
\begin{gather*}
  \int_0^{t}\d{t}\begin{pmatrix}
    \sin^2\bar{\omega}t\\
    \cos^2\bar{\omega}t\\
    \sin\bar{\omega}t\cos\bar{\omega}t\\
  \end{pmatrix}
  =
  \frac{1}{2\bar{\omega}}\begin{pmatrix}
    \bar{\omega}t - \frac{\sin 2\bar{\omega}t}{2}\\
    \bar{\omega}t + \frac{\sin 2\bar{\omega}t}{2}\\
    \sin^2\bar{\omega}t
  \end{pmatrix},
\end{gather*}
:::
Computing this all is rather tedious, and better left for a computer.  When the dust
settles, we have the generating function:
\begin{gather*}
  S_0(q, P, t) =
  \bar{\omega}\frac{P^{2} + q^{2} e^{2 \lambda t}}
  {2 \tan{\bar{\omega} t}} 
  - \frac{P \bar{\omega} q e^{\lambda t}}{\sin\bar{\omega}t} 
  + \lambda\frac{P^{2} - q^{2} e^{2 \lambda t}}{2},\\
  P = q(0) = q e^{\lambda t} \cos\bar{\omega}t
  - (\lambda q e^{\lambda t} + p e^{- \lambda t})
    \frac{\sin\bar{\omega}t}{\bar{\omega}} = a = q(0),\\
  Q = - \dot{q}(0) =
  -pe^{-\lambda t} \cos\bar{\omega}t
  -
  \left(
    \omega_0^{2} q e^{\lambda t}
    + \lambda p e^{-\lambda t} 
  \right)\frac{\sin\bar{\omega}t}{\bar{\omega}} 
  = \lambda a - \bar{\omega}b,\\
  q = \frac{e^{- \lambda t}}{\bar{\omega}}
  \Bigl(
    P(\bar{\omega} \cos\bar{\omega}t + \lambda\sin\bar{\omega}t) 
    - Q \sin\bar{\omega} t
  \Bigr),\\
  p = \frac{-e^{\lambda t}}{\bar{\omega}}
  \Bigl(
    Q(\bar{\omega} \cos\bar{\omega}t - \lambda\sin\bar{\omega} t)
    + P\omega_0^2\sin\bar{\omega}t 
  \Bigr)
\end{gather*}

```{code-cell}
:tags: [hide-cell]

from IPython.display import Latex
import sympy
from sympy import sqrt, exp, sin, cos, I, Eq, S

def disp(*v, **kw):
    """Simple display function."""
    for var, expr in v + tuple(kw.items()):
        display(Latex(f"${var}={sympy.latex(expr)}$"))
        
a, b, P, Q, p, q = sympy.var(r'a,b,P,Q,p,q', real=True)
lam, w_, w0, t = sympy.var(r'\lambda,\bar{\omega},\omega_0,t', positive=True)
s, c = sin(w_*t), cos(w_*t)
w0 = sqrt(w_**2+lam**2)
q_t = exp(-lam*t)*(a*c + b*s)

# Check equation of motion
assert (q_t.diff(t, t) + 2*lam*q_t.diff(t) + w0**2*q_t).simplify() == 0

# Compute L0
H0 = exp(-2*lam*t)*p**2/2 + exp(2*lam*t)*w0**2*q**2/2
dq = sympy.var(r'\dot{q}')
p_ = sympy.solve(H0.diff(p) - dq, p)[0]
L0 = (p_*dq - H0).subs([(p, p_)]).simplify()
disp(L_0=L0)

# Now use solution
dq_t = q_t.diff(t)
p_t = p_.subs([(dq, dq_t)])
L0 = L0.subs([(q, q_t), (dq, dq_t)]).expand().simplify()
#L0 = (exp(2*lam*t)*(dq_t**2 - w0**2*q_t**2)/2).expand().simplify()
disp(L_0=L0.collect([sin(2*w_*t), cos(2*w_*t)]))

# Compute S0
S0 = L0.integrate((t, 0, t)).collect([sin(2*w_*t), cos(2*w_*t)]).simplify()
disp(S_0=S0)

# Find a and b in terms of P and q
ab_subs = sympy.solve([q_t.limit(t, 0) - P, q_t - q], [a, b])
S0 = S0.subs(ab_subs).simplify()
disp(S_0=S0)

# Check that this works!
PQ_subs = sympy.solve([S0.diff(q) - p, S0.diff(P) - Q], [P, Q])
pq_subs = sympy.solve([S0.diff(q) - p, S0.diff(P) - Q], [p, q])
P_, Q_ = PQ_subs[P].simplify(), PQ_subs[Q].simplify()
p_, q_ = pq_subs[p].simplify(), pq_subs[q].simplify()
disp(P=P_, Q=Q_)
disp(p=p_.collect([P, Q]).simplify(), 
     q=q_.collect([P, Q]).simplify())
disp(P=P_.subs([(q, q_t), (p, p_t)]).simplify(), 
     Q=Q_.subs([(q, q_t), (p, p_t)]).simplify())
K0 = (H0 + S0.diff(t)).subs(PQ_subs).simplify()
assert K0 == 0
```

We can now express the perturbation in terms of the Hamiltonian $H_1$:

\begin{gather*}
  H_1(Q, P, t) = 
  \bigl(
    P\bar{\omega}\cos\bar{\omega}t 
    + 
    (P\lambda - Q)\sin\bar{\omega}t 
  \bigr)^{2}
  \frac{\omega_0^2}{2\bar{\omega}^2}
  \cos\omega_{p}t.
\end{gather*}
The derivatives required for canonical perturbation theory is thus
\begin{gather*}
  \epsilon
  \begin{pmatrix}
    \partial{H_1}/\partial{P}\\
    -\partial{H_1}/\partial{Q}
  \end{pmatrix}
  =
  \epsilon
  \bigl(P\bar{\omega}\cos\bar{\omega}t + (P\lambda - Q)\sin\bar{\omega}t\bigr)
  \begin{pmatrix}
    \bar{\omega}\cos\bar{\omega}t + \lambda\sin\bar{\omega}t\\
    \sin\bar{\omega}t
  \end{pmatrix}
  \frac{\omega_0^2}{\bar{\omega}^2}
  \cos\omega_{p}t.
\end{gather*}

## Example: Anharmonic Oscillator 1

We use the same approach for the anharmonic oscillator starting with $q(0) = q_0$ and
$\dot{q}(0) = 0$:
\begin{gather*}
  \ddot{q} + \omega_0^2(q + \epsilon q^3) = 0,\qquad
  H_1(q, p, t) = \omega_0^2\frac{q^4}{4}.
\end{gather*}

### Canonical Transformation

The canonical transformation is
\begin{gather*}
  q = \frac{(\alpha P - Q\sin\bar{\omega}t)}{\bar{\omega}},\qquad
  p = \frac{-(\alpha Q + P\omega_0^2\sin\bar{\omega}t)}{\bar{\omega}},\\
  \alpha = \bar{\omega}\cos\bar{\omega} t.
\end{gather*}


We use the same approach for the anharmonic oscillator starting with $q(0) = q_0$ and
$\dot{q}(0) = 0$:
\begin{gather*}
  \ddot{q} + 2\lambda \dot{q} + \omega_0^2(q + \epsilon q^3) = 0,\qquad
  H_1(q, p, t) = \omega_0^2\frac{q^4}{4}.
\end{gather*}
First we clean up the notation a bit.



Thus, the perturbation and canonical series is:
\begin{gather*}
  H_1(Q, P, t) = 
  \frac{\omega_0^2}{\bar{\omega}^4e^{4\lambda t}}
  \frac{(\alpha P - Q \sin\bar{\omega}t)^4}{4},\qquad
  X = \frac{\omega_0^2}{\bar{\omega}^4e^{4\lambda t}}
      (\alpha P - Q \sin\bar{\omega}t)^3,\\
  \begin{pmatrix}
    \partial H_1/\partial P\\
    -\partial H_1/\partial Q\\
  \end{pmatrix}
  =
  X
  \begin{pmatrix}
    \alpha\\
    \sin\bar{\omega}t
  \end{pmatrix}.
\end{gather*}
Plugging in our zero'th order solution $P_0 = q(0) = q_0$, $Q_0 = -\dot{q}(0) = 0$, we
 have the first order correction
\begin{align*}
  X_0 &= \frac{\omega_0^2q_0^3}{\bar{\omega}^4e^{4\lambda t}}\alpha^3,\\
  \dot{Q}_1 &= \alpha X_0 
  = \frac{\omega_0^2q_0^3}{\bar{\omega}^4e^{4\lambda t}}
    (\bar{\omega}\cos\omega t - \lambda \sin\bar{\omega}t)^4,\\
  \dot{P}_1 &= X_0\sin\bar{\omega}t 
  = \frac{\omega_0^2q_0^3}{\bar{\omega}^4e^{4\lambda t}}
    (\bar{\omega}\cos\omega t - \lambda \sin\bar{\omega}t)^3
    \sin\bar{\omega}t.
\end{align*}
In the limit of no damping $\lambda \rightarrow 0$ we have
\begin{align*}
  \alpha &= \omega_0\cos\omega_0 t\\
  \dot{Q}_1 &= 
  \omega_0^2q_0^3\cos^4\omega_0 t,\\
  \dot{P}_1 &=
  \frac{\omega_0^2q_0^3}{\omega_0}
  \cos^3\omega t\sin\omega_0t,\\
  Q_1 &= 
  \omega_0^2q_0^3\frac{12\omega_0 t + 8 \sin 2\omega_0 t + \sin 4\omega_0 t}
                      {32\omega_0},\\
  \omega_0 P_1 &=
  \omega_0^2q_0^3
  \frac{1-\cos^4 \omega_0 t}{4\omega_0}.
\end{align*}
Converting back to our original coordinates, we have
\begin{align*}
  q &= q_0 + \epsilon \frac{\omega_0 P_1\cos\omega_0 t - Q_1 \sin\omega_0t}{\omega_0}\\
    &= q_0 + \epsilon q_0^3
       \frac{\cos 3\omega_0 t - \cos \omega_0 t - 12 \omega_0 t \sin \omega_0 t}{32}
  ,\\
  p &= p_0 + \epsilon (Q_1\cos\omega_0 t + \omega_0 P_1\sin\omega_0t).
\end{align*}



First we clean up the notation a bit.



## Example: Anharmonic Oscillator

We use the same approach for the anharmonic oscillator starting with $q(0) = q_0$ and
$\dot{q}(0) = 0$:
\begin{gather*}
  \ddot{q} + 2\lambda \dot{q} + \omega_0^2(q + \epsilon q^3) = 0,\qquad
  H_1(q, p, t) = \omega_0^2\frac{q^4}{4}.
\end{gather*}
First we clean up the notation a bit.



The canonical transformation is
\begin{gather*}
  q = \frac{e^{-\lambda t}}{\bar{\omega}}(\alpha P - Q\sin\bar{\omega}t),\qquad
  p = \frac{-e^{\lambda t}}{\bar{\omega}}(\alpha Q + P\omega_0^2\sin\bar{\omega}t),\\
  \alpha = \bar{\omega}\cos\bar{\omega} t - \lambda \sin\bar{\omega}t.
\end{gather*}
Thus, the perturbation and canonical series is:
\begin{gather*}
  H_1(Q, P, t) = 
  \frac{\omega_0^2}{\bar{\omega}^4e^{4\lambda t}}
  \frac{(\alpha P - Q \sin\bar{\omega}t)^4}{4},\qquad
  X = \frac{\omega_0^2}{\bar{\omega}^4e^{4\lambda t}}
      (\alpha P - Q \sin\bar{\omega}t)^3,\\
  \begin{pmatrix}
    \partial H_1/\partial P\\
    -\partial H_1/\partial Q\\
  \end{pmatrix}
  =
  X
  \begin{pmatrix}
    \alpha\\
    \sin\bar{\omega}t
  \end{pmatrix}.
\end{gather*}
Plugging in our zero'th order solution $P_0 = q(0) = q_0$, $Q_0 = -\dot{q}(0) = 0$, we
 have the first order correction
\begin{align*}
  X_0 &= \frac{\omega_0^2q_0^3}{\bar{\omega}^4e^{4\lambda t}}\alpha^3,\\
  \dot{Q}_1 &= \alpha X_0 
  = \frac{\omega_0^2q_0^3}{\bar{\omega}^4e^{4\lambda t}}
    (\bar{\omega}\cos\omega t - \lambda \sin\bar{\omega}t)^4,\\
  \dot{P}_1 &= X_0\sin\bar{\omega}t 
  = \frac{\omega_0^2q_0^3}{\bar{\omega}^4e^{4\lambda t}}
    (\bar{\omega}\cos\omega t - \lambda \sin\bar{\omega}t)^3
    \sin\bar{\omega}t.
\end{align*}
In the limit of no damping $\lambda \rightarrow 0$ we have
\begin{align*}
  \alpha &= \omega_0\cos\omega_0 t\\
  \dot{Q}_1 &= 
  \omega_0^2q_0^3\cos^4\omega_0 t,\\
  \dot{P}_1 &=
  \frac{\omega_0^2q_0^3}{\omega_0}
  \cos^3\omega t\sin\omega_0t,\\
  Q_1 &= 
  \omega_0^2q_0^3\frac{12\omega_0 t + 8 \sin 2\omega_0 t + \sin 4\omega_0 t}
                      {32\omega_0},\\
  \omega_0 P_1 &=
  \omega_0^2q_0^3
  \frac{1-\cos^4 \omega_0 t}{4\omega_0}.
\end{align*}
Converting back to our original coordinates, we have
\begin{align*}
  q &= q_0 + \epsilon \frac{\omega_0 P_1\cos\omega_0 t - Q_1 \sin\omega_0t}{\omega_0}\\
    &= q_0 + \epsilon q_0^3
       \frac{\cos 3\omega_0 t - \cos \omega_0 t - 12 \omega_0 t \sin \omega_0 t}{32}
  ,\\
  p &= p_0 + \epsilon (Q_1\cos\omega_0 t + \omega_0 P_1\sin\omega_0t).
\end{align*}

```{code-cell}
:tags: [hide-cell]

wp = sympy.var(r'\omega_p', positive=True)
H1 = exp(2*lam*t)*w0**2 * cos(wp*t)*q**2/2
disp(H_1=H1.subs(pq_subs).simplify())
disp(('H_{1,P}', H1.subs(pq_subs).diff(P).simplify()),
     ('-H_{1,Q}', H1.subs(pq_subs).diff(Q).simplify()))
```

### Anharmonic Oscillator Response

+++

$$
  V(\theta) = \frac{\omega_0^2}{8\theta_{\min}^2}(\theta^2 - \theta_{\min}^2)^2\\
  V'(\theta) = \frac{\omega_0^2}{2\theta_{\min}^2}\theta(\theta^2 - \theta_{\min}^2)
             = -\alpha \theta + \beta \theta^3.
$$

In terms of the parameters in [Duffing equation][]:

$$
  \alpha = \frac{\omega_0^2}{2}, \qquad
  \beta = \frac{\omega_0^2}{2\theta_{\min}^2},\\
  \omega_0 = \sqrt{2\alpha}, \qquad
  \theta_{\min} = \sqrt{\frac{\alpha}{\beta}}\\
$$

[Duffing equation]: <https://en.wikipedia.org/wiki/Duffing_equation>

```{code-cell} ipython3
x = np.linspace(-1, 1)
alpha, beta, delta, gamma, w = 1, 5, 0.02, 8.0, 0.5
V = -alpha * x**2 / 2 + beta * x**4/4
th_min = np.sqrt(alpha/beta)
plt.plot(x, V)
plt.axvline([th_min])
```

```{code-cell}
#:tags: [hide-cell]

class AnharmonicOscillator(Pendulum):
    """Anharmonic oscillator."""
    v1 = -0.1
    v3 = 1.0
    theta_min = 0.1  # Minimum of double well.
    
    w_d0 = 8.0
    w_d1 = 10.0
    T = 1000.0
    
    theta0 = 0.1
    dtheta0 = 0
    
    def w_d(self, t):
        return self. w_d0 + (self.w_d1 - self.w_d0) * (1-np.cos(2*np.pi * t/self.T))/2
    
    def f_t(self, t):
        return self.f * np.cos(self.w_d(t)*t)
        
    def V_w2(self, theta, d=0):
        """Return the specific potential `V(theta)/m/l/w0**2` and its derivatives."""
        if d == 0:
            return (theta**2 - self.theta_min**2)**2/8/self.theta_min**2
        elif d == 1:
            return theta*(theta**2 - self.theta_min**2)/2/self.theta_min**2
        else:
            raise NotImplementedError(f"{d=}")
            
    def solve_and_plot(self, axs=None, label="", window_shape=5):
        """Solve and plot.  Return `axs`."""
        t, theta, dtheta = self.solve(T=self.T)
        if axs is None:
            fig, axs = plt.subplots(2, 1)
        w0 = self.w0
        
        axs[0].plot(t, theta, label=label)
        
        
        from numpy.lib.stride_tricks import sliding_window_view
        
        theta_view = sliding_window_view(theta, window_shape=window_shape)
        ws = self.w_d(t)
        w_view = sliding_window_view(ws, window_shape=window_shape)
        dtheta = np.max(theta_view, axis=1) - np.min(theta_view, axis=1)
        ws = np.mean(w_view, axis=1)
        axs[1].plot(ws/self.w0, dtheta)
        #ax.plot(w0*t/2/np.pi, theta, label=label)
        #ax.set(xlabel=r"$t/(2\pi/\omega_0)$", 
        #      ylabel=r"$\theta(t)$",
        #      title=", ".join([
        #          fr"$f/\omega_0^2={self.f/w0**2:.2g}\cos({self.w_d/w0:.2g}\omega_0 t)$",
        #          fr"$\omega^2/\omega_0^2=1+{self.h:.2g}\cos({self.w_p/w0:.2g}\omega_0t)$",
        #          fr"$\lambda/\omega_0={self.damping:.2g}$",
        #      ]))
        return axs
    
    def plot_poincare(self, N, skip=0, Np=100, method="BDF", **kw):
        """Plot Poincaré sections for N periods.
        
        Arguments
        ---------
        N : int
            Number of periods to plot.
        Np : int
            Number of points per period.
        skip : int
            Number of periods to skip before plotting.
        """
        assert self.w_d0 == self.w_d1
        w_d = self.w_d0
        T = 2*np.pi / w_d
        y0 = (self.theta0, self.dtheta0)
        t0 = 0
        if skip > 0:
            sol = solve_ivp(
                self.compute_dy_dt, y0=y0, t_span=(t0, T*skip), method=method, **kw)
            y0 = sol.y[:, -1]
            t0 = T*skip
            
        t_eval = t0 + np.arange(N*Np)*T/Np
        sol = solve_ivp(
            self.compute_dy_dt, y0=y0, 
            t_span=(t0, t0 + N*T), 
            t_eval=t_eval, method=method, **kw)
        
        t = sol.t
        theta, dtheta = sol.y.reshape((2, Np, N))
        plt.plot(theta.T, dtheta.T, '.', ms=0.11)
        return (t, theta, dtheta)
            

for theta0 in [0, 0.1, 0.2]:
    # Duffing alpha=1 beta=5 delta=0.02 K=8 omega=0.5
    # alpha = 
    p = AnharmonicOscillator(h=0, f=0.2, damping=0.02, theta_min=0.1, w_d0=2.0, w_d1=2.0, T=20, theta0=theta0)
    #p = AnharmonicOscillator(h=0, f=0.2, damping=0.2, theta_min=0.1, w_d0=2.0, w_d1=2.0, T=20, theta0=theta0)
    #p.solve_and_plot(window_shape=100)
    t, th, dth = p.plot_poincare(1000, skip=5000, Np=200);
```

```

fig, ax = plt.subplots(figsize=(4, 3))
for lam in lams:
    chi = 1/(w0**2 - w**2 + 2j*w*lam)
    l, = ax.semilogy(w/w0, abs(chi*w0**2)**2, label=f"$\lambda={lam:.2g}\omega_0$")
    if lam > 0:
        w_max = np.sqrt(w0**2 - 2*lam**2)
        chi2_max = 1/(4*lam**2*(w0**2-lam**2))
        ax.plot([w_max/w0], [chi2_max*w0**4], 'o', c=l.get_c())

lams = np.linspace(0, w0, 100)[1:-1]
w_max = np.sqrt(w0**2 - 2*lams**2)
chi2_max = 1/(4*lams**2*(w0**2-lams**2))
ax.plot(w_max/w0, chi2_max*w0**4, 'k:')
ax.legend()
ax.set(xlabel=r"$\omega/\omega_0$",
       ylabel=r"$|\chi|^2\omega_0^4$",
       ylim=(0, 40))

if glue: glue("fig:LinearResponse", fig);
plt.tight_layout()
fig.savefig(FIG_DIR / "LinearResponse.svg")
```


[Fine-structure constant]: <https://en.wikipedia.org/wiki/Fine-structure_constant>
[Borel resummation]: <https://en.wikipedia.org/wiki/Borel_summation>
[Big $O$ notation]: <https://en.wikipedia.org/wiki/Big_O_notation>
