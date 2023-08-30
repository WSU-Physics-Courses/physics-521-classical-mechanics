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
import logging;logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
```

(sec:PendulumWorksheet)=
# The Pendulum

The ideal simple pendulum -- a massless rigid rod of length $l$ with a point mass $m$
fixed to move about a pivot -- is a familiar and somewhat intuitive physical systems
that contains within it a wealth of physics.

## Intuitive Questions

You should be able to justify these intuitive questions using the formal results derived
below.  Use the formalism to test and check your intuition.  For all of these questions,
be prepared to explain you answer both quantitatively using the formal results, and
intuitively.

1. How does the oscillation frequency of a pendulum depend on its amplitude?
2. Your frisbee is stuck in a tall narrow tree.  It is fairly precariously balanced, so
   you are sure you can recover it if you shake the tree hard enough.  You give the tree
   a nudge and notice the natural oscillation frequency.  Should you vibrate the tree at
   a slightly higher frequency or a slightly lower frequency to maximize the chance of
   recovering your frisbee?
3. You have a tall vase on your table.  If you vibrate the table at the resonant
   frequency of the vase, it might fall over.  Do you need to be concerned about
   vibrating the table at any other frequency, like twice or half of the resonant
   frequency?
4. A pendulum with very little friction sits at equilibrium.  You can vibrate the
   pendulum at a fixed frequency of your choice (with low amplitude), but you cannot
   change the frequency once you choose it.  Can you get the pendulum to complete a full
   revolution (swing over the top)?
5. Describe how the amplitude of an oscillating pendulum with low damping changes if you
   slowly change the length of the pendulum.  (E.g., consider a mass oscillating at the
   end of a long rope as you pull the rope up through a hole.) What about if you have a
   pendulum of fixed length in an elevator that gradually changes acceleration? (Is this
   the same problem, or is there a difference?)
6. Consider balancing a ruler standing on its end.  You probably know that you can
   balance it if you allow your hand to move laterally - i.e. chasing the ruler as it
   starts to fall.  Can you somehow balance the ruler by only moving your hand up and
   down?  (You can clasp the ruler so that you can push it up and pull it down, but your
   fingers do not have enough strength to apply significant torque to the ruler if you
   cannot move your hand laterally.)

## Model
The complete model we shall consider is

\begin{gather*}
  m\ddot{\theta} = -\frac{mg}{l}\omega^2\sin\theta - 2m\lambda\dot{\theta} + \Gamma
\end{gather*}
where all quantities might possibly depend on time:
:::{margin}
*Here we also specify their physical dimensions in terms of mass $M$, distance $D$, and
time $T$.*
:::
* $[\theta] = 1$: Angle of the pendulum from equilibrium (hanging down).  This is the
  dynamical variable.  When we generalize the problem, or discuss the small amplitude
  limit, we shall use the general coordinate notation $q=\theta$.
* $[m] = M$: Mass of the pendulum bob.  As there is only one dynamical mass in this
  problem, we will remove it below.
* $[g] = D/T^2$: Acceleration due to gravity, also known as **the gravitational field**.
* $[l] = D$: Length of the pendulum.
* $[\lambda] = 1/T$: Damping, either due to friction at high speed, wind resistance,
  or dragging an object through a viscous fluid.  Note: this is **not** standard $F_f =
  \mu F_N$ friction (which makes another interesting problem).
* $[\Gamma] = M/T^2$: Driving torque ($F = l\Gamma$ if the force is directed
  perpendicular to the pendulum rod).

Anticipating the solution in the small amplitude limit, we shall cancel the common
factor of mass and write this as: 
\begin{gather*}
  \ddot{\theta} + 2\lambda\dot{\theta} + \omega^2_0\sin\theta = f, 
  \qquad \omega^2_0 = \frac{g}{l}, \qquad f = \frac{\Gamma}{m}.
\end{gather*}
* $[\omega_0] = 1/T$: Only the combination $g/l$ has physical significance for this
  system if the driving force is appropriately defined.  This is sometimes called the
  natural resonant frequency of the system.  It is the angular frequency in the small
  amplitude limit with no damping: i.e. the harmonic oscillator.
* $[f] = 1/T^2$: Driving force/torque expressed as an acceleration, torque per unit
  mass, or force per unit mass and unit length.

With these re-definitions, everything is specified in terms of time/frequency.  One
might further define $\omega_0=1$ or similar rendering the problem dimensionless, but we
will not do so here as it obscures some of the physics.

Many more details will be discussed in {ref}`sec:PendulumExample`, but provide here a
set of problems for you to work through to test your understanding of mechanics. I have
tried to express these simply so that you can try to attack them without knowing
specific techniques like the Hamilton-Jacobi equation.  My hope is that, after trying
these and making mistakes, you will appreciate better the power of some of the more
formal techniques of classical mechanics.

## Harmonic Oscillator

Here we consider small amplitude oscillations $\theta \ll 1$, keeping only the first
term in
\begin{gather*}
  \sin\theta = \theta - \frac{\theta^3}{3!} + O(\theta^5).
\end{gather*}

:::{margin}
Where there is only one frequency in a problem like this, we will drop the subscript
$\omega_0 \equiv \omega$.
:::
:::{doit}
Find the general solution for a harmonic oscillator:

\begin{gather*}
  \ddot{q} + \omega^2 q = 0, \qquad
  q(0) = q_0, \qquad
  \dot{q}(0) = \dot{q}_0.
\end{gather*}
:::

:::{solution}
\begin{gather*}
  q(t) = q_0\cos\omega t + \frac{\dot{q}_0}{\omega}\sin \omega t.
\end{gather*}
:::

:::{solution} Express the Hamiltonian and Lagrangian formulations.
\begin{gather*}
  T = \frac{\dot{q}^2}{2}, \qquad V = \frac{\omega^2q^2}{2}, \qquad
  L(q, \dot{q}, t) = T - V = \frac{\dot{q}^2}{2} - \frac{\omega^2q^2}{2}\\
  p = L_{,\dot{q}} = \dot{q}, \qquad
  H = p\dot{q} - L = \frac{p^2}{2} + \frac{\omega^2q^2}{2}.
\end{gather*}
:::

### Damped Harmonic Oscillator

Make sure you understand how to solve general homogeneous linear ODEs like this one.

:::{margin}
You should know this in your sleep.  If not, study these thoroughly.
:::
:::{margin}
This checks your ability to solve a homogeneous second-order linear differential
equation.
:::
:::{doit}
Find the general solution for a damped harmonic oscillator:

\begin{gather*}
  \ddot{q} + 2\lambda \dot{q} + \omega^2 q = 0, \qquad
  q(0) = q_0, \qquad
  \dot{q}(0) = \dot{q}_0.
\end{gather*}
:::

:::{solution}
\begin{gather*}
  q(t) = e^{-\lambda t}\left(
    q_0\cos\bar{\omega} t 
    + \frac{\dot{q}_0 + \lambda q_0}{\bar{\omega}}\sin \bar{\omega} t
  \right), \qquad
  \bar{\omega} = \sqrt{\omega^2 - \lambda^2}.
\end{gather*}
:::

:::{solution} Express the Hamiltonian and Lagrangian formulations.
\begin{gather*}
  T = \frac{\dot{q}^2}{2}, \qquad V = \frac{\omega^2q^2}{2}, \qquad
  L(q, \dot{q}, t) = e^{2\lambda t}
  \left(\frac{\dot{q}^2}{2} - \frac{\omega^2q^2}{2}\right)\\
  p = L_{,\dot{q}} = e^{2\lambda t}\dot{q}, \qquad
  H = p\dot{q} - L = e^{-2\lambda t}\frac{p^2}{2} + e^{2\lambda t}\frac{\omega^2q^2}{2}.
\end{gather*}

This is probably not obvious.  Check explicitly that you get the correct equation of
motion.

\begin{gather*}
  \overbrace{2\lambda e^{2\lambda t}\dot{q} + e^{2\lambda t}\ddot{q}}^{\dot{p}} 
  = \overbrace{-e^{2\lambda t}\omega^2q}^{L_{,q}},\\
  2\lambda \dot{q} + \ddot{q} = -\omega^2q.
\end{gather*}
:::

### Driven Damped Harmonic Oscillator

:::{margin}
This checks your ability to solve an inhomogeneous second-order linear differential
equation.
:::
:::{doit}
Find the general solution for a damped harmonic oscillator:

\begin{gather*}
  \ddot{q} + 2\lambda \dot{q} + \omega^2 q = f(t).
\end{gather*}

Make sure you have a general solution, but don't worry about expressing the
coefficients.  Your answer should be expressed in terms of various integrals.
:::

:::{solution}
The general solution can be written as
\begin{multline*}
  q(t) = 
  e^{-\lambda t}\cos \bar{\omega} t
  \int \frac{-f(t)e^{-\lambda t}\sin\bar{\omega}t}{\bar{\omega}}\;\d{t}\\
  + 
  e^{-\lambda t}\sin \bar{\omega} t
  \int \frac{f(t)e^{-\lambda t}\cos\bar{\omega}t}{\bar{\omega}}\;\d{t}.
\end{multline*}
The constants of integration provide the general solution.

This follows from the fact that, given two independent solutions $A(t)$ and $B(t)$ of a
homogeneous linear differential equation, the solution $q(t)$ to an inhomogeneous
equation can be formed as follows:
\begin{gather*}
  \ddot{A} + g_1(t)\dot{A} + g_0(t) A = 0,\\
  \ddot{B} + g_1(t)\dot{B} + g_0(t) B = 0,\\
  \ddot{q} + g_1(t)\dot{q} + g_0(t) q = f(t),
\end{gather*}
\begin{gather*}
 q = aA + bB,\\
 \dot{a} = -\frac{fB}{W},\qquad
 \dot{b} = \frac{fA}{W},\\
  W = A\dot{B} - B\dot{A}.
\end{gather*}

To check, just compute:
\begin{gather*}
  q = aA + bB,\\
  \dot{q} = \underbrace{\dot{a}A + \dot{b}B}_{0} + a\dot{A} + b\dot{B}
          = a\dot{A} + b\dot{B},\\
  \ddot{q} = \underbrace{\dot{a}\dot{A} + \dot{b}\dot{B}}_{f} + a \ddot{A}  + b\ddot{A}
           = f + a \ddot{B} + b\ddot{A}.
\end{gather*}
The key is to notice that the remaining terms when inserted into the original equation,
simply fulfill the homogeneous equation, with the inhomogeneous term leftover.
\begin{multline*}
  \ddot{q} + g_1(t)\dot{q} + g_0(t) q = 
  a\overbrace{(\ddot{A} + g_1(t)\dot{A} + g_0(t)A)}^{0} \\ 
  +
  b\underbrace{(\ddot{B} + g_1(t)\dot{B} + g_0(t)B)}_{0}
  + 
  f(t) = f(t).
\end{multline*}
The key is to choose the coefficients in each step to eliminate the derivatives of the
coefficients, and in the last step, reproduce the inhomogeneous term. For a second-order
differential equation, this gives the following two-dimensional linear system, which can
be solved by Cramer's rule
\begin{gather*}
  \begin{pmatrix}
    A & B\\
    \dot{A} & \dot{B}
  \end{pmatrix}
  \begin{pmatrix}
    \dot{a}\\
    \dot{b}
  \end{pmatrix}
  =
  \begin{pmatrix}
    0\\
    f
  \end{pmatrix}, \\
  \begin{pmatrix}
    \dot{a}\\
    \dot{b}
  \end{pmatrix}
  =
  \begin{pmatrix}
    A & B\\
    \dot{A} & \dot{B}
  \end{pmatrix}^{-1}
  \begin{pmatrix}
    0\\
    f
  \end{pmatrix}
  =
  \frac{1}{A\dot{B} - B\dot{A}}
  \begin{pmatrix}
    \dot{B} & -B\\
    -\dot{A} & A
  \end{pmatrix}
  \begin{pmatrix}
    0\\
    f
  \end{pmatrix}.
\end{gather*}
This method easily generalizes to higher order.
:::

:::{margin}
This is called the linear response because the amplitude $a$ depends linearly on the
driving acceleration $f$.  For the harmonic oscillator, this linear dependence works for
all amplitudes, but in other systems, this linear behaviour generally only applies for
small amplitude driving forces $f$.

We change notations here slightly so that $\omega_0$ is the natural resonance of the
system, and $\omega$ is the drive frequency.
:::
:::{doit}
Compute the linear response $\chi$ of the damped harmonic oscillator:
\begin{gather*}
  \ddot{q} + 2\lambda \dot{q} + \omega_0^2 q = f \cos \omega t.
\end{gather*}
After a time long enough for any transients to die off, the system will simply
oscillate with some amplitude $a$ at the drive frequency $\omega$ but with a phase
offset $\phi$ form the driving force:
\begin{gather*}
  q(t) = a\cos(\omega t + \phi).
\end{gather*}
The density linear response is the complex quantity
\begin{gather*}
  \chi = \frac{a}{f}e^{\I\phi}.
\end{gather*}
Plot $\abs{\chi}^2$ as a function of $\omega$.  At what angular frequency $\bar{\omega}$
is this a maximum?  What is the phase shift $\phi$ at this maximum?
:::

::::{solution}
This problem is most easily solved using Fourier techniques.  The equations of motion
can be expressed as the real part of
\begin{gather*}
  \ddot{q} + 2\lambda \dot{q} + \omega_0^2 q = f e^{\I \omega t}, \qquad
  q(t) = \chi f e^{\I\omega t} = a e^{\I \omega t + \phi},\\
  (-\omega^2 + 2 \I\omega \lambda  + \omega_0^2)\chi fe^{\I\omega t} = fe^{\I\omega t},\\
  \chi = \frac{1}{\omega_0^2 - \omega^2 + 2 \I \omega \lambda}
  = \frac{\omega_0^2 - \omega^2 - 2 \I\omega \lambda}
         {(\omega_0^2 - \omega^2)^2 + 4\omega^2\lambda^2},\\
  \abs{\chi}^2 = \frac{1}{(\omega_0^2 - \omega^2)^2 + 4\omega^2\lambda^2},
  \qquad
  \tan\phi = \frac{2\omega\lambda}{\omega^2 - \omega_0^2}.
\end{gather*}
Completing the square, we have
\begin{gather*}
  \abs{\chi}^2 = \frac{1}
  {\bigl(\omega^2 - (\omega_0^2 - 2\lambda^2)\bigr)^2 
   + 4\lambda^2(\omega_0^2 - \lambda^2)}.
\end{gather*}
Note that the resonance peak occurs at the shifted frequency
\begin{gather*}
  \omega_{\max}^2 = \omega_0^2 - 2\lambda^2 \leq \omega_0^2,\qquad
  \abs{\chi_{\max}}^2 = \frac{1}
  {4\lambda^2(\omega_0^2 - \lambda^2)}.
\end{gather*}
which is not the same as the natural frequency $\bar{\omega}^2 = \omega_0^2 - \lambda^2$
of the damped oscillator.  At resonance, the phase shift is
\begin{gather*}
  \tan\phi = -\sqrt{\frac{\omega_0^2}{\lambda^2} - 2}.
\end{gather*}

Note that if the damping is small $\lambda \ll \omega_0$, then
\begin{gather*}
  \tan\phi \approx -\frac{\omega_0}{\lambda}, \qquad
  \phi \approx -\frac{\pi}{2} + \frac{\lambda}{\omega_0} 
    + O\left(\frac{\lambda^2}{\omega_0^2}\right).
\end{gather*}

:::{glue:figure} fig:LinearResponse
Linear response of a damped harmonic oscillator for various amounts of damping.  The
dotted line shows $\abs{\chi_\max}^2$ vs $\omega_{\max}$.
:::
::::

```{code-cell}
:tags: [hide-cell]

try: from myst_nb import glue
except: glue = None

w0 = 1.0
w = np.linspace(0, 2*w0, 500)
lams = [0, 0.1, 0.2, 0.3]

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
```

## Anharmonic Oscillator

:::{margin}
For the pendulum, the first correction is $\epsilon = -1/3! = -1/6$.
:::
En-route to a pendulum, consider adding an anharmonic perturbation:
\begin{gather*}
  \ddot{q} + 2\lambda \dot{q} + \omega_0^2(q + \epsilon q^3) = 0.
\end{gather*}

:::{doit}
Treating $\epsilon \ll 1$ as a small parameter, use some sort of perturbation theory to
estimate how this perturbation changes the natural frequency of the oscillations to
lowest order in $\epsilon$.  If you were to compute higher order corrections
(i.e. express $\omega$ as a power series in $\epsilon$), what would you expect the
radius of convergence in $\epsilon$ to be?
:::
:::{solution}
:show:

First some qualitative discussion. The corresponding potential is
\begin{gather*}
  V(q) = \omega_0^2\frac{q^2}{2} + \epsilon \omega_0^2 \frac{q^4}{4}.
\end{gather*}
This becomes more and more confining as the amplitude grows, so I would expect that the
frequency should increase with increasing amplitude.

Suppose that we could compute the power-series for $\omega = \omega_0 + \epsilon
\omega_1 + \epsilon^2 \omega_2 + \cdots$ where $\omega_{n\geq 1}(E)$ depend on the
amplitude of the oscillations.  If this converges, it should converge in a
region of the complex plane with $\abs{\epsilon} < R_\epsilon$ where $R_\epsilon$ is the
radius of convergence.  However, if $\epsilon < 0$, then the potential is unbounded
below, and once oscillations become sufficiently large, there likely ceases to be a
reasonable frequency.  This might indicate that the radius of converges is zero
(i.e. the series is an divergent asymptotic series.)

Starting with the case of $\lambda = 0$ and applying naïve perturbation theory to the
solution with initial conditions $q(0) = q_0$ and $\dot{q}(0) = 0$ gives the solution
(cf. {cite:p}`Fetter:2006` (7.29))
\begin{gather*}
  q(t) = q_0\cos \omega_0 t + \epsilon\omega_0^2 \frac{q_0^3}{32}(
    \cos 3\omega_0 t - \cos\omega_0 t - 12 \omega t \sin \omega_0 t)
  + O(\epsilon^2).
\end{gather*}
Expanding for small $t$, we have 
\begin{gather*}
  q(t) = q_0 - \frac{\omega_0^2 t^2}{2} + \epsilon\omega_0^4 \frac{q_0^3}{2}t^2 
  + O(\epsilon^2)
\end{gather*}
suggesting that
\begin{gather*}
  \omega^2 \approx \omega_0^2\left(1 - \epsilon \omega_0^2 q_0^3\right)Re
\end{gather*}

Instead, we apply canonical perturbation theory using the classical action
\begin{gather*}
\end{gather*}


:::









[Fine-structure constant]: <https://en.wikipedia.org/wiki/Fine-structure_constant>
[Borel resummation]: <https://en.wikipedia.org/wiki/Borel_summation>
[Big $O$ notation]: <https://en.wikipedia.org/wiki/Big_O_notation>
