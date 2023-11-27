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

```{code-cell}
:tags: [hide-input]

import mmf_setup; mmf_setup.nbinit(quiet=True)
import logging;logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
```

# Assignment 6: Hamiltonian Dynamics

+++

**Due: 11:59pm Friday 4 December 2023**

## Arbitrary Dispersion: Negative Mass and Special Relativity

Consider a particle moving in 1D with  kinetic energy $E(p)$ moving under a constant
force $F$.  Use Hamilton's equations to find the general solution for the position
$x(t)$ of the particle.  Check your answer with the familiar solution for $E(p) =
p^2/2m$.  Discuss the physical meaning of $E'(p)$ and $E''(p)$ in terms of Newton's law
and the particle motion.

:::{note}
  This approach also works for $E(p) = \sqrt{p^2c^2 + m^2c^4}$ where $c$ is the speed of
  light. This gives the motion of a particle under constant force in special relativity
  The corresponding coordinate transformation into a co-moving constantly accelerating
  frame gives rise to [Rindler
  coordinates](https://en.wikipedia.org/wiki/Rindler_coordinates), which are applicable
  close to the surface of the earth.  These have interesting properties associated with
  general relativity, including time-dilation at different heights, and an event horizon
  at a distance $d=mc^2/F$ below the observer.  For the earth, this distance is about
  $d\approx0.3$[pc](https://en.wikipedia.org/wiki/Parsec), well past the limit where the
  approximation of a constant gravitational field breaks down. 
:::


## Hamilton Jacobi Equation

Your assignment is to analyze the motion of a harmonic oscillator using the Hamiltonian
formalism.  Please follow the outline given below for analyzing a free particle and
complete the same type of analysis for the harmonic oscillator.

### Free Particle (Sample Analysis)

#### Lagrangian Analysis

*Analyze the problem using the Lagrangian formalism.*

1. We use the generalized coordinate $q = x$ and velocity $\dot{q} = v$ so that the Lagrangian is:

   $$
     \mathcal{L}(q, \dot{q}, t) = \frac{m}{2}\dot{q}^2.
   $$

2. The canonical momentum is:

   $$
     p = \pdiff{\mathcal{L}}{\dot{q}} = m\dot{q}.
   $$

3. The Euler-Lagrange equation is:

   $$
     \dot{p} = m\ddot{q} = \pdiff{\mathcal{L}}{q} = 0
   $$

4. The general solution is

   $$
      \dot{q}(t) = v, \qquad
      q(t) = q_0 + v_0t.
   $$

+++

#### Hamiltonian Analysis

+++

1. From above, we can form the Hamiltonian using the Legendre transform.  First we invert $p = m\dot{q}$ to find $\dot{q} = p/m$, then we perform the Legendre transformation:

   $$
     H(q, p, t) = p \dot{q}(p) - \mathcal{L}(q, \dot{q}(p), t) = \frac{p^2}{2m}.
   $$

2. Now we can express the problem in terms of Hamilton's equation of motion:

   \begin{align*}
     \begin{pmatrix}
       \dot{q}\\
       \dot{p}
     \end{pmatrix}
     &=
     \begin{pmatrix}
       0 & 1\\
       -1 & 0
     \end{pmatrix}
     \cdot
     \begin{pmatrix}
       \pdiff{H}{q}\\
       \pdiff{H}{p}
     \end{pmatrix}
     =
     \begin{pmatrix}
       0 & 1\\
       -1 & 0
     \end{pmatrix}
     \cdot
     \begin{pmatrix}
       0\\
       p/m
     \end{pmatrix}\\
     &=
     \frac{1}{m}
     \begin{pmatrix}
       0 & 1\\
       0 & 0
     \end{pmatrix}
     \cdot
     \begin{pmatrix}
       q\\
       p
     \end{pmatrix}.
   \end{align*}

3. The solution can be expressed as:

   \begin{align*}
    \begin{pmatrix}
       q\\
       p
     \end{pmatrix}
     &=
     e^{\left(\begin{smallmatrix} 0 & 1\\ 
                                  0 & 0\end{smallmatrix}\right)\frac{t}{m}}
    \cdot
    \begin{pmatrix}
       q_0\\
       p_0
     \end{pmatrix}                                 
     =
     \left(
     \mat{1} 
     + \begin{pmatrix} 
         0 & 1\\ 
         0 & 0
       \end{pmatrix}\frac{t}{m}
     \right)
    \cdot
    \begin{pmatrix}
       q_0\\
       p_0
     \end{pmatrix}
     =
     \begin{pmatrix} 
       1 & \frac{t}{m}\\ 
       0 & 1
     \end{pmatrix}
    \cdot
    \begin{pmatrix}
       q_0\\
       p_0
     \end{pmatrix}\\
     &=
    \begin{pmatrix}
       q_0 + p_0t/m\\
       p_0
     \end{pmatrix}                                 
   \end{align*}
   
   where we have used the fact that $\mat{A}^n = \mat{0}$ for $n>1$ where
   $\mat{A}=\bigl(\begin{smallmatrix} 0 & 1\\ 0 & 0\end{smallmatrix}\bigr)$.
   
   
4. The Hamilton-Jacobi equation is:

   $$
     H\left(q, \pdiff{S}{q}, t\right) + \pdiff{S}{t}  
     = \frac{1}{2m}\left(\pdiff{S}{q}\right)^2 + \pdiff{S}{t} = 0.
     = \frac{1}{2m}S_{,q}^2 + S_{,t} = 0.
   $$
   
5. This equation is separable, and we may place the $q$'s on one side, and the $t$'s on the other to obtain:

   $$
     \frac{1}{2m}S_{,q}^2 = E = -S_{,t}.
   $$
   
   Integrating each side, we obtain:
   
   $$
      S(q, t) = \sqrt{2mE} q + f(t), \qquad
      S(q, t) = W(q) - Et,
   $$
   
   where $f(t)$ and $W(q)$ are the integration constants of each of the pieces.  The solution is thus:
   
   $$
     S(q,t) = \sqrt{2mE}q - Et + S_0.
   $$
   
   Note that $W(q) = \sqrt{wmE}q + S_0$ is called "*Hamilton's characteristic function*" (i.e. in Fetter and Walecka) or sometimes the "*abbreiviated action*" (Landau and Lifshitz) and the form $S(q,t) = W(q) - Et$ is always valid when $H(q, p, t) = H(q, p)$ is independent of time.

6. We are free to choose any new coordinate $Q$ as long as the invertability requirement still holds:

   $$
     \left(\frac{\partial^2 S}{\partial q \partial Q}\right) = \sqrt{\frac{m}{2E(Q)}}E'(Q) \neq 0.
   $$
   
   Since the new Hamiltonian $H'(Q, P, t) = H(q, p, t) + S_{,t} = 0$ by construction, the equations of motion are $\dot{Q} = \dot{P} = 0$ and $P$ and $Q=E$ are constants of motion.
   
   A convenient choice is $E(Q) = Q$: i.e. introducing the energy $E$ as the new coordinate.
   
7. After this choice is made, we have the *generating function* for the canonical transformation:

   $$
     S(q, Q, t) = \sqrt{2mQ}q - Qt + S_0.
   $$   
     
   The canonical momentum follows from the following, which may be inverted to express $q(Q, P, t)$:
   
   $$
     P = - \pdiff{S(q,Q, t)}{Q} = t - \sqrt{\frac{m}{2E}} q\\
     q(Q, P, t) = \sqrt{\frac{2E}{m}}(t - P) = v_0(t - t_0) = q_0 + v_0t, \\
     p(Q, P, t) = S_{,q} = \sqrt{2mE} = mv_0.
   $$
   
   Thus, we see that the canonical momentum $P$ to the energy $Q=E$ is the initial time $t_0$.
   
8. Inverting these, we have the explicit canonical transformation:

   $$
     Q(q, p, t) = E = \frac{p^2}{2m}, \qquad
     P(q, p, t) = t - \sqrt{\frac{m}{2E}}q = t - \frac{mq}{p}.
   $$
   
   We can now explicitly check that the Poisson bracket (I am using the convention of
   Landau and Lifshitz here) satisfies the canonical commutation relationships:
   
   $$
     [P, Q] = \pdiff{P}{p}\pdiff{Q}{q} - \pdiff{P}{q}\pdiff{Q}{p}
            = \frac{mq}{p^2}\cdot 0 - \left(-\frac{m}{p}\right)\cdot \frac{p}{m}
            = 1.
   $$

9. *(This analysis is a little harder to do for the oscillator, so do not feel you have
   to do it.)* Armed with the solutions, we may construct the action function for the
   path connecting $(q_0, t_0)$ to $(q_1, t_1)$:
   

   $$
     q(t) = q_0 + \frac{q_1 - q_0}{t_1 - t_0}(t-t_0), \qquad
     \dot{q}(t) = \frac{q_1 - q_0}{t_1 - t_0}.     
   $$
   
   Hence, we have the action:
   
   $$
     \bar{S}(q_0, t_0; q, t) = \int_{t_0}^{t} \mathcal{L}\d{t}
     = \int_{t_0}^{t} \frac{m}{2}\frac{(q_1 - q_0)^2}{(t_1 - t_0)^2}\d{t}
     = \frac{m}{2}\frac{(q_1 - q_0)^2}{t - t_0}.
   $$
   
   This allows us to construct the general solution to any initial-value problem for the
   Hamilton-Jacobi equation:
   
   $$
     H(q, S_{,q}(q, t), t) + S_{,t}(q,t) = 0, \qquad S(q, t_0) = S_0(q)
   $$
   
   as
   
   $$
     S(q, t) = S_0(q_0) + \int_{t_0}^{t} L(q(t), \dot{q}(t), t)\;\d{t}
   $$
   
   where the action is computed over the trajectory starting from $q(t_0) = q_0$ with
   initial momentum $p_0 = S_0'(q_0)$ and ending at $q(t) = q$ at time $t$.  For this
   problem $p_0 = mv_0$ so we have 
   
   $$
     q(t) = q_0 + v_0(t - t_0) = q_0 + \frac{S'_0(q_0)}{m}(t - t_0)
   $$
   
   which must be inverted to find $q_0 = q_0(q, t, t_0)$.  The explicit solution here is expressed in terms of this:
   
   $$
     S(q, t) = S_0(q_0) + \frac{1}{2m}[S'_0(q_0)]^2 (t - t_0).
   $$
   
   Since $H$ is independent of time, we can take $t_0 = 0$ without loss of generality.  Now, consider an example problem from Arnold where he asks for the solution to this problem with initial conditions $S_0(q) = \frac{mq^2}{2T}$ (though he chooses units where $m=T=1$).  This can be explicity constructed:
   
   $$
     S'_0(q_0) = \frac{m q_0}{T} = mv_0, \qquad
     q = q_0 + \frac{q_0}{T}t, \qquad
     q_0 = \frac{q}{1 + \frac{t}{T}} = \frac{q T}{T + t}.
   $$
   
   The explicit solution is thus
   
   $$
     S(q, t) = \frac{m q^2 T}{2(T + t)^2} + \frac{1}{2m}\left(\frac{mq}{T + t}\right)^2 t
     =  \frac{mq^2}{2(T + t)}.
   $$
   
   Note that this does *not* have the same form as the separable solution we constructed above.  This is due to the choice of initial conditions $S_0(q)$.  In particular, our separable solution corresponds with the initial conditions $S_0(q) = m v_0 q$ instead.  Mathematically, however, the Hamilton-Jacobi equations can be solved with any arbitrary initial conditions.

+++

### Harmonic Oscillator

+++

Your assignment is to repeat a similar analysis with the harmonic oscillator.

+++

#### Lagrangian Analysis

+++

*Analyze the problem using the Lagrangian formalism.*

1. Use the generalized coordinate $q = x$ and velocity $\dot{q} = v$ so that the Lagrangian is:

   $$
     \mathcal{L}(q, \dot{q}, t) = \frac{m}{2}\dot{q}^2 - \frac{k}{2}q^2.
   $$

2. Find the the canonical momentum?

   $$
     p = ?
   $$

3. Write the Euler-Lagrange equation:

   $$
     \dot{p} = ?
   $$

4. Write down the general solution:

   $$
      q(t) = ?
   $$

+++

#### Hamiltonian Analysis

+++

*Analyze the problem using the Hamiltonian formalism*.

1. Use the Legendre transform to write the Hamiltonian:

   $$
     H(q, p, t) = ?
   $$

2. Now we can express the problem in terms of Hamilton's equation of motion:

   $$
     \begin{pmatrix}
       \dot{q}\\
       \dot{p}
     \end{pmatrix}
     = ?
     = \mat{A}\cdot
    \begin{pmatrix}
       q\\
       p
     \end{pmatrix}.
   $$
   
3. The solution can be expressed as:

   $$
    \begin{pmatrix}
       q\\
       p
     \end{pmatrix}
     =
     e^{\mat{A}t}
    \cdot
    \begin{pmatrix}
       q_0\\
       p_0
     \end{pmatrix}                                 
     = ?
   $$
   
   where you have used the properties of the matrix $\mat{A}^n$ to explicitly compute the matrix exponential.

4. Write the Hamilton-Jacobi equation is:

   $$
     H\left(q, \pdiff{S}{q}, t\right) + \pdiff{S}{t}  = ? = 0.
   $$
   
5. This equation is separable, and we may place the $q$'s on one side, and the $t$'s on
   the other.  Connect the two equations with the constant $E$ and integrate each side
   to obtain both the *abbreviated action $W(q)$ and the generating function $S(q,t)$:

   $$
      W(Q) = ?, \\
      S(q, t) = W(q) - Et = ?.
   $$
   
   *Note: to complete the integrals here you will probably want to make a trignometric
   substitution $\sqrt{k/2E}q = \sin\theta$.* 
   
6. You are free to choose any new coordinate $Q$ as long as the invertability
   requirement still holds.  Compute this a choose a reasonable coordinate $Q$: 

   $$
     \left(\frac{\partial^2 S}{\partial q \partial Q}\right) = ? \neq 0.
   $$
   
7. Now express the *generating function* $S(q,Q,t)$ for the canonical transformation in terms of your chosen coordinate:

   $$
     S(q, Q, t) = ?.
   $$   
     
   Use this to determine the canonical momenta:
   
   $$
     P = ?\\
     q(Q, P, t) = ?\\
     p(Q, P, t) = ?.
   $$

8. Invert these to get the explicit canonical transformation:

   $$
     Q(q, p, t) = ?\\
     P(q, p, t) = ?.
   $$
   
   Explicitly check that the Poisson bracket (I am using the convention of Landau and Lifshitz here) satisfies the canonical commutation reationships:
   
   $$
     [P, Q] = ? = 1.
   $$

9. Armed with the generating function designed so that $H'(Q, P, t) = 0$, argue that $Q$ and $P$ are constant, and hence use your result from step 7. to write down the complete solution and compare with your previous results.

   $$
     q(t) = ?\\
     p(t) = ?.
   $$

10. Since this problem is both separable and periodic, we can express the solution in terms of *action-angle variables*.  Using this formalism, compute the fundamental frequency of the system.

11. *(Bonus)* Use the same action-angle formalism to compute the fundamental frequencies for the simple pendulum of mass $m$ on a massless rod of length $l$, which has the following Hamiltonian expressed in terms of the angle $q=\theta$ from the vertical:

  $$
    H(q,p) = \frac{p^2}{2ml^2} + mgl(1 - \cos q).
  $$
  
  Hints:
  1. Don't try to analytically evaluate the integrals – just get the answer expressed in terms of definite integrals.  
  2. Remember that there are two types of periodic orbits here: *librations* (if $1 - \cos q < 1$ always) and *rotations* where $q$ takes on all values.  These will have different endpoints in your integrals.
