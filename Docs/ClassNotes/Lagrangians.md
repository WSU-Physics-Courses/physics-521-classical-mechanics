Lagrangian Dynamics
===================

```{contents} Contents
:local:
:depth: 3
```

## When is $L = K - V$?

I make a big deal about how the prescription of forming the Lagrangian $L = K-V$ as the
difference between kinetic and potential energies is only valid for Newtonian
mechanics.  How does one know if this is appropriate?

The way I think about it is to imagine writing the full Lagrangian for all of the
particles in terms of their Cartesian inertial coordinates $\vect{r}_i = (x_i, y_i,
z_i)$, with all other complications inserted as constraints with Lagrange multipliers
etc.  If, in this case, the kinetic energy is quadratic in the velocities, then you can
use the $L = K - V$ prescription after transforming to whatever (non-singular)
coordinates make the problem easiest to solve:

\begin{gather*}
  K = \frac{1}{2}
  \vect{v}^T\cdot
  \mat{M}
  \cdot
  \vect{v}, \qquad
  \vect{v} = \begin{pmatrix}
    \dot{x}_0\\
    \dot{y}_0\\
    \dot{z}_0\\
    \dot{x}_1\\
    \dot{y}_2\\
    \dot{z}_3\\
    \vdots\\\
    \dot{x}_{N-1}\\
    \dot{y}_{N-1}\\
    \dot{z}_{N-1}
  \end{pmatrix}.
\end{gather*}

Note: the "mass matrix" $\mat{M}$ does not need to be diagonal.

To see how this fails, consider special relativity.  Here, it is natural to start from
the Hamiltonian formulation with the relativistic energy-momentum relation (here, with
no potential $V(x)=0$)

\begin{gather*}
  H(x, p) = E(p) = \sqrt{p^2c^2 + m^2c^4}, \qquad
  v = \dot{x} = E'(p) = \frac{pc^2}{\sqrt{p^2c^2 + m^2c^4}}.
\end{gather*}

Solving for $p(v)$ we have the relativistic kinetic energy in terms of the time-dilation
factor $\gamma(v/c)$:

\begin{gather*}
  p = mv\sqrt{1 - \frac{v^2}{c^2}}, \qquad
  E(v) = mc^2\overbrace{\frac{1}{\sqrt{1-\frac{v^2}{c^2}}}}^{\gamma(v/c)}.
\end{gather*}

However, with a little bit of work, one can find that the [Lagrangian for a relativistic
particle](https://en.wikipedia.org/wiki/Relativistic_Lagrangian_mechanics) should
actually be:

\begin{gather*}
  L[x, \dot{x}] = p\dot{x} - H = -mc^2\sqrt{1-\frac{\dot{x}^2}{c^2}} - V(x),
\end{gather*}

which can be obtained via the Legendre transform, but is not of the form $E(v) - V(x)$.
