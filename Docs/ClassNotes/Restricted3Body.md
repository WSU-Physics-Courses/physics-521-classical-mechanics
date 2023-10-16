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
:tags: [hide-cell]

%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
import mmf_setup;mmf_setup.nbinit()
```

(sec:Restricted3BodyProblem)=
Restricted 3-Body Problem
=========================

```{contents} Contents
:local:
:depth: 3
```

The restricted 3-body problem considers the motion of a satellite between to orbiting
bodies like the Sun and the Earth, or the Earth and the Moon.  It makes the
approximation that the two large objects orbit each other with circular orbits, so that
they are stationary in a rotating frame centered on their center of mass with angular
velocity $\vect{\Omega}$ pointing out of the plane.

## Rotating Frame

Let's define the rotating frame such that one mass $M$ (say the Sun) sits to the left of the origin
$\vect{R} = (-R, 0)$, while the other mass $m$ (say the earth) sits to the right of the
origin at $\vect{r} = (r, 0)$.  We take the origin to be at at the center of mass:
\begin{gather*}
  m\vect{r} + M\vect{R} = 0, \qquad
  mr = MR, \qquad
  \eta = \frac{M}{m} = \frac{r}{R}.
\end{gather*}

## Dimensional Analysis

Here we introduce the dimensionless parameter $\eta$ which characterizes the mass
ratio.  As we shall show shortly, it turns out that this is the only intrinsic
dimensionless parameter for the system, completely characterizing the qualitative
behaviour.  To see this, consider the parameters of the problem: $M$, $m$, $R$, $r$,
$\Omega$ and Newton's gravitational constant $[G] = D^3/M/T^2$.  Naïvely, one has three
dimensionless parameters, however, as we have already seen, the mass ratio and distance
ratios are both described by $\eta$:
\begin{gather*}
  \eta = \frac{M}{m} = \frac{r}{R}.
\end{gather*}
Another ratio is dimensionless:
\begin{gather*}
  \frac{GM}{R^3\Omega^2},
\end{gather*}
however, this is fixed by Newton's law for the two orbiting bodies, whose force must
match the centrifugal force so that they remain fixed in the moving frame.  Thus, this
dimensionless constant is also expressed in terms of $\eta$:
\begin{gather*}
  \frac{GmM}{(r+R)^2} = mr\Omega^2 = MR\Omega^2, \\
  \frac{GM}{R^3\Omega^2} = \frac{r}{R}\left(\frac{r}{R}+1\right)^2
  = \eta(1+\eta)^2.
\end{gather*}

## Lagrange Points

The Lagrange points occur where the net force on a small object of mass $m_0$ at
$\vect{x}$ is zero in the rotating frame:

\begin{gather*}
  \frac{\vect{F}_{M}}{m_0} = GM\frac{\vect{R} - \vect{x}}{\norm{\vect{R}-\vect{x}}^3},\qquad
  \frac{\vect{F}_{m}}{m_0} = Gm\frac{\vect{r} - \vect{x}}{\norm{\vect{r}-\vect{x}}^3},\qquad
  \frac{\vect{F}_{C}}{m_0} = \Omega^2 \vect{x},\\
  \vect{F}_{M} + \vect{F}_{m} + \vect{F}_{C} = 0.
\end{gather*}

In components, we have
\begin{gather*}
  \frac{GM(-R - x)}{\Bigl((-R-x)^2 + y^2\Bigr)^{3/2}} 
  + \frac{Gm(r - x)}{\bigl((r-x)^2 + y^2\bigr)^{3/2}} + \Omega^2 x = 0,\\
  \frac{-GMy}{\Bigl((-R-x)^2 + y^2\Bigr)^{3/2}} 
  + \frac{-Gmy}{\bigl((r-x)^2 + y^2\bigr)^{3/2}} + \Omega^2 y = 0.
\end{gather*}

### $L_1$, $L_2$, and $L_3$

The first three Lagrange points have $y=0$.  Their location on the $x$-axis is given by:
\begin{gather*}
  \frac{M(x+R)}{\abs{x+R}^3} + \frac{m(x - r)}{\abs{x-r}^3} = \frac{\Omega^2}{G} x.
\end{gather*}

### $L_4$ and $L_5$

The remaining points have $\abs{y} > 0$, and satisfy:

\begin{gather*}
  \frac{M(x + R)}{\Bigl((x+R)^2 + y^2\Bigr)^{3/2}} 
  + \frac{m(x - r)}{\bigl((x-r)^2 + y^2\bigr)^{3/2}} = \frac{\Omega^2}{G} x,\\
  \frac{M}{\Bigl((x+R)^2 + y^2\Bigr)^{3/2}} 
  + \frac{m}{\bigl((x-r)^2 + y^2\bigr)^{3/2}} = \frac{\Omega^2}{G}.
\end{gather*}

It is not obvious, but these have $x = (r - R)/2$, halfway between the two objects:

\begin{gather*}
  x + R = \frac{r+R}{2}, \qquad
  x - r = -\frac{r+R}{2},\\
  \frac{M}{\Bigl(\frac{(r+R)^2}{4} + y^2\Bigr)^{3/2}} 
  + \frac{m}{\bigl(\frac{(r+R)^2}{4} + y^2\bigr)^{3/2}} = \frac{\Omega^2}{G} \frac{r-R}{r+R},\\
  \frac{M}{\Bigl(\frac{(r+R)^2}{4} + y^2\Bigr)^{3/2}} 
  + \frac{m}{\bigl(\frac{(r+R)^2}{4} + y^2\bigr)^{3/2}} = \frac{\Omega^2}{G}.
\end{gather*}





