---
jupytext:
  formats: ipynb,md:myst,py:light
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

import mmf_setup; mmf_setup.nbinit()
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
#import manim.utils.ipython_magic
#!manim --version
```

(sec:LagrangeMultipliers)=
Lagrange Multipliers
====================

## Unconstrained Minimization

:::{margin}
Numerically, one often takes finite steps:
\begin{gather*}
  \vect{r}_{n+1} = \vect{r}_{n} - \gamma\vect{\nabla}E(\vect{r}_{n}).
\end{gather*}
This can work very well, but there are cases -- especially in long valleys -- where this
approach can get stuck.  More sophisticated variants like the [conjugate gradient
method][] can be used in these cases.
:::
Consider minimizing a function $E(\vect{r})$: for concreteness, think about $E(\vect{r})
= mgh(\vect{r})$ as the gravitational potential energy if you are walking in mountainous
terrain.  Your goal is to find the lowest point and a local solution is to walk in the
direction of the negative gradient $\dot{\vect{r}} \propto -\vect{\nabla}E(\vect{r})$,
stopping when the gradient vanishes $\vect{\nabla}E(\vect{r}) = \vect{0}$.  This
algorithm is called [gradient descent][] and provides a very good first step to finding
the minimum of a function.

This outlines the usual mathematical strategy for extremization of finding the roots of
$\vect{\nabla}E(\vect{r}) = \vect{0}$, but beware that this may also locate maxima and
saddle points.

## Constraints

Now suppose that your motion is constraint by a function $g(\vect{r}) = 0$.  The example
I used in class was to imagine you are a goat tethered to a stake with a rod such that
you must maintain a fixed distance $R$ from the stake:
\begin{gather*}
  g(\vect{r}) = \abs{\vect{r}}^2 - R^2.
\end{gather*}

Intuitively, you are only allowed to move perpendicular to $\vect{\nabla} g(\vect{r})$.

This case can be solved by the method of [Lagrange multipliers][].  The idea here is
that, at the minimum, the gradients of $E$ and the gradients of $g$ must be parallel,
otherwise there is an allowed direction of motion that will reduce the energy.  Thus
\begin{gather*}
  \vect{\nabla}E(\vect{r}) = \mu \vect{\nabla}g(\vect{r}).
\end{gather*}

The method of Lagrange multipliers is this general expressed as extremizing
\begin{gather*}
  \mathcal{L}(\vect{r}, \mu) = E(\vect{r}) - \mu g(\vect{r}).
\end{gather*}
Note: many constraints can be included this way -- just include more terms.

The solution thus has the form
\begin{gather*}
  \vect{\nabla}\Bigl(E(\vect{r}) - \mu g(\vect{r})\bigr) = \vect{0}.
\end{gather*}
This will give the solution $\vect{r}(\mu)$ as a function of the Lagrange multiplier,
which then must be adjusted to satisfy the constraint:
\begin{gather*}
  g\bigl(\vect{r}(\mu)\bigr) = 0.  
\end{gather*}

:::{admonition} Example

Minimize $E(x) = (x-1)^2$ subject to the constraint that $g(x) = x^2 - 4 = 0$:

1. Subtract the constraint from the objective with a factor of the Lagrange multiplier
   $\mu$:   
   \begin{gather*}
     \mathcal{L}(x, \mu) = E(x) - \mu g(x) = (1-\mu)x^2 - 2x + 1 + 4\mu.
   \end{gather*}
2. Find the extremal points by setting the gradient to zero:
   \begin{gather*}
     \pdiff{\mathcal{L}}{x} = 2(1-\mu)x - 2, \qquad
      x = \frac{1}{1-\mu}.
   \end{gather*}
   Note that the solution depends on the Lagrange multiplier  $\mu$.
3. Adjust the Lagrange multiplier to satisfy the constraint:
   \begin{gather*}
     0 = g(x) = \frac{1}{(1-\mu)^2}-4, \qquad
     1-\mu = \pm \tfrac{1}{2},\\
     \mu \in \{\tfrac{1}{2}, \tfrac{3}{2}\}.
   \end{gather*}
4. Find the potential solutions and check which are minima:
   \begin{gather*}
     x = \in \{-2, 2\}, \qquad E(-2) = 9, \qquad E(2) = 1.
   \end{gather*}
   Hence, $x=2$ is the minimum, and $x=-2$ is the maximum, subject to the constraints.
:::

:::{warning}

In class, we considered this example with $E(x) = x^2$ and ran into a problem:
\begin{gather*}
  \mathcal{L}(x, \mu) = (1-\mu)x^2 + 4\mu, \qquad 2(1-\mu)x = 0.
\end{gather*}
This seems to imply that $x = 0$, which is the global minimum.  However, note that if
$\mu = 1$, then **all solutions** $x$ are allowed.  In this case, the constraint and the
objective have the same contours, so this is highly degenerate and $x$ does not depend
on $\mu$.  Here you just have to find the $x = \pm 2$ that satisfy the constraint, and
check these.
:::










[gradient descent]: <https://en.wikipedia.org/wiki/Gradient_descent>
[conjugate gradient method]: <https://en.wikipedia.org/wiki/Conjugate_gradient_method>
[Lagrange multipliers]: <https://en.wikipedia.org/wiki/Lagrange_multiplier>
