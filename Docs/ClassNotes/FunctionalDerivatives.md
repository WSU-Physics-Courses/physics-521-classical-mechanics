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

Functional Derivatives
======================


## Catenary

What is the shape of a thin rope of linear mass density $\lambda$ and fixed length $L_0$
hanging between two hooks?  This is a classic problem for calculus of variations.
Expressed as an optimization problem, our goal is to minimize the potential energy of
the rope while holding the length constant:
\begin{gather*}
  \min_{y(x)} E[y] \quad \Big|\quad [y(x_0), y(x_1)] = [y_0, y_1], \quad \text{and}\quad
  L[y] = L_0.
\end{gather*}

The first task is to parameterize the rope.  A straightforward approach is to express it
as a function $y(x)$ subject to the boundary conditions $y(x_0) = y_0$ and $y(x_1) = y_1$.
The length and potential energy are then given by the [functional][]s $L[y]$ and $E[y]$
respectively:
\begin{gather*}
  L[y] = \int_{0}^{L}\d{L}, \qquad
  E[y] = \int_{0}^{M}\d{m}\; gy.
\end{gather*}
The differentials can be expressed as:
\begin{gather*}
  \d{L} = \sqrt{\d{x}^2 + \d{y}^2} = \sqrt{1 + (y')^2}\d{x}, \qquad
  \d{m} = \lambda \d{L},
\end{gather*}
giving the definite integrals
\begin{gather*}
  L[y] = \int_{x_0}^{x_1}\d{x}\;\sqrt{1+(y')^2}, \qquad
  E[y] = \lambda g \int_{x_0}^{x_1}\d{x}\; y\sqrt{1+(y')^2}.
\end{gather*}

## Direct Approach

The constraint can be enforced by using the method of {ref}`sec:LagrangeMultipliers`,
thus we look for extremal points of the functional
\begin{gather*}
  I[y] = E[y] - \mu L[y] = 
  \int_{x_0}^{x_1}\d{x}\; \underbrace{(\lambda g y - \mu)\sqrt{1+(y')^2}}_{\mathcal{L}(y, y', x)}.
\end{gather*}

To simplify this, we write the integrand in terms of the function $\mathcal{L}(y, y',
x)$, which is the analogue of the Lagrangian in classical mechanics.
\begin{gather*}
  I[y] = \int_{x_0}^{x_1}\d{x}\; \mathcal{L}(y, y', x).
\end{gather*}
The solution to the extremization problem follows the usual approach of [calculus of
variations][], which yields the [Euler-Lagrange equation][]:
\begin{gather*}
  \diff{}{x}\pdiff{\mathcal{L}}{y'} = \pdiff{\mathcal{L}}{y}.
\end{gather*}
Using our functional form, this gives the following solution for the [catenary][]:
\begin{gather*}
  \diff{}{x}\left(\frac{y'(\lambda g y - \mu)}
                       {\sqrt{1+(y')^2}}\right) = \lambda g \sqrt{1+(y')^2}.
\end{gather*}

Expanding, then simplifying, we have
\begin{gather*}
  \frac{y''(\lambda g y - \mu) + \lambda g (y')^2}{\sqrt{1+(y')^2}}
  - \frac{(y')^2y''(\lambda g y - \mu)}{\sqrt{1+(y')^2}^3}= \lambda g \sqrt{1+(y')^2},\\
  y''(\lambda g y - \mu) 
  = \lambda g \frac{1 + (y')^2}{1 - (y')^2}.
\end{gather*}

Expressing, $y'' = \d{y'}/d{y} y'$, this separates:
\begin{gather*}
  \int y' \frac{1 - (y')^2}{1 + (y')^2}\d{y'} 
  = \int\frac{\lambda g}{\lambda g y - \mu}\d{y}.
\end{gather*}
However, even after solving this, we still have another integration to perform.  Maybe
there is a better way...

## Alternative Strategies

Let's see if some other formulations might have helped us.
Consider two expressions of the solution: $y(x)$ as we have done above with the
independent variable $x$ (the analog of $t$ in Lagrangian mechancs), and the inverse
$x(y)$ which switches the roll so that $y$ is the independent variable.  We can express
the extremization problem in terms of the following two Lagrangians:
\begin{align*}
  \mathcal{L}_x(y, y', x) = (\lambda g y - \mu)\sqrt{1 + (y')^2},\\
  \mathcal{L}_y(x, x', y) = (\lambda g y - \mu)\sqrt{1 + (x')^2}.
\end{align*}

## Conservation Laws

Notice that $\mathcal{L}_x(y, y', x)$ is independent of $x$.  This is the equivalent of
a classical Lagrangian being time-independent, and Noether's theorem tells us that the
corresponding Hamiltonian is conserved:
\begin{gather*}
  H_x = y'\pdiff{\mathcal{L}_x(y, y', x)}{y'} - \mathcal{L}_x(y, y', x) = \text{const}.
\end{gather*}

Similarly, $\mathcal{L}_y(x, x', y)$ is independent of $x$.  This makes the conserved
quantity a little more obvious since, from the Euler-Lagrange equations:
\begin{gather*}
  \diff{}{y}\left(\pdiff{\mathcal{L}_y(x, x', y)}{x'}\right) = \pdiff{\mathcal{L}_y(x,
  x', y)}{x} = 0,\\
  \pdiff{\mathcal{L}_y(x, x', y)}{x'} = \text{const}.
\end{gather*}

It turns out that these two conserved quantities are equivalent, and this trick of
exchanging the dependent and independent variables works in classical mechanics to
derive the conserved Hamiltonian.  In any case, it should be pretty apparent that this
latter form is simpler to evaluate:
\begin{gather*}
  \pdiff{\mathcal{L}_y(x, x', y)}{x'} = \frac{x'(\lambda g y - \mu)}{\sqrt{1 + (x')^2}}
  = \text{const}.
\end{gather*}

:::{margin}
Doing this, we assume things like $x' > 0$.  This is find for getting a general
expression, but we must check at the end of the day that these manipulates are
justified.  If we are not careful, we may throw the baby out with the bathwater by
multiplying or dividing by zero.
:::
Simplifying, and replacing $x'(y) = 1/y'(x)$, we obtain
\begin{gather*}
  \lambda g y - \mu = c^{-1}\sqrt{1 + (y')^2},
\end{gather*}
where $c$ is a constant.

:::{note}
This is considerably simpler than the direct approach. When presented with a
somewhat complicated problem, it can be worth your time to first consider the general
form of the problem and try a few general approaches to see if you can find one that
might be computationally simpler before delving into the algebraic details.
:::

This is also separable:
\begin{gather*}
  \d{x} = \frac{\d{y}}{\sqrt{c^{2}(\lambda g y - \mu)^2 - 1}}.
\end{gather*}

The integral here has the form
\begin{gather*}
  \int \frac{1}{\sqrt{ay^2 + by + c}}\d{y}
\end{gather*}
which can be found in [common tables](https://www.integral-table.com/).

:::{admonition} Full Solution

Once can also check a partially remembered answer.  I recall that the solution for a
[catenary][] is somehow related to the [hyperbolic functions][].  Practical experience
with hanging ropes suggest it is probably something of the form:
\begin{gather*}
  y(x) = a\cosh(bx + d), \qquad
  y'(x) = ab\sinh(bx + d).
\end{gather*}
If $ab = 1$, then $1+(y')^2 = 1+\sinh^2(bx+c) = \cosh^2(bx+c)$ which is almost right,
but won't get the constant $\mu$.  We can get this by adding a constant, and a bit more
through suggest the following form which has a minimum at $(x_\min, y_\min)$: 
\begin{gather*}
  y(x) = a\Bigl(\cosh\bigl((x-x_\min)/a\bigr) - 1\Bigr) + y_\min, \\
  y'(x) = \sinh\Bigl((x-x_\min)/a\Bigr), \\
  \begin{aligned}
  \sqrt{1 + (y')^2} &= \cosh\Bigl((x-x_\min)/a\Bigr) = \frac{y(x) + y_\min}{a} - 1 \\
                    &= \frac{\lambda g y - \mu}{c}.
  \end{aligned}
\end{gather*}

This is exactly the solution if
\begin{gather*}
  a = \frac{c}{\lambda g}, \qquad
  \frac{y_\min}{a} - 1 = -\frac{\mu}{c}.
\end{gather*}
This has three parameters $a$, $x_\min$, and $y_\min$, allowing us to satisfy the three
constraints $y(x_0) = y_0$, $y(x_1) = y_1$, and $L[y] = L_0$.  The length equation
becomes:
\begin{gather*}
  L[y] = \int_{x_0}^{x_1}\d{x}\; \sqrt{1+(y')^2}
       = \int_{x_0}^{x_1}\d{x}\; \cosh\tfrac{x-x_\min}{a}\\
       = \left.a\sinh\tfrac{x-x_\min}{a}\right|_{x_0}^{x_1} = L_0.
\end{gather*}

The algebra can be simplified considerably by choosing the origin at the minimum $x_\min
= 0$, $y_\min = 0$.  This implies $x_0 = -x_1$ and $y_0 = y_1$:
\begin{gather*}
  c = \mu, \qquad a = \frac{\mu}{\lambda g},\\
  L_0 = 2a\sinh\tfrac{x_1}{a}
\end{gather*}


As a final check, one should make sure that no errors were made multiplying or dividing
by zero, and that this is indeed a minimum, not a maximum or saddle point.  The latter
check is fairly easy on physical grounds if we make sure $a, x_1 >0$.  The maximum energy has
a form with $a, x_1<0$ -- an inverted [catenary][].
:::

As a final check, let's compute the conserved Hamiltonian from the first approach:
\begin{align*}
  H_x &= \overbrace{\frac{(y')^2(\lambda g y - \mu)}{\sqrt{1 +
  (y')^2}}}^{y'\partial\mathcal{L}_x/\partial y'}
  -\overbrace{(\lambda g y - \mu)\sqrt{1 + (y')^2}}^{\mathcal{L}_x} = \text{const},\\
    &= \frac{(y')^2 - \Bigl(1 + (y')^2\Bigr)}{\sqrt{1 + (y')^2}}(\lambda g y - \mu),\\
    &= \frac{\lambda g y - \mu}{\sqrt{1 + (y')^2}},\\
\end{align*}
which gives the same equation as the second approach:
\begin{gather*}
  \lambda g y - \mu = c^{-1}\sqrt{1 + (y')^2}.
\end{gather*}

<!-- ## Parameterized Curve -->

<!-- We consider one final approach, using a curve $\vect{z}(s) = [x(s), y(s)]$ parameterized by the -->
<!-- arc-length: -->
<!-- \begin{gather*} -->
<!--   E[\vect{z}] = \int_0^{L} g \lambda y(s)\d{s}. -->
<!-- \end{gather*} -->
<!-- Note that this only depends on the coordinate $y$, so the solution is trivial: -->
<!-- \begin{gather*} -->
<!--   E[y] = g \lambda \int_0^{L} \overbrace{\mathcal{L}_s(y, y', s)}^{\mathcal{L} = y}\d{s},\\ -->
<!--   \diff{}{s}\pdiff{\mathcal{L}_s(y, y', s)}{y'} = \pdiff{\mathcal{L}_s(y, y', s)}{y},\\ -->
<!--   0 = g\lambda. -->
<!-- \end{gather*} -->





[calculus of variations]: <https://en.wikipedia.org/wiki/Calculus_of_variations>
[functional]: <https://en.wikipedia.org/wiki/Functional_(mathematics)>
[Euler-Lagrange equation]: <https://en.wikipedia.org/wiki/Calculus_of_variations#Euler%E2%80%93Lagrange_equation>
[catenary]: <https://en.wikipedia.org/wiki/Catenary>

[gradient descent]: <https://en.wikipedia.org/wiki/Gradient_descent>
[conjugate gradient method]: <https://en.wikipedia.org/wiki/Conjugate_gradient_method>
[Lagrange multipliers]: <https://en.wikipedia.org/wiki/Lagrange_multiplier>
[hyperbolic functions]: <https://en.wikipedia.org/wiki/Hyperbolic_functions>
