---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3 (phys-521-2021)
  language: python
  name: phys-521-2021
---

```{code-cell} ipython3
:tags: [hide-input]

import mmf_setup;mmf_setup.nbinit(quiet=True)
import logging;logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
%pylab inline --no-import-all
```

# Assignment 3: Lagrangian Dynamics

+++

**Due: 11:59pm Friday 8 October 2021**

+++

## Brachistochrone Problem

+++

Consider a bead of mass $m$ sliding down a wire from the point $\vect{P} = (x_0, y_0)$.

1. Write and expression for the total time $T$ it takes for the bead to slide from
   $\vect{P}$ to the origin as a functional of the path of the wire $y(x)$.

2. Write the Euler-Lagrange equations for the solution.

3. Show that the solution is a a [cycloid](https://en.wikipedia.org/wiki/Cycloid) (or
   solve the problem directly).

+++

## Constrained Optimization

+++

Find the maximum of the function $f(x,y) = -x^2-y^2$ subject to the constraint that
$(x-x_0)^2 + y^2 = R^2$ using the method of Lagrange multipliers.

```{code-cell} ipython3
x0 = 5
R = 1
x = np.linspace(-10,10,100)[:, None]
y = np.linspace(-10,10,100)[None, :]
f = -x**2 - y**2
c = (x-x0)**2 + y**2

fig, ax = plt.subplots(figsize=(10,10))
cs = ax.contour(x.ravel(), y.ravel(), f.T, 
                20,
                linestyles='-', colors='k')
plt.clabel(cs, fmt='%1.0f')
cs = ax.contour(x.ravel(), y.ravel(), c.T, 
                levels=[0,1,2,3,4,5], linestyles='-', colors='b')
plt.clabel(cs, fmt='%1.0f')
ax.set(xlabel='x', ylabel='y', aspect=1);
```

## Bead on a Circular Wire

+++

Derive the equations of motion for a bead of mass $m$ moving on a frictionless circular wire of radius $a$ (problem 3.1 but for now consider the wire to be stationary).  Do this three ways:

1. Use Newtonian mechanics by projecting the forces as appropriate.
2. Using the Euler-Lagrange equations for the coordinate $\theta(t)$ shown in class or in Fig. 16.1 in the book.
3. Using the Euler-Lagrange equations for the coordinates $r(t)$ and $\theta(t)$ but introducing the constraint $r(t) = a$ with the method of Lagrange multipliers.  Show how to use this formulation to find an expression for force of constraint exerted by the wire to keep the bead in place.  (I.e. the force you would feel if you were to hold the wire in place while the bead is moving.)
4. Derive the equations of motion for the bead if the wire moves up and down with a time-dependent function $h(t)$.  (This is a driven pendulum.  We will analyze it later.)

+++

## Problem 3.1: Bead on a Rotating Circular Wire

+++

> **3.1** A point mass $m$ slides without friction along a wire bent into a vertical circle of radius $a$.
> The wire rotates with constant angular velocity $\Omega$ about the vertical diameter, and the apparatus is placed in a uniform gravitational field $\vect{g}$ parallel to the axis of rotation.
> * **a)** Construct the lagrangian for the point mass using the angle $\theta$ (angular displacement measured from the downward vertical) as generalized coordinate.
> * **b)** Show that the condition for an equilibrium circular orbit at angle $\theta_0$ is $\cos\theta_0 = g/a\Omega^2$. Explain this result by balancing forces in a co-rotating coordinate system.
> * **c)** Demonstrate that this orbit is stable against small displacements along the wire and show that the angular frequency of oscillation about the equilibrium orbit is given by $\omega^2 = \Omega^2\sin^2\theta_0$. *Hint:* Write $\theta = \theta_0 +\eta(t)$, where $\eta(t)$ is a small quantity.
>* **d)** What happens if $a\Omega^2 < g$?
