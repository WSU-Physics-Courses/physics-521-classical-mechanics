---
jupytext:
  formats: ipynb,md:myst
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

# Assignment 5: Rotations

+++

**Due: 11:59pm Friday 12 November 2021**

+++

## Rolling Coin
Consider a coin rolling in a circle.  Treat the coin as a flat disk of radius $R$ and mass $M$.  Assume that the coin rolls without slipping.  Determine all properties of the motion (period of the orbit, radius of the circle, angle of the coin, angular velocities etc.)  Be sure to clearly sketch the geometry of your problem, perform a dimensional analysis, and check your solution by considering various limits.

+++

## Nutation
Using the parametrization of the Euler angles in Landau and Lifshitz Volume 1, describe nutation in the small amplitude limit (i.e. treat this as an application of the normal-mode theory discussed in class.)  Clearly state all steps in formulating the problem, i.e. deriving the equations (use the Lagrangian approach, and conserved quantities to obtain an effective 1D theory with an appropriate effective potential), find the "stationary" solution - this corresponds to simple **precession**.  Check this by considering the time-evolution of the angular momentum $\dot{\vec{L}} = \vec{\Omega}\times\vec{L} = \vec{\tau}$.  Then, describe the deviations from the stationary solution to derive the normal modes.  Try to identify all of the modes shown in Fig. 49, and check your answer numerically using the analytic solution (7).  (References are to the 1st edition, which is available online.)  Let me know if you need any help setting up the numerical checks.  I recommend using the [`scipy.integration.quad`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html) routine.
