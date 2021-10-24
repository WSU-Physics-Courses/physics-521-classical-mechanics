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

```{code-cell}
:tags: [hide-input]

import mmf_setup;mmf_setup.nbinit(quiet=True)
import logging;logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
```

# Assignment 4: Small Oscillations

+++

**Due: 11:59pm Friday 29 October 2021**

+++

## Masses and Springs 1

+++

Consider an equiliateral triangle of three equal masses $m$ and springs with spring constant $k$ and equilibrium length $l$ as discussed in class.  Consider this free to move in three dimensions (i.e. on the International Space Station).

1. How many normal modes do you expect for this system?
2. How many different frequencies do you expect?
3. What would be the degeneracy of these frequencies?  Describe the modes with a sketch.
4. Compute at least one non-zero frequency.

+++

## Masses and Springs 2

+++

Repeat the process for a tetrahedron with four equal masses and four 6 equal springs.
You may want to check your results with the code on CoCalc.

+++

## 2 Masses

Completely

+++

## Double Pendulum

+++

Consider a double pendulum as shown below:

```{code-cell}
:tags: [hide-input]

import sympy
sympy.init_printing()

from phys_521_2021 import double_pendulum
double_pendulum.draw_figure()
```

The formal task here is to characterize and quantify the normal modes of this system,
but I would like you to take this further and perform a thorough analysis of this
system, investigating both analytic features (such as the normal modes) and features
that are only apparent numerically.  Upon request, I will provide code examples to help
you analyze this system, but would like you to first try to solve it from scratch
yourself.  (Please use the notation given above so that we can compare answers.) 

Here are some recommendations for analyzing this system.  Note, I recommend first
finding out as much information as possible using techniques that take very little of
**your** time and effort.  These quick solutions might be approximate, or very slow to
run on a computer but get you something you can use to check your more refined solutions
later.  Once you have a set of quick and dirty results that you know are correct, you
can start to implement a more elegant or efficient solution, checking it against the
other solutions.  *As an example, consider performing a nasty integral - the integral
might admit a nice analytic form after performing some clever substitutions, but you
might first try simply approximating it numerically to see if you can get an idea of the
answer.  Then as you implement the more sophisticated transformations, you can keep
checking to make sure that you have not make any mistakes.* 

**Brief Exploration:** Spend a few minutes thinking about the problem, maybe writing a
Lagrangian, getting familiar with the coordinates and structure.  The equations for this
system are moderately complicated, so correctly deriving them and implementing them can
be prone to errors, thus it is crucial that you think about how you can check your
answers.  I will suggest some general approaches here: not all will be relevant or
useful for this problem.

**Dimensionless Parameters:** Identify the dimensionless parameters of the problem.
What will set the scale for periods?  When will the motion be simple, when will it be
complicated?  In your normal mode analysis, what will constitute "small"?  These
parameters will characterize the different qualitative behaviors of the system, allowing
you to classify them.

**Limiting Cases:** Identify simple limiting cases.  Can you reduce the system to a
simpler system with appropriate limits?  Write these limits down explicitly, and if
possible, determine how the leading order corrections will scale as you move away from
the ideal limits.  This will provide a very useful way of quantitatively testing your
code.  *(One issue is that some useful limits might be impossible to implement
numerically because of divergence or singularities.  Thus your numerical implementation
will have to simulate a system close to, but not at, the limit.  Since the system cannot
be exactly situated at the limit in these cases, understanding how the errors will scale
is crucial for testing.)*

**Conservation Laws and Symmetry:** Identify conserved quantities, or relationships that
follow from symmetry.

**Brute Force Solution:** Is there a simple way to implement brute force solution?
Sometimes a problem can be formulated in a dead simple way that takes **you** almost no
time, but perhaps is very slow for the computer, or can only give an approximate answer.
If so, get a simple or limited version of your code working.

**CAS:** If your problem is messy, can you use a computer algebra system
[CAS](https://en.wikipedia.org/wiki/Computer_algebra_system) like
[SymPy](http://www.sympy.org/en/index.html) or [Sage](http://www.sagemath.org/) (both
can be used on CoCalc) to make sure your results are correct?  Sometimes these can even
be used to generate the functions used in your numerical code.  *(Warning: Although I
really like this approach in principle, I find that often I waste far more time trying
to automatically generate code than I would spend if I just sat down and did the thing
by hand once.  However, if you need to generate many similar but different pieces of
code with changing functional forms, then code generation becomes a very viable option.
Keep in mind, however, that the final functional form generated by a CAS will likely not
be optimal in terms of performance, or roundoff error.  Often hand-manipulation will
greatly improve the result, but this is where I end up wasting lots of time - trying to
automate these simplifications.)*

**Numerical Implementation:** Once you have a numerical implementation, test it
carefully against the limiting case you found earlier.  Also check that your units
etc. are correct by running the code with different scalings of the parameters, but the
same dimensionless combinations.  The results should be identical, but with different
units.

**Testing:** Check that your code obeys all the conservation laws and symmetry
properties you derived earlier.  If you find something that seems wrong, how do the
errors behave?  If they are smooth, there is probably some systematic error (missing
factor, etc.).  If they appear to fluctuate rapidly or randomly, they might be numerical
errors due to roundoff, or an algorithm not converging.  *(With
[`solve_ivp`](https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.integrate.solve_ivp.html)
or similar integrators, you might have problems if you take to large steps.  These can
be fixed by taking smaller step sizes, or increasing the equivalent of the `max_step`
parameter to allow the algorithm to do more work for each time step.)*

**Exploration:** Armed with a validated numerical solution, start exploring various
aspects of the physics.  Check your intuition, and try to test analytic results and
approximations.  If you find something surprising, can you quantify it or understand
what is happening qualitatively from the analytics?

```{margin}
[<img style="float:right; padding-left:1em" width="150"
src="https://upload.wikimedia.org/wikipedia/commons/8/87/Double_pendulum_flips_graph.png">](https://commons.wikimedia.org/wiki/File:Double_pendulum_flips_graph.png)
```

**Generalization**:
Consider generalizing the problem: are there any ways you can generalize your solution
with a minimal amount of work to extend its range of validity?  Can you test other
aspects of physics or apply your solution to related problems?  Once you have a working
numerical solution to a problem, and have put some time into understanding the behavior
of the system, it is a good idea to see if you can get some mileage out of it.  *(For
example, the system under consideration here exhibits chaotic dynamics.   It is thus
provides a great opportunity for exploring and learning about
[chaos](https://en.wikipedia.org/wiki/Chaos_theory), [Poincaré
sections](https://en.wikipedia.org/wiki/Poincar%C3%A9_map), [Lyapunov
exponents](https://en.wikipedia.org/wiki/Lyapunov_exponent), etc.:*
