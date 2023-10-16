---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (phys-521)
  language: python
  name: phys-521
---

```{code-cell} ipython3
:tags: [hide-input]

import sys
try: import mmf_setup;mmf_setup.nbinit(quiet=True)
except: 
    !{sys.executable} -m pip install --user mmf-setup
    import mmf_setup;mmf_setup.nbinit()  # You might need to restart the kernel for this to work.
```

# Assignment 1: Newtonian Mechanics

+++

Due: Start of class on **Monday 25 September**.

:::{note}

Look for a CoCalc notebook version of this assignment with some numerical checks for
your answers to be pushed to your student project.

:::

+++

## Rockets

Consider a rocket of mass $m(t)$ which ejects fuel at a rate of $\dot{m}(t) \leq 0$.
Assume that all of the fuel is ejected with speed $v_e$ directed in the $-x$ direction
relative to the rocket.

1. Carefully justify the Tsiolkovsky rocket equation for a rocket moving in one
   dimension without gravity (or air resistance):

   \begin{gather*}
     v(t) = v(0) + v_e\ln\frac{m(0)}{m(t)}.  
   \end{gather*}
   
2. This formula is independent of the rate $\dot{m}(t)$ at which fuel is expelled.
   Explain how this result is consistent with the simple formula for the velocity of the
   rocket if all of the fuel were to be immediately eject as one blob with speed $v_e$:
   
   \begin{gather*}
     v(t>0) = v_i + v_e\frac{m(0) - m(t)}{m(t)}.
   \end{gather*}
   
3. Derive the equation of motion for the rocket moving vertically in a gravitational
   field.
4. Solve these equations for a rocket moving vertically in a constant gravitational
   field.  Assume that $\dot{m}(t) = \dot{m}$ is constant and find the height $z(t)$.
5. **Bonus**: Briefly estimate how much energy is required to place a payload of $1$kg
   into a geosynchronous orbit.  How does this depend on the overall mass of the rocket
   (i.e. is it more efficient to send several small rockets or a single large rocket?

## Tides

Give a plausible physical argument as to why the distance between the Earth and the Moon
is slowly increasing.

## Elliptical Orbits

As Kepler showed†, a particle orbiting in gravitational potential $V(r) = \alpha/r$ will
move along an ellipse.  Will the center of mass of an extended object also move in a
perfect ellipse?  Provide a **concise** and convincing argument that this will be the
case, or provide a simple counter example.

† *I do not require you to show it here, but I also expect **you** to be able to derive
and explain all of Kepler's laws from Newton's law, reducing the 6 degrees of freedom of
the original 2-body problem to a single effective equation for the relative coordinate
$r$ in terms of the reduced mass, etc.  I will likely ask you about this during one of
your exams.*

## Central Potentials

Throughout the course we will visit the problem of a Harmonic Oscillator: i.e. the
motion of a particle of mass $m$ in a potential $V(r) = \tfrac{1}{2}kr^2$ which might
represent a ball connected to an anchored spring with spring constant $k$.  We shall
revisit this problem in all formalisms and use it as a basis for understanding chaotic
dynamics.

1. Use the effective potential to show that all orbits are bound and that $E$ must
   exceed $E_{\text{min}} = \sqrt{kl^2/m}$ where $l$ is the angular momentum of the
   system.
2. Verify that the orbit is a closed ellipse with the origin at the center of the
   potential.  (Compare your result with the formulas in the book for problem 1.10 (b).)
3. Prove that the period is independent of the energy and angular momentum.  Could you
   have anticipated this from simple arguments? Discuss the significance of this result.

## Scattering

Do problem 1.17 from {cite:p}`Fetter:2003`.

> A uniform beam of particles with energy $E$ is scattered by an attractive (top-hat or
> spherical square-well) central potential:
>
> $$
    V(r) = \begin{cases}
      -V_0 & r < a\\
      0 & r \geq a
    \end{cases}
  $$
> 
> Show that the orbit of a particle is identical to a light ray refracted by a sphere of
> radius $a$ with a particular index of refraction $n$ (see the book). Compute the
> differential cross-section and show that it is
>
> $$
    \diff{\sigma}{\Omega} = \frac{n^2 a^2}{4\cos(\tfrac{1}{2}\theta)}
    \frac{\bigl[n\cos(\tfrac{1}{2}\theta) - 1\bigr](n-\cos\tfrac{1}{2}\theta)}
         {(1 + n^2 - 2n \cos\tfrac{1}{2}\theta)^2}
  $$
>
> Compute the total cross-section $\sigma$.
