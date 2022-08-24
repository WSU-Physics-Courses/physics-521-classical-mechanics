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

import mmf_setup;mmf_setup.nbinit(quiet=True)
```

# Assignment 0: Forced Oscillations

Due: **Friday 2 September**.  Please insert your solutions into your OneNote Class in
the **Homework** section on a page named **Assignment 0: Forced Oscillations**.

## Forced Damped Oscillations

This assignment is meant to complement the reading assignment from {cite:p}`LL1:1976`
§26 "Forced oscillations under friction".  Your goal is to find the complete solution to
the problem of a mass on a spring with a linear damping force and a drive:

\begin{gather*}
  m \ddot{x}  = \underbrace{-kx}_{\text{spring}} 
                -\underbrace{\alpha\dot{x}}_{\text{damping}}
                + F(t), \qquad
  F(t) = f\cos(\gamma t)
\end{gather*}

where $F(t)$ is a driving force which we will take to be a pure cosine wave as shown.

1. Sketch the setup and suggest some plausible mechanisms for the various terms.
2. Write this equation in the form of (26.1):

   \begin{gather*}
     \ddot{x} + 2\lambda \dot{x} + \omega_0^2 x = \frac{f}{m}\cos(\gamma t).
   \end{gather*}
   
   Express these parameters in terms of the physical parameters above.  From now on,
   work with these so we can compare with results from the book.
3. Perform a dimensional analysis of the problem.  Make sure you express the dimensions
   of all the parameters, separate out those parameters that are intrinsic to the system, and
   those parameters that are part of the initial state.

   How many dimensionless parameters are there upon which the qualitative behavior of
   the system might depend?
4. Discuss how you expect the qualitative behavior to depend on these dimensionless
   parameters.  Include a discussion about "transient" behavior and long-term
   "steady-state" behavior.  Arrange your dimensionless parameters so that you can
   identify the parameters which will affect the steady-state behavior.
5. Find the general solution to the problem $x(t)$ in terms of the initial position
   $x_0$ and the initial velocity $\dot{x}_0$.  Enter your solution into CoCalc and make
   sure that your solution agrees with the numerical solution given there.  Include this
   comparison with your final solution.
   
   :::{important}
   In as many assignments as possible, I will provide numerical solutions against which
   you can check your formula on CoCalc.  In order for me to grade your homework, I
   expect:
   
   1. Either you get the correct result and your solution matches the numerics.
   2. You make an appointment with me **before the assignment is due** to discuss what
      might be going wrong.
      
   If you hand in an assignment with an incorrect solution, and have not contacted me
   before-hand to discuss what might be going wrong, then you may get **zero** on the
   assignment.
   
   If you have difficulty, please first speak with your classmates.  If you cannot
   resolve the issue, then at this point it will be productive for a group to make an
   appointment with me to discuss.
   
   In Fall 2022, we plan to hold open iSciMath lab/office hours in Spark 235 from 4:30pm
   to 7pm most days of the week: Monday and Wednesday for sure (after Physics 555) and
   other days by appointment.  You may stop by these sessions to discuss any issues you
   have.  I am also available for appointment by Zoom.
   :::
   
   
   
   
(Eq. (26.1) in the text).






Consider a rocket of mass $m(t)$ which ejects fuel at a rate of $\dot{m}(t) \leq 0$.  Assume that all of the fuel is ejected with speed $v_e$ directed in the $-x$ direction relative to the rocket.

1. Carefully justify the Tsiolkovsky rocket equation derived in class for a rocket moving in one dimension without gravity (or air resistance):

   $$
     v(t) = v(0) + v_e\log\frac{m(0)}{m(t)}.
   $$
   
2. This formula is independent of the rate $\dot{m}(t)$ at which fuel is expelled.  Explain how this result is consistent with the simple formula for the velocity of the rocket if all of the fuel were to be immediately eject as one blob with speed $v_e$:
   
   $$
     v(t>0) = v_i + v_e\frac{m(0) - m(t)}{m(t)}.
   $$
   
3. Derive the equation of motion for the rocket moving vertically in a gravitational field.
4. Solve these equations for a rocket moving vertically in a constant gravitational field.  Assume that $\dot{m}(t) = \dot{m}$ is constant and find the height $z(t)$.
5. **Bonus**: Briefly estimate how much energy is required to place a payload of $1$kg into a geosynchronous orbit.  How does this depend on the overall mass of the rocket (i.e. is it more efficient to send several small rockets or a single large rocket?

+++ {"tags": ["remove-cell"]}

## Tides

Give a plausible physical argument as to why the distance between the Earth and the Moon is slowly increasing.

+++

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

+++

## Central Potentials

Throughout the course we will visit the problem of a Harmonic Oscillator: i.e. the motion of a particle of mass $m$ in a potential $V(r) = \tfrac{1}{2}kr^2$ which might represent a ball connected to an anchored spring with spring constant $k$.  We shall revisit this problem in all formalisms and use it as a basis for understanding chaotic dynamics.

1. Use the effective potential to show that all orbits are bound and that $E$ must exceed $E_{\text{min}} = \sqrt{kl^2/m}$ where $l$ is the angular momentum of the system.
2. Verify that the orbit is a closed ellipse with the origin at the center of the potential.  (Compare your result with the formulas in the book for problem 1.10 (b).)
3. Prove that the period is independent of the energy and angular momentum.  Could you have anticipated this from simple arguments? Discuss the significance of this result.

+++

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
