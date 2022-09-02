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
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
```

(sec:a0)=
# Assignment 0: Forced Oscillations

Due: **Friday 2 September**.  Please insert your solutions into your OneNote Class in
the **Homework** section on a page named **Assignment 0: Forced Oscillations**.

## Forced Damped Oscillations

This assignment is meant to complement the reading assignment from {cite:p}`LL1:1976`
§26 "Forced oscillations under friction".  Your goal is to find the complete solution to
the problem of a mass on a spring with a linear damping force and a cosine driving
force.

:::{note}
I recommend the following approach:

1. Quickly skim through the reading so you know what the topic is.  Look at some of the
   equations and figures to see if they make sense.  Once you get a rough idea, put the
   reading aside and see if you can derive the results on your own.  You have likely
   seen some of this material before.
2. If you get stuck, go back to the material and see what they do.  Does this provide
   any insight that helps you?  Come back to your own work as soon as you have some
   insight.
3. After you have a good idea of what is going on, start reading the material in detail
   to see if there are any finer points you have missed.
4. I find it very helpful to work through the equations numerically to make sure I
   understand what they mean.  Feel free to use the numerical solution provided on
   [CoCalc] to do this.  For example, I while checking my numerical solution, I realized
   that equation (26.2) was not what I thought it was originally.
:::

\begin{gather*}
  m \ddot{x}  = \underbrace{-kx}_{\text{spring}} 
                \overbrace{-\alpha\dot{x}}^{\text{damping}}
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
   of all the parameters, separate out those parameters that are **intrinsic** to the
   system, and those parameters that are connected to the **initial state**.

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
6. Derive and explain the meaning of the resonance curve in Fig. 31 (shown below).
   Discuss all the qualitatively different behavior as characterized by the intrinsic
   dimensionless parameters you found above.  Explain why the initial state is
   irrelevant.  Try to explain in terms of simple intuitive arguments why the phase
   shift $\delta$ has the value it does at resonance.

   Feel free to use or modify the following code (also in your CoCalc assignment) to
   help, but be sure to explain what it is doing.
   
```{code-cell}

%matplotlib inline
import numpy as np, matplotlib.pyplot as plt

f_m = 1.2
lam = 0.5
w0 = 1.5

gammas = np.linspace(0, 3*w0, 1000)[1:]

# Explain where these come from!
B = f_m / (w0**2 - gammas**2 + 2j*lam*gammas)
delta = np.angle(B)
B0 = f_m / (2j*lam*gammas)
I = abs(B**2)
I0 = abs(B0)**2

fig, ax = plt.subplots()
lines = []
lines.extend(ax.plot(gammas/w0, I/I0, 'C0-', 
                     label=r"$I/I_0$"))
axr = ax.twinx()
lines.extend(axr.plot(gammas/w0, delta, 'C1--', 
                      label=r"Phase shift $\delta$"))
ax.set(xlabel="$\gamma/\omega_0$", ylabel=r"$I/I_0$",
       title=f"$\lambda={lam}$, $f/m={f_m}$, $\omega_0={w0}$")
axr.set(ylabel=r"$\delta$", ylim=(-np.pi, 0),
        yticks=[-np.pi, -np.pi/2, 0], 
        yticklabels=["$-\pi$", "$-\pi/2$", "0"]) 
labels = [_l.get_label() for _l in lines]
ax.grid("on", c="C0")
axr.grid("on", c="C1", ls="--")
ax.legend(loc="lower left");
axr.legend(loc="lower right");
```
