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
import logging; logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
%matplotlib inline
import matplotlib
import numpy as np, matplotlib.pyplot as plt
plt.rc('figure', figsize=(10.0, 8.0))
plt.rc('animation', html='html5')
#import manim.utils.ipython_magic
#!manim --version
```

Chaos and Feigenbaum's Constant
===============================

:::{margin}
For a history of Feigenbaum, see [Mitchell Feigenbaum (1944‑2019),
4.66920160910299067185320382…](https://writings.stephenwolfram.com/2019/07/mitchell-feigenbaum-1944-2019-4-66920160910299067185320382/)
by Stephen Wolfram.
:::
Here we consider the phenomena of period doubling in chaotic systems, which leads to
universal behavior {cite:p}`Feigenbaum:1978`.  The quintessential system is that of the
[Logistic map]:

\begin{gather*}
  x \mapsto f_r(x) = r x(1-x),
\end{gather*}

which is a crude model for population growth.  The interepretation is that $r$ is
proportional to the growth rate, and that $x \in [0,1]$ is the total population as a
fraction of the carrying capacity of the environment.  For small $x$, the population
grows exponentially without bound $x \mapsto r x$, while for large $x \approx 1$ the
population rapidly declines as the food is quickly exhausted and individuals starve.

```{code-cell}
def f(x, r, d=0):
    """Logistic growth function (or derivative)"""
    if d == 0:
        return r*x*(1-x)
    elif d == 1:
        return r*(1-2*x)
    elif d == 2:
        return -2*r
    else:
       return 0
```

```{code-cell}
:tags: [hide-input]

x = np.linspace(0, 1)

fig, ax = plt.subplots()
ax.plot(x, x, '-k')
for r in [1.0, 2.0, 3.0, 4.0]:
    ax.plot(x, f(x, r=r), label=f"$r={r}$")
ax.legend()
ax.set(xlabel="$x$", ylabel="$f_r(x)$");
```

The behaviour of this system is often demonstrated with a [cobweb plot]:

```{code-cell}
:tags: [hide_input]

from matplotlib.animation import FuncAnimation
from IPython.display import HTML

class CobWeb:
    """Class to draw and animate cobweb diragrams.
    
    Parameters
    ----------
    x0 : float
       Initial population `0 <= x0 <= 1`
    N0, N : int
       Skip the first `N0` iterations then plot `N` iterations.
    """
    def __init__(self, x0=0.2, N0=0, N=1000):
        self.x0 = x0
        self.N0 = N0
        self.N = N
        self.fig, self.ax = plt.subplots()
        self.artists = None
        
    def init(self):
        return self.cobweb(r=1.0)

    def cobweb(self, r):
        """Draw a cobweb diagram.

        Arguments
        ---------
        r : float
            Growth factor 0 <= r <= 4.

        Returns 
        -------
        artists : list
            List of updated artists.
        """
        # Generate population
        x0 = self.x0
        xs = [x0]
        for n in range(self.N0 + self.N+1):
            x0 = f(x0, r=r)
            xs.append(x0)

        xs = xs[self.N0:]  # Skip N0 initial steps

        # Manipulate data for plotting
        Xs = np.empty((len(xs), 2))
        Ys = np.zeros((len(xs), 2))
        Xs[:, 0] = xs
        Xs[:, 1] = xs    
        Ys[1:, 0] = xs[1:]
        Ys[:-1, 1] = xs[1:]    
        Xs = Xs.ravel()[:-2]
        Ys = Ys.ravel()[:-2]

        if self.N0 > 0:
            Xs = Xs[1:]
            Ys = Ys[1:]

        x = np.linspace(0,1,200)
        y = f(x, r)
        title = f"$r={r:.2f}$"
        if self.artists is None:
            artists = self.artists = []
            artists.extend(self.ax.plot(Xs, Ys, 'r', lw=0.5))
            artists.extend(self.ax.plot(x, y, 'C0'))
            artists.extend(self.ax.plot(x, x, 'k'))
            self.ax.set(xlim=(0, 1), ylim=(0, 1), title=title)
            artists.append(self.ax.title)
        else:
            artists = self.artists[:2] + self.artists[-1:]
            artists[0].set_data(Xs, Ys)
            artists[1].set_data(x, y)
            artists[2].set_text(title)
        return artists
    
    def animate(self, 
                rs=None,
                interval_ms=20):
        if rs is None:
            # Slow down as we get close to 4.
            x = np.sin(np.linspace(0.0, 1.0, 100)*np.pi/2)
            rs = 1.0 + 3.0*x
        animation = FuncAnimation(
            self.fig, 
            self.cobweb, 
            frames=rs,
            interval=interval_ms,
            init_func=self.init,
            blit=True)
        display(HTML(animation.to_jshtml()))
        plt.close('all')

cobweb = CobWeb()
cobweb.animate()
```

Notice that as the growth constant increases, the population first stabilizes on a
single fixed-point, but then this expands into a cycle of period 2, then a cycle of
period 4 etc.  This is called period doubling, and this behaviour can be summarized in a 
[bifircation diagram].

## Bifurcation Diagrams

Here we develop an algorithm for drawing a high-quality bification diagram.  We start
with a simple algorithm that simply accumulates $N$ iterations starting with $x_0 = 0.2$
at a set of $N_r$ points between $r_0=2.0$ and $r_1=4.0$.

```{code-cell}
fig, ax = plt.subplots()

r0 = 2.0
r1 = 4
Nr = 500
N = 200
x0 = 0.2

rs = np.linspace(r0,r1,Nr)
for r in rs:
    x = x0
    xs = []
    for n in range(N):
        x = f(x, r=r)
        xs.append(x)
    ax.plot([r]*len(xs), xs, '.', ms=0.1, c='k')
ax.set(xlabel='$r$', ylabel='$x$');
```

This works reasonably well, but has several artifacts.  First, note the various curves -
these are reminants of the choice of the initial point.  Instead, we should iterate
through some $N_0$ iterations to allow the system to stabilize, then start
accumulating.

```{code-cell}
fig, ax = plt.subplots()

N0 = 100

rs = np.linspace(r0,r1,Nr)
for r in rs:
    x = x0
    for n in range(N0):
        x = f(x, r=r)
    xs = []
    for n in range(N):
        x = f(x, r=r)
        xs.append(x)
    ax.plot([r]*len(xs), xs, '.', ms=0.1, c='k')
ax.set(xlabel='$r$', ylabel='$x$');
```

This is better, but the bifurcation points are still blurry because it takes a long time
for the system to settle down there.  One solution is to increase $N_0$, but this will
increase the computational cost.  Another is to use the last iteration $x$ from the
previous $x$ as $x_0$ which should put us close to the final attractor.  This latter
solution works well without increasing the computational cost:

```{code-cell}
fig, ax = plt.subplots()

N0 = 100
x = x0
rs = np.linspace(r0,r1,Nr)
for r in rs:
    # x = x0   # Use the previous value of x
    for n in range(N0):
        x = f(x, r=r)
    xs = []
    for n in range(N):
        x = f(x, r=r)
        xs.append(x)
    ax.plot([r]*len(xs), xs, '.', ms=0.1, c='k')
ax.set(xlabel='$r$', ylabel='$x$');
```

There are further refinements one could make: for example, when the system converges
quickly to a low period cycle, we still perform $N$ iterations and plot $N$ points.  One
could implement a system that checks to see if the system has converged, then finds the
period of the cycle so that only these points are plotted.  Such a system should fall
back to the previous method only when the period becomes too large.  Another
sophisticated approach would be to track the curves on the bifurcation plot and plot
these as smooth curves.  These require some book keeping and we keep our current
algorithm which we now code as a function.  A final modification we add is allowing the
points to be transparent which enables us to see the relative density of points.

```{code-cell}
def bifurcation(r0=2, r1=4, Nr=500, x0=0.2, N0=500, N=2000, alpha=0.1, 
                rs_xs=None):
    """Draw a bifurcation diagram"""
    if rs_xs is not None:
        rs, xs = rs_xs
    else:
        rs = np.linspace(r0,r1,Nr)
        xs = []
        x = x0
        for r in rs:
            for n in range(N0):
                x = f(x, r=r)
            _xs = [x]
            for n in range(N):
                _xs.append(f(_xs[-1], r=r))
            xs.append(_xs)
    _rs = []
    _xs = []
    for r, x in zip(rs, xs):
        _rs.append([r]*len(x))
        _xs.append(x)
    plt.plot(_rs, _xs, '.', ms=0.1, c='k', alpha=alpha)
    plt.axis([r0, r1, 0, 1])
    plt.xlabel('r')
    plt.ylabel('x')

    return rs, xs

rs_xs = bifurcation()
```

## Fixed Points

The fixed points can be easily deduced by finding the roots of $ x = f^{(n)}(x)$ where
the power here means iteration $f^{(2)}(x) = f(f(x))$

```{code-cell}
:tags: [hide-cell]

import sympy
sympy.init_printing()
x, r = sympy.var('x, r')

def fn(n, x=x):
    for _n in range(n):
        x = f(x, r)
    return x
```

### Period 1

Here are the period 1 fixed points.  The solution at $x=0$ is trivial, but there is a
nontrivial solution $x = 1 - r^{-1}$ if $r\geq 1$:

```{code-cell}
:tags: [hide-cell]

r1s = sympy.solve(fn(1) - x, r)
r1 = sympy.lambdify([x], r1s[0], 'numpy')  # Makes the solution a function
r1s
```

```{code-cell}
:tags: [hide-input]

xs = np.linspace(0, 1.0, 100)[1:-1]
plt.plot(r1(xs), xs, 'r')
bifurcation(rs_xs=rs_xs);
```

### Period 2

For period 2 we have the solutions for the period 1 equations since these also have
period 2, and two new solutions of which only one is positive:

\begin{gather*}
  r_{\pm} = \frac{1-x \pm \sqrt{(1-x)(1+3x)}}{2x(1-x)}
\end{gather*}

```{code-cell}
:tags: [hide-cell]

r2s = sympy.solve((fn(2) - x)/(r-r1s[0])/x/(x-1), r)
r2 = sympy.lambdify([x], r2s[0], 'numpy')
r2s
```

```{code-cell}
:tags: [hide-input]

xs = np.linspace(0, 1.0, 100)[1:-1]
plt.plot(r1(xs), xs, 'r')
plt.plot(r2(xs), xs, 'g')
bifurcation(rs_xs=rs_xs);
```

### Period 3 Implies Chaos

:::{margin}
Symbolically this is a mess, but we can convert the result to a
polynomial and find the roots efficiently that way.  The first period 3 cycle begins at
$r_0=1+\sqrt{8} \approx 3.8284$ (see [The Birth of Period 3,
Revisited](http://www.jstor.org/stable/2690665).
I found the value by trial and error and then using the [Inverse Symbolic Calculator:
3.82842712475](http://isc.carma.newcastle.edu.au/standardCalc?input=3.82842712475).
:::
In 1975, Li and Yorke proved that [Period 3 Implies
Chaos](https://www.jstor.org/stable/2318254) in 1D systems by showing that if there
exists cycles with period 3, then there are cycles of all orders.  Here we demonstrate
the period 3 solution.

```{code-cell}
:tags: [hide-input]

res = sympy.factor((fn(3) - x)/x/(r-r1s[0])/(x-1))

coeffs = sympy.lambdify([r], sympy.Poly(res, x).coeffs(), 'numpy')

def get_x3(r, _coeffs=coeffs):
    return sorted([_r for _r in np.roots(coeffs(r)) if np.isreal(_r)])


xs = np.linspace(0, 1, 100)[1:-1]
plt.plot(r1(xs), xs, 'r')
plt.plot(r2(xs), xs, 'g')

r0 = 1+np.sqrt(8)
rs = np.linspace(r0+1e-12, 4, 100)
xs = np.array(list(map(get_x3, rs)))
plt.plot(rs, xs, 'b');
bifurcation(rs_xs=rs_xs);
```

## Exact solution for $r=4$: Angle doubling on the unit circle

The chaos at $r=4$: $x\mapsto 4x(1-x)$ is special in that the evolution admits an [exact
solution](https://en.wikipedia.org/wiki/Logistic_map#Solution_in_some_cases).  Start
with the identity: 

\begin{gather*}
  \sin^2(2\theta) = (2\sin\theta\cos\theta)^2 = 4\sin^2\theta\cos^2\theta = 4\sin^2\theta(1-\sin^2\theta).
\end{gather*}

This suggests letting $x_n = \sin^2\theta_n$ and $x_{n+1} = \sin^2\theta_{n+1}$ which
now yields the trivial map $\theta \mapsto 2\theta$ which has the explicit solution
$\theta_n = 2^n\theta_0$.  Hence the logistic map with $r=4$ has the solution:

\begin{gather*}
  x_n = \sin^2(2^n\theta_0), \qquad x_0 = \sin^2(\theta_0).
\end{gather*}

This evolution is equivalent to moving arround a unit circle by doubling the angle at
each step.  If $\theta_0$ is a rational fraction of $2\pi$ then the motion will
ultimately become periodic, but if $\theta_0$ is an irrational factor of $2\pi$, then
the motion will never be periodic.

## Feigenbaum Constants

First we define some notation.  Let $r_{n}$ be the smallest value of the growth parameter where
the period bifurcates from $2^{n-1}$ to $2^{n}$.  Define the iterated function as:

\begin{gather*}
  f^{(n)}_{r} = \overbrace{f_{r}(f_{r}(\cdots f_{r}(f_{r}}^{n\times}(x))\cdots)).
\end{gather*}

:::{margin}
If $x_* = f(x_*)$ is a fixed point, we can determine the stability by perturbing:
\begin{gather*}
  f(x_*+\epsilon) \approx f(x_*) + \epsilon f'(x_*)\\
  = x_* + \epsilon f'(x_*).
\end{gather*}

Iff $\abs{f'(x_*)} < 1$, then iterates will converge to $x_*$.
:::
The first bifurcation point is at $r_{1} = 3$, $x_{1} = 2/3$, i.e. the solution to $x_1
= f_{r_1}(x_1)$.  At this point, the fixed point becomes unstable:
$\abs{f'_{r_{1}}(x_{1})} = 1$.  Now, since $x_{1}$ is a fixed point of $f_{r_1}(x)$, it
must also be a fixed point of $f^{(2)}_{r_1}(x)$.  We plot both of these below, as well
as $f^{(2)}_{r_1+\epsilon}(x)$ for a slightly larger $r = r_1 + \epsilon$:

```{code-cell}
:tags: [hide-input]

x = np.linspace(0, 1)
r_1 = 3.0
epsilon = 0.2
r_1a = r_1 + epsilon

fig, ax = plt.subplots()
ax.plot(x, x, 'k')
ax.plot(x, f(x, r=r_1), 'C0', label="$f_{r_1}(x)$")
ax.plot(x, f(f(x, r=r_1), r=r_1), 'C1', label="$f^{(2)}_{r_1}(x)$")
ax.plot(x, f(f(x, r=r_1a), r=r_1a), 'C1', ls='--', label=f"$f^{{(2)}}_{{r_1+{epsilon:.1f}}}(x)$")
ax.legend()
ax.set(xlabel="$x$", title=f"$r_1={r_1}$");
```

Notice the behaviour that, at this shared fixed point, $f^{(2)}_{r_1}{}'(x) =
\bigl(f'_{r_1}(x)\bigr)^2 = 1$.  As we increase $r = r_1+\epsilon$, this steepens and
the two new fixed points move outward.

:::{margin}
Let $g(x) = f_{r_2}(x)$, $g_2(x) = g(g(x))$, and call the two fixed points be $x_1$ and $x_2$:
\begin{gather*}
  x_2 = g(x_1)= g_2(x_2), \\
  x_1 = g(x_2) = g_2(x_1).
\end{gather*}

At the first bifurcation point, the iteration becomes unstable:
\begin{gather*}
  g_2'(x_1) = g'\bigl(g(x_1)\bigr)g'(x_1) \\
  = g'(x_2)g'(x_1) = 1,
\end{gather*}

but since $g_2'(x_2) = g'(x_1)g'(x_2)$ the second fixed point also becomes unstable at
the same $r = r_2$.
:::
We now look at $f^{(2)}_{r_2}(x)$ and $f^{(4)}_{r_2}(x)$, where we see the same
behaviour about both of the new fixed points:

```{code-cell}
:tags: [hide-input]

x = np.linspace(0, 1, 300)
r_2 = 1+np.sqrt(6)
def f2(x, r=r_2):
    return f(f(x, r), r)
epsilon = 0.1
r_2a = r_2 + epsilon

fig, ax = plt.subplots()
ax.plot(x, x, 'k')
ax.plot(x, f2(x), 'C0', label="$f^{(2)}_{r_2}(x)$")
ax.plot(x, f2(f2(x)), 'C1', label="$f^{(4)}_{r_2}(x)$")
ax.plot(x, f2(f2(x, r=r_2a), r=r_2a), 'C1', ls='--', label=f"$f^{{(4)}}_{{r_2+{epsilon:.1f}}}(x)$")
ax.legend()
ax.set(xlabel="$x$", title=f"$r_2={r_2:.6f}$");
```

```{code-cell}
:tags: [hide-input]

x = np.linspace(0, 1, 10000)
rr = 3.5699456718709449018420051513864989367638369115148323781079755299213628875001367775263210

def fn(x, n, r=rr):
    for n in range(n):
        x = f(x, r)
    return x

fig, ax = plt.subplots()
ax.plot(x, x, 'k')
for n in range(10):
    ax.plot(x, fn(x, 2**n), 'C1', lw=0.1)
ax.plot(x, fn(x, 1024), 'C0', label="$f^{(1024)}_{r_*}(x)$")
ax.legend()
ax.set(xlabel="$x$", title=f"$r_*={rr:.6f}$");
```

\begin{gather*}
  g(x) = -\alpha g\Bigl(g(\tfrac{x}{\alpha})\Bigr)\\
  g(x) = -\alpha g\Bigl(g(\tfrac{x}{\alpha})\Bigr)
  g(1) = -\alpha g\Bigl(g(\tfrac{1}{\alpha})\Bigr)
\end{gather*}





(see [Weisstein, Eric W. "Logistic Map"]).



| $n$ | $r_n$    | $\delta_n = \frac{r_{n}-r{n-1}}{r_{n+1}-r{n}}$ |
|-----|----------|------------------------------------------------|
| 0   | 1        |
| 1   | 3        | 
| 2   | 3.449490 |
| 3   | 3.544090 |
| 4   | 3.564407 |
| 5   | 3.568750 |
| 6   | 3.56969  |


```
rn = np.array([1.0, 3.0,  1+np.sqrt(6), 
               3.5440903595519228536, 3.5644072660954325977735575865289824,
               3.568750, 3.56969, 3.56989, 3.569934, 3.569943, 3.5699451, 
               3.569945557
               3.5699456718709449018420051513864989367638369115148323781079755299213628875001367775263210342163
               ])

```
    




Mitchell Feigenbaum {cite:p}`Feigenbaum:1978` poin

[logistic map]: <https://en.wikipedia.org/wiki/Logistic_map>
[cobweb plot]: <https://en.wikipedia.org/wiki/Cobweb_plot>
[bifircation diagram]: <https://en.wikipedia.org/wiki/Bifurcation_diagram>
[Weisstein, Eric W. "Logistic Map"]: <https://mathworld.wolfram.com/LogisticMap.html>
