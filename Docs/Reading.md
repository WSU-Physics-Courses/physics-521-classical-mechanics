---
substitutions:
  Fasano: |-
    A. Fasano, S. Marmi, and B. Pelloni: "Analytical Mechanics : An
    Introduction". Oxford Graduate Texts (2006).
  Fetter: |-
    Alexander L. Fetter and John Dirk Walecka: "Theoretical Mechanics
    of Particles and Continua", Dover (2003). {cite:p}`Fetter:2003`.
  FetterSupp: |-
    Alexander L. Fetter and John Dirk Walecka: "Nonlinear Mechanics: A
    Supplement to Theoretical Mechanics of Particles and Continua", Dover (2006).
  Gregory: |-
    R. Douglas Gregory: "Classical Mechanics". Cambridge University
    Press (2006).
  LL1: |-
    L. D. Landau and E. M. Lifshitz: "Mechanics", Pergamon Press (1969).
  SMC: CoCalc (<http://cocalc.com>)
  Sussman: |-
    G. J. Sussman and J. Wisdom: "Structure and Interpretation of
    Classical Mechanics". MIT Press (2015). (Available online through Library.)
  deLange: |-
    O. L. deLange and J. Pierrus: "Solved Problems in Classical
    Mechanics: Analytical and Numerical Solutions with
    Comments". Oxford University Press (2010). (Available online through Library.)
  Arnold: |-
    Vladimir I. Arnol'd: "Mathematical Methods of Classical Mechanics".
    Springer-Verlag, New York, 1989, translated by K. Vogtmann and A. Weinstein.
    {cite:p}`Arnold:1989`

---

Readings
========

## Required Textbooks

{{ Fetter }}<!-- This comment needed for definition list for some reason  -->
: This textbook provides a concise and thorough introduction to classical mechanics,
  including a discussion of fluids and elastic materials that is missing from many other
  texts.  We will only cover roughly the first half of the text in this course, but it
  is a worthwhile book to have.
  ([Available from Amazon as a Dover Edition](
      http://www.amazon.com/Theoretical-Mechanics-Particles-Continua-Physics/dp/0486432610).)

{{ FetterSupp }}
: The supplement provides a complete derivation of the Lorenz equations - the first
  example of a chaotic system - and presents a derivation of the KAM theorem,
  demonstrating some modern results in classical mechanics.
  ([Available from Amazon as a Dover Edition](
      http://www.amazon.com/Nonlinear-Mechanics-Supplement-Theoretical-Particles/dp/0486450317/).)

{{ LL1 }}
: Classic textbook.  Very concise introduction to the key concepts in classical
  mechanics, including some topics omitted from other work such as parametric resonances
  and the adiabatic theorem.
  ([Available online through the WSU Library.](
      https://ntserver1.wsulibs.wsu.edu:2061/book/9780750628969/mechanics))
  
Additional required readings will be made on {{ Perusall }} with a forum for discussion.

## Supplementary Material

{{ Gregory }}<!-- This comment needed for definition list for some reason  -->
:  Undergraduate textbook.  Not very insightful, but provides a good review and contains
   quite a few problems. 
   ([Available online through the WSU Library.](
       https://ntserver1.wsulibs.wsu.edu:2171/lib/wsu/detail.action?docID=422398).)

{{ Fasano }}
: Texbook at a comparable level to Fetter and Walecka with many problems.  For me, the
  presentation seems a little formal, and the problems seem to interfere with the flow,
  so I am not sure it is the best book to learn from.  It is great, however, for finding
  problems to test ones understanding.
   ([Available online through the WSU Library.](
    https://ntserver1.wsulibs.wsu.edu:2171/lib/wsu/detail.action?docID=422398))

{{ deLange }}
: Large collection of problems and solutions including numerical problems.  The
  numerical problems are particularly interest since they go beyond what is typically
  seen in purely analytic texts.
  ([Available online through the WSU Library.](
      https://ntserver1.wsulibs.wsu.edu:2171/lib/wsu/detail.action?docID=584565).)

{{ Sussman }}
: Quite a different presentation of the core results from a more formal perspective with
  many interesting numerical examples.  The notation may not be completely familiar -
  the authors use functional notation and the Scheme programming language.  For example,
  the usual Euler Lagrange equations become

  $$
    D(\partial_2 L\circ \Gamma[q]) - \partial_1L \circ \Gamma[q] = 0 \\ \Gamma[q] = (t, q(t), Dq(t), \dots).
  $$

  The ability to be able to recognize and understand results in a slightly different
  "language" can be extremely valuable for checking one's understanding and cementing
  concepts.  Thus, while I do not recommend learning from this text, I highly recommend
  reading it to check that you really understand the concepts.
  ([Available online through the WSU Library.](
      https://ntserver1.wsulibs.wsu.edu:2171/lib/wsu/detail.action?docID=3339940).)

{{ Arnold }}
: Mathematical formulation of classical mechanics with a heavy emphasis on differential
  geometry (differential forms etc.).  A fun book to read *after* you understand the basics.
  
[John Baez: Classical Mechanics](https://math.ucr.edu/home//baez/classical/)
: Lecture notes with a more mathematical flavor (similar to {cite:p}`Arnold:1989`) but
  still quite accessible.  The lecture notes move from classical mechanics to gauge
  theories and relativity.

