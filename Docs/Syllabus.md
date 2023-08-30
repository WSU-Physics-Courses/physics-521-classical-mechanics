---
myst:
  substitutions:
    Fasano: |-
      A. Fasano, S. Marmi, and B. Pelloni: "Analytical Mechanics : An
      Introduction". Oxford Graduate Texts (2006).
    Fetter: |-
      Alexander L. Fetter and John Dirk Walecka: "Theoretical Mechanics
      of Particles and Continua", Dover (2003).
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
---

(sec:sylabus)=
# Syllabus

## Course Information

- **Instructor(s):** Michael McNeil Forbes [`m.forbes+521@wsu.edu`](mailto:m.forbes+521@wsu.edu)
- **Course Assistants:** 
- **Office:** Webster 947F
- **Office Hours:** TBD
- **Course Homepage:** https://schedules.wsu.edu/List/Pullman/20233/Phys/521/01
- **Class Number:** 521
- **Title:** Phys 521: Classical Mechanics
- **Credits:** 3
- **Recommended Preparation**: Undergraduate mechanics and calculus including calculus of
  variations, Newton's laws, Kepler's laws, conservation of momentum, energy,
  angular momentum, moment of inertia, torque, angular motion, friction, etc.
- **Meeting Time and Location:** MWF, 11:10am - 12:00pm, [Webster 941](https://sah-archipedia.org/buildings/WA-01-075-0008-13), Washington State University (WSU), Pullman, WA
- **Grading:** Grade based on assignments and project presentation.

```{contents}
```

### Textbooks and Resources

#### Required

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
  
Additional required readings will be made on [Perusall] with a forum for discussion.

#### Additional Resources

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
    https://ntserver1.wsulibs.wsu.edu:2171/lib/wsu/detail.action?docID=422398).)

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
  the usual Euler Lagrange equations

  \begin{gather*}
    %\newcommand{\diff}[2]{\frac{\mathrm{d}{#1}}{\mathrm{d}{#2}}}
    %\newcommand{\pdiff}[2]{\frac{\partial{#1}}{\partial{#2}}}
    \diff{}{t} \pdiff{L}{\dot{q}^i} - \pdiff{L}{q^i}
  \end{gather*}

  becomes

  \begin{gather*}
    D(\partial_2 L\circ \Gamma[q]) - \partial_1L \circ \Gamma[q] = 0 \\ \Gamma[q] = (t, q(t), Dq(t), \dots).
  \end{gather*}


  The ability to be able to recognize and understand results in a slightly different
  "language" can be extremely valuable for checking one's understanding and cementing
  concepts.  Thus, while I do not recommend learning from this text, I highly recommend
  reading it to check that you really understand the concepts.
  ([Available online through the WSU Library.](
      https://ntserver1.wsulibs.wsu.edu:2171/lib/wsu/detail.action?docID=3339940).)

Canvas
: The course material is hosted on the WSU [Canvas] system
  <https://canvas.wsu.edu>. Check the webpage there for changes to the schedule.

Computation Platform: {{ SMC }}
: This will be used for assignment distribution and for numerical work.


### Student Learning Outcomes

The main objective of this course is to enable students to explain physical
phenomena within the realm of classical mechanics, making appropriate
simplifying approximations to formulate and solve for the behavior of
mechanical systems using mathematical models, and communicating these results
to peers.

By the end of this course, the students should be able to take a particular
physical system of interest and:

1. **Understand the Physics:** Identify the appropriate quantities required to
   describe the system, making simplifying assumptions where appropriate with a
   quantitative understanding as to the magnitude of the errors incurred by
   making these assumptions.
2. **Define the Problem:** Formulate a well-defined model describing the
   dynamics of the system, with confidence that the model is solvable.  At this
   point, one should be able to describe a brute force solution to the problem
   that would work given sufficient computing resources and precision.
3. **Formulate the Problem:** Simplify the mathematical formulation of the
   problem as much as possible using appropriate theoretical frameworks such as
   the Lagrangian or Hamiltonian frameworks.
4. **Solve the Problem:** Use analytic and numerical techniques to solve the
   problem at hand.
5. **Assess the Solution:** Assess the validity of the solutions by applying
   physical principles such as conservation laws and dimensional analysis, use
   physical intuition to make sure quantities are of a reasonable magnitude and sign,
   and use various limiting cases to check the validity of the obtained
   solutions.
6. **Communicate and Defend the Solution:** Communicate the results with peers,
   defending the approximations made, the accuracy of the techniques used, and
   the assessment of the solutions.  Demonstrate insight into the nature of the
   solutions, making appropriate generalizations, and providing intuitive
   descriptions of the quantitative behavior found from solving the problem.

A further outcome relates to the department requirement for students to
demonstrate this proficiency through a series of general examinations, and a
more general requirement for the students to interact face-to-face with other
physicists.

7. **Proficiency**: Be able to demonstrate proficiency with these skills.  In
   particular, be able to rapidly formulate and analyze many classical
   mechanics problems without external references.


### Expectations for Student Effort

Students are expected to:

1. Stay up to date with reading assignments as outlined in the [Reading
   Schedule][reading schedule].

2. Participate in the online forums, both asking questions and addressing peers
   questions.

3. Identify areas of weakness, work with peers to understand difficult
   concepts, then present remaining areas of difficulty to the instructor for
   personal attention or for discussion in class.

4. Complete assignments on time, providing well crafted solutions to the posed
   problems that demonstrate mastery of the material through the [Learning
   Outcomes][learning outcomes] 1-6.  Final solutions much be written using proper English,
   including **complete sentences** with a clear logical progression through
   all steps of the solution.  Excessive verbosity is not required, but the
   progression through the solution must be clear to the reader, along with a
   justification of all assumptions and approximations made.

   Submitted solutions should not contain incomplete or random attempts at
   solving a problem: they should contain a streamline approach proceeding
   directly and logically from the formulation of the problem to the solution.
   (Student's are encouraged to discuss their intended approach with peers and
   with the instructor **well before the deadline** in order to obtain the
   feedback required to formulate a proper solution for submission).

5. Find or formulate exam problems at a level appropriate for completion of the
   physics department comprehensive examinations, and practice solving these
   under exam conditions, seeking help from the instructor as required to
   develop the required proficiency of the material.

6. Choose a topic for the final project, and obtain approval from the
   instructor by **November 1**.

7. Complete the final project, and present at the end of semester (typically
   one evening during the last week of classes, but the final date will chosen
   by polling everyone's schedules.

8. Successfully complete both the midterm and final examinations.

For each hour of lecture equivalent, students should expect to have a minimum
of two hours of work outside class.


### Assessment and Grading Policy

The learning outcomes will be assessed as follows:

**Assignments:**

: Throughout the course, students will be expected to demonstrate outcomes 1-6
  applied to well-formulated problems demonstrating the techniques currently
  being taught (see the following [Course Outline]).  Successful completion
  of the assignments will assess the student's ability with these skills while
  they have access to external resources such as the textbook, and without
  stringent time constraints.  A peer-grading component of the course will
  help ensure that written solutions effectively communicate the results as
  per outcome 6.
  
   :::{important}
   In as many assignments as possible, I will provide numerical solutions against which
   you can check your formula on [CoCalc].  In order for me to grade your homework, I 
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
   
   In Fall 2023, we plan to hold open iSciMath lab/office hours in the Band Room Friday
   2-4:30pm.  You may stop by these sessions to discuss any issues you have.  I am also
   available for appointment by Zoom.
   :::
  
**Exams:**

: The proficiency of the students to rapidly apply these skills without
  external resources (outcome 7) will be assessed through time-limited midterm
  and final examinations.  Time-permitting, exams will contain both written and oral
  portions to simulate our comprehensive exam procedure.  The oral portion will help
  assess the student's communication skills (outcome 6).

**Readings:**

: Students will be expected to keep up with the reading assignments.  This will be
  assesed by students participating in the class Perusall discussions, with at least 2
  annotations per reading assignment, or with a submission in the form of a summary of
  the reading, the student's own derivation of a key result, etc.  This will assess
  their ability to communicate about classical mechanics.

**Final Project:**

: The ability of the students to analysis an unstructured mechanics problem in
  an open-ended context will be assessed through their completion and defense
  of a final class project in an area of their choosing.  This will give the
  students a chance to exercise their skills in a context much closer to that
  in which they will encounter while performing physics research.

The final grade will be converted to a letter grade using the following scale: 

| Percentage P       | Grade |
| ------------------ | ----- |
| 90.0% ≤ P          | A     |
| 85.0% ≤ P \< 90.0% | A-    |
| 80.0% ≤ P \< 85.0% | B+    |
| 75.0% ≤ P \< 80.0% | B     |
| 70.0% ≤ P \< 75.0% | B-    |
| 65.0% ≤ P \< 70.0% | C+    |
| 60.0% ≤ P \< 65.0% | C     |
| 55.0% ≤ P \< 60.0% | C-    |
| 50.0% ≤ P \< 55.0% | D+    |
| 40.0% ≤ P \< 50.0% | D     |
| P \< 40.0%         | F     |

The following table shows how many point you may earn at most from each component of the
course:

- 10: Reading Assignments
- 40: Homework (or 20 Homework + 20 Project)
- 25: Midterm Exams:
- 25: Final Exam

### Optional Project

There will be an optional project in this course which may be used for 25 of the
homework points.  Further details will be discussed later: you may choose the topic, but
must run your proposal by the instructor.

### Exams

The exams will be administered in two parts, similar to how the department qualifying
exams are administered.  First you will submit a written portion.  You will then
schedule an oral exam with instructor during which you will be asked questions about
your written work, and given an opportunity to explain your reasoning.  If you do not
feel that the instructor has arrived at an accurate assessment of your exam performance
after the oral portion, you may opt to have your written exam graded in detail and will
receive that as your grade (with the exception being if it is discovered during the oral
portion that your written portion is not your work).  However, our experience is that
the oral portion generally improves ones grade.

The final exam is scheduled for Tuesday 12 December 2023 at 10:30am.

### Attendance and Make-up Policy 

While there is no strict attendance policy, students are expected attend an participate
in classroom activities and discussion. Students who miss class are expected to cover
the missed material on their own, e.g. by borrowing their classmates notes, reviewing
recorded lectures (if available), etc.


### Course Timeline
<!-- 16 Weeks -->

The following details the content of the course.  It essentially follows the main
textbook.  Content from the supplement will be inserted as appropriate
throughout. Details and further resources will be included on the lecture pages on the
[Canvas] server.

### Course Outline

1. Introduction and Basic Principles  (~1 week)

    - Why study classical mechanics?
    - Newtonian mechanics.
    - Symmetry and Conservation.
    - Central Forces
    - Kepler
    - Scattering

2. Accelerated Coordinate Systems (~1 week)

    - Change of coordinates
    - Centripetal acceleration
    - Coriolis effect

3. Lagrangian Dynamics (~2 weeks)

    - Why another formulation?
    - Constraints
    - Euler-Lagrange Equations
    - Calculus of Variations
    - Hamilton's Principle
    - Generalized momenta
    - The Path Integral approach to Quantum Mechanics

4. Small Oscillations (~1 week)

    - Normal modes
    - Linear Equations
    - Stability

5. Rigid Bodies (~1 week)

    - Moment of Inertia
    - Euler's Equations

6. Hamilton Dynamics (~2 weeks)

    - Canonical Transformations
    - Hamilton-Jacobi Theory
    - Action-Angle Variables
    - The Canonical Quantization approach to Quantum Mechanics

7. Strings, Waves, and Drums (~1 week)

    - Lagrangian for continuous systems
    - Boundary conditions
    - Numerical solutions of the wave equation

10. Non-linear Mechanics (SIII: Discrete Dynamical Systems) (~2 weeks)

    These topics will be introduced as we progress through the course,
    inserted into the appropriate locations.

11. Canonical Perturbation Theory (~2 weeks)

    These topics will be introduced as we progress through the course,
    inserted into the appropriate locations.

12. Special topics and review.

    - How these topics will be covered depends on interest.  One option
      is to discuss superfluidity with some numerical examples
      demonstrating vortices, vortex dynamics, and related phenomena.
    - Duffing Oscillator
    - Stability Analysis
    - Chaos
    - Fluids
    - Special Relativity

## Other Information

### Policy for the Use of Large Language Models (LLMs) or Generative AI in Physics Courses

The use of LLMs or Generative AI such as Chat-GPT is becoming prevalent, both in education and in industry.  As such, we believe that it is important for students to recognize the capabilities and inherent limitations of these tools, and use them appropriately.

To this end, **please submit 4 examples of your own devising:**
* Two of which demonstrate the phenomena of "hallucination" -- Attempt to use the tool to learn something you know to be true, and catch it making plausible sounding falsehoods.
* Two of which demonstrate something useful (often the end of a process of debugging and correcting the AI).

Note: one can find plenty of examples online of both cases.  Use these to better understand the capabilities and limitations of the AIs, but for your submission, please find your own example using things you know to be true. *If you are in multiple courses, you may submit the same four examples for each class, but are encouraged to tailor your examples to the course.*

Being able to independently establish the veracity of information returned by a search, an AI, or indeed any publication, is a critical skill for a scientist.  **If you are the type of employee who can use tools like ChatGPT to write prose, code etc., but not accurately validate the results, then you are exactly the type of employee that AI will be able to replace.** 

Any use of Generative AI or similar tools for submitted work must include:
1. **A complete description of the tool.** (E.g. *"ChatGPT Version 3.5 via CoCalc's interface"* or *Chat-GPT 4 through Bing AI using the Edge browser"*, etc.)
2. **A complete record of the queries issued and response provided.**  (This should be provided as an attachment, appendices, or supplement.)
3. **An attribution statement consistent with the following:**
   *“The author generated this <text/code/etc.> in part with <GPT-3, OpenAI’s large-scale language-generation model/etc.> as documented in appendix <1>. Upon generating the draft response, the author reviewed, edited, and revised the response to their own liking and takes ultimate responsibility for the content.”*

### COVID-19 Statement
Per the proclamation of Governor Inslee on August 18, 2021, **masks that cover both the
nose and mouth must be worn by all people over the age of five while indoors in public
spaces.**  This includes all WSU owned and operated facilities. The state-wide mask mandate
goes into effect on Monday, August 23, 2021, and will be effective until further
notice. 
 
Public health directives may be adjusted throughout the year to respond to the evolving
COVID-19 pandemic. Directives may include, but are not limited to, compliance with WSU’s
COVID-19 vaccination policy, wearing a cloth face covering, physically distancing, and
sanitizing common-use spaces.  All current COVID-19 related university policies and
public health directives are located at
[https://wsu.edu/covid-19/](https://wsu.edu/covid-19/).  Students who choose not to
comply with these directives may be required to leave the classroom; in egregious or
repetitive cases, student non-compliance may be referred to the Center for Community
Standards for action under the Standards of Conduct for Students.

### Academic Integrity

Academic integrity is the cornerstone of higher education.  As such, all members of the
university community share responsibility for maintaining and promoting the principles
of integrity in all activities, including academic integrity and honest
scholarship. Academic integrity will be strongly enforced in this course.  Students who
violate WSU's Academic Integrity Policy (identified in Washington Administrative Code
(WAC) [WAC 504-26-010(4)][wac 504-26-010(4)] and -404) will fail the course, will not
have the option to withdraw from the course pending an appeal, and will be Graduate:
6300, 26300 reported to the Office of Student Conduct.

Cheating includes, but is not limited to, plagiarism and unauthorized collaboration as
defined in the Standards of Conduct for Students, [WAC 504-26-010(4)][wac
504-26-010(4)]. You need to read and understand all of the [definitions of
cheating][definitions of cheating].  If you have any questions about what is and is not
allowed in this course, you should ask course instructors before proceeding.

If you wish to appeal a faculty member's decision relating to academic integrity, please
use the form available at \_communitystandards.wsu.edu. Make sure you submit your appeal
within 21 calendar days of the faculty member's decision.

Academic dishonesty, including all forms of cheating, plagiarism, and fabrication, is
prohibited. Violations of the academic standards for the lecture or lab, or the
Washington Administrative Code on academic integrity

### Students with Disabilities

Reasonable accommodations are available for students with a documented
disability. If you have a disability and need accommodations to fully
participate in this class, please either visit or call the Access
Center at (Washington Building 217, Phone: 509-335-3417, E-mail:
<mailto:Access.Center@wsu.edu>, URL: <https://accesscenter.wsu.edu>) to schedule
an appointment with an Access Advisor. All accommodations MUST be
approved through the Access Center. For more information contact a
Disability Specialist on your home campus.

### Campus Safety

Classroom and campus safety are of paramount importance at Washington
State University, and are the shared responsibility of the entire
campus population. WSU urges students to follow the “Alert, Assess,
Act,” protocol for all types of emergencies and the “[Run, Hide, Fight]”
response for an active shooter incident. Remain ALERT (through direct
observation or emergency notification), ASSESS your specific
situation, and ACT in the most appropriate way to assure your own
safety (and the safety of others if you are able).

Please sign up for emergency alerts on your account at MyWSU. For more
information on this subject, campus safety, and related topics, please
view the FBI’s [Run, Hide, Fight] video and visit [the WSU safety
portal][the wsu safety portal].

### Students in Crisis - Pullman Resources 

If you or someone you know is in immediate danger, DIAL 911 FIRST! 

* Student Care Network: https://studentcare.wsu.edu/
* Cougar Transit: 978 267-7233 
* WSU Counseling and Psychological Services (CAPS): 509 335-2159 
* Suicide Prevention Hotline: 800 273-8255 
* Crisis Text Line: Text HOME to 741741 
* WSU Police: 509 335-8548 
* Pullman Police (Non-Emergency): 509 332-2521 
* WSU Office of Civil Rights Compliance & Investigation: 509 335-8288 
* Alternatives to Violence on the Palouse: 877 334-2887 
* Pullman 24-Hour Crisis Line: 509 334-1133 

[communitystandards.wsu.edu]: https://communitystandards.wsu.edu/
[definitions of cheating]: https://apps.leg.wa.gov/WAC/default.aspx?cite=504-26-010
[run, hide, fight]: https://oem.wsu.edu/emergency-procedures/active-shooter/
[the wsu safety portal]: https://oem.wsu.edu/about-us/
[wac 504-26-010(4)]: https://apps.leg.wa.gov/WAC/default.aspx?cite=504-26
[SSH]: <https://en.wikipedia.org/wiki/Secure_Shell> "SSH on Wikipedia"
[CoCalc]: <https://cocalc.com> "CoCalc: Collaborative Calculation and Data Science"
[GitLab]: <https://gitlab.com> "GitLab"
[GitHub]: <https://github.com> "GitHub"
[Git]: <https://git-scm.com> "Git"
[Anki]: <https://apps.ankiweb.net/> "Powerful, intelligent flash cards."

[Official Course Repository]: <https://gitlab.com/wsu-courses/physics-521-classical-mechanics> "Official Course Repository hosted on GitLab"
[Shared CoCalc Project]: <https://cocalc.com/c5a0bdae-e17c-46a1-ada4-78c6556d9429/> "Shared CoCalc Project"
[WSU Courses CoCalc project]: <https://cocalc.com/projects/c31d20a3-b0af-4bf7-a951-aa93a64395f6>
[Perusall]: <https://app.perusall.com/courses/2023-fall-physics-521-pullm-1-01-03206-classical-mechanics-i-471434971>
[Canvas]: <https://wsu.instructure.com/courses/1567877/>
