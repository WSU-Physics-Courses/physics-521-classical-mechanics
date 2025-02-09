Phys 521 - Classical Mechanics
==============================
[![Documentation Status](https://readthedocs.org/projects/physics-521-classical-mechanics-i/badge/?version=latest)](https://physics-521-classical-mechanics-i.readthedocs.io/en/latest/?badge=latest)
[![gitlab pipeline status](https://gitlab.com/wsu-courses/physics-521-classical-mechanics/badges/main/pipeline.svg)](https://gitlab.com/wsu-courses/physics-521-classical-mechanics/-/commits/main)
[![gitlab coverage report](https://gitlab.com/wsu-courses/physics-521-classical-mechanics/badges/main/coverage.svg)](https://gitlab.com/wsu-courses/physics-521-classical-mechanics/-/commits/main)

This is the main project for the [WSU Physics][] course
**Phys 521: Classical Mechanics** first offered in [Fall 2023](https://schedules.wsu.edu/List/Pullman/20233/Phys/521/01).

Physics has a successful track record of providing effective solutions to complex
problems outside its specific domain. This course will focus on using efficient
numerical techniques inspired by physics to solve challenging problems in a wide variety
of applications.  Techniques will be chosen from physics applications, but also applied
to problems outside of the physics domain including economics, biology, sociology, etc.
Students will be introduced to powerful numerical toolkits based on the
[SciPy](https://www.scipy.org/) and [NumFocus](https://numfocus.org) ecosystem. Using
the [CoCalc](https://cocalc.com/) platform will enable rapid development and prototyping
with an explicit path to stable, tested, and performant codes capable of supporting
research, or industry applications.

[![Documentation Status](https://readthedocs.org/projects/physics-521-classical-mechanics-i/badge/?version=latest)](https://physics-521-classical-mechanics-i.readthedocs.io/en/latest/?badge=latest)

Main project for the [WSU Physics][] course **Physics 521: Classical Mechanics I**.

## Installation

This will generate an environment you can use to work with the project.  Once
this is done you can make the documentation, tests, etc. with commands like:

### Preliminaries

The latter will host the documentation on https://localhost:8000 and auto-update when you
make changes. If you want to manually interact with the environment, then you can run:

```bash
make shell
```

Note: the `LC_*` environment variables set above allow us to customize the environment
later so that multiple users can share the same environment.  This deals with a quirk of
[CoCalc] in that one connects to a project with a specialized username.  This means that
everyone has the same username, making it difficult for different users to commit
changes to version control.  To remedy this, my [`mmf-setup`] packages modifies the
`~/.bash_alises` file to use the variables sent above.

## CoCalc

To build the documentation on CoCalc, open the following file:

* [`SphinxAutoBuildServer.term`](SphinxAutoBuildServer.term)

This will build the documentation, and serve it on 
`https://cocalc.com/c5a0bdae-e17c-46a1-ada4-78c6556d9429/raw/.repositories/phys-521-classical-mechanics/Docs/_build/html/index.html`.



## Developer Notes

To install everything on [CoCalc], do the following after connecting to your project
with SSH as described above:

1. Make a `repositories` directory and clone the project:

See `Docs/index.md` for more details.

To use this repository:

1. *(Optional)* 
   * Create accounts on [CoCalc][] and [GitLab][], a project on [CoCalc][], and a
   repo on [GitLab][].  Send your [GitLab][] account name to your instructor.
   * Create [Create SSH
   keys](https://doc.cocalc.com/project-settings.html#ssh-keys), and add them [to your
   CoCalc account](https://doc.cocalc.com/account/ssh.html) and [to your GitLab
   account](https://docs.gitlab.com/ee/ssh/).
   * SSH into your [CoCalc][] project for the remaining steps.
2. Clone this repo and initialize it:

   ```bash
   mkdir repositories
   cd repositories
   git clone git@gitlab.com:wsu-courses/physics-521-classical-mechanics.git
   ```
2. Run `make init`:
  
   ```bash
   cd ~/repositories/physics-521-classical-mechanics
   git pull   # Make sure repo is up to date.
   make init
   ```
   
   This will create a [Conda][] environment you can activate `conda activate envs/phys-521`,
   and a Jupyter kernel called `phys-521` that you can select from notebooks.
   
     ```bash
     python3 -m pip install --user --upgrade mmf-setup
     mmf_setup cocalc
     ```
     
     The last step installs some useful files including `~/.inputrc` (arrows for command
     completion), `~/.bash_aliases` (defines some useful commands like `findpy`, and
     `finda`, uses the `LC_*` variables defined above, etc.)
   * Depending on the state of the `USE_ANACONDA2020` flag, one of the following is
     done. If `USE_ANACONDA2020` is `true`, then the `anaconda2020` environment on
     [CoCalc] is activated, otherwise [Miniconda] is installed in `~/.miniconda3` and
     the `base` environment is updated with `conda`, `anaconda-project`, and `mamba`.
     This is roughly equivalent to the following (but also checking hashes, using proper
     paths, using `/tmp`, disabling prompts, etc.)

     ```bash
     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
     bash Miniconda3-latest-Linux-x86_64.sh
     conda update conda
     conda install anaconda-project
     conda install --override-channels --channel conda-forge mamba
     conda clean --all
     ```
   * Once [Conda] is installed, [Anaconda Project] is used to setup the specific project
     environment:

     ```bash
     cd ~/repositories/physics-521-classical-mechanics
     anaconda-project prepare
     anaconda-project run init
     ```

     This installs all of the dependencies listed in `anaconda-project.yaml` into an
     environment in `envs/phys-521`, and then creates a Jupyter kernel called
     `phys-521` that the [CoCalc] notebooks can find.  The latter step is roughly
     equivalent to running the following with the appropriate python installed in
     `envs/phys-521`:

     ```bash
     python3 -m ipykernel install --user --name "phys-521" --display-name "Python 3 (phys-521)"
     ```

   * Finally, a line is added to the end of `~/.bash_aliases` to activate the
     environment when you ssh to the [CoCalc] project.  If you logout and log back in,
     you should now see the following prompt:

     ```bash
     $ ssh cc521shared
     (phys-521) ~$ 
     ```
 
 You should now be able to use the `phys-521` kernel in notebooks.

This will run [`sphinx-autobuild`](https://github.com/executablebooks/sphinx-autobuild)
which will launch a webserver on http://127.0.0.1:8000 and rebuild the docs whenever you
save a change.

Here is the play-by-play for setting up the documentation.

```bash
cd Docs
sphinx-quickstart
wget https://brand.wsu.edu/wp-content/themes/brand/images/pages/logos/wsu-signature-vertical.svg -O _static/wsu-logo.svg 
cp -r ../envs/default/lib/python3.9/site-packages/sphinx_book_theme/_templates/* _templates
```

I then edited the `conf.py`

```bash
hg add local.bib _static/ _templates/
```

## CoCalc Setup


* [Purchase a license](https://cocalc.com/settings/licenses) with 2 projects to allow
  the course and [WSU Courses CoCalc project][] and [Shared CoCalc Project][] to run.  This
  approach requires the students to pay $14 for access four the term (4 months).  They
  can optionally use any license they already have instead.
   
  Optionally, one might opt to purchase a license for $n+2$ projects where $n$ is the
  number of students, if there is central funding available.  See [Course Upgrading
  Students](https://doc.cocalc.com/teaching-upgrade-course.html#course-upgrading-students)
  for more details.
  
* Next, [create a course](https://doc.cocalc.com/teaching-create-course.html).  I do
  this in my [WSU Courses CoCalc project][].



* Create a [Shared CoCalc Project][] and activate the license for this project so that it
  can run.  I then add the SSH key to may `.ssh/config` files so I can quickly login.

* Clone the repos into the shared project and initialize the project.  Optional, but
  highly recommend -- use my [`mmf-setup`][] project to provide some useful features

  ```bash
  ssh smc<project name>       # My alias in .ssh/config
  python3 -m pip install mmf_setup
  mmf_setup cocalc
  ```
  
  This provides some instructions on how to use the CoCalc configuration.  The most
  important is to forward your user agent and set your `hg` and `git` usernames:
  
  ```bash
  ~$ mmf_setup cocalc
  ...
  If you use version control, then to get the most of the configuration,
  please make sure that you set the following variables on your personal
  computer, and forward them when you ssh to the project:

      # ~/.bashrc or similar
      LC_HG_USERNAME=Your Full Name <your.email.address+hg@gmail.com>
      LC_GIT_USEREMAIL=your.email.address+git@gmail.com
      LC_GIT_USERNAME=Your Full Name

  To forward these, your SSH config file (~/.ssh/config) might look like:

      # ~/.ssh/config
      Host cc_phys-521-classical-mechanics
        User c5a0bdaee17c46a1ada478c6556d9429
    
      Host cc_phys-521-classical-mechanics
        HostName ssh.cocalc.com
        ForwardAgent yes
        SendEnv LC_HG_USERNAME
        SendEnv LC_GIT_USERNAME
        SendEnv LC_GIT_USEREMAIL
        SetEnv LC_EDITOR=vi
  ```
  
  Logout and log back in so we have the forwarded credentials, and now clone the repos.
  
  ```bash
  git clone https://gitlab.com/wsu-courses/physics-521-classical-mechanics.git phys-521-classical-mechanics
  cd phys-521-classical-mechanics
  make
  ```
  
  The last step runs `git clone git@gitlab.com:wsu-courses/physics-521-classical-mechanics_resources.git _ext/Resources` which puts the resources folder in `_ext/Resources`.

* Create an environment:

  ```bash
  ssh cc_phys-521-classical-mechanics
  cd phys-521-classical-mechanics
  anaconda2021
  anaconda-project prepare
  conda activate envs/phys-521
  python -m ipykernel install --user --name "phys-521" --display-name "Python 3 (phys-521)"
  ```

  This will create a Conda environment as specified in `anaconda-project.yml` in `envs/phys-521`.


# Funding

<a href="https://www.nsf.gov"><img width="10%"
src="https://nsf.widen.net/content/txvhzmsofh/png/" />
</a>
<br>

Some of the material presented here is based upon work supported by the National Science
Foundation under [Grant Number 2012190](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2012190). Any opinions, findings, and conclusions or
recommendations expressed in this material are those of the author(s) and do not
necessarily reflect the views of the National Science Foundation.


<!-- Links -->
[CoCalc]: <https://cocalc.com> "CoCalc: Collaborative Calculation and Data Science"
[WSU Physics]: <https://physics.wsu.edu> "WSU Physics Department"
[GitLab]: <https://gitlab.com> "GitLab"
[GitHub]: <https://github.com> "GitHub"
[Git]: <https://git-scm.com> "Git"
[Mercurial]: <https://www.mercurial-scm.org> "Mercurial"
[hg-git]: <https://hg-git.github.io> "The Hg-Git mercurial plugin"
[Heptapod]: <https://heptapod.net> "Heptapod: is a community driven effort to bring Mercurial SCM support to GitLab"
[Jupyter]: <https://jupyter.org> "Jupyter"
[Jupytext]: <https://jupytext.readthedocs.io> "Jupyter Notebooks as Markdown Documents, Julia, Python or R Scripts"
[Resources project]: <https://gitlab.com/wsu-courses/physics-521-classical-mechanics_resources> "Private course resources repository."
[Official Course Repository]: <https://gitlab.com/wsu-courses/physics-521-classical-mechanics/> "Official Physics 581 Repository hosted on GitLab"
[file an issue]: <https://gitlab.com/wsu-courses/physics-521-classical-mechanics/-/issues> "Issues on the class GitLab project."
[Conda]: <https://docs.conda.io/en/latest/> "Conda: Package, dependency and environment management for any language—Python, R, Ruby, Lua, Scala, Java, JavaScript, C/ C++, FORTRAN, and more."
[`mmf-setup`]: <https://pypi.org/project/mmf-setup/>
[Miniconda]: <https://docs.conda.io/en/latest/miniconda.html> "Miniconda is a free minimal installer for conda."
[MyST]: <https://myst-parser.readthedocs.io/en/latest/> "MyST - Markedly Structured Text"
[Read the Docs]: <https://readthedocs.org> "Read the Docs homepage"
[WSU Physics]: <https://physics.wsu.edu> "WSU Department of Physics and Astronomy"
[WSU Mathematics]: <https://www.math.wsu.edu/> "WSU Department of Mathematics and Statistics"
[`anaconda-project`]: <https://anaconda-project.readthedocs.io> "Anaconda Project: Tool for encapsulating, running, and reproducing data science projects."
[`anybadge`]: <https://github.com/jongracecox/anybadge> "Python project for generating badges for your projects"
[`conda-forge`]: <https://conda-forge.org/> "A community-led collection of recipes, build infrastructure and distributions for the conda package manager."
[`genbadge`]: <https://smarie.github.io/python-genbadge/> "Generate badges for tools that do not provide one."
[`mmf-setup`]: <https://pypi.org/project/mmf-setup/> "PyPI mmf-setup page"
[`pytest`]: <https://docs.pytest.org> "pytest: helps you write better programs"
[hg-git]: <https://hg-git.github.io> "The Hg-Git mercurial plugin"
[GitLab test coverage visualization]: <https://docs.gitlab.com/ee/user/project/merge_requests/test_coverage_visualization.html>



[Official Course Repository]: <https://gitlab.com/wsu-courses/physics-521-classical-mechanics> "Official Course Repository hosted on GitLab"
[Shared CoCalc Project]: <https://cocalc.com/c5a0bdae-e17c-46a1-ada4-78c6556d9429/> "Shared CoCalc Project"
[WSU Courses CoCalc project]: <https://cocalc.com/projects/c31d20a3-b0af-4bf7-a951-aa93a64395f6>


[GitHub Mirror]: <https://github.com/WSU-Physics-Courses/physics-521-classical-mechanics> "GitHub mirror"
[GitLab public repo]: <https://gitlab.com/wsu-courses/physics-521-classical-mechanics> "GitLab public repository."
[Gitlab private resources repo]: <https://gitlab.com/wsu-courses/physics-521-classical-mechanics_resources> "Private resources repository."
[file an issue]: <https://gitlab.com/wsu-courses/physics-521-classical-mechanics/-/issues> "Issues on the GitLab project."

<!-- End Links -->
