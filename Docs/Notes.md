Instructor Notes
================

These are notes about creating and maintaining this repository.  They may be in a state
of flux.

To Do
-----
* Fix `{{ course_package }}` replacement in `Syllabus.md`.




## Maintainer Notes
Try to keep this upper-level project as clean as possible, matching the layout expected
for the students.  This will be turned into a skeleton at some point.


## Tools

### Cookie Cutter
* https://github.com/cookiecutter/cookiecutter
* https://cookiecutter-hypermodern-python.readthedocs.io/

### Anaconda Project

```bash
anaconda-project init
mv anaconda-project.yml anaconda-project.yaml  # https://stackoverflow.com/a/22268973/1088938
anaconda-project add-packages python=3.9 scipy matplotlib sphinx notebook ipykernel nb_conda
anaconda-project add-lpackages conda-forge::uncertainties
anaconda-project add-packages conda-forge::sphinx-panels conda-forge::sphinx-book-theme conda-forge::myst-nb
anaconda-project add-packages pytest-cov pytest-flake8 pytest-xdist 
anaconda-project add-packages --pip sphinxcontrib-zopeext sphinxcontrib-bibtex mmf-setup
```

Then I changed all occurrences of `default` to `phys-581-2021` in
`anaconda-project.yaml` so that the environments have a consistent naming convention.


To clean:

```bash
anaconda-project clean
```

[`nbrr`]
: Maybe look at https://github.com/pyoceans/nbrr as a way of generating dependencies.
  This is used by [PyVis examples](https://examples.pyviz.org/make_project.html).

[`nbrr`]: <https://github.com/pyoceans/nbrr> "Jupyter Notebook Reproducible Repositories" 

## Repository Setup

Can use GitHub, GitLab, or Heptapod.  With automatic pushing to GitHub, one and run the
following CI's:
* LGTM.com

* One course repo.  Students clone their own fork and pull changes.  Assignments
  distributed as tests.
* How to grade?  Student's can keep projects private (but probably will not have access
  to badges.)  Run tests on Student's CoCalc servers or with CI?
  
### Things Students Need to Change
These are things that each student must change in their repo.  These should be fields in
the final skeleton project.

* `pyproject.toml`: `description`, `repository`, `documentation`, and ultimately
  `version`, `license`.
* 

## Best Practices
* Use Jupytext and version control the associated python files.  Only commit the full
  notebooks (with output) when you want to archive documentation.
  * Maybe do this on an "output" branch or something so the main repo does not get
    cluttered?


## Tests and Grading

Tests are setup using [`pytest`].  The configuration is in `pyprojects.toml`.  Each
assignment will have an official test-suite called
`tests/assignmentX/test_official_assignmentX.py` where `X` is the assignment number.  These
official tests will be marked with the special marker `assignmentX`, so that the
official tests can be run with:

```bash
pytest -k test_official_assignmentX --no-cov  # Disable coverage.
```

Students should be instructed not to put tests in `tests/assignmentX/test_official_assignmentX.py`


## Docs

We use [MyST] for writing the documentation, which is the compiled into HTML, PDF,
etc. with [Sphinx] using the ideas of [JupyerBook] but simply using [Sphinx].  The
documents live in `Docs/` with the exception of `README.md` which lives at the top level
and provides the landing page for the
[GitLab](https://gitlab.com/wsu-courses/physics-581-physics-inspired-computation) and
[GitHub](https://github.com/WSU-Physics-Courses/physics-581-physics-inspired-computation)
repos.

To build the documents interactively:

```bash
make doc-server
```

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

I then edited the `conf.py`.

```bash
hg add local.bib _static/ _templates/
```

Substitutions can be used for parameters such as the `{{ class_room }} = ` {{ class_room
}}.  These can be defined in `cony.py` in the `myst_substitutions` dictionary and follow
Jinja2 conventions.  I use these to factor common information which will ultimately be
part of the templates.

### Read The Docs

The documents are hosted at [Read the
Docs](https://readthedocs.org/projects/physics-581-physics-inspired-computational-techniques/) (RtD)
where they should be build automatically whenever the main branch is pushed.  To get
this working, one needs to tell RtD which packages to install, and they [recommend using
a configuration file](https://docs.readthedocs.io/en/stable/config-file/v2.html) for
this called `.readthedocs.yaml`.

### Gotchas

* Be careful not to use [MyST] features in the `README.md` file as this forms the
  landing page on the
  [GitLab](https://gitlab.com/wsu-courses/physics-581-physics-inspired-computation) and
  [GitHub](https://github.com/WSU-Physics-Courses/physics-581-physics-inspired-computation)
  repos, neither of which support [MyST].  Check these to be sure that they look okay.
* We literally include the top-level `README.md` files as the first page of the
  documentation in `Docs/index.md`.  This as one side effect that when running `make
  doc-server` (`sphinx-autobuild`), edits to `README.md` do not trigger rebuilding of
  the `index.md` file.  While actively editing, I include `README.md` as a separate
  file, and view that.
* We are using [`anaconda-project`], but [Read the Docs] does not directly support
  provisioning from this, however, you can make the `anaconda-project.yaml` file look
  like an `environment.yaml` file if you [change `packages:` to `dependencies:` as long
  as you can ensure `anaconda-project>=0.8.4`](https://github.com/Anaconda-Platform/anaconda-project/issues/265#issuecomment-903206709).  This allows one to simply install
  this with `conda env --file anaconda-project.yaml`.  
* Since we are just doing a Conda install on RtD, we don't run `anaconda-project run
  init` and the kernel used by our notebooks does not get installed.  We can do this in
  the Sphinx `Docs/conf.py` file:
  
  ```python
  # Docs/conf.py
  ...
  def setup(app):
      import subprocess

      subprocess.check_call(["anaconda-project", "run", "init"])
  ```
  
## CoCalc Setup

* [Purchase a license](https://cocalc.com/settings/licenses) with 2 projects to allow
  the course and [WSU Courses CoCalc project] and [Shared CoCalc Project] to run.  This
  approach requires the students to pay $14 for access four the term (4 months).  They
  can optionally use any license they already have instead.
   
  Optionally, one might opt to purchase a license for $n+2$ projects where $n$ is the
  number of students, if there is central funding available.  See [Course Upgrading
  Students](https://doc.cocalc.com/teaching-upgrade-course.html#course-upgrading-students)
  for more details.
  
* Next, [create a course](https://doc.cocalc.com/teaching-create-course.html).  I do
  this in my [WSU Courses CoCalc project] called
  [581-2021](https://cocalc.com/projects/c31d20a3-b0af-4bf7-a951-aa93a64395f6/files/PhysicsInspiredComputation/581-2021.course).
  
  Open this course and create the [Shared CoCalc Project].  Activate the license for
  this project so that it can run.  I then add the SSH key to may `.ssh/config` files so
  I can quickly login.

* Clone the repos into the shared project and initialize the project.  Optional, but
  highly recommend -- use my [`mmf-setup`] project to provide some useful features

  ```bash
  ssh smc581shared       # My alias in .ssh/config
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
      Host cc-project1
        User ff1cb986f...
    
      Host cc*
        HostName ssh.cocalc.com
        ForwardAgent yes
        SendEnv LC_HG_USERNAME
        SendEnv LC_GIT_USERNAME
        SendEnv LC_GIT_USEREMAIL
        SetEnv LC_EDITOR=vi
  ```
  
  Logout and log back in so we have the forwarded credentials, and now clone the repos.
  
  ```bash
  git clone git@gitlab.com:wsu-courses/physics-581-physics-inspired-computation.git
  cd physics-581-physics-inspired-computation
  make
  ```
  
  The last step runs `git clone
  git@gitlab.com:wsu-courses/physics-581-physics-inspired-computation_resources.git
  _ext/Resources` which puts the resources folder in `_ext/Resources`.

* Create an environment:

  ```bash
  ssh smc581shared
  cd physics-581-physics-inspired-computation
  anaconda2020
  anaconda-project prepare
  conda activate envs/default
  python -m ipykernel install --user --name "PHYS-581-2021" --display-name "Python 3 (PHYS-581-2021)"
  ```

  This will create a Conda environment as specified in `anaconda-project.yaml` in `envs/default`.

## GitHub Mirror

[GitHub] has a different set of tools, so it is useful to mirror the repo there so we
can take advantage of these:

* [GitLab Main Repo](https://gitlab.com/wsu-courses/physics-581-physics-inspired-computation)
* [GitHub Mirror](https://github.com/WSU-Physics-Courses/physics-581-physics-inspired-computation)

To do this:

1. Create an empty [GitHub
   project](https://github.com/forbes-group/physics-581-physics-inspired-computation).
   Disable
   [Features](https://github.com/WSU-Physics-Courses/physics-581-physics-inspired-computation/settings)
   like `Wikis`, `Issues`, `Projects`, etc. which are not desired for a mirror.
2. Get a personal token from [GitHub] as [described
   here](https://hg.iscimath.org/help/user/project/repository/repository_mirroring#setting-up-a-push-mirror-from-gitlab-to-github).
   Create a token here [**Settings > Developer settings > Personal access
tokens**](https://github.com/settings/tokens) with `repo` access, `admin:repo_hook`
   access, and `delete_repo` access.  Copy the key. 
3. Go to your [**Settings > Repository > Mirroring
   respositories**](https://gitlab.com/wsu-courses/physics-581-physics-inspired-computation/-/settings/repository)
   in you GitLab repo and use the URL to your GitHub repo using the following format:
   `https://<your_github_username>@github.com/<your_github_group>/<your_github_project>.git`.
   I.e.:
   
   ```
   https://mforbes@github.com/WSU-Physics-Courses/physics-581-physics-inspired-computation.git
   ```
   
   Include your name here (the user associated with the key you just generated) and
   use the key as the password.  Choose **Mirror direction** to be **Push**.
   Optionally, you can choose to mirror only protected branches: this would be a good
   choice if you were mirroring a private development repo and only wanted to public
   branches to be available on [GitHub].

Now whenever you push changes to GitLab, they will be mirrored on GitHub, allowing you
to use GitHub features like their CI, Notebook viewer etc.


[WSU Courses CoCalc project]: <https://cocalc.com/projects/c31d20a3-b0af-4bf7-a951-aa93a64395f6>
[Shared CoCalc Project]: (https://cocalc.com/projects/74852aba-2484-4210-9cf0-e7902e5838f4/) "581-2021 Shared CoCalc Project"
[GitHub]: <https://github.com> "GitHub"
[`pytest`]: <https://docs.pytest.org> "pytest: helps you write better programs"
[Read the Docs]: <https://readthedocs.org> "Read the Docs homepage"
[`anaconda-project`]: <https://anaconda-project.readthedocs.io> "Anaconda Project: Tool for encapsulating, running, and reproducing data science projects."
