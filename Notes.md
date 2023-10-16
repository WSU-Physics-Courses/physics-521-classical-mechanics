Instructor Notes
================

These are notes about creating and maintaining this repository.  They may be in a state
of flux.

To Do
-----

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

### Read The Docs

The documents are hosted at [Read the
Docs](https://readthedocs.org/projects/physics-521-classical-mechanics-i/) (RtD)
where they should be build automatically whenever the main branch is pushed.  To get
this working, one needs to tell RtD which packages to install, and they [recommend using
a configuration file](https://docs.readthedocs.io/en/stable/config-file/v2.html) for
this called `.readthedocs.yaml`.

### Gotchas

* Be careful not to use [MyST] features in the `README.md` file as this forms the
  landing page on the
  [GitLab](https://gitlab.com/wsu-courses/physics-521-classical-mechanics) and
  [GitHub](https://github.com/WSU-Physics-Courses/physics-521-classical-mechanics)
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
  
## GitHub Mirror

[GitHub] has a different set of tools, so it is useful to mirror the repo there so we
can take advantage of these:

* [GitLab Main Repo](https://gitlab.com/wsu-courses/physics-521-classical-mechanics)
* [GitHub Mirror](https://github.com/WSU-Physics-Courses/physics-521-classical-mechanics)

To do this:

1. Create an empty [GitHub
   project](https://github.com/WSU-Physics-Courses/physics-521-classical-mechanics).
   Disable
   [Features](https://github.com/WSU-Physics-Courses/physics-521-classical-mechanics/settings)
   like `Wikis`, `Issues`, `Projects`, etc. which are not desired for a mirror.
2. Get a personal token from [GitHub] as [described
   here](https://hg.iscimath.org/help/user/project/repository/repository_mirroring#setting-up-a-push-mirror-from-gitlab-to-github).
   Create a token here [**Settings > Developer settings > Personal access
tokens**](https://github.com/settings/tokens) with `repo` access, `admin:repo_hook`
   access, and `delete_repo` access.  Copy the key. 
3. Go to your [**Settings > Repository > Mirroring
   respositories**](https://gitlab.com/wsu-courses/physics-521-classical-mechanics/-/settings/repository)
   in you GitLab repo and use the URL to your GitHub repo using the following format:
   `https://<your_github_username>@github.com/<your_github_group>/<your_github_project>.git`.
   I.e.:
   
   ```
   https://mforbes@github.com/WSU-Physics-Courses/physics-521-classical-mechanics.git
   ```
   
   Include your name here (the user associated with the key you just generated) and
   use the key as the password.  Choose **Mirror direction** to be **Push**.
   Optionally, you can choose to mirror only protected branches: this would be a good
   choice if you were mirroring a private development repo and only wanted to public
   branches to be available on [GitHub].

Now whenever you push changes to GitLab, they will be mirrored on GitHub, allowing you
to use GitHub features like their CI, Notebook viewer etc.


[WSU Courses CoCalc project]: <https://cocalc.com/projects/c31d20a3-b0af-4bf7-a951-aa93a64395f6>
[GitHub]: <https://github.com> "GitHub"
[`pytest`]: <https://docs.pytest.org> "pytest: helps you write better programs"
[Read the Docs]: <https://readthedocs.org> "Read the Docs homepage"
[`anaconda-project`]: <https://anaconda-project.readthedocs.io> "Anaconda Project: Tool for encapsulating, running, and reproducing data science projects."
