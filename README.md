Physics 521: Classical Mechanics I
==================================

[![Documentation Status](https://readthedocs.org/projects/physics-521-classical-mechanics-i/badge/?version=latest)](https://physics-521-classical-mechanics-i.readthedocs.io/en/latest/?badge=latest)

Main project for the [WSU Physics] course **Physics 521: Classical Mechanics I**.

## Installation

This project is intended to be installed on [CoCalc], but should work on other
computers.

### Preliminaries

Before performing the main install, you should complete the following preliminaries,
which will ensure that you can access the repository with SSH etc.  These may vary by
platform, but after you are done

1. Create a [CoCalc] project in which to work. (We will use the class [Shared CoCalc
   Project] for this example.)
2. (Optional) Create a [GitLab] account and inform your instructor of your username so
   that they can add you to the class private repository.
3. On your computer, create an SSH key, authenticate to this, and optionally, add this
   to your keychain manager (e.g. `Keychain Access.app` on Mac OS X).
4. Copy the associated public key to your [GitLab] account and to your [CoCalc] account.
5. Add the appropriate entry to your `~/.ssh/config` file so you can `ssh` to the
   [CoCalc] project, forwarding your keys.  For example, with the class [Shared CoCalc
   Project], I do this with the following configuration:
   
   ```ini
   # ~/.ssh/config
   ...
   Host cc521shared
     User 31c120c9295644209d6f374a6ee32df3
   Host cc*
     HostName ssh.cocalc.com
     ForwardAgent yes
     SetEnv LC_HG_USERNAME=Michael McNeil Forbes <michael.forbes+python@gmail.com>
     SetEnv LC_GIT_USERNAME=Michael McNeil Forbes
     SetEnv LC_GIT_USEREMAIL=michael.forbes+github@gmail.com
     SetEnv LC_EDITOR=vi
   Host *
     IgnoreUnknown UseKeychain
     UseKeychain yes
     AddKeysToAgent yes
     AddressFamily inet
     # Force IPv4
     # https://www.electricmonk.nl/log/2014/09/24/
     #         ssh-port-forwarding-bind-cannot-assign-requested-address/
   ```

Once these are done, you should be able to ssh directly to the [CoCalc] project **after
first starting the project via the web***:

```bash
ssh smc521shared
```

Note: the `LC_*` environment variables set above allow us to customize the environment
later so that multiple users can share the same environment.  This deals with a quirk of
[CoCalc] in that one connects to a project with a specialized username.  This means that
everyone has the same username, making it difficult for different users to commit
changes to version control.  To remedy this, my [`mmf-setup`] packages modifies the
`~/.bash_alises` file to use the variables sent above.

The remainder of the instructions should be performed after connecting to your [CoCalc]
project in this way.
   
### CoCalc Installation

To install everything on [CoCalc], do the following after connecting to your project
with SSH as described above:

1. Make a `repositories` directory and clone the project:

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
   
   This does the following (see the `Makefile` for the exact details.)

   * Try to clone the private class [Resources project] into `_ext/Resources`.  If you
     do not have access to this project, then a warning will be displayed, but the
     initialization should continue.
   * [`mmf-setup`] is installed in `~/.local/bin/mmf_setup`.
   
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
     environment in `envs/phys-521-2022`, and then creates a Jupyter kernel called
     `phys-521-2022` that the [CoCalc] notebooks can find.  The latter step is roughly
     equivalent to running the following with the appropriate python installed in
     `envs/phys-521-2022`:

     ```bash
     python3 -m ipykernel install --user --name "phys-521-2022" --display-name "Python 3 (phys-521-2022)"
     ```

   * Finally, a line is added to the end of `~/.bash_aliases` to activate the
     environment when you ssh to the [CoCalc] project.  If you logout and log back in,
     you should now see the following prompt:

     ```bash
     $ ssh cc521shared
     (phys-521-2022) ~$ 
     ```
 
 You should now be able to use the `phys-521-2022` kernel in notebooks.

[Shared CoCalc Project]: <https://cocalc.com/projects/31c120c9-2956-4420-9d6f-374a6ee32df3> "581-2021 Shared CoCalc Project"
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
[Anaconda Project]: <https://anaconda-project.readthedocs.io/en/latest/> "Reproducible and executable project directories"
