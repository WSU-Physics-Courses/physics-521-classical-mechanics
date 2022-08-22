# Modelled after
# https://github.com/simoninireland/introduction-to-epidemics/blob/master/Makefile
SHELL = /bin/bash
DOCS ?= Docs

COOKIECUTTER_URL ?= git+https://gitlab.com/forbes-group/cookiecutters.git
COOKIECUTTER ?= cookiecutter $(COOKIECUTTER_URL) --directory project
COOKIECUTTER_YAML ?= .cookiecutter.yaml

GET_RESOURCES = git clone git@gitlab.com:wsu-courses/physics-521-classical-mechanics_resources.git _ext/Resources

# Currently, even the new method uses too much memory...
USE_ANACONDA2020 ?= true

# ------- Tools -------
ifdef ANACONDA2020
  # If this is defined, we assume we are on CoCalc
  ifeq ($(USE_ANACONDA2020), true)
    # Old approach using anaconda-project in the ANACONDA2020 environment.
    # Due to the /ext/anaconda2020.02/.condarc issue, we must use mamba in this case
    # https://github.com/Anaconda-Platform/anaconda-project/issues/334#issuecomment-911918761
    CONDA_EXE = $$ANACONDA2020/bin/mamba
    ACTIVATE ?= source $$ANACONDA2020/bin/activate
  else
    # New approach - use our own miniconda
    MINICONDA = ~/.miniconda3
    CONDA_EXE = $(MINICONDA)/bin/conda
    ACTIVATE ?= source $(MINICONDA)/bin/activate
  endif

  #ANACONDA_PROJECT ?= $(ACTIVATE) root && CONDA_EXE=$(CONDA_EXE) anaconda-project
	AP_PRE ?= CONDA_EXE=$(CONDA_EXE)
  ANACONDA_PROJECT ?= $$ANACONDA2020/bin/anaconda-project

else
  ACTIVATE ?= eval "$$(conda shell.bash hook)" && conda activate
  AP_PRE ?= CONDA_EXE=$(CONDA_EXE)
  ANACONDA_PROJECT ?= anaconda-project
endif

ENV ?= phys-521-2022
ENV_PATH ?= $(abspath envs/$(ENV))
ACTIVATE_PROJECT ?= $(ACTIVATE) $(ENV_PATH)
JUPYTEXT ?= $(ANACONDA_PROJECT) run jupytext

# Manim is not ready for ARM yet, so we use an intel build and rosetta
ifeq ($(shell uname -p),arm)
  AP_PRE += CONDA_SUBDIR=osx-64
endif

# ------- Top-level targets  -------
# Default prints a help message
help:
	@make usage

usage:
	@echo "$$HELP_MESSAGE"

521-Docs.tgz: $(DOCS)/*
	@make html
	tar -s "|$(DOCS)/_build/html|521-Docs|g" -zcvf $@ $(DOCS)/_build/html

init:  _ext/Resources  ~/.local/bin/mmf_setup anaconda-project.yaml $(MINICONDA)
	$(AP_PRE) $(ANACONDA_PROJECT) prepare
	$(AP_PRE) $(ANACONDA_PROJECT) run init  # Custom command: see anaconda-project.yaml
ifdef ANACONDA2020
	if ! grep -Fq '$(ACTIVATE_PROJECT)' ~/.bash_aliases; then \
	  echo '$(ACTIVATE_PROJECT)' >> ~/.bash_aliases; \
	fi
	@make sync
endif

# Jupytext
sync:
	$(AP_PRE) find . -name ".ipynb_checkpoints" -prune -o \
	                 -name "_ext" -prune -o \
	                 -name "envs" -prune -o \
	                 -name "*.ipynb" -o -name "*.md" \
	                 -exec $(JUPYTEXT) --sync {} + 2> >(grep -v "is not a paired notebook" 1>&2)
# See https://stackoverflow.com/a/15936384/1088938 for details

clean:
	-find . -name "__pycache__" -exec $(RM) -r {} +
	-$(RM) -r _htmlcov .coverage .pytest_cache
	-$(ACTIVATE) root && conda clean --all -y


realclean:
	$(AP_PRE) $(ANACONDA_PROJECT) run clean || true  # Custom command: see anaconda-project.yaml
	$(AP_PRE) $(ANACONDA_PROJECT) clean || true
	$(RM) -r envs


test:
	$(AP_PRE) $(ANACONDA_PROJECT) run test


html:
	$(AP_PRE) $(ANACONDA_PROJECT) run make -C $(DOCS) html

# We always rebuild the index.md file in case it literally includes the top-level README.md file.
# However, I do not know a good way to pass these to sphinx-autobuild yet.
ALWAYS_REBUILD ?= $(shell find $(DOCS) -type f -name "*.md" -exec grep -l '```{include}' {} + )

doc-server:
ifdef ANACONDA2020
	$(AP_PRE) $(ANACONDA_PROJECT) run sphinx-autobuild --re-ignore '_build|_generated' $(DOCS) $(DOCS)/_build/html --host 0.0.0.0 --port 8000 
else
	$(AP_PRE) $(ANACONDA_PROJECT) run sphinx-autobuild --re-ignore '_build|_generated' $(DOCS) $(DOCS)/_build/html
endif

# ------- Experimental targets  -----
hg-update-cookiecutter:
	hg update cookiecutter-base
	rm -rf Docs/_templates/ Docs/_static/
	$(COOKIECUTTER) --config-file $(COOKIECUTTER_YAML) --overwrite-if-exists --no-input
	hg commit --addremove -m "BASE: Updated cookiecutter skeleton"
	hg update default
	hg merge cookiecutter-base
	hg commit -m "Merge in cookiecutter updates"

hg-amend-cookiecutter:
	hg update cookiecutter-base
	rm -rf Docs/_templates/ Docs/_static/
	$(COOKIECUTTER) --config-file $(COOKIECUTTER_YAML) --overwrite-if-exists --no-input
	hg amend --addremove

.PHONY: hg-update-cookiecutter, hg-amend-cookiecutter

# ------- Auxilliary targets  -------
MINICONDA_SH = https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
MINICONDA_HASH = 1ea2f885b4dbc3098662845560bc64271eb17085387a70c2ba3f29fff6f8d52f

$(MINICONDA):
	wget  $(MINICONDA_SH) -qO /tmp/_miniconda.sh
	echo "$(MINICONDA_HASH)  /tmp/_miniconda.sh" > /tmp/_miniconda.shasum
	shasum -a 256 -c /tmp/_miniconda.shasum && bash /tmp/_miniconda.sh -b -p $@
	rm /tmp/_miniconda.sh*
	$@/bin/conda update -y conda
	$@/bin/conda install -y anaconda-project

	# Dropping defaults allows this to work with < 1GB
	$@/bin/conda install --override-channels --channel conda-forge -y mamba
	$@/bin/conda clean -y --all


# Special target on CoCalc to prevent re-installing mmf_setup.
~/.local/bin/mmf_setup:
ifdef ANACONDA2020
	python3 -m pip install --user --upgrade mmf-setup
	mmf_setup cocalc
endif


_ext/Resources:
	-$(GET_RESOURCES)
	@if [ ! -d "$@" ]; then \
	  echo "$$RESOURCES_ERROR_MESSAGE"; \
	fi
 

$(DOCS)/environment.yaml: anaconda-project.yaml Makefile
	$(AP_PRE) $(ANACONDA_PROJECT) run export 1> $@


.PHONY: clean realclean init sync doc doc-server help test

# ----- Usage -----
define HELP_MESSAGE

This Makefile provides several tools to help initialize the project.  It is primarly designed
to help get a CoCalc project up an runnning, but should work on other platforms.

Variables:
   ANACONDA2020: (= "$(ANACONDA2020)")
                     If defined, then we assume we are on CoCalc and use this to activate
                     the conda base envrionment. Otherwise, you must make sure that the ACTIVATE
                     command works properly.
   ACTIVATE: (= "$(ACTIVATE)")
                     Command to activate a conda environment as `$$(ACTIVATE) <env name>`
                     Defaults to `conda activate`.
   ANACONDA_PROJECT: (= "$(ANACONDA_PROJECT)")
                     Command to run the `anaconda-project` command.  If you need to first
                     activate an environment (as on CoCalc), then this should do that.
                     Defaults to `anaconda-project`.
   AP_PRE: (= "$(AP_PRE)")
                     Pre-commands (like setting `CONDA_EXEC=mamba`) to be run before commands
                     executed with `$(ANACONDA_PROJECT)`.
   DOCS: (= "$(DOCS)")
                     Name of the documentation directory.
                     Defaults to `Docs`.
   ENV: (= "$(ENV)")
                     Name of the conda environment user by the project.
                     (Customizations have not been tested.)
                     Defaults to `phys-521-2022`.
   ENV_PATH: (= "$(ENV_PATH)")
                     Path to the conda environment user by the project.
                     (Customizations have not been tested.)
                     Defaults to `envs/$$(ENV)`.
   ACTIVATE_PROJECT: (= "$(ACTIVATE_PROJECT)")
                     Command to activate the project environment in the shell.
                     Defaults to `$$(ACTIVATE)  $$(ENV)`.

Experimental Variables: (These features are risky or have not been full tested.)
   COOKIECUTTER_URL: (= "$(COOKIECUTTER_URL)")
                     Location of source project for cookiecutter skeletons.  Usually this is
                     `git+https://gitlab.com/forbes-group/cookiecutters.git` but can point to
                     a local directory if you have a clone or are testing changes.
   COOKIECUTTER: (= "$(COOKIECUTTER)")
                     Cookiecutter command, including `--directory` if needed.
   COOKIECUTTER_YAML: (= "$(COOKIECUTTER_YAML)")
                     Local cookiecutter yaml file for the project.

Initialization:
   make init         Initialize the environment and kernel.  On CoCalc we do specific things
                     like install mmf-setup, and activate the environment in ~/.bash_aliases.
                     This is done by `make init` if ANACONDA2020 is defined.

Testing:
   make test         Runs the general tests.

Maintenance:
   make clean        Call conda clean --all: saves disk space.
   make reallyclean  delete the environments and kernel as well.
   make hg-update-cookiecutter (EXPERIMENTAL)
                     Update the base branch with any pushed cookiecutter updates.  Note: this
                     assumes several things, including that you have a `default` and
                     `cookiecutter-base` base branch, as discussed in the docs, that you are
                     using mercurial, and will attempt to automatically merge the changes.
                     You may need to intervene, so try a few times manually before using this.
   make hg-amend-cookiecutter (EXPERIMENTAL)
                     Run hg amend rather than commit and does not merge

Documentation:
   make html         Build the html documentation in `$$(DOCS)/_build/html`
   make doc-server   Build the html documentation server on http://localhost:8000
                     Uses Sphinx autobuild
   521-Docs.tgz  Package documentation for upload to Canvas.
endef
export HELP_MESSAGE


define RESOURCES_ERROR_MESSAGE

*************************************************************
WARNING: The `_ext/Resources` folder could not be created with

  $(GET_RESOURCES)

Likely this is because this repository is private and requires registration in the class.
If you believe that you should have access, please contact your instructor.

These resources are not crucial for the project, but are important for the course.
*************************************************************

endef
export RESOURCES_ERROR_MESSAGE
