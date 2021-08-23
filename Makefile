# Modelled after
# https://github.com/simoninireland/introduction-to-epidemics/blob/master/Makefile
SHELL := /bin/bash

RESOURCES = git@gitlab.com:wsu-courses/physics-521-classical-mechanics_resources.git

# ------- Tools -------
ifdef ANACONDA2020
  # If this is defined, we assume we are on CoCalc
  ACTIVATE := source $$ANACONDA2020/bin/activate
	ANACONDA_PROJECT := $(ACTIVATE) root && anaconda-project
else
  ACTIVATE := conda activate
	ANACONDA_PROJECT := anaconda-project
endif

ENV := phys-521-2021
ENV_PATH := $(abspath envs/$(ENV))
ACTIVATE_PROJECT := $(ACTIVATE) $(ENV_PATH)

# ------- Top-level targets  -------

# Default prints a help message
help:
	@make usage


usage:
	@echo "$$HELP_MESSAGE"


init: _ext/Resources $(ENV_PATH)
	$(ANACONDA_PROJECT) prepare
	$(ANACONDA_PROJECT) run init	# Custom command: see anaconda-project.yaml
ifdef ANACONDA2020
	@make cocalc-init
endif


cocalc-init:
	python3 -m pip install --user mmf-setup
	mmf_setup cocalc
	if ! grep -Fq '$(ACTIVATE_PROJECT)' ~/.bash_aliases; then \
	  echo '$(ACTIVATE_PROJECT)' >> ~/.bash_aliases; \
  fi
	@make sync


$(ENV_PATH): anaconda-project.yaml
	$(ANACONDA_PROJECT) prepare
	$(ANACONDA_PROJECT) run init


_ext/Resources:
	-git clone $(RESOURCES) $@
	@if [ ! -d "$@" ]; then \
	  echo "$$RESOURCES_ERROR_MESSAGE"; \
  fi


Docs/environment.yaml: anaconda-project.yaml Makefile
	$(ANACONDA_PROJECT) run export 1> $@


sync:
	find . -name ".ipynb_checkpoints" -prune -o \
	       -name "_ext" -prune -o \
	       -name "envs" -prune -o \
	       -name "*.ipynb" -exec jupytext --sync {} +


reallyclean:
	$(ANACONDA_PROJECT) run clean || true	# Custom command: see anaconda-project.yaml
	$(ANACONDA_PROJECT) clean || true
	$(RM) -r envs


clean:
	find . -name "__pycache__" -exec $(RM) {} +
	$(RM) _htmlcov .coverage .pytest_cache
	$(ACTIVATE) root && conda clean --all -y


doc-server:
	sphinx-autobuild Docs Docs/_build/html


.PHONY: clean realclean init cocalc-init sync doc-server help


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
   ENV: (= "$(ENV)")
                     Name of the conda environment user by the project.
                     (Customizations have not been tested.)
                     Defaults to `phys-521-2021`.
   ENV_PATH: (= "$(ENV_PATH)")
                     Path to the conda environment user by the project.
                     (Customizations have not been tested.)
                     Defaults to `envs/$$(ENV)`.
   ACTIVATE_PROJECT: (= "$(ACTIVATE_PROJECT)")
                     Command to activate the project environment in the shell.
                     Defaults to `$$(ACTIVATE)  $$(ENV)`.

Initialization:
   make init         Initialize the environment and kernel.
   make cocalc-init  Do some CoCalc-specific things like install mmf-setup, and activate the
                     environment in ~/.bash_aliases.  This is called by `make init`
                     if ANACONDA2020 is defined, so usually does not need to be called explicitly.

Maintenance:
   make clean        Call conda clean --all: saves disk space.
   make reallyclean  delete the environments and kernel as well.

endef
export HELP_MESSAGE


define RESOURCES_ERROR_MESSAGE

*************************************************************
WARNING: The `_ext/Resources` folder could not be cloned from

  $(RESOURCES)

Likely this is because this repository is private and requires registration in the class.
If you believe that you should have access, please contact your instructor, and provide
your GitLab username.

These resources are not crucial for the project, but are important for the course.
*************************************************************

endef
export RESOURCES_ERROR_MESSAGE
