# Modelled after
# https://github.com/simoninireland/introduction-to-epidemics/blob/master/Makefile
SHELL := /bin/bash

RESOURCES = git@gitlab.com:wsu-courses/physics-521-classical-mechanics_resources.git

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
    # New approach - use our on miniconda
    MINICONDA = ~/.miniconda3
    CONDA_EXE = $(MINICONDA)/bin/conda
    ACTIVATE ?= source $(MINICONDA)/bin/activate
  endif

  #ANACONDA_PROJECT ?= $(ACTIVATE) root && CONDA_EXE=$(CONDA_EXE) anaconda-project
  ANACONDA_PROJECT ?= CONDA_EXE=$(CONDA_EXE) $$ANACONDA2020/bin/anaconda-project

else
  ACTIVATE ?= eval "$$(conda shell.bash hook)" && conda activate
  ANACONDA_PROJECT ?= CONDA_EXE=$(CONDA_EXE) anaconda-project
endif

ENV ?= phys-521-2021
ENV_PATH ?= $(abspath envs/$(ENV))
ACTIVATE_PROJECT ?= $(ACTIVATE) $(ENV_PATH)

# ------- Top-level targets  -------
521-Docs.tgz: Docs/*
	cd Docs && make html
	tar -s "|Docs/_build/html|521-Docs|g" -zcvf $@ Docs/_build/html

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

# Special target on CoCalc to prevent re-installing mmf_setup.
~/.local/bin/mmf_setup:
ifdef ANACONDA2020
	python3 -m pip install --user --upgrade mmf-setup
	mmf_setup cocalc
endif

init: _ext/Resources ~/.local/bin/mmf_setup anaconda-project.yaml $(MINICONDA)
	@make _init
ifdef ANACONDA2020
	if ! grep -Fq '$(ACTIVATE_PROJECT)' ~/.bash_aliases; then \
	  echo '$(ACTIVATE_PROJECT)' >> ~/.bash_aliases; \
	fi
	@make sync
endif

_init:
	$(ANACONDA_PROJECT) prepare
	$(ANACONDA_PROJECT) run init  # Custom command: see anaconda-project.yaml

.PHONY: _init

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
	       -name "*.ipynb" -o -name "*.md" \
	       -exec jupytext --sync {} + 2> >(grep -v "is not a paired notebook" 1>&2)
# See https://stackoverflow.com/a/15936384/1088938 for details

clean:
	find . -name "__pycache__" -exec $(RM) -r {} +
	$(RM) -r _htmlcov .coverage .pytest_cache
	$(ACTIVATE) root && conda clean --all -y


reallyclean:
	$(ANACONDA_PROJECT) run clean || true  # Custom command: see anaconda-project.yaml
	$(ANACONDA_PROJECT) clean || true
	$(RM) -r envs


doc-server:
	sphinx-autobuild --ignore Docs/_build Docs Docs/_build/html


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
   make init         Initialize the environment and kernel.  On CoCalc we do specific things
                     like install mmf-setup, and activate the environment in ~/.bash_aliases.
                     This is done by `make init` if ANACONDA2020 is defined.

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
