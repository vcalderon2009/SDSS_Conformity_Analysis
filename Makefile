.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

# General Project variables
PROJECT_DIR       := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET             = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE            = default
PROJECT_NAME       = conformity
PYTHON_INTERPRETER = python3
ENVIRONMENT_FILE   = environment.yml

# Directories
SRC_DIR      = $(PROJECT_DIR)/src/data
ONE_HALO_DIR = $(SRC_DIR)/One_halo_conformity
TWO_HALO_DIR = $(SRC_DIR)/Two_halo_conformity
DATA_DIR     = $(PROJECT_DIR)/data

# Function variables
CPU_FRAC          = 0.7
REMOVE_FILES      = "True"
REMOVE_WP         = "False"
SHUFFLE_TYPE_CENS = "normal"
CLF_METHOD        = 1
CLF_SEED          = 1235
HALOTYPE          = 'fof'
HOD_N             = 0
PERF_OPT          = "False"
SAMPLE            = "19"
DV                = 1.0
VERBOSE           = "False"
CATL_TYPE         = "mr"


ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

##############################################################################
# VARIABLES FOR COMMANDS                                                     #
##############################################################################
src_pip_install:=pip install -e .

src_pip_uninstall:= pip uninstall --yes src

cosmo_utils_pip_install:=pip install cosmo-utils

cosmo_utils_pip_upgrade:= pip install --upgrade cosmo-utils

cosmo_utils_pip_uninstall:= pip uninstall cosmo-utils

##############################################################################
# COMMANDS                                                                   #
##############################################################################

## Deletes all build, test, coverage, and Python artifacts
clean: clean-build clean-pyc clean-test

## Removes Python file artifacts
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

## Remove build artifacts
clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

## Remove test and coverage artifacts
clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

## Lint using flake8
lint:
	flake8 --exclude=lib/,bin/,docs/conf.py .

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

## Set up python interpreter environment - Using environment.yml
environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
		# conda config --add channels conda-forge
		conda env create -f $(ENVIRONMENT_FILE)
		$(cosmo_utils_pip_install)
endif
	$(src_pip_install)

## Update python interpreter environment
update_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
		conda env update -f $(ENVIRONMENT_FILE)
		$(cosmo_utils_pip_upgrade)
endif
	$(src_pip_uninstall)
	$(src_pip_install)

## Delete python interpreter environment
remove_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, removing conda environment"
		conda env remove -n $(ENVIRONMENT_NAME)
		$(cosmo_utils_pip_uninstall)
endif
	$(src_pip_uninstall)

## Import local source directory package
src_env:
	$(src_pip_install)

## Updated local source directory package
src_update:
	$(src_pip_uninstall)
	$(src_pip_install)

## Remove local source directory package
src_remove:
	$(src_pip_uninstall)

## Installing cosmo-utils
cosmo_utils_install:
	$(cosmo_utils_pip_install)

## Upgrading cosmo-utils
cosmo_utils_upgrade:
	$(cosmo_utils_pip_upgrade)

## Removing cosmo-utils
cosmo_utils_remove:
	$(cosmo_utils_pip_uninstall)

#################################################################################
# PROJECT FUNCTIONS                                                                 #
#################################################################################

## Figures
plot_figures:
	# 1-halo
	@python $(ONE_HALO_DIR)/one_halo_conformity_quenched_fractions_make.py \
	-a plots -hod_model_n $(HOD_N) -halotype $(HALOTYPE) \
	-clf_method $(CLF_METHOD) -clf_seed $(CLF_SEED)
	@python $(ONE_HALO_DIR)/one_halo_mark_correlation_make.py -a plots \
	-hod_model_n $(HOD_N) -halotype $(HALOTYPE) -clf_method $(CLF_METHOD) \
	-clf_seed $(CLF_SEED)
	# 2-halo
	@python $(TWO_HALO_DIR)/two_halo_conformity_quenched_fractions_make.py \
	-a plots -hod_model_n $(HOD_N) -halotype $(HALOTYPE) \
	-clf_method $(CLF_METHOD) -clf_seed $(CLF_SEED)
	@python $(TWO_HALO_DIR)/two_halo_mark_correlation_make.py -a plots \
	-hod_model_n $(HOD_N) -halotype $(HALOTYPE) -clf_method $(CLF_METHOD) \
	-clf_seed $(CLF_SEED)

## 1-halo Quenched Fractions - Calculations
1_halo_fracs_calc: download_dataset
	# 1-halo
	@python $(ONE_HALO_DIR)/one_halo_conformity_quenched_fractions_make.py \
	-a calc -cpu_frac $(CPU_FRAC) -remove $(REMOVE_FILES) \
	-hod_model_n $(HOD_N) -halotype $(HALOTYPE) -clf_method $(CLF_METHOD) \
	-clf_seed $(CLF_SEED)

## 1-halo Marked Correlation Function - Calculations
1_halo_mcf_calc: download_dataset
	# 1-halo
	@python $(ONE_HALO_DIR)/one_halo_mark_correlation_make.py -a calc \
	-cpu_frac $(CPU_FRAC) -remove $(REMOVE_FILES) -remove-wp $(REMOVE_WP) \
	-hod_model_n $(HOD_N) -halotype $(HALOTYPE) -clf_method $(CLF_METHOD) \
	-clf_seed $(CLF_SEED)

## 2-halo Quenched Fractions - Calculations
2_halo_fracs_calc: download_dataset
	# 2-halo
	@python $(TWO_HALO_DIR)/two_halo_conformity_quenched_fractions_make.py \
	-a calc -cpu_frac $(CPU_FRAC) -remove $(REMOVE_FILES) \
	-shuffle_type $(SHUFFLE_TYPE_CENS) -remove-wp $(REMOVE_WP) \
	-hod_model_n $(HOD_N) -halotype $(HALOTYPE) -clf_method $(CLF_METHOD) \
	-clf_seed $(CLF_SEED)

## 2-halo Marked Correlation Function - Calculations
2_halo_mcf_calc: download_dataset
	# 2-halo
	@python $(TWO_HALO_DIR)/two_halo_mark_correlation_make.py -a calc \
	-cpu_frac $(CPU_FRAC) -remove $(REMOVE_FILES) -remove-wp $(REMOVE_WP) \
	-hod_model_n $(HOD_N) -halotype $(HALOTYPE) -clf_method $(CLF_METHOD) \
	-clf_seed $(CLF_SEED)

## Remove Plot screen session
remove_plot_screens:
	screen -S "One_Halo_FRAC_STAT_conformity_plots" -X quit || echo ""
	screen -S "One_Halo_MCF_conformity_plots" -X quit || echo ""
	screen -S "Two_Halo_FRAC_STAT_conformity_plots" -X quit || echo ""
	screen -S "Two_Halo_MCF_conformity_plots" -X quit || echo ""

## Remove Calc. screen session
remove_calc_screens:
	screen -S "One_Halo_FRAC_STAT_conformity_calc" -X quit || echo ""
	screen -S "One_Halo_MCF_conformity_calc" -X quit || echo ""
	screen -S "Two_Halo_FRAC_STAT_conformity_calc" -X quit || echo ""
	screen -S "Two_Halo_MCF_conformity_calc" -X quit || echo ""

## Download required Dataset
download_dataset:
	# Downloading dataset
	@python $(SRC_DIR)/download_dataset.py -hod_model_n $(HOD_N) \
	-halotype $(HALOTYPE) -clf_method $(CLF_METHOD) -clf_seed $(CLF_SEED) \
	-dv $(DV) -sample $(SAMPLE) -abopt $(CATL_TYPE) \
	-perf $(PERF_OPT) -v $(VERBOSE)

## Remove downloaded catalogues
remove_catalogues:
	find $(DATA_DIR) -type f -name '*.hdf5' -delete

##############################################################################
# Self Documenting Commands                                                  #
##############################################################################

.DEFAULT_GOAL := show-help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
#   * save line in hold space
#   * purge line
#   * Loop:
#       * append newline + line to hold space
#       * go to next line
#       * if line starts with doc comment, strip comment character off and loop
#   * remove target prerequisites
#   * append hold space (+ newline) to line
#   * replace newline plus comments by `---`
#   * print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: show-help
show-help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')

