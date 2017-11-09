.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = conformity
PYTHON_INTERPRETER = python3
ENVIRONMENT_FILE = environment.yml

SRC_DIR = $(PROJECT_DIR)/src/data
ONE_HALO_DIR = $(SRC_DIR)/One_halo_conformity
TWO_HALO_DIR = $(SRC_DIR)/Two_halo_conformity

# CPU-Fraction
CPU_FRAC = 0.7
REMOVE_FILES = "True"
SHUFFLE_TYPE_CENS = "normal"

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	pip install -r requirements.txt

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py

## Delete all compiled Python files
clean:
	find . -name "*.pyc" -exec rm {} \;

## Lint using flake8
lint:
	flake8 --exclude=lib/,bin/,docs/conf.py .

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	@pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

## Set up python interpreter environment - Using environment.yml
environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
		# conda config --add channels conda-forge
		conda env create -f $(ENVIRONMENT_FILE)
endif

## Update python interpreter environment
update_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
		conda env update -f $(ENVIRONMENT_FILE)
endif

## Delete python interpreter environment
remove_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, removing conda environment"
		conda env remove -n $(PROJECT_NAME)
endif

#################################################################################
# PROJECT FUNCTIONS                                                                 #
#################################################################################

## Figures
plot_figures:
	# 1-halo
	@python $(ONE_HALO_DIR)/one_halo_conformity_quenched_fractions_make.py -a plots
	@python $(ONE_HALO_DIR)/one_halo_mark_correlation_make.py -a plots
	# 2-halo
	@python $(TWO_HALO_DIR)/two_halo_conformity_quenched_fractions_make.py -a plots
	@python $(TWO_HALO_DIR)/two_halo_mark_correlation_make.py -a plots

## 1-halo Quenched Fractions - Calculations
1_halo_fracs_calc:
	# 1-halo
	@python $(ONE_HALO_DIR)/one_halo_conformity_quenched_fractions_make.py -a calc -cpu_frac $(CPU_FRAC) -remove $(REMOVE_FILES)

## 1-halo Marked Correlation Function - Calculations
1_halo_mcf_calc:
	# 1-halo
	@python $(ONE_HALO_DIR)/one_halo_mark_correlation_make.py -a calc -cpu_frac $(CPU_FRAC) -remove $(REMOVE_FILES)

## 2-halo Quenched Fractions - Calculations
2_halo_fracs_calc:
	# 2-halo
	@python $(TWO_HALO_DIR)/two_halo_conformity_quenched_fractions_make.py -a calc -cpu_frac $(CPU_FRAC) -remove $(REMOVE_FILES) -shuffle_type $(SHUFFLE_TYPE_CENS)

## 2-halo Marked Correlation Function - Calculations
2_halo_mcf_calc:
	# 2-halo
	@python $(TWO_HALO_DIR)/two_halo_mark_correlation_make.py -a calc -cpu_frac $(CPU_FRAC) -remove $(REMOVE_FILES)

## Remove Plot screen session
remove_plot_screens:
	screen -S "One_Halo_FRAC_STAT_conformity_plots" -X quit
	screen -S "One_Halo_MCF_conformity_plots" -X quit
	screen -S "Two_Halo_FRAC_STAT_conformity_plots" -X quit
	screen -S "Two_Halo_MCF_conformity_plots" -X quit

## Remove Calc. screen session
remove_calc_screens:
	screen -S "One_Halo_FRAC_STAT_conformity_calc" -X quit
	screen -S "One_Halo_MCF_conformity_calc" -X quit
	screen -S "Two_Halo_FRAC_STAT_conformity_calc" -X quit
	screen -S "Two_Halo_MCF_conformity_calc" -X quit

## Download required Dataset
download_dataset:
	# Downloading dataset
	python $(SRC_DIR)/download_dataset.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

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

