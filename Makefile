PYTHON ?= python

.PHONY: pipeline build_dataset models report

pipeline:
	$(PYTHON) scripts/run_pipeline.py

build_dataset:
	$(PYTHON) scripts/build_dataset.py

models:
	$(PYTHON) scripts/run_models.py

report: models
