PYTHON ?= python

.PHONY: pipeline build_dataset

pipeline:
	$(PYTHON) scripts/run_pipeline.py

build_dataset:
	$(PYTHON) scripts/build_dataset.py
