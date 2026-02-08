PYTHON ?= python

.PHONY: pipeline build_dataset models report report-pdf

pipeline:
	$(PYTHON) scripts/run_pipeline.py

build_dataset:
	$(PYTHON) scripts/build_dataset.py

models:
	$(PYTHON) scripts/run_models.py

report: models

report-pdf:
	$(PYTHON) scripts/build_latex_tables.py
	@if command -v latexmk >/dev/null 2>&1; then \
		cd report && latexmk -pdf -bibtex -interaction=nonstopmode report.tex; \
	else \
		cd report && pdflatex -interaction=nonstopmode report.tex && bibtex report && pdflatex -interaction=nonstopmode report.tex && pdflatex -interaction=nonstopmode report.tex; \
	fi
