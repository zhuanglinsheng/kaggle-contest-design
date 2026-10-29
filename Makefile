.PHONY: setup test models figure3 synthetic-recovery model-comparison chapter4 estimate rerun-failed \
	finalize-estimation counterfactuals precision-design validate-main validate-sensitivity \
	chapter5-audit chapter5-report build-chapter6-data estimate-chapter6 report-chapter6 \
	sync-paper-assets reproduce-derived release-manifest verify-release paper clean-paper

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
SYSTEM_PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

# Keep language, plotting and build caches with the other intermediate files.
export PYTHONPYCACHEPREFIX := $(CURDIR)/tmp/cache/python
export MPLCONFIGDIR := $(CURDIR)/tmp/cache/matplotlib
export XDG_CACHE_HOME := $(CURDIR)/tmp/cache

setup:
	$(SYSTEM_PYTHON) -m venv .venv
	.venv/bin/python -m pip install --only-binary=:all: -r requirements.txt

test:
	$(PYTHON) -m compileall -q pipeline metakaggle synthetic_data tests
	$(PYTHON) -m unittest discover -s tests -v

models:
	$(PYTHON) -m pipeline.compile_models

figure3:
	$(PYTHON) -m pipeline.synthetic.optimal_generate

synthetic-recovery:
	$(PYTHON) -m pipeline.synthetic.recovery

model-comparison:
	$(PYTHON) -m pipeline.synthetic.model_comparison

.PHONY: richer-recovery
richer-recovery:
	$(PYTHON) -m pipeline.synthetic.richer_recovery

chapter4:
	$(PYTHON) -m pipeline.synthetic.optimal_generate

estimate:
	$(PYTHON) -m pipeline.empirical.estimate

rerun-failed:
	$(PYTHON) -m pipeline.empirical.rerun_failed

finalize-estimation:
	$(PYTHON) -m pipeline.empirical.finalize

counterfactuals:
	$(PYTHON) -m pipeline.empirical.counterfactual

precision-design:
	$(PYTHON) -m pipeline.empirical.precision_design

validate-main:
	$(PYTHON) -m pipeline.validation.chapter5

validate-sensitivity:
	$(PYTHON) -m pipeline.validation.chapter5_robust
	$(PYTHON) -m pipeline.validation.chapter5_influence

chapter5-audit:
	$(PYTHON) -m pipeline.validation.chapter5_final_audit

chapter5-report:
	$(PYTHON) -m pipeline.validation.chapter5_report

build-chapter6-data:
	$(PYTHON) -m pipeline.robustness.chapter6_data

estimate-chapter6:
	$(PYTHON) -m pipeline.robustness.chapter6_fit

report-chapter6:
	$(PYTHON) -m pipeline.robustness.chapter6_report --scenarios pair L1 L2

# Current publication commands read current_release.toml. No estimation is run.
sync-paper-assets:
	$(PYTHON) -m pipeline.jk9 verify

# The historical all-in-one pipeline would overwrite current manuscript assets.
# Derived regeneration for the current release is documented stage by stage.
reproduce-derived:
	$(PYTHON) -m pipeline.jk9 tables

release-manifest:
	$(PYTHON) -m pipeline.jk9 lock

verify-release:
	$(PYTHON) -m pipeline.jk9 verify

verify-chains:
	$(PYTHON) -m pipeline.jk9 verify --check-chains

export-chain-index:
	$(PYTHON) -m pipeline.jk9 snapshot

# Historical workflows retain their original paths and scientific specification.
# Select them deliberately, e.g. make LEGACY=1 chapter5-report.
legacy-context:
	@test "$(LEGACY)" = "1" || { echo "Historical result paths: use LEGACY=1 in an isolated checkout. See docs/REPRODUCIBILITY.md."; exit 2; }

estimate rerun-failed finalize-estimation counterfactuals precision-design validate-main validate-sensitivity chapter5-audit chapter5-report build-chapter6-data estimate-chapter6 report-chapter6: | legacy-context

.PHONY: verify-chains export-chain-index legacy-context

paper:
	mkdir -p tmp/latex
	cd paper && latexmk -pdf -outdir=../tmp/latex -interaction=nonstopmode -halt-on-error PaperJK9.tex
	cp tmp/latex/PaperJK9.pdf paper/PaperJK9.pdf

clean-paper:
	cd paper && latexmk -c -outdir=../tmp/latex PaperJK9.tex

.PHONY: compact-caches restore-cache jk9-models
compact-caches:
	$(PYTHON) -m pipeline.cache_store pack --prune

restore-cache:
	$(PYTHON) -m pipeline.cache_store restore $(if $(CONTEST),--contest $(CONTEST),)

jk9-models:
	$(PYTHON) -m pipeline.jk9 compile-models

.PHONY: prediction-summary prediction-baselines
prediction-summary:
	$(PYTHON) -m pipeline.jk9_prediction summary

prediction-baselines:
	$(PYTHON) -m pipeline.jk9_prediction baselines

.PHONY: storage-plan storage-audit
storage-plan:
	$(PYTHON) -m pipeline.organize

storage-audit:
	$(PYTHON) -m pipeline.organize --audit
