.PHONY: setup test models figure3 synthetic-recovery model-comparison chapter4 estimate rerun-failed \
	finalize-estimation counterfactuals precision-design validate-main validate-sensitivity \
	chapter5-audit chapter5-report build-chapter6-data estimate-chapter6 report-chapter6 \
	sync-paper-assets reproduce-derived release-manifest verify-release paper clean-paper

PYTHON ?= .venv/bin/python
SYSTEM_PYTHON ?= python3

setup:
	$(SYSTEM_PYTHON) -m venv .venv
	$(PYTHON) -m pip install -r requirements-lock.txt

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
	$(PYTHON) -m pipeline.publication --sync-assets

# The historical all-in-one pipeline would overwrite current manuscript assets.
# Derived regeneration for the current release is documented stage by stage.
reproduce-derived:
	@echo "No all-in-one rebuild for the current empirical revision. See docs/REPRODUCIBILITY.md."
	@exit 2

release-manifest:
	$(PYTHON) -m pipeline.publication --write-manifest

verify-release:
	$(PYTHON) -m pipeline.publication --check

verify-chains:
	$(PYTHON) -m pipeline.publication --check --check-chains

export-chain-index:
	$(PYTHON) -m pipeline.publication --export-chain-index

# Historical workflows retain their original paths and scientific specification.
# Select them deliberately, e.g. make LEGACY=1 chapter5-report.
legacy-context:
	@test "$(LEGACY)" = "1" || { echo "Historical result paths: use LEGACY=1 in an isolated checkout. See docs/REPRODUCIBILITY.md."; exit 2; }

estimate rerun-failed finalize-estimation counterfactuals precision-design validate-main validate-sensitivity chapter5-audit chapter5-report build-chapter6-data estimate-chapter6 report-chapter6: | legacy-context

.PHONY: verify-chains export-chain-index legacy-context

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error PaperJK7.tex

clean-paper:
	cd paper && latexmk -c PaperJK7.tex
