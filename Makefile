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

sync-paper-assets:
	cp results/synthetic/optimal_hourly/nonsteady/figure3.png paper/figure3.png
	cp results/theory/numerical_joint_plane_moderate_dense/three_class_comparison/numerical_regions_bw.pdf paper/numerical_regions_bw.pdf
	cp results/robustness/chapter6/reports/pair_L1_L2/validate_lambda_pair.pdf paper/validate_lambda_pair.pdf
	cp results/robustness/chapter6/reports/pair_L1_L2/validate_sigma_pair.pdf paper/validate_sigma_pair.pdf

reproduce-derived: finalize-estimation counterfactuals precision-design validate-main validate-sensitivity chapter5-audit chapter5-report report-chapter6 sync-paper-assets paper

release-manifest:
	$(PYTHON) -m pipeline.release --write-manifest

verify-release:
	$(PYTHON) -m pipeline.release --check-manifest

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error PaperJK7.tex

clean-paper:
	cd paper && latexmk -c PaperJK7.tex
