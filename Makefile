.PHONY: test models figure3 synthetic-recovery estimate rerun-failed finalize-estimation \
	counterfactuals validate-main build-robustness-data estimate-robustness \
	finalize-robustness validate-robustness paper clean-paper

test:
	python -m compileall -q pipeline metakaggle synthetic_data tests
	python -m unittest discover -s tests -v

models:
	python -m pipeline.compile_models

figure3:
	python -m pipeline.synthetic.figure3

synthetic-recovery:
	python -m pipeline.synthetic.recovery

estimate:
	python -m pipeline.empirical.estimate

rerun-failed:
	python -m pipeline.empirical.rerun_failed

finalize-estimation:
	python -m pipeline.empirical.finalize

counterfactuals:
	python -m pipeline.empirical.counterfactual

validate-main:
	python -m pipeline.validation.chapter5

build-robustness-data:
	python -m pipeline.robustness.build_data

estimate-robustness:
	python -m pipeline.robustness.estimate

finalize-robustness:
	python -m pipeline.robustness.finalize

validate-robustness:
	python -m pipeline.validation.robustness

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error PaperJK7.tex

clean-paper:
	cd paper && latexmk -c PaperJK7.tex
