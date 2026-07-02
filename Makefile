# ECG-C reproducibility entry points.
# Data (PTB-XL@100Hz + NSTDB, ~2 GB) must be under data/ first -- see data/README.md.
.PHONY: setup freeze smoke eval eval-seeds analysis test all clean-results

setup:            ## editable install + dependencies
	pip install -e .
	pip install -r requirements.txt

freeze:           ## LOCK exact versions from the current (working) env -> commit this
	pip freeze > requirements.lock.txt
	@echo "Wrote requirements.lock.txt -- commit it so results/ are reproducible."

smoke:            ## fast wiring check: dry-run plan, then a 200-record train/eval
	python -m src.run_eval --dry-run
	python -m src.run_eval --seeds 0 --limit 200 --results-dir results/smoke

eval:             ## single-seed full grid -> results/grid.csv + results/preds/ (reproduces the report)
	python -m src.run_eval --seeds 0

eval-seeds:       ## 5 training seeds -> results/seed<S>/ + results/grid_multiseed.csv (ranking stability)
	python -m src.run_eval --seeds 0 1 2 3 4

eval-full:        ## CPU zoo + deep GPU models (InceptionTime, ResNet), 5 seeds  [needs GPU deep extras]
	python -m src.run_eval --models minirocket rocket hydra catch22_ridge inceptiontime resnet --seeds 0 1 2 3 4
	@echo "Mantis: wire MantisProbe(embed_fn=<encoder>) then add 'mantis' to --models."

analysis:         ## regenerate leaderboard, summary.md, and all figures from results/
	python -m src.analysis

test:             ## run the test suite
	pytest -q

all: eval analysis  ## end-to-end: evaluate then analyse
