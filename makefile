.PHONY: figures

analysis:
	python scripts/analysis/1a_epoch_fungi_data.py
	python scripts/analysis/1b_epoch_plantae_data.py
	python scripts/analysis/1c_epoch_animalia_data.py
	python scripts/analysis/2_spectral_analysis.py

figures:
	python scripts/figures/figure_2.py
	python scripts/figures/figure_3.py
	python scripts/figures/figure_4.py

all: analysis figures

