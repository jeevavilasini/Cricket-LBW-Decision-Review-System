.PHONY: help env test demo video experiments quick clean figures

PY ?= ./drs-env/bin/python
VENV ?= drs-env

help:
	@echo "DRS ball tracking - available targets"
	@echo ""
	@echo "  make env          create the virtual environment and install dependencies"
	@echo "  make test         run the test suite"
	@echo "  make demo         one delivery end to end, with figures"
	@echo "  make video        the full vision pipeline on rendered footage"
	@echo "  make experiments  the complete experimental programme"
	@echo "  make quick        a fast smoke test of the experiments"
	@echo "  make app          launch the Streamlit interface"
	@echo "  make clean        remove generated outputs"

env:
	python3 -m venv --system-site-packages $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -r requirements.txt

test:
	$(PY) -m pytest tests/ -q

demo:
	$(PY) scripts/run_demo.py --preset leg_spin_marginal

video:
	$(PY) scripts/run_video.py --synthetic good_length_middle

experiments:
	$(PY) scripts/run_experiments.py

quick:
	$(PY) scripts/run_experiments.py --quick

app:
	$(PY) -m streamlit run app/streamlit_app.py

clean:
	rm -rf outputs/synth_frames outputs/synth
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '.pytest_cache' -type d -prune -exec rm -rf {} +
	@echo "Cleaned generated frames and caches (outputs/results and outputs/figures kept)."
