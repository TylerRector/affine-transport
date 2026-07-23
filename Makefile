PYTHON ?= python3
KITCHEN_ARRAYS ?= selected_arrays.npz

.PHONY: all data bench figures spec clean

all: bench figures

data:
	$(PYTHON) src/scenes.py --kitchen-dump $(KITCHEN_ARRAYS)

bench:
	$(PYTHON) src/bench.py

figures:
	$(PYTHON) src/figures.py

spec:
	lean --run spec/AffineTransport.lean
	$(PYTHON) tools/check_spec.py

clean:
	rm -rf src/__pycache__ tools/__pycache__ results figures
