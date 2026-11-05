PY ?= python

.PHONY: install build test eval eval-smoke serve

install:
	$(PY) -m pip install -e ".[dev]"

build:
	$(PY) -m perkpilot build

test:
	$(PY) -m pytest -q

eval:
	$(PY) -m eval.run_eval

eval-smoke:
	$(PY) -m eval.run_eval --smoke

serve: build
	$(PY) -m perkpilot serve
