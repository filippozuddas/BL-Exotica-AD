.PHONY: install train search inject test lint

install:
	pip install -e ".[dev]"

train:
	python scripts/train.py --config configs/training/default.yaml

search:
	python scripts/search.py --config configs/search/default.yaml

inject:
	python scripts/inject_recover.py --config configs/training/default.yaml

test:
	pytest tests/ -v

lint:
	ruff check src/ scripts/ tests/
	mypy src/
