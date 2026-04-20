.PHONY: style check_code_quality

export PYTHONPATH = .
check_dirs := roboflow

style:
	ruff format $(check_dirs)
	ruff check $(check_dirs) --fix

check_code_quality:
	ruff format $(check_dirs) --check
	ruff check $(check_dirs)
	mypy $(check_dirs)
