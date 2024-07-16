.PHONY: style check_code_quality publish

export PYTHONPATH = .
check_dirs := roboflow

style:
	ruff format $(check_dirs)
	ruff check $(check_dirs) --fix

check_code_quality:
	ruff format $(check_dirs) --check
	ruff check $(check_dirs)
	mypy $(check_dirs)

publish:
	python setup.py sdist bdist_wheel
	twine check dist/*
	twine upload dist/* -u ${PYPI_USERNAME} -p ${PYPI_PASSWORD} --verbose
