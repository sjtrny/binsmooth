#!/usr/bin/env bash
black  --preview --config pyproject.toml ./
isort binsmooth.py
flake8 --exclude=.git,*migrations*,venv,docs
pydocstyle --convention=numpy --add-ignore=D412 binsmooth.py
pytest
