#!/usr/bin/env bash
black  --config pyproject.toml ./
flake8 --exclude=.git,*migrations*,venv,docs
pydocstyle --convention=numpy --add-ignore=D412 binsmooth.py
pytest
#make -C docs/ html