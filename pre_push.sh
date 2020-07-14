#!/usr/bin/env bash
black  --config pyproject.toml ./
flake8 --exclude=.git,*migrations*,venv,docs
pytest
#make -C docs/ html