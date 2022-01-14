#!/bin/bash
venv_name=venv

virtualenv -p python3 $venv_name
$venv_name/bin/python3 -m pip install -r requirements_build.txt

