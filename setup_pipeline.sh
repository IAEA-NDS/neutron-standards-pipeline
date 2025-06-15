#!/bin/bash

current_commit=$(git rev-parse HEAD)

setup_commit=""
if [ -e "venvs/setup_commit" ]; then
    setup_commit=$(cat venvs/setup_commit)
fi

if [ "$current_commit" = "$setup_commit" ]; then
    echo "venvs are already set up for $current_commit"
    exit 1
fi

gmapy_dir="deps/neutron-standards-evaluation/gmapy"
datpy_dir="deps/datpy"

gmapy_venv_dir="venvs/venv_gmapy"
datpy_venv_dir="venvs/venv_datpy"

git submodule update --init --recursive
if [ "$?" -ne 0 ]; then
    echo "Unable to update submodules"
    exit 2
fi

mkdir -p venvs
if [ "$?" -ne 0 ]; then
    echo "Unable to create venv root dir"
    exit 3
fi


# create gmapy venv
python3.9 -m venv "$gmapy_venv_dir" && \
source venvs/venv_gmapy/bin/activate && \
python -m pip install "$gmapy_dir" && \
deactivate
if [ "$?" -ne 0 ]; then
    echo "Unable to create venv for gmapy"
    exit 4
fi

# create datpy venv 
python3.9 -m venv "$datpy_venv_dir" && \
source venvs/venv_datpy/bin/activate && \
python -m pip install "$datpy_dir" && \
deactivate
if [ "$?" -ne 0 ]; then
    echo "Unable to create venv for datpy"
    exit 5
fi


echo $current_commit > venvs/setup_commit
