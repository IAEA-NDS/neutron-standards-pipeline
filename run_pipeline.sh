#!/bin/bash

source_eval_dir="deps/neutron-standards-evaluation/evaluation"

mkdir -p work/data && \
mkdir -p work/evaluation && \

cp "input/reduced.json" work/data/data.json && \
cp "$source_eval_dir/01_model_preparation.py" work/evaluation/ && \
cp "$source_eval_dir/02_parameter_optimization.py" work/evaluation && \
source venvs/venv_gmapy/bin/activate && \
cd work/evaluation

if [ "$?" -ne 0 ]; then
    echo "problem preparing data and code infrastructure"
    exit 1
fi

python 01_model_preparation.py && \
python 02_parameter_optimization.py

if [ "$?" -ne 0 ]; then
    echo "problem running tensorflow pipeline"
    exit 2
fi
