#!/bin/bash

# FILENAMES
datpy_input="work/data/upd_full_input.json"
datpy_output="work/data/upd_full_input_reduced.json"
# ATTENTION: The last part "data.json" will be used as a regex pattern below.
#            Pay attention to characters having special meaning in regex patterns. 
gma_database="deps/neutron-standards-evaluation/data/data.json"
upd_gma_database="work/data/upd_data.json"

model_prep_output="work/evaluation/output/01_model_preparation_output.pkl"
param_optim_output="work/evaluation/output/02_parameter_optimization_output.pkl"


# PREPARARE DIRECTORY STRUCTURE

mkdir -p work/data && \
mkdir -p work/evaluation && \

# PERFORM DATA REDUCTION WITH DATPY

if [ ! -f "$datpy_input" ]; then
    cp input/full_input.json $datpy_input 
fi

source venvs/venv_datpy/bin/activate && \
python -m datpy.datpy \
    --input $datpy_input \
    --output $datpy_output 

# UPDATE GMA DATABASE WITH RESULT OF REDUCTION

source venvs/venv_datpy/bin/activate && \
python transfer_to_gmadb.py \
    --reduced-input $datpy_output \
    --gma-database $gma_database \
    --gma-database-out $upd_gma_database

# PERFORM EVALUTION WITH GMAPY

source_eval_dir="deps/neutron-standards-evaluation/evaluation"
gma_basename=$(basename $gma_database)
upd_gma_basename=$(basename $upd_gma_database)

save_cwd="$(pwd)"

cp "$source_eval_dir/01_model_preparation.py" work/evaluation/ && \
sed -i -e "s/\/$gma_basename/\/$upd_gma_basename/" work/evaluation/01_model_preparation.py && \
cp "$source_eval_dir/02_parameter_optimization.py" work/evaluation && \
source venvs/venv_gmapy/bin/activate && \
cd work/evaluation

if [ "$?" -ne 0 ]; then
    echo "problem preparing data and code infrastructure"
    exit 1
fi
cd $save_cwd

if [ -f "$model_prep_output" ] || [ -f "$param_optim_output" ]; then
    echo
    echo "ERROR: gmapy output files exist---aborting"
    exit 2
fi

# python 01_model_preparation.py && \
# python 02_parameter_optimization.py
 
if [ "$?" -ne 0 ]; then
    echo "problem running tensorflow pipeline"
    exit 2
fi


# UPDATE THE PRIOR VALUES FOR A REFINED REDUCTION 

source venvs/venv_gmapy/bin/activate && \
python update_prior.py \
    --model-preparation-output $model_prep_output \
    --parameter-optimization-output $param_optim_output \
    --datpy-input "$datpy_input" 
