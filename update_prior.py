import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from gmapy.data_management.object_utils import load_objects
import gmapy.data_management.json as json
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-preparation-output', type=str, help='pickle file with output objects from model preparation stage')
    parser.add_argument('--parameter-optimization-output', type=str, help='pickle file with output objects from optimization stage')
    parser.add_argument('--datpy-input', type=str, help='json file used as input for data reduction by datpy')
    args = parser.parse_args()

    prep_outfile = args.model_preparation_output
    optim_outfile = args.parameter_optimization_output
    datpy_inpfile = args.datpy_input

    priortable, = load_objects(prep_outfile, 'priortable')
    params = load_objects(optim_outfile, 'params') 
    params = np.squeeze(np.array(params))

    sel = priortable.NODE != 'fis'
    assert len(priortable[sel]) == len(params)
    priortable.loc[sel, 'POST'] = params 

    with open(datpy_inpfile) as f:
        db = json.load(f)

    prior = db['prior']
    for p in prior:
        if 'ID' not in p:
            continue
        curid = p['ID']
        updvals = priortable.loc[priortable.NODE == f'xsid_{curid}', 'POST']
        p['CS'][1:-1] = updvals

    with open(datpy_inpfile, 'w') as f:
        json.dump(db, f, indent=2)
