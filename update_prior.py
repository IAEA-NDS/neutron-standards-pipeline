import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from gmapy.data_management.object_utils import load_objects
import gmapy.data_management.json as json


priortable, = load_objects('work/evaluation/output/01_model_preparation_output.pkl', 'priortable')
params = load_objects('work/evaluation/output/02_parameter_optimization_output.pkl', 'params') 
params = np.squeeze(np.array(params))

sel = priortable.NODE != 'fis'
assert len(priortable[sel]) == len(params)
priortable.loc[sel, 'POST'] = params 

with open('input/input.json') as f:
    db = json.load(f)


prior = db['prior']
for p in prior:
    if 'ID' not in p:
        continue
    curid = p['ID']
    updvals = priortable.loc[priortable.NODE == f'xsid_{curid}', 'POST']
    p['CS'][1:-1] = updvals


with open('input/upd_input.json', 'w') as f:
    json.dump(db, f, indent=2)
