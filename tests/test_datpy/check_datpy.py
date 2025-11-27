import numpy as np
from datpy.datpy import (
    reduce_database
)
from gmapy.data_management.database_IO import (
    read_gma_database
)
import json
from helpers import compare_values
from pathlib import Path

input_path = Path('input')

gmadb_ref = read_gma_database(input_path / 'DAT_REF.RES')
gmadb_ref = {'prior': gmadb_ref['prior_list'], 'datablocks': gmadb_ref['datablock_list']}

with open(input_path / 'GMDATA.JSON', 'r') as f:
    crd_db = json.load(f)

gmadb_new = reduce_database(crd_db)

if not compare_values(gmadb_ref, gmadb_new):
    raise ValueError('Differences between DAT and datpy reduction detected')
else:
    print('\nCHECK PASSED: Python code `datpy` and Fortran code DATP` produce equivalent results')

# with open('DAT_NEW.RES', 'w') as f:
#     json.dump(gmadb_new, f)
