import datpy.data_io.json as json
import numpy as np
from pathlib import Path
import argparse

# convenience

def get_prior_index(database):
    prior_index = {}
    for p in database['prior']:
        if p['type'] != 'legacy-fission-spectrum':
            prior_index[p['ID']] = p
    return prior_index


def get_dataset_index(database):
    dataset_index = {}
    for blck in database['datablocks']:
        for ds in blck['datasets']:
            dataset_index[ds['NS']] = ds
    return dataset_index


# simple checks 

def is_sacs(ds):
    if ds.get('MT') in (6,10):
        return True
    if ds.get('quantity','') in ('legacy_sacs', 'legacy_sacs_ratio'):
        return True
    return False


def is_rrr_fit(ds):
    full_label = (ds.get('CLABL','') + ds.get('BREF','')).replace(' ', '')
    return any(s in full_label for s in ('Hale', 'Chen', 'EDA', 'RAC', 'Derrien'))


def is_tnc_data(ds):
    if 'E' not in ds:
        return False
    return len(ds['E']) == 1 and ds['E'][0] == 2.53e-8


def deal_with_rrr_fit(ds, prior_index):
    dataset_id = ds['NS']
    print(f'dataset {dataset_id} with MT={ds["MT"]} is an external RRR fit and will be treated in an ad-hoc way')
    if ds['MT'] != 1:
        raise ValueError('RRR dataset must be of type MT=1')
    reac_id = ds['NT'][0]
    min_en = min(ds['E'])
    max_en = max(ds['E'])
    p = prior_index[reac_id]
    new_energies = [e for e in p['EN'] if min_en <= e and e <= max_en]
    # import matplotlib.pyplot as plt
    # import matplotlib
    # matplotlib.use('Qt5Agg')
    # plt.plot(ds['E'], ds['CSS'])
    # plt.title(p['CLAB'])
    # plt.show()
    new_css = np.interp(new_energies, ds['E'], ds['CSS'])
    uncs = np.array(ds['CO'])
    new_uncs = np.zeros((len(new_css), uncs.shape[1]), dtype=float)
    for i in range(new_uncs.shape[1]):
        new_uncs[:,i] = np.interp(new_energies, ds['E'], uncs[:,i])
    ds['E'] = new_energies
    ds['CSS'] = new_css
    ds['CO'] = new_uncs


def deal_with_mysterious_dataset(ds):
    dataset_id = ds['NS']
    label = ds['CLABL']
    ref = ds['BREF']
    full_label = (label + ref).replace(' ', '')
    year = ds['YEAR']
    if 'n_TOF_EAR1' in full_label and 'Mingrone' in full_label and year == 2014:
        print(f'Skipping mysterious n_TOF dataset from Mingrone (dataset id: {dataset_id})')
    elif 'n_TOF_EAR1' in full_label and 'Wright' in full_label and year == 2014:
        print(f'Skipping mysterious n_TOF dataset from Wright (dataset id: {dataset_id})')
    else:
        raise ValueError(f'dataset {dataset_id} has no update, but should have one!')


# specific actions

def _myassert(field, ds, ds_upd, dataset_id):
    if ds[field] != ds_upd[field]:
        raise ValueError(f'dataset {dataset_id}---{field} mismatch: {ds[field]} vs {ds_upd[field]}')


def modify_dataset(ds, ds_upd):
    ns = ds['NS']
    if ns == 1450:
        print(f'TODO: Take care of MT mismatch for dataset {ns} (using reduced anyway)')
    else:
        if ns in (8010, 8011):
            print(f'TODO: Take care of MT mismatch for dataset {ns} (using reduced anyway)')
        else:
            _myassert('MT', ds, ds_upd, ns)
        _myassert('NT', ds, ds_upd, ns)
    _myassert('NS', ds, ds_upd, ns)
    _myassert('YEAR', ds, ds_upd, ns)
    source_energies = ds_upd['E']
    target_energies = ds['E']
    energy_idcs = []
    for e in target_energies: 
        if e in source_energies:
            energy_idcs.append(source_energies.index(e))
        else:
            print(f'Skipping energy {e} in dataset {ns} because not found in update')
    ds['E'] = []
    ds['CSS'] = []
    ds['CO'] = []
    for idx in energy_idcs:
        ds['E'].append(ds_upd['E'][idx])
        ds['CSS'].append(ds_upd['CSS'][idx])
        ds['CO'].append(ds_upd['CO'][idx])


# database management 

def read_database(database_file):
    with open(database_file, 'r') as f:
        database = json.load(f)
    return database


def save_database(database, database_file):
    with open(database_file, 'w') as f:
        json.dump(database, f, indent=2)


def perform_dataset_action(ds, prior_index, dataset_index):
    dataset_id = ds['NS'] if 'NS' in ds else ds['identifier']
    ds_upd = dataset_index.get(dataset_id)
    if ds_upd is not None:
        modify_dataset(ds, ds_upd)
    elif is_rrr_fit(ds):
        deal_with_rrr_fit(ds, prior_index)
    elif is_tnc_data(ds):
        print(f'dataset {dataset_id} with MT={ds["MT"]} is a TNC dataset (hence no reduction update available and not necessary)')
    elif is_sacs(ds):
        print(f'keeping SACS dataset {dataset_id} as is') 
    else:
        deal_with_mysterious_dataset(ds)
        return False
    return True


def update_database(database, database_update): 
    dataset_index = get_dataset_index(database_update)
    prior_index = get_prior_index(database_update)
    datablocks = gmadb['datablocks']
    new_datablocks = []
    while datablocks:
        blck = datablocks.pop(0)
        datasets = blck['datasets']
        new_datasets = []
        while datasets:
            dataset = datasets.pop(0)
            if perform_dataset_action(dataset, prior_index, dataset_index):
                new_datasets.append(dataset)
        if new_datasets:
            blck['datasets'] = new_datasets
            new_datablocks.append(blck)
    gmadb['datablocks'] = new_datablocks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gma-database', type=str, help='path to GMA database file')
    parser.add_argument('--reduced-input', type=str, help='path to reduced input (result of datpy invocation)')
    parser.add_argument('--gma-database-out', type=str, help='output file with update gma database')
    args = parser.parse_args()

    gmadb_file = args.gma_database
    reddb_file = args.reduced_input
    outdb_file = args.gma_database_out

    gmadb = read_database(gmadb_file)
    reddb = read_database(reddb_file)
    update_database(gmadb, reddb)

    save_database(gmadb, outdb_file) 
