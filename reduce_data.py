import json
import pandas as pd
import numpy as np
from gmapy.data_management.database_IO import read_gma_database
from gmapy.legacy.data_extraction_functions import read_gma_result
from gmapy.legacy.legacy_gmap import run_gmap as run_legacy_gmap
from gmapy.tf_uq.gmap_tf import evaluate_gma_database
from gmapy.tf_uq.reduction_tf import reduce_database_iteratively
from gmapy.gmap import run_gmap_simplified
from pathlib import Path
import sys
sys.path.insert(0, 'input/std2017/approach0')
from run_approach import run_fortran_pipeline


# ------------------------------------------------------------
# Perform reduction and evaluation with Fortran legacy codes
# ------------------------------------------------------------

run_fortran_pipeline('input/std2017/approach0')
legacy_result_df = read_gma_result('input/std2017/approach0/fortran_results_inner_iter3/002_evaluation/gma.res')

legacy_results = []
for i in range(2, 61, 2):
    print(i)
    cur_idx = '{:03d}'.format(i)
    cur_df = read_gma_result(f'input/std2017/approach0/fortran_results_inner_iter3/{cur_idx}_evaluation/gma.res')
    legacy_results.append(cur_df)

legacy_results2 = []
for i in range(2, 39, 2):
    print(i)
    cur_idx = '{:03d}'.format(i)
    cur_df = read_gma_result(f'input/std2017/approach0/fortran_results_inner_iter10/{cur_idx}_evaluation/gma.res')
    legacy_results2.append(cur_df)

# quick check of convergence
r1 = legacy_results[1]['RESULT'].to_numpy()
r2 = legacy_results[29]['RESULT'].to_numpy()
r3 = legacy_results[1].copy()
r3['RESULT2'] = r2
r3['RELDIFF'] = np.abs((r3.RESULT2 - r3.RESULT) / r3.RESULT)
r3.sort_values('RELDIFF')

# prepare the iteration history
legacy_hist = np.stack([r['RESULT'].to_numpy() for r in legacy_results])
legacy_hist2 = np.stack([r['RESULT'].to_numpy() for r in legacy_results2])

# plots of the iteration history for selected observables
for i in range(0, legacy_hist.shape[1], 10):
    plt.title(r3.at[i, 'REAC_STRING'] + ' at ' + str(r3.at[i, 'ENERGY']) + ' MeV')
    plt.plot(legacy_hist[:, i] / np.mean(legacy_hist[:, i]))
    plt.show()

# case 1:
# show 6Li(n,a) for Fortran/FGLS3, Fortran/FGLS10 and TF/FGLS and TF/FGLS3
# show 10 B(n,a0) for Fortran/FGLS3, Fortran/FGLS10 and TF/FGLS and TF/FGLS3 at

# ad-hoc comparison between Python TensorFlow and Fortran DATP+GMAP
import matplotlib.pyplot as plt
lrdf = legacy_result_df
for i, row in legacy_result_df[::].iterrows():
    # nodestr = 'xsid_10'
    nodestr = row['NODE']
    if row['REAC_STRING'] != 'U8(n,f)':
        continue
    if row['ENERGY'] < 1.4 or row['ENERGY'] > 1.6:
        continue
    # energy = 5.4e-1
    energy = row['ENERGY']
    leg_idx = lrdf.index[(lrdf.NODE == nodestr) & (lrdf.ENERGY == energy)].item()
    new_idx = tbl.index[(tbl.NODE == nodestr) & (tbl.ENERGY == energy)].item()
    fig, ax = plt.subplots()
    ax.set_title(lrdf.at[leg_idx, 'REAC_STRING'] + ' at ' + str(lrdf.at[leg_idx, 'ENERGY']) + ' MeV')
    ax.set_ylabel('cross section [barn]')
    ax_right = ax.twinx()
    ax_right.set_ylabel('relative change')
    scale = np.mean(legacy_hist[:, leg_idx])
    ax.plot(legacy_hist[:, leg_idx], label='Fortran/FGLS3')
    ax.plot(legacy_hist2[:, leg_idx], label='Fortran/FGLS10')
    ax.plot(hist_fgls[1:, new_idx], label='TensorFlow/FGLS')
    ax.plot(hist_fgls3[1:, new_idx], label='TensorFlow/FGLS3')
    # ax.plot(hist_mle[1:, new_idx], label='TensorFlow/MLE')
    # ax.plot(hist_chisquare[1:, new_idx], label='TensorFlow/chisquare')
    ax_right.set_ylim(
        ax.get_ylim()[0] / scale,
        ax.get_ylim()[1] / scale
    )
    ax.set_xlabel('iteration')
    ax.legend()
    plt.show()


# ------------------------------------------------------------
# 1) Perform reduction with legacy code
# but perform evaluation with Python legacy mode
# ------------------------------------------------------------

# We do the final evaluation with the legacy mode of gmapy
legacy_mode_result = run_legacy_gmap(
    'input/std2017/approach0/04_evaluation/data.gma', dbtype='legacy', num_iter=3, remove_dummy=False,
    correct_ppp=True, fix_ppp_bug=False, fix_sacs_jacobian=False, legacy_integration=True,
)

df1 = legacy_mode_result['table']
df1 = df1[df1.NODE.str.match('^xsid_')].reset_index(drop=True)

assert (df1.NODE == legacy_result_df.NODE).all()
assert (df1.ENERGY == legacy_result_df.ENERGY).all()
np.allclose(df1.POST, legacy_result_df.RESULT)
df1['FORTRAN_POST'] = legacy_result_df.RESULT
df1['RELDIFF'] = np.abs((df1['POST'] - df1['FORTRAN_POST']) / df1['POST'])

with pd.ExcelWriter('sheets/scenario_01.xlsx') as writer:
    df1[['NODE', 'REAC', 'DESCR', 'ENERGY', 'POST', 'FORTRAN_POST', 'RELDIFF']].to_excel(writer, index=False)

# ------------------------------------------------------------
# 2) Perform reduction with Python datpy code
# and perform evaluation with Python legacy mode
# ------------------------------------------------------------

legacy_mode_result_from_json_db = run_legacy_gmap(
    'input/std2017/approach0/04_evaluation/data.json', dbtype='json', num_iter=3, remove_dummy=False,
    correct_ppp=True, fix_ppp_bug=False, fix_sacs_jacobian=False, legacy_integration=True,
)

df2 = legacy_mode_result_from_json_db['table']
df2 = df2[df2.NODE.str.match('^xsid_')].reset_index(drop=True)

assert (df2.NODE == legacy_result_df.NODE).all()
assert (df2.ENERGY == legacy_result_df.ENERGY).all()
np.allclose(df2.POST, legacy_result_df.RESULT)

df2['FORTRAN_POST'] = legacy_result_df['RESULT']
df2['RELDIFF'] = np.abs((df2['POST'] - df2['FORTRAN_POST'])  / df2['POST'])
np.where(df2['RELDIFF'] == np.max(df2['RELDIFF']))
df2.loc[176]
df2.sort_values('RELDIFF')

with pd.ExcelWriter('sheets/scenario_02.xlsx') as writer:
    df2[['NODE', 'REAC', 'DESCR', 'ENERGY', 'POST', 'FORTRAN_POST', 'RELDIFF']].to_excel(writer, index=False)


# ------------------------------------------------------------
# 3) Perform reduction with Python datpy code
# and perform evaluation with Python legacy mode
#   - without bugs
# ------------------------------------------------------------

legacy_mode_result_from_json_db = run_legacy_gmap(
    'input/std2017/approach0/04_evaluation/data.json', dbtype='json', num_iter=3, remove_dummy=False,
    correct_ppp=True, fix_ppp_bug=True, fix_sacs_jacobian=True, legacy_integration=False,
)

df3 = legacy_mode_result_from_json_db['table']
df3 = df3[df3.NODE.str.match('^xsid_')].reset_index(drop=True)

assert (df3.NODE == legacy_result_df.NODE).all()
assert (df3.ENERGY == legacy_result_df.ENERGY).all()
np.allclose(df3.POST, legacy_result_df.RESULT)

df3['FORTRAN_POST'] = legacy_result_df['RESULT']
df3['RELDIFF'] = np.abs((df3['POST'] - df3['FORTRAN_POST'])  / df3['POST'])

with pd.ExcelWriter('sheets/scenario_03.xlsx') as writer:
    df3[['NODE', 'REAC', 'DESCR', 'ENERGY', 'POST', 'FORTRAN_POST', 'RELDIFF']].to_excel(writer, index=False)


# ------------------------------------------------------------
# 4) Perform reduction with Python datpy code
# and perform evaluation with modern Python mode
# (equivalent to Python legacy mode without bugs)
# ------------------------------------------------------------

modern_mode_result_from_json_db = run_gmap_simplified(
    dbfile='input/std2017/approach0/04_evaluation/data.json', dbtype='json',
    num_iter=3, correct_ppp=True, remove_dummy=False
)

df4 = legacy_mode_result_from_json_db['table']
df4 = df4[df4.NODE.str.match('^xsid_')].reset_index(drop=True)

assert (df4.NODE == df3.NODE).all()
assert (df4.ENERGY == df3.ENERGY).all()
assert np.allclose(df4.POST, df3.POST, rtol=1e-10)

df4['PY_LEGACY_POST'] = df3['POST']
df4['RELDIFF'] = np.abs((df4['POST'] - df4['PY_LEGACY_POST'])  / df4['POST'])

with pd.ExcelWriter('sheets/scenario_04.xlsx') as writer:
    df4[['NODE', 'REAC', 'DESCR', 'ENERGY', 'POST', 'PY_LEGACY_POST', 'RELDIFF']].to_excel(writer, index=False)


# ------------------------------------------------------------
# 5) Perform reduction with Python datpy code
# and perform evaluation with modern Python mode
#   - more iterations
# ------------------------------------------------------------

modern_mode_result_from_json_db = run_gmap_simplified(
    dbfile='input/std2017/approach0/04_evaluation/data.json', dbtype='json',
    num_iter=10, correct_ppp=True, remove_dummy=False
)

df5 = modern_mode_result_from_json_db['table']
df5 = df5[df5.NODE.str.match('^xsid_')].reset_index(drop=True)
np.allclose(df5['POST'], df4['POST'], rtol=1e-4)

df5['POST4'] = df4['POST']
df5['RELDIFF'] = np.abs((df5['POST'] - df5['POST4'])  / df4['POST'])

with pd.ExcelWriter('sheets/scenario_05.xlsx') as writer:
    df5[['NODE', 'REAC', 'DESCR', 'ENERGY', 'POST', 'POST4', 'RELDIFF']].to_excel(writer, index=False)


# ------------------------------------------------------------
# 6) Perform reduction with Python datpy code
# and perform evaluation with modern Python mode
#   - more iterations
#   - remove dummy data
#   - alternative regularization scheme
# ------------------------------------------------------------

modern_mode_result_from_json_db = run_gmap_simplified(
    dbfile='input/std2017/approach0/04_evaluation/data.json', dbtype='json',
    num_iter=10, correct_ppp=True, remove_dummy=True, reg=1e-6
)

df6 = modern_mode_result_from_json_db['table']
df6 = df6[df6.NODE.str.match('^xsid_')].reset_index(drop=True)

assert (df6.NODE == df5.NODE).all()
assert (df6.ENERGY == df5.ENERGY).all()
df6['POST5'] = df5['POST']
df6['RELDIFF'] = np.abs((df6['POST'] - df6['POST5']) / df6['POST'])

with pd.ExcelWriter('sheets/scenario_06.xlsx') as writer:
    df6[['NODE', 'REAC', 'DESCR', 'ENERGY', 'POST', 'POST5', 'RELDIFF']].to_excel(writer, index=False)


# ------------------------------------------------------------
# 7) Perform reduction with Python datpy code
# and perform evaluation with gmapy tensorflow mode
#   - more iterations
#   - remove dummy data
#   - alternative regularization scheme
# ------------------------------------------------------------

gma_db = read_gma_database('input/std2017/approach0/04_evaluation/data.json')
tf_result = evaluate_gma_database(gma_db['prior_list'], gma_db['datablock_list'], remove_dummy=True, optim_opts={'max_iters':10, 'rel_tol': 1e-8})

df7 = tf_result['table']
df7 = df7[df7.NODE.str.match('^xsid_')].reset_index(drop=True)

assert (df7.NODE == df6.NODE).all()
assert (df7.ENERGY == df6.ENERGY).all()
assert np.allclose(df7.POST, df6.POST)

df7['POST6'] = df6['POST']
df7['RELDIFF'] = np.abs((df7['POST'] - df7['POST6']) / df7['POST'])
df7['FORTRAN_POST'] = legacy_result_df.RESULT
df7['FORTRAN_RELDIFF'] = np.abs((df7['POST'] - df7['FORTRAN_POST']) / df1['POST'])

with pd.ExcelWriter('sheets/scenario_07.xlsx') as writer:
    df7[['NODE', 'REAC', 'DESCR', 'ENERGY', 'POST', 'POST6', 'RELDIFF', 'FORTRAN_POST', 'FORTRAN_RELDIFF']].to_excel(writer, index=False)


# ------------------------------------------------------------
# 8) Perform reduction with Python datpy code
# and perform evaluation with gmapy tensorflow mode
#   - more iterations
#   - remove dummy data
#   - chisquare minimization instead of GLS
# ------------------------------------------------------------

gma_db = read_gma_database('input/std2017/approach0/04_evaluation/data.json')
tf_result = evaluate_gma_database(
    gma_db['prior_list'], gma_db['datablock_list'], remove_dummy=True,
    optim_type='chisquare', optim_opts={
        'max_inner_iters': 1000, 'max_outer_iters': 20,
    }
)

df8 = tf_result['table']
df8 = df8[df8.NODE.str.match('^xsid_')].reset_index(drop=True)

assert (df8.NODE == df7.NODE).all()
assert (df8.ENERGY == df7.ENERGY).all()
assert np.allclose(df8.POST, df7.POST)

df8['POST7'] = df7['POST']
df8['RELDIFF'] = np.abs((df8.POST - df8.POST7) / df8.POST)
df8['FORTRAN_POST'] = legacy_result_df.RESULT
df8['FORTRAN_RELDIFF'] = np.abs((df8['POST'] - df8['FORTRAN_POST']) / df8['POST'])

with pd.ExcelWriter('sheets/scenario_08.xlsx') as writer:
    df8[['NODE', 'REAC', 'DESCR', 'ENERGY', 'POST', 'POST7', 'RELDIFF', 'FORTRAN_POST', 'FORTRAN_RELDIFF']].to_excel(writer, index=False)


# Change condition to `True` for small ad-hoc test
# that we indeed found a chisquare minimum
if False:
    def get_chisq(x):
        C = tf_result['likelihood'].get_covariance_linop(x).to_dense().numpy()
        y = tf_result['likelihood'].get_model_prediction(x).numpy()
        e = tf_result['expvals'].numpy()
        d = (e-y).reshape(-1,1)
        chisq = d.T @ np.linalg.solve(C, d)
        return chisq.item()

    def is_max_chisq(x, i, chisq_ref):
        x1 = x.copy()
        x1[i] *= 1+1e-4
        chisq1 = get_chisq(x1)
        x2 = x.copy()
        x2[i] *= 1-1e-4
        chisq2 = get_chisq(x2)
        return chisq1 > chisq_ref and chisq2 > chisq_ref

    # check if indeed chisquare was minimized
    x_ref = tf.constant(df.loc[df.NODE != 'fis', 'POST'], dtype=tf.float64).numpy()
    chisq_ref = get_chisq(x_ref)
    for i in range(0, 1127, 10):
        is_good = is_max_chisq(x_ref, 10, chisq_ref)
        print(f'i: {i} - is_good: {is_good}')


C = tf_result['likelihood'].get_covariance_linop(optres).to_dense().numpy()

# ------------------------------------------------------------
# 9) Perform reduction with Python datpy code
# and perform evaluation with gmapy tensorflow mode
#   - more iterations
#   - remove dummy data
#   - Maximum Likelihood Estimation (MLE)
# ------------------------------------------------------------

gma_db = read_gma_database('input/std2017/approach0/04_evaluation/data.json')
tf_result = evaluate_gma_database(
    gma_db['prior_list'], gma_db['datablock_list'], remove_dummy=True,
    optim_type='mle', optim_opts={
        'max_inner_iters': 1000, 'max_outer_iters': 20,
    }
)

df9 = tf_result['table']
df9 = df9[df9.NODE.str.match('^xsid_')].reset_index(drop=True)
df9['FORTRAN_POST'] = legacy_result_df.RESULT
df9['FORTRAN_RELDIFF'] = np.abs((df9['POST'] - df9['FORTRAN_POST']) / df9['POST'])

with pd.ExcelWriter('sheets/scenario_09.xlsx') as writer:
    df9[['NODE', 'REAC', 'DESCR', 'ENERGY', 'POST', 'FORTRAN_POST', 'FORTRAN_RELDIFF']].to_excel(writer, index=False)


# ------------------------------------------------------------
# 10) Perform iterative reduction with Python TensorFlow code
# and perform evaluation with gmapy tensorflow mode
#   - iterative GLS approach
#   - one iteration only
# ------------------------------------------------------------

with open('input/std2017/approach0/fortran_results_inner_iter3/001_reduction/gmdata.json', 'r') as f:
    unred_gmadb = json.load(f)

tf_reduction_result = reduce_database_iteratively(
    unred_gmadb, max_iters=30, rel_tol=1e-6, remove_dummy=True, optim_type='iterative-gls', save_history=True,
    optim_opts={'max_iters':3, 'rel_tol': 1e-6, 'must_converge': False}
)

hist = np.stack(tf_reduction_result['param_hist'])

gma_db = tf_reduction_result['new_gmadb']
tf_result = evaluate_gma_database(gma_db['prior'], gma_db['datablocks'], remove_dummy=True, optim_opts={'max_iters':2, 'rel_tol': 1e-8, 'must_converge': False})

df10 = tf_result['table']
df10 = df10[df10.NODE.str.match('^xsid_')].reset_index(drop=True)
df10['FORTRAN_POST'] = legacy_result_df.RESULT
df10['FORTRAN_RELDIFF'] = np.abs((df10['POST'] - df10['FORTRAN_POST']) / df10['POST'])

with pd.ExcelWriter('sheets/scenario_10d.xlsx') as writer:
    df10[['NODE', 'REAC', 'DESCR', 'ENERGY', 'POST', 'FORTRAN_POST', 'FORTRAN_RELDIFF']].to_excel(writer, index=False)


# ad-hoc code
from datpy.datpy import reduce_database
from copy import deepcopy
from gmapy.tf_uq.gmap_tf import prepare_stat_model
tmp = prepare_stat_model(
    gma_db['prior'], gma_db['datablocks'], remove_dummy=True, mt6_ppp=False,
    relative=True, optim_type='iterative-gls', optim_opts=None,
)
tbl = tmp['priortable']
tbl = tbl[tbl.NODE != 'fis'].reset_index(drop=True)

# Call for np.save
# tf_reduction_result = reduce_database_iteratively(
#     unred_gmadb, max_iters=30, rel_tol=1e-6, remove_dummy=True, optim_type='iterative-gls', save_history=True,
#     optim_opts={'max_iters':3, 'rel_tol': 1e-6, 'must_converge': False}
# )
# np.save('sheets/scenario_10g.npy', hist)
hist_fgls3 = np.load('sheets/scenario_10g.npy')

# Call for np.save
# tf_reduction_result = reduce_database_iteratively(
#     unred_gmadb, max_iters=30, rel_tol=1e-6, remove_dummy=True, optim_type='mle', save_history=True,
#     optim_opts={'max_inner_iters': 1000, 'max_outer_iters': 15, 'nugget': 1e-4, 'must_converge': False}
# )
# np.save('sheets/scenario_10f.npy', hist)
hist_mle = np.load('sheets/scenario_10f.npy')

# Call for np.save
# tf_reduction_result = reduce_database_iteratively(
#     unred_gmadb, max_iters=30, rel_tol=1e-6, remove_dummy=True, optim_type='chisquare', save_history=True,
#     optim_opts={'max_inner_iters': 1000, 'max_outer_iters': 15, 'nugget': 1e-4, 'must_converge': False}
# )
# np.save('sheets/scenario_10e.npy', hist)
hist_chisquare = np.load('sheets/scenario_10e.npy')


# Call for np.save
# tf_reduction_result = reduce_database_iteratively(
#     unred_gmadb, max_iters=30, rel_tol=1e-6, remove_dummy=True, optim_type='iterative-gls', save_history=True,
#     optim_opts={'max_iters':50, 'rel_tol': 1e-6, 'must_converge': False}
# )
# np.save('sheets/scenario_10d.npy', hist)
hist_fgls = np.load('sheets/scenario_10d.npy')

tbl = tf_result['table'].copy()
tbl = tbl[tbl.NODE != 'fis'].reset_index(drop=True)

for i in range(0, hist.shape[1]):
    if not tbl.at[i, 'NODE'].startswith('xsid_'):
        continue
    plot_title = tbl.at[i, 'DESCR'] + ' at ' + str(tbl.at[i, 'ENERGY']) + ' MeV'
    plt.plot(hist[:,i] / np.mean(hist[:,i]))
    plt.title(plot_title)
    plt.xlabel('iteration')
    plt.ylabel('cross section [rel]')
    plt.savefig(f'sheets/plots10/{plot_title}_rel.png')
    plt.close()
    plt.plot(hist[:,i])
    plt.title(plot_title)
    plt.xlabel('iteration')
    plt.ylabel('cross section')
    plt.savefig(f'sheets/plots10/{plot_title}.png')
    plt.close()


# ------------------------------------------------------------
# 11) Perform iterative reduction with Python TensorFlow code
# and perform evaluation with gmapy tensorflow mode
#   - iterative GLS approach
#   - several iterations (5)
# ------------------------------------------------------------

with open('input/std2017/approach0/01_reduction/gmdata.json', 'r') as f:
    unred_gmadb = json.load(f)

tf_reduction_result = reduce_database_iteratively(
    unred_gmadb, max_iters=5, rel_tol=1e-6, remove_dummy=True, optim_type='iterative-gls',
    optim_opts={'max_iters':30, 'rel_tol': 1e-8}
)

gma_db = tf_reduction_result['new_gmadb']
tf_result = evaluate_gma_database(
    gma_db['prior'], gma_db['datablocks'], remove_dummy=True,
    optim_type='iterative-gls', optim_opts={'max_iters':30, 'rel_tol': 1e-8}
)

df11 = tf_result['table']
df11 = df11[df11.NODE.str.match('^xsid_')].reset_index(drop=True)
df11['FORTRAN_POST'] = legacy_result_df.RESULT
df11['FORTRAN_RELDIFF'] = np.abs((df11['POST'] - df11['FORTRAN_POST']) / df11['POST'])

with pd.ExcelWriter('sheets/scenario_11.xlsx') as writer:
    df11[['NODE', 'REAC', 'DESCR', 'ENERGY', 'POST', 'FORTRAN_POST', 'FORTRAN_RELDIFF']].to_excel(writer, index=False)

# ------------------------------------------------------------
# 12) Perform iterative reduction with Python TensorFlow code
# and perform evaluation with gmapy tensorflow mode
#   - iterative GLS approach
#   - several iterations (10)
# ------------------------------------------------------------

with open('input/std2017/approach0/01_reduction/gmdata.json', 'r') as f:
    unred_gmadb = json.load(f)

tf_reduction_result = reduce_database_iteratively(
    unred_gmadb, max_iters=10, rel_tol=1e-6, remove_dummy=True, optim_type='iterative-gls',
    optim_opts={'max_iters':100, 'rel_tol': 1e-8}
)

gma_db = tf_reduction_result['new_gmadb']
tf_result = evaluate_gma_database(
    gma_db['prior'], gma_db['datablocks'], remove_dummy=True,
    optim_type='iterative-gls', optim_opts={'max_iters':100, 'rel_tol': 1e-8})

df12 = tf_result['table']
df12 = df12[df12.NODE.str.match('^xsid_')].reset_index(drop=True)
df12['FORTRAN_POST'] = legacy_result_df.RESULT
df12['FORTRAN_RELDIFF'] = np.abs((df12['POST'] - df12['FORTRAN_POST']) / df12['POST'])

with pd.ExcelWriter('sheets/scenario_12.xlsx') as writer:
    df12[['NODE', 'REAC', 'DESCR', 'ENERGY', 'POST', 'FORTRAN_POST', 'FORTRAN_RELDIFF']].to_excel(writer, index=False)


# ------------------------------------------------------------
# 13) Perform iterative reduction with Python TensorFlow code
# and perform evaluation with gmapy tensorflow mode
#   - chisquare approach
#   - one iteration
# ------------------------------------------------------------

with open('input/std2017/approach0/01_reduction/gmdata.json', 'r') as f:
    unred_gmadb = json.load(f)

tf_reduction_result = reduce_database_iteratively(
    unred_gmadb, max_iters=1, rel_tol=1e-6, remove_dummy=True,
    optim_type='chisquare', optim_opts={'max_inner_iters':1000, 'max_outer_iters': 30}
)

gma_db = tf_reduction_result['new_gmadb']
tf_result = evaluate_gma_database(
    gma_db['prior'], gma_db['datablocks'], remove_dummy=True,
    optim_type='chisquare', optim_opts={'max_inner_iters':1000, 'max_outer_iters': 30}
)

df13 = tf_result['table']
df13 = df13[df13.NODE.str.match('^xsid_')].reset_index(drop=True)
df13['FORTRAN_POST'] = legacy_result_df.RESULT
df13['FORTRAN_RELDIFF'] = np.abs((df13['POST'] - df13['FORTRAN_POST']) / df13['POST'])

with pd.ExcelWriter('sheets/scenario_13.xlsx') as writer:
    df13[['NODE', 'REAC', 'DESCR', 'ENERGY', 'POST', 'FORTRAN_POST', 'FORTRAN_RELDIFF']].to_excel(writer, index=False)

# ------------------------------------------------------------
# 14) Perform iterative reduction with Python TensorFlow code
# and perform evaluation with gmapy tensorflow mode
#   - Maximum Likelihood Estimation
#   - one iteration
# ------------------------------------------------------------

with open('input/std2017/approach0/01_reduction/gmdata.json', 'r') as f:
    unred_gmadb = json.load(f)

tf_reduction_result = reduce_database_iteratively(
    unred_gmadb, max_iters=1, rel_tol=1e-6, remove_dummy=True,
    optim_type='mle', optim_opts={'max_inner_iters':1000, 'max_outer_iters': 30}
)

gma_db = tf_reduction_result['new_gmadb']
tf_result = evaluate_gma_database(
    gma_db['prior'], gma_db['datablocks'], remove_dummy=True,
    optim_type='mle', optim_opts={'max_inner_iters':1000, 'max_outer_iters': 30}
)

df14 = tf_result['table']
df14 = df14[df14.NODE.str.match('^xsid_')].reset_index(drop=True)
df14['FORTRAN_POST'] = legacy_result_df.RESULT
df14['FORTRAN_RELDIFF'] = np.abs((df14['POST'] - df14['FORTRAN_POST']) / df14['POST'])

with pd.ExcelWriter('sheets/scenario_14.xlsx') as writer:
    df14[['NODE', 'REAC', 'DESCR', 'ENERGY', 'POST', 'FORTRAN_POST', 'FORTRAN_RELDIFF']].to_excel(writer, index=False)

