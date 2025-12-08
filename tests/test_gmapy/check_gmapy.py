from pathlib import Path
from gmapy.gmap import run_gmap_simplified
from gmapy.data_management.database_IO import read_gma_database
import pandas as pd
import numpy as np
from helpers import (
    read_gma_result,
    evaluate_gma_database2,
)
from gmapy.mappings.priortools import (
    remove_dummy_datasets
)
from gmapy.legacy.legacy_gmap import run_gmap as run_gmap_legacy 
from gmapy.data_management.tablefuns import (
    create_experiment_table,
)
from gmapy.tf_uq.gmap_tf import evaluate_gma_database as evaluate_gma

input_path = Path('input')

###############################################################
# CHECK THAT RUN_GMAP_LEGACY REPRODUCES FORTRAN GMA RESULT
###############################################################

gma_res_raw = read_gma_result(input_path / 'gma.res')
gma_res = gma_res_raw[['NODE', 'REAC', 'ENERGY', 'RESULT']].copy()
gma_res = gma_res.rename(columns={'RESULT': 'GMA_RESULT'})

gmapy_res2_raw = run_gmap_legacy(
    dbfile=input_path / 'data.gma', dbtype='legacy', num_iter=3,
    correct_ppp=True, remove_dummy=False, legacy_output=False, 
    fix_ppp_bug=False, fix_sacs_jacobian=False, legacy_integration=True,
)

df1 = gma_res.copy()
df2 = gmapy_res2_raw['table']
df2 = df2[df2.NODE.str.startswith('xsid_')].copy()

(df1['NODE'] == df2['NODE']).all()
(df1['ENERGY'] == df2['ENERGY']).all()
np.allclose(df1['GMA_RESULT'], df2['POST'], rtol=1e-4)

# Check passed -> Fortran GMA and Python gmapy code yield exactly the
#                 same result if gmapy run in legacy mode

###############################################################
# CHECK THAT REDUCED DATA IN JSON DOES NOT ALTER RESULT 
###############################################################

gma_res_raw = read_gma_result(input_path / 'gma.res')
gma_res = gma_res_raw[['NODE', 'REAC', 'ENERGY', 'RESULT']].copy()
gma_res = gma_res.rename(columns={'RESULT': 'GMA_RESULT'})

gmapy_res2_raw = run_gmap_legacy(
    dbfile=input_path / 'DAT_NEW.JSON', dbtype='json', num_iter=3,
    correct_ppp=True, remove_dummy=False, legacy_output=False, 
    fix_ppp_bug=False, fix_sacs_jacobian=False, legacy_integration=True,
)

df1 = gma_res.copy()
df2 = gmapy_res2_raw['table']
df2 = df2[df2.NODE.str.startswith('xsid_')].copy()

(df1['NODE'] == df2['NODE']).all()
(df1['ENERGY'] == df2['ENERGY']).all()
np.where(~np.isclose(df1['GMA_RESULT'], df2['POST'], rtol=3e-3))

# Check failed -> all differences below 0.3% except for
#                 10B(n,a0) at 0.0065 MeV where the difference
#                 is about 1%                 

###############################################################
# CHECK THE VALIDITY OF THE PPP CORRECTION IN LEGACY MODE
###############################################################

gmapy_res1_raw = run_gmap_legacy(
    dbfile=input_path / 'DAT_NEW.JSON', dbtype='json', num_iter=5,
    correct_ppp=True, remove_dummy=False, legacy_output=False, 
    fix_ppp_bug=True, fix_sacs_jacobian=True, legacy_integration=False,
)

gmapy_res2_raw = run_gmap_legacy(
    dbfile=input_path / 'DAT_NEW.JSON', dbtype='json', num_iter=5,
    correct_ppp=True, remove_dummy=False, legacy_output=False, 
    fix_ppp_bug='test', fix_sacs_jacobian=True, legacy_integration=False,
)

df1 = gmapy_res1_raw['table']
df2 = gmapy_res2_raw['table']
df1['POST2'] = df2['POST']
df1['RELDIFF'] = ((df1['POST2'] - df1['POST']) / df1['POST']).abs()
df1.sort_values('RELDIFF', inplace=True)

# Check passed -> The PPP correction is consistently implemented,
#                 meaning the minimal invasive fix of the PPP bug
#                 gives idential results to how it has been implemented
#                 at the time.

##############################################################################
# CHECK THAT THE IMPACT OF SACS JACOBIAN BUG AND LEGACY INTEGRATION IS MINOR 
##############################################################################

gmapy_res1_raw = run_gmap_legacy(
    dbfile=input_path / 'DAT_NEW.JSON', dbtype='json', num_iter=5,
    correct_ppp=True, remove_dummy=False, legacy_output=False, 
    fix_ppp_bug=True, fix_sacs_jacobian=True, legacy_integration=False,
)

gmapy_res2_raw = run_gmap_legacy(
    dbfile=input_path / 'DAT_NEW.JSON', dbtype='json', num_iter=5,
    correct_ppp=True, remove_dummy=False, legacy_output=False, 
    fix_ppp_bug='test', fix_sacs_jacobian=False, legacy_integration=True,
)

df1 = gmapy_res1_raw['table']
df2 = gmapy_res2_raw['table']
df1['POST2'] = df2['POST']
df1['RELDIFF'] = ((df1['POST2'] - df1['POST']) / df1['POST']).abs()
df1.sort_values('RELDIFF', inplace=True)

# Check passed -> The difference in result due to the bug in the 
#                 SACS Jacobian evaluation and legacy integration
#                 routine for SACS is negligible, for all evaluated
#                 values below 0.04% except for three PU9(n,f) SACS
#                 where the difference is about 0.2%

##############################################################################
# CHECK THAT THE SIMPLIFIED GMAP ROUTINE YIELDS THE SAME RSULT AS LEGACY ONE
##############################################################################

gmapy_res1_raw = run_gmap_simplified(
    dbfile=input_path / 'DAT_NEW.JSON', dbtype='json', num_iter=5,
    correct_ppp=True, remove_dummy=False,
)

gmapy_res2_raw = run_gmap_legacy(
    dbfile=input_path / 'DAT_NEW.JSON', dbtype='json', num_iter=5,
    correct_ppp=True, remove_dummy=False, legacy_output=False, 
    fix_ppp_bug=True, fix_sacs_jacobian=True, legacy_integration=False,
)

df1 = gmapy_res1_raw['table']
df2 = gmapy_res2_raw['table']
df1['POST2'] = df2['POST']
df1['RELDIFF'] = ((df1['POST2'] - df1['POST']) / df1['POST']).abs()
df1.sort_values('RELDIFF', inplace=True)

# Check passed -> The legacy GMAP routine (bugs disabled) and the
#                 simplified GMAP routine yield the same result.

##############################################################
# CHECK GMAP AND TENSORFLOW APPROACH YIELD SAME RESULT
##############################################################

gmapy_res1_raw = run_gmap_simplified(
    dbfile=input_path / 'DAT_NEW.JSON', dbtype='json', num_iter=2,
    correct_ppp=True, remove_dummy=True, reg=1e-4,
)

gmadb = read_gma_database(input_path / 'DAT_NEW.JSON')
remove_dummy_datasets(gmadb['datablock_list'])
exptable = create_experiment_table(gmadb['datablock_list'])

# gmapy_res2_raw = evaluate_gma_database2(
#     gmadb['prior_list'], gmadb['datablock_list'],
#     rel_tol=1e-40, max_iters=3, remove_dummy=True, mt6_ppp=False,
#     rel_damp_unc=np.sqrt(1e4),
# )
gmapy_res2_raw = evaluate_gma(
    gmadb['prior_list'], gmadb['datablock_list'],
    remove_dummy=True, mt6_ppp=False,
    optim_type='iterative-gls',
    optim_opts = {
        'max_iters': 3, 'rel_tol': 1e-40, 'rel_damp_unc': np.sqrt(1e4), 'must_converge': False,
    }
)

# gmapy_res2_raw = evaluate_gma(
#     gmadb['prior_list'], gmadb['datablock_list'],
#     remove_dummy=True, mt6_ppp=False, relative=True,
#     optim_type='chisquare',
#     optim_opts = {
#         'max_inner_iters': 1000, 'max_outer_iters': 100, 'nugget': 1e-5, 'must_converge': True
#     }
# )

df1 = gmapy_res1_raw['table'].copy()
df1 = df1[df1.NODE.str.match('^xsid_|^norm_')].reset_index(drop=True)

df2 = gmapy_res2_raw['table'].copy()
df2 = df2[df2.NODE != 'fis'].reset_index(drop=True)

(df1.NODE == df2.NODE).all()
(df1.ENERGY == df2.ENERGY).all()
np.allclose(df1.POST, df2.POST)

# Check passed -> Simplified GMA approach yields same result
#                 as TensorFlow approach

##############################################################################
# CHECK CONVERGENCE WITH DUMMY DATA BY USING DIFFERENT NUMBER OF ITERATIONS (USING DUMMY DATA)
##############################################################################

gmapy_res1_raw = run_gmap_simplified(
    dbfile=input_path / 'DAT_NEW.JSON', dbtype='json', num_iter=10,
    correct_ppp=True, remove_dummy=False,
)

gmapy_res2_raw = run_gmap_simplified(
    dbfile=input_path / 'DAT_NEW.JSON', dbtype='json', num_iter=20,
    correct_ppp=True, remove_dummy=False,
)

df1 = gmapy_res1_raw['table']
df2 = gmapy_res2_raw['table']
df1['POST2'] = df2['POST']
df1['RELDIFF'] = ((df1['POST2'] - df1['POST']) / df1['POST']).abs()
df1.sort_values('RELDIFF', inplace=True)

# Check passed -> The result vector converges as the difference
#                 with 10 iterations and 20 iterations is negligible.

#############################################################################
# CHECK THAT WE OBTAIN CONVERGENCE FOR A SPECIFIC CHOICE OF REGULARIZATION 
#############################################################################

gmapy_res1_raw = run_gmap_simplified(
    dbfile=input_path / 'DAT_NEW.JSON', dbtype='json', num_iter=60,
    correct_ppp=True, remove_dummy=True, reg=1e-5,
)

gmapy_res2_raw = run_gmap_simplified(
    dbfile=input_path / 'DAT_NEW.JSON', dbtype='json', num_iter=80,
    correct_ppp=True, remove_dummy=True, reg=1e-5,
)

df1 = gmapy_res1_raw['table']
df2 = gmapy_res2_raw['table']
(df1.NODE == df2.NODE).all()
(df1.REAC == df2.REAC).all()
(df1.ENERGY == df2.ENERGY).all()
np.allclose(df1.POST, df2.POST, rtol=1e-10)

# TODO


df1['POST2'] = df2['POST']
df1['RELDIFF'] = ((df1['POST2'] - df1['POST']) / df1['POST']).abs()
df1.sort_values('RELDIFF', inplace=True)

# Check passed (sort of) -> It seems to be the case that with an
#                           increasing number of iterations the differences
#                           between results become smaller, but convergence
#                           seems also to be very slow.

##############################################################
# CHECK THAT DIFFERENT REGULARIZATIONS GIVE THE SAME RESULT 
##############################################################

gmapy_res1_raw = run_gmap_simplified(
    dbfile=input_path / 'DAT_NEW.JSON', dbtype='json', num_iter=40,
    correct_ppp=True, remove_dummy=True, reg=1e-2,
)

gmapy_res2_raw = run_gmap_simplified(
    dbfile=input_path / 'DAT_NEW.JSON', dbtype='json', num_iter=40,
    correct_ppp=True, remove_dummy=True, reg=1e-6
)

df1 = gmapy_res1_raw['table'].reset_index(drop=True)
df2 = gmapy_res2_raw['table'].reset_index(drop=True)
df_merged = df1.merge(
    df2[['NODE', 'REAC', 'ENERGY', 'POST']],
    on=['NODE', 'REAC', 'ENERGY'],
    how='left'
)
df_merged['RELDIFF'] = ((df_merged['POST_x'] - df_merged['POST_y']) / df_merged['POST_x']).abs()
df_merged.sort_values('RELDIFF', inplace=True)
pd.set_option('display.max_rows', None)
df_merged

# Check passed -> The two results with (reg=1e-2 and reg=1e-6) converge
#                 to the same result (relative difference below 1e-9

###############################################################################
# CHECK THAT DUMMY DATA REGULARIZATION EQUIVALENT TO NEW REGULARIZATION SCHEME
###############################################################################

gmapy_res1_raw = run_gmap_simplified(
    dbfile=input_path / 'DAT_NEW.JSON', dbtype='json', num_iter=20,
    correct_ppp=True, remove_dummy=False, reg=0,
)

gmapy_res2_raw = run_gmap_simplified(
    dbfile=input_path / 'DAT_NEW.JSON', dbtype='json', num_iter=20,
    correct_ppp=True, remove_dummy=True, reg=1e-4
)

df1 = gmapy_res1_raw['table'].reset_index(drop=True)
# remove the dummy data
df1 = df1[~df1.NODE.str.startswith('exp_90')].reset_index(drop=True)
df2 = gmapy_res2_raw['table'].reset_index(drop=True)

np.all(df1['NODE'] == df2['NODE'])
np.all(df1['REAC'] == df2['REAC'])
np.all(df1['ENERGY'] == df2['ENERGY'])
np.allclose(df1['POST'], df2['POST'], rtol=1e-2)
np.where(~np.isclose(df1['POST'], df2['POST'], rtol=3e-2))

df1.loc[[68, 148]]
df2.loc[[68, 148]]

# Check passed (sort of) -> Both results converge with relative difference below 1e-10 except for
#                           6Li(n,a) and 6Li(n,n) at 0.98 MeV energy where the difference is not greater than 3%
#                           (0.245809 vs 0.238680 for 6Li(n,a) and 1.034064 versus 1.041194 for 6Li(n,n)

###################################################################################
# CHECK THAT WE OBTAIN CONVERGENCE MIXING DUMMY DATA AND REGULARITZATION
# FOR BOTH RUN_GMAP_SIMPLIFIED AND TENSORFLOW
###################################################################################

gmapy_res1_raw = run_gmap_simplified(
    dbfile=input_path / 'DAT_NEW.JSON', dbtype='json', num_iter=0,
    correct_ppp=True, remove_dummy=True, reg=1e-6,
)

gmadb = read_gma_database(input_path / 'DAT_NEW.JSON')
gmapy_res2_raw = evaluate_gma_database2(
    gmadb['prior_list'], gmadb['datablock_list'],
    rel_tol=1e-40, max_iters=1, remove_dummy=True, mt6_ppp=False,
    rel_damp_unc=np.sqrt(1e6),
)

df1 = gmapy_res1_raw['table'].copy()
df1 = df1[df1.NODE.str.match('norm_|xsid_')].reset_index(drop=True)
df2 = gmapy_res2_raw.copy()
df2 = df2[df2.NODE.str.match('norm_|xsid_')].reset_index(drop=True)
(df1.NODE == df2.NODE).all()
(df1.REAC == df2.REAC).all()
(df1.ENERGY == df2.ENERGY).all()
np.allclose(df1.POST, df2.RESULT)


###################################################################################
# CHECK THAT WE OBTAIN CONVERGENCE IRRESPECTIVE OF REGULARIZATION WITH TENSORFLOW  
###################################################################################

gmadb = read_gma_database(input_path / 'DAT_NEW.JSON')
# remove_dummy_datasets(gmadb['datablock_list'])

gmapy_res1_raw = evaluate_gma_database2(
    gmadb['prior_list'], gmadb['datablock_list'],
    rel_tol=1e-40, max_iters=40, remove_dummy=True, mt6_ppp=True,
    rel_damp_unc=np.sqrt(0.1),
)
gmapy_res1 = gmapy_res1_raw[gmapy_res1_raw.NODE.str.startswith('xsid_')]
gmapy_res1 = gmapy_res1[['NODE', 'REAC', 'ENERGY', 'RESULT']].copy()
gmapy_res1 = gmapy_res1.rename(columns={'RESULT': 'RESULT1'})

gmapy_res2_raw = evaluate_gma_database2(
    gmadb['prior_list'], gmadb['datablock_list'],
    rel_tol=1e-40, max_iters=80, remove_dummy=True, mt6_ppp=True,
    rel_damp_unc=np.sqrt(1e1),
)
gmapy_res2 = gmapy_res2_raw[gmapy_res2_raw.NODE.str.startswith('xsid_')]
gmapy_res2 = gmapy_res2[['NODE', 'REAC', 'ENERGY', 'RESULT']].copy()
gmapy_res2 = gmapy_res2.rename(columns={'RESULT': 'RESULT2'})

gmapy_res3_raw = evaluate_gma_database2(
    gmadb['prior_list'], gmadb['datablock_list'],
    rel_tol=1e-40, max_iters=50, remove_dummy=False, mt6_ppp=True,
    rel_damp_unc=np.sqrt(1),
)
gmapy_res3 = gmapy_res3_raw[gmapy_res3_raw.NODE.str.startswith('xsid_')]
gmapy_res3 = gmapy_res3[['NODE', 'REAC', 'ENERGY', 'RESULT']].copy()
gmapy_res3 = gmapy_res3.rename(columns={'RESULT': 'RESULT3'})

(gmapy_res1.NODE == gmapy_res2.NODE).all()
(gmapy_res2.NODE == gmapy_res3.NODE).all()
(gmapy_res1.REAC == gmapy_res2.REAC).all()
(gmapy_res2.REAC == gmapy_res3.REAC).all()
(gmapy_res1.ENERGY == gmapy_res2.ENERGY).all()
(gmapy_res2.ENERGY == gmapy_res3.ENERGY).all()
np.allclose(gmapy_res1.RESULT1, gmapy_res2.RESULT2)
np.allclose(gmapy_res2.RESULT2, gmapy_res3.RESULT3)
np.where(~np.isclose(gmapy_res2.RESULT2, gmapy_res3.RESULT3, rtol=3e-3))

# Check passed if relying purely on regularization (not via dummy data)
# gmapy_res1 and gmapy_res2 yield numerically identical results.

# Check failed if mixing dummy data and regularization term (gmapy_res2 != gmapy_res3) 
# All differences below 0.3% except for:
#         NODE        REAC  ENERGY   RESULT2    RESULT3
# 68    xsid_1   MT:1-R1:1  0.9800  0.238680  0.277106
# 148   xsid_2   MT:1-R1:2  0.9800  1.041195  1.002817
# 228   xsid_3   MT:1-R1:3  0.9800  0.102393  0.102008
# 229   xsid_3   MT:1-R1:3  1.0000  0.129525  0.128955
# 371   xsid_6   MT:1-R1:6  0.0025  3.164842  2.905605
# 835  xsid_10  MT:1-R1:10  0.5200  0.000922  0.000898
