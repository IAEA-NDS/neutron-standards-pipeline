import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from copy import deepcopy
from datpy.datpy import (
    reduce_database
)
from gmapy.mappings.tf.restricted_map import RestrictedMap
from gmapy.data_management.tablefuns import (
    create_prior_table,
)
from gmapy.mappings.tf.compound_map_tf import (
    CompoundMap as CompoundMapTF,
)
from gmapy.data_management.uncfuns import (
    create_prior_covmat
)
from gmapy.tf_uq.custom_distributions import (
    MultivariateNormalLikelihoodWithCovParams,
)
from gmapy.mappings.priortools import (
    remove_dummy_datasets,
    attach_shape_prior,
    initialize_shape_prior,
)
from gmapy.tf_uq.inference import iterative_gls_estimate
from reduce_helpers import (
    prepare_experiment_info,
    create_cov_linop_fun,
    update_unreduced_gmadb_prior,
    reduce_database_iteratively,
)


with open('input/full_input.json') as f:
    orig_gmadb = json.load(f)

remove_dummy_datasets(orig_gmadb['datablocks'])

############################################################
#   INITIAL REDUCTION FOR BUILDING THE STATISTICAL MODEL
###########################################################

new_gmadb = reduce_database(orig_gmadb)

############################################################
#   BUILD THE STATISTICAL MODEL
###########################################################

priortable = create_prior_table(new_gmadb['prior'])
priorcov = create_prior_covmat(new_gmadb['prior'])
exptable, expvals, expcov_list = prepare_experiment_info(new_gmadb)

# build the covariance linear operator fun
cov_linop_fun = create_cov_linop_fun(expcov_list)

# construct priortable and mapping to experimental data
priortable, priorcov = attach_shape_prior((priortable, exptable), covmat=priorcov, raise_if_exists=False)
compmap = CompoundMapTF((priortable, exptable), reduce=True)
initialize_shape_prior((priortable, exptable), compmap)

# some convenient shortcuts
is_adj = priorcov.diagonal() != 0.
priorvals = priortable.PRIOR.to_numpy()

# define model propagation and jacobian function
adj_idcs = np.where(is_adj)[0]
fixed_idcs = np.where(~is_adj)[0]
restrimap = RestrictedMap(
    len(priorvals), compmap.propagate, compmap.jacobian,
    fixed_params=priorvals[fixed_idcs], fixed_params_idcs=fixed_idcs
)

# define the start values
likelihood = MultivariateNormalLikelihoodWithCovParams(
    len(adj_idcs), 0, restrimap.propagate, restrimap.jacobian, expvals, cov_linop_fun, approximate_hessian=True, relative=True
)

# functions used by optimization routine
propfun = tf.function(likelihood.get_model_prediction)
jacfun = tf.function(likelihood.get_model_jacobian)
cov_linop_fun = tf.function(likelihood.get_covariance_linop)

# Loop over the following stages

startvals = tf.constant(priorvals[is_adj], dtype=tf.float64)

optim_func = lambda x: iterative_gls_estimate(
    x, propfun, jacfun, expvals, cov_linop_fun,
    ret_optres=True, max_iters=300, rel_tol=1e-6, rel_damp_unc=1000
)

# Optimize in several stages. Only done for comparing convergence visually.

orig_gmadb1, new_gmadb1 = reduce_database_iteratively(
    startvals, optim_func, priortable, is_adj, orig_gmadb, new_gmadb,
    expvals, expcov_list, max_iters=3, rel_tol=1e-6
)

orig_gmadb2, new_gmadb2 = reduce_database_iteratively(
    startvals, optim_func, priortable, is_adj, orig_gmadb1, new_gmadb1,
    expvals, expcov_list, max_iters=3, rel_tol=1e-6
)

orig_gmadb3, new_gmadb3 = reduce_database_iteratively(
    startvals, optim_func, priortable, is_adj, orig_gmadb2, new_gmadb2,
    expvals, expcov_list, max_iters=3, rel_tol=1e-6
)


for xsid in range(10): 
    # prepare early stopping result
    cur_prior1 = new_gmadb['prior'][xsid]
    cur_ens1 = cur_prior1['EN']
    cur_xs1 = cur_prior1['CS']
    cur_prior_df1 = pd.DataFrame({'EN': cur_ens1, 'CS': cur_xs1})
    cur_prior_df1 = cur_prior_df1.query('EN >= 0.1 & EN <= 20')
    # prepare intermediate stopping result
    cur_prior2 = new_gmadb2['prior'][xsid]
    cur_ens2 = cur_prior2['EN']
    cur_xs2 = cur_prior2['CS']
    cur_prior_df2 = pd.DataFrame({'EN': cur_ens2, 'CS': cur_xs2})
    cur_prior_df2 = cur_prior_df2.query('EN >= 0.1 & EN <= 20')
    # prepare late stopping result
    cur_prior3 = new_gmadb3['prior'][xsid]
    cur_ens3 = cur_prior3['EN']
    cur_xs3 = cur_prior3['CS']
    cur_prior_df3 = pd.DataFrame({'EN': cur_ens3, 'CS': cur_xs3})
    cur_prior_df3 = cur_prior_df3.query('EN >= 0.1 & EN <= 20')
    # some sanity checks
    assert cur_prior1['CLAB'] == cur_prior2['CLAB']
    assert cur_prior2['CLAB'] == cur_prior3['CLAB']
    # plot the results
    plt.plot(cur_prior_df1.EN, cur_prior_df1.CS)
    plt.plot(cur_prior_df2.EN, cur_prior_df2.CS)
    plt.plot(cur_prior_df3.EN, cur_prior_df3.CS)
    plt.title(cur_prior1['CLAB'])
    plt.show()

