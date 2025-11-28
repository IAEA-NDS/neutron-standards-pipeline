import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from copy import deepcopy
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
    MultivariateNormalLikelihood,
    MultivariateNormalLikelihoodWithCovParams,
)
from gmapy.mappings.priortools import (
    remove_dummy_datasets,
    attach_shape_prior,
    initialize_shape_prior,
)
from gmapy.tf_uq.inference import iterative_gls_estimate
from gmapy.data_management.tablefuns import (
    create_experiment_table,
)
from gmapy.data_management.uncfuns import (
    create_experimental_covmat,
)


def read_gma_result(filename: str) -> dict:
    """Read results from GMA result file."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    matchstr = '1   RESULT'
    results = {}
    reac_order = {}
    pos = -1
    while True: 
        pos += 1
        if pos == len(lines):
            break
        line = lines[pos]
        if not line.startswith(matchstr):
            continue
        cur_reac = line[len(matchstr):].strip()
        results[cur_reac] = {"energy": [], "xs": []}
        reac_order.setdefault(cur_reac, len(reac_order))
        pos += 4
        while True: 
            pos += 1
            if pos == len(lines):
                break
            line = lines[pos]
            if line[11:13].strip() != '':
                break
            energy, xs = line.split()[:2]
            energy = float(energy)
            xs = float(xs)
            results[cur_reac]['energy'].append(energy)
            results[cur_reac]['xs'].append(xs)

    # combine to dataframe
    df_list = []
    for reac, cont in results.items():
        xsid = reac_order[reac] + 1
        curdf = pd.DataFrame({
            'NODE': f'xsid_{xsid}',
            'REAC': f'MT:1-R1:{xsid}',
            'REAC_STRING': reac,
            'ENERGY': cont['energy'],
            'RESULT': cont['xs'],
        })
        df_list.append(curdf)

    df = pd.concat(df_list, ignore_index=True)
    return df


def create_cov_chol_op(expcov_list):
    expchol_list = [tf.linalg.cholesky(x) for x in expcov_list]
    expchol_op_list = [tf.linalg.LinearOperatorLowerTriangular(
            x, is_non_singular=True, is_square=True
        ) for x in expchol_list]
    expcov_chol = tf.linalg.LinearOperatorBlockDiag(
        expchol_op_list, is_non_singular=True, is_square=True)
    return expcov_chol


def create_cov_linop_fun(expcov_list):
    """Create a linear covariance operator function.

    Create a function that returns a TensorFlow LinearOperator
    representing a covariance matrix. The generated function
    expects formally a 1d input tensor as argument for 
    compatibility with the gmapy 
    `MultivariateNormalLikelihoodWithCovParams` class but does
    not use it. 
    """
    def cov_linop_fun(x):
        expchol_list = [tf.linalg.cholesky(x) for x in expcov_list]
        expchol_op_list = [tf.linalg.LinearOperatorLowerTriangular(
                x, is_non_singular=True, is_square=True
            ) for x in expchol_list]
        expcov_chol = tf.linalg.LinearOperatorBlockDiag(
            expchol_op_list, is_non_singular=True, is_square=True)
        expcov_linop = tf.linalg.LinearOperatorComposition(
            [expcov_chol, expcov_chol.adjoint()],
            is_self_adjoint=True, is_positive_definite=True
        )
        return expcov_linop
    return cov_linop_fun


def evaluate_gma_database(
    prior: list, datablocks: list, remove_dummy: bool=True,
    rel_tol=1e-6, max_iters: int=10
) -> pd.DataFrame:
    # with open('input/full_input.json') as f:

    if remove_dummy:
        remove_dummy_datasets(datablocks)

    priortable = create_prior_table(prior)
    priorcov = create_prior_covmat(prior)

    exptable = create_experiment_table(datablocks)
    expcov = create_experimental_covmat(datablocks, relative=True).toarray()
    exptable['UNC'] = np.sqrt(expcov.diagonal())

    # speed up the pdf log_prob calculations exploiting the block diagonal structure
    block_lens = exptable['DB_IDX'].value_counts().sort_index().to_numpy()
    block_stops = np.cumsum(block_lens)
    block_starts = np.concatenate([[0], block_stops[:-1]], axis=0)

    expcov_list = []
    for (sta, sto) in zip(block_starts, block_stops):
        curmat = tf.constant(expcov[sta:sto, sta:sto], dtype=tf.float64)
        if curmat.shape == (0, 0):
            continue
        expcov_list.append(tf.Variable(curmat))

    # build the covariance linear operator fun
    cov_chol_op = create_cov_chol_op(expcov_list)

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

    expvals = tf.constant(exptable.DATA.to_numpy(), dtype=tf.float64)

    # define the start values
    likelihood = MultivariateNormalLikelihood(
        len(adj_idcs), restrimap.propagate, restrimap.jacobian, expvals, cov_chol_op, approximate_hessian=True, relative=True
    )

    # functions used by optimization routine
    propfun = tf.function(likelihood.get_model_prediction)
    jacfun = tf.function(likelihood.get_model_jacobian)
    # cov_linop_fun = tf.function(likelihood.get_covariance_linop)
    cov_linop_fun = likelihood.get_covariance_linop

    startvals = tf.constant(priorvals[is_adj], dtype=tf.float64)

    optim_func = lambda x: iterative_gls_estimate(
        x, propfun, jacfun, expvals, cov_linop_fun,
        ret_optres=False, max_iters=max_iters,
        rel_tol=rel_tol, rel_damp_unc=1e4, must_converge=False
    )

    # Optimize in several stages. Only done for comparing convergence visually.
    optres = optim_func(startvals)
    result_df = priortable.copy()
    result_df.loc[is_adj, 'RESULT'] = optres.numpy()
    return result_df


# TODO: Ugly, remove this function and parametrize
#       the inner optimization routine...
def evaluate_gma_database2(
    prior: list, datablocks: list, remove_dummy: bool=True,
    rel_tol=1e-6, max_iters: int=10, mt6_ppp: bool=False, rel_damp_unc=1e4
) -> pd.DataFrame:
    # with open('input/full_input.json') as f:

    prior = deepcopy(prior)
    datablocks = deepcopy(datablocks)

    if remove_dummy:
        remove_dummy_datasets(datablocks)

    priortable = create_prior_table(prior)
    priorcov = create_prior_covmat(prior)

    exptable = create_experiment_table(datablocks)
    expcov = create_experimental_covmat(datablocks, relative=True).toarray()
    exptable['UNC'] = np.sqrt(expcov.diagonal())

    # speed up the pdf log_prob calculations exploiting the block diagonal structure
    block_lens = exptable['DB_IDX'].value_counts().sort_index().to_numpy()
    block_stops = np.cumsum(block_lens)
    block_starts = np.concatenate([[0], block_stops[:-1]], axis=0)

    expcov_list = []
    for (sta, sto) in zip(block_starts, block_stops):
        curmat = tf.constant(expcov[sta:sto, sta:sto], dtype=tf.float64)
        if curmat.shape == (0, 0):
            continue
        expcov_list.append(tf.Variable(curmat))

    # build the covariance linear operator fun
    cov_linop_fun = create_cov_linop_fun(expcov_list)

    # construct priortable and mapping to experimental data
    priortable, priorcov = attach_shape_prior((priortable, exptable), covmat=priorcov, raise_if_exists=False)
    compmap = CompoundMapTF((priortable, exptable), reduce=True)

    # some convenient shortcuts
    is_adj = priorcov.diagonal() != 0.
    priorvals = priortable.PRIOR.to_numpy()
    refvals = priorvals.copy()
    reluncs = exptable['UNC'].to_numpy()

    initialize_shape_prior((priortable, exptable), compmap, refvals=refvals, uncs=reluncs)

    # define model propagation and jacobian function
    adj_idcs = np.where(is_adj)[0]
    fixed_idcs = np.where(~is_adj)[0]
    restrimap = RestrictedMap(
        len(priorvals), compmap.propagate, compmap.jacobian,
        fixed_params=priorvals[fixed_idcs], fixed_params_idcs=fixed_idcs
    )

    expvals = tf.constant(exptable.DATA.to_numpy(), dtype=tf.float64)

    # determine the MT6 (SACS) experiment indices
    mt6_idcs=None
    if not mt6_ppp:
        mt6_idcs = np.where(exptable.REAC.str.startswith('MT:6-'))[0]

    # determine indices of normalization factors
    norm_sel = priortable.loc[is_adj, 'NODE'].str.match('norm_').to_numpy()
    norm_idcs = np.where(norm_sel)[0]
    norm_values = np.ones(len(norm_idcs), dtype=int)

    # define the start values
    likelihood = MultivariateNormalLikelihoodWithCovParams(
        len(adj_idcs), 0, restrimap.propagate, restrimap.jacobian,
        expvals, cov_linop_fun, approximate_hessian=True, relative=True,
        no_ppp_idcs=mt6_idcs, cov_freeze_param_idcs=norm_idcs, cov_freeze_param_values=norm_values
    )

    # functions used by optimization routine
    propfun = tf.function(likelihood.get_model_prediction)
    jacfun = tf.function(likelihood.get_model_jacobian)
    cov_linop_fun = tf.function(likelihood.get_covariance_linop)

    startvals = tf.constant(priorvals[is_adj], dtype=tf.float64)

    optim_func = lambda x: iterative_gls_estimate(
        x, propfun, jacfun, expvals, cov_linop_fun,
        ret_optres=False, max_iters=max_iters,
        rel_tol=rel_tol, rel_damp_unc=rel_damp_unc, must_converge=False
    )

    # Optimize in several stages. Only done for comparing convergence visually.
    optres = optim_func(startvals)
    result_df = priortable.copy()
    result_df.loc[is_adj, 'RESULT'] = optres.numpy()
    return result_df
