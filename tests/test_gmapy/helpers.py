# NOTE: The functionality of this module is already fully
#       integrated into the gmapy package (as of 2025-12-08)

import json
from typing import Optional
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
    ChiSquarePseudoDistWithCovParams,
)
from gmapy.mappings.priortools import (
    remove_dummy_datasets,
    attach_shape_prior,
    initialize_shape_prior,
)
from gmapy.tf_uq.inference import (
    iterative_gls_estimate,
    determine_MAP_estimate,
)
from gmapy.data_management.tablefuns import (
    create_experiment_table,
)
from gmapy.data_management.uncfuns import (
    create_experimental_covmat,
)
from datpy.datpy import (
    reduce_database
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



def update_expvals(
    exptable: pd.DataFrame, expvals: Optional[tf.Variable]=None
) -> tf.Variable:
    """Create or update the experimental values."""
    if expvals is None:
        return tf.Variable(exptable.DATA.to_numpy(), dtype=tf.float64)
    else:
        expvals.assign(exptable.DATA.to_numpy())
        return expvals


def update_expcov_list(
    exptable: pd.DataFrame, expcov: np.array,
    expcov_list: Optional[list[tf.Variable]]=None
) -> list[tf.Variable]:
    """Create or update list of covariance matrix blocks."""
    block_lens = exptable['DB_IDX'].value_counts().sort_index().to_numpy()
    block_stops = np.cumsum(block_lens)
    block_starts = np.concatenate([[0], block_stops[:-1]], axis=0)

    must_create = False
    if expcov_list is None:
        must_create = True
        expcov_list = []

    for i, (sta, sto) in enumerate(zip(block_starts, block_stops)):
        curmat = tf.constant(expcov[sta:sto, sta:sto], dtype=tf.float64)
        if must_create:
            expcov_list.append(tf.Variable(curmat))
        else:
            expcov_list[i].assign(curmat)

    return expcov_list


def prepare_stat_model(
    prior: list, datablocks: list, remove_dummy: bool=True, mt6_ppp=False,
    relative=True, optim_type='iterative-gls', optim_opts=None,
) -> dict:
    """Prepare all quantities for statistical inference."""
    if optim_opts is None:
        optim_opts = {}

    prior = deepcopy(prior)
    datablocks = deepcopy(datablocks)

    if remove_dummy:
        remove_dummy_datasets(datablocks)

    priortable = create_prior_table(prior)
    priorcov = create_prior_covmat(prior)

    exptable = create_experiment_table(datablocks)
    expcov = create_experimental_covmat(datablocks, relative=relative).toarray()
    exptable['UNC'] = np.sqrt(expcov.diagonal())

    expvals = update_expvals(exptable, None)
    expcov_list = update_expcov_list(exptable, expcov, None)

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

    # determine the MT6 (SACS) experiment indices
    mt6_idcs=None
    if not mt6_ppp:
        mt6_idcs = np.where(exptable.REAC.str.startswith('MT:6-'))[0]

    # determine indices of normalization factors
    norm_sel = priortable.loc[is_adj, 'NODE'].str.match('norm_').to_numpy()
    norm_idcs = np.where(norm_sel)[0]
    norm_values = np.ones(len(norm_idcs), dtype=int)

    distfun = (
        ChiSquarePseudoDistWithCovParams if optim_type.lower() == 'chisquare'
        else MultivariateNormalLikelihoodWithCovParams  # for MLE and iterative GLS
    )

    propfun = tf.function(restrimap.propagate)
    jacfun = tf.function(restrimap.jacobian)

    likelihood = distfun(
        len(adj_idcs), 0, propfun, jacfun,
        expvals, cov_linop_fun, approximate_hessian=True, relative=relative,
        no_ppp_idcs=mt6_idcs, cov_freeze_param_idcs=norm_idcs, cov_freeze_param_values=norm_values
    )

    if optim_type.lower() == 'iterative-gls':
        optim_func = lambda x: iterative_gls_estimate(
            x,
            likelihood.get_model_prediction,
            likelihood.get_model_jacobian,
            expvals,
            likelihood.get_covariance_linop,
            ret_optres=False, **optim_opts
        )
    elif optim_type.lower() in ('mle', 'chisquare'):
        neg_log_prob_and_gradient = tf.function(likelihood.neg_log_prob_and_gradient)
        neg_log_prob_hessian = likelihood.log_prob_hessian

        optim_func = lambda x: determine_MAP_estimate(
            x, neg_log_prob_and_gradient, neg_log_prob_hessian, ret_optres=False, **optim_opts
        )
    else:
        raise ValueError('Unknown optimization type')

    return {
        'optim_func': optim_func,
        'priortable': priortable,
        'exptable': exptable,
        'expcov_rel': expcov,
        'likelihood': likelihood,
        'is_adj': is_adj,
        'expvals': expvals,
        'expcov_list': expcov_list,
    }


def evaluate_gma(
    prior: list, datablocks: list, remove_dummy: bool=True, mt6_ppp=False,
    relative=True, optim_type='iterative-gls', optim_opts=None,
) -> dict:

    model_info = prepare_stat_model(
        prior, datablocks, remove_dummy, mt6_ppp,
        relative, optim_type, optim_opts
    )

    is_adj = model_info['is_adj']
    priorvals = model_info['priortable']['PRIOR']
    startvals = tf.constant(priorvals[is_adj], dtype=tf.float64)
    optim_func = model_info['optim_func']

    optres = optim_func(startvals)
    result_df = model_info['priortable'].copy()
    result_df.loc[is_adj, 'RESULT'] = optres.numpy()
    result_df.loc[~is_adj, 'RESULT'] = result_df.loc[~is_adj, 'PRIOR']
    return {
        'table': result_df,
        'likelihood': model_info['likelihood'],
        'exptable': model_info['exptable'],
        'expcov_rel': model_info['expcov_rel'],
        'expvals': model_info['expvals'],
        'expcov_list': model_info['expcov_list'],
    }


def update_unreduced_gmadb_prior(gmadb, priortable, is_adj, newvals):
    """Update the values in the prior used for reduction.

    Copy the values from the `PRIOR` column in `priortable`
    to the appropriate location in the `gmadb` dict.
    Importantly, `gmadb` contains the data *before reduction*.

    Args:
        gmadb (dict): Data structure with the prior and the
            unreduced experimental data.
        priortable (pd.DataFrame): Dataframe with the prior
            information and additionally a column `PRIOR`
            with updated prior values.

    Returns:
        None: Modifications of gmadb are performed inplace.
    """
    priortable = priortable.copy()
    priortable.loc[is_adj, 'POST'] = newvals
    xsprior_df = priortable[priortable.NODE.str.startswith('xsid_')]
    for xsidstr, group_df in xsprior_df.groupby('NODE'):
        prior_id = int(xsidstr.split('_')[1]) - 1
        postxs = group_df['POST'].to_list()
        curprior = gmadb['prior'][prior_id]
        # The original prior for reduction has
        # two limit points more
        assert len(curprior['EN'])-2 == len(postxs)
        assert curprior['EN'][1:-1] == group_df['ENERGY'].to_list()
        curprior['CS'][1:-1] = postxs
        curprior['CS'][0] = postxs[0]
        curprior['CS'][-1] = postxs[-1]


def reduce_database_iteratively(
    orig_gmadb, max_iters=10, rel_tol=1e-6,
    remove_dummy=True, mt6_ppp=False,
    optim_type='iterative-gls', optim_opts=None
):
    """Reduce experimental data."""
    orig_gmadb = deepcopy(orig_gmadb)
    if remove_dummy:
        remove_dummy_datasets(orig_gmadb['datablocks'])

    new_gmadb = reduce_database(orig_gmadb)

    model_info = prepare_stat_model(
        new_gmadb['prior'], new_gmadb['datablocks'],
        mt6_ppp=mt6_ppp, optim_type=optim_type, optim_opts=optim_opts
    )

    is_adj = model_info['is_adj']
    priortable = model_info['priortable']
    expvals = model_info['expvals']
    expcov_list = model_info['expcov_list']
    optim_func = model_info['optim_func']

    startvals = tf.constant(
        priortable.loc[is_adj, 'PRIOR'].to_numpy(),
        dtype=tf.float64
    )

    for i in range(max_iters):
        print(f'Outer iteration: {i}')
        # determine posterior
        optres = optim_func(startvals)
        # udpate prior values in unreduced database
        update_unreduced_gmadb_prior(
            orig_gmadb, priortable, is_adj, optres.numpy()
        )
        # perform reduction to obtain GMA database
        new_gmadb = reduce_database(orig_gmadb)
        # retrieve updated reduced experimental data.
        old_expvals = expvals.numpy()
        exptable = create_experiment_table(new_gmadb['datablocks'])
        expcov = create_experimental_covmat(
            new_gmadb['datablocks'], relative=True
        ).toarray()
        # update the tf.Variables expvals and expcov_list.
        # these changes are also registered in the likelihood instance
        expvals = update_expvals(exptable, expvals)
        expcov_list = update_expcov_list(exptable, expcov, expcov_list)

        delta = expvals.numpy() - old_expvals
        delta_norm = np.linalg.norm(delta)
        expvals_norm = np.linalg.norm(expvals)
        relative_change = delta_norm / expvals_norm
        print(f'relative change: {relative_change}')
        if relative_change <= rel_tol:
            break
        # use the current best estimate
        # as starting value in next iteration.
        startvals = optres

    return orig_gmadb, new_gmadb
