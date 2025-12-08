from copy import deepcopy
import numpy as np
import tensorflow as tf
from datpy.datpy import (
    reduce_database
)
from gmapy.data_management.tablefuns import (
    create_experiment_table,
)
from gmapy.data_management.uncfuns import (
    create_experimental_covmat,
)


def prepare_experiment_info(gmadb, expvals=None, expcov_list=None): 
    """Prepare the exptable, expvals, expcov_list.

    Args:
        gmadb: A GMA database dictionary 
        expvals (tf.Variable): A tf.Variable with experimental values to update.
            If `None`, the tf.Variable will be created.
        expcov_list (list[tf.Variable]): Covariance matrix blocks.
            If `None`, the list of covariance matrix blocks will be created.

    Returns:
        tuple: A 3-tuple containing:
        pd.DataFrame: Data frame with experimental information.
        tf.Variable: Experimental values. If arg `expvals` was provided,
            this data structure will be updated.
        list[tf.Variable]: List of TensorFlow variables representing
            covariance matrices. If arg `expcov_list` was provided,
            this data structure will be updated.
    """
    exptable = create_experiment_table(gmadb['datablocks'])
    expcov = create_experimental_covmat(gmadb['datablocks'], relative=True).toarray()
    exptable['UNC'] = np.sqrt(expcov.diagonal())

    # speed up the pdf log_prob calculations exploiting the block diagonal structure
    block_lens = exptable['DB_IDX'].value_counts().sort_index().to_numpy()
    block_stops = np.cumsum(block_lens)
    block_starts = np.concatenate([[0], block_stops[:-1]], axis=0)

    must_build = False
    if expcov_list is None:
        must_build = True
        expcov_list = [] 
    i = 0
    for (sta, sto) in zip(block_starts, block_stops):
        curmat = tf.constant(expcov[sta:sto, sta:sto], dtype=tf.float64)
        if curmat.shape == (0, 0):
            continue
        if must_build:
            expcov_list.append(tf.Variable(curmat))
        else:
            expcov_list[i].assign(curmat)
        i += 1

    expvals_tf = tf.constant(exptable.DATA.to_numpy(), dtype=tf.float64)
    if expvals is None:
        expvals = tf.Variable(expvals_tf)
    else:
        expvals.assign(expvals_tf)

    return exptable, expvals, expcov_list


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


def create_cov_chol_op(expcov_list):
    expchol_list = [tf.linalg.cholesky(x) for x in expcov_list]
    expchol_op_list = [tf.linalg.LinearOperatorLowerTriangular(
            x, is_non_singular=True, is_square=True
        ) for x in expchol_list]
    expcov_chol = tf.linalg.LinearOperatorBlockDiag(
        expchol_op_list, is_non_singular=True, is_square=True)
    return expcov_chol


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
    startvals, optim_func, priortable, is_adj, orig_gmadb, new_gmadb,
    expvals, expcov_list, max_iters=10, rel_tol=1e-6
):
    """Reduce experimental data.

    """
    orig_gmadb = deepcopy(orig_gmadb)
    new_gmadb = deepcopy(new_gmadb)
    for i in range(max_iters):
        # determine posterior
        optres = optim_func(startvals)
        # udpate prior values in unreduced database
        update_unreduced_gmadb_prior(orig_gmadb, priortable, is_adj, optres.position.numpy())
        # perform reduction to obtain GMA database
        new_gmadb = reduce_database(orig_gmadb)
        # retrieve updated reduced experimental data.
        # as expvals and expcov_list contain tf.Variables
        # which are tracked, this automatically also updates
        # the statistical model, which is represented by
        # the `likelihood` instance.
        old_expvals = expvals.numpy()
        exptable, expvals, expcov_list = prepare_experiment_info(
            new_gmadb, expvals=expvals, expcov_list=expcov_list
        )
        delta = expvals.numpy() - old_expvals
        delta_norm = np.linalg.norm(delta)
        expvals_norm = np.linalg.norm(expvals)
        relative_change = delta_norm / expvals_norm
        print(f'relative change: {relative_change}')
        if relative_change <= rel_tol:
            break
        # use the current best estimate as starting value
        # in next iteration.
        startvals = optres.position

    return orig_gmadb, new_gmadb
