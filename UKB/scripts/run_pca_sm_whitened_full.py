
import argparse
import hail as hl
from hail.methods.pca import (_make_tsm_from_call, _pca_and_moments)

hl.init(tmp_dir='gs://ukb-data/tmp/ukb-full')


def _get_whiten_config(args):
    arg_dict = vars(args)
    mean_centered = arg_dict['mean_center']
    hwe_normalized = arg_dict['hwe_normalize']
    normalized_after = arg_dict['normalize_after_whiten']
    if all([mean_centered, not hwe_normalized, normalized_after]):
        # 1st config: mean_center=True, hwe_normalize=False, normalize_after_whiten=True
        config = 1
    elif all([not mean_centered, hwe_normalized, not normalized_after]):
        # 2nd config: mean_center=False, hwe_normalize=True, normalize_after_whiten=False
        config = 2
    else:
        config = None
    return config


bucket = 'gs://ukb-data'
version = 'genotypes'
samples = '406696-samples' # '337111-samples'
gcs_prefix = f'{bucket}/{version}/{samples}'
mt_name = 'gt_147604_406696.mt' # 'gt_147604_337111.mt'
n_parts_scores = 8
n_parts_loadings = 8
overwrite = False

# Set args to pass to _make_tsm_from_call()
whiten_ws = 0
block_size = 100
partition_size = 1000
make_tsm_from_call_args = {
    'block_size': block_size,
    'partition_size': partition_size
}

# Need to specify these args when performing whitening
if whiten_ws != 0:
    make_tsm_from_call_args['whiten_window_size'] = whiten_ws
    make_tsm_from_call_args['whiten_block_size'] = 64

# Set args to pass to _pca_and_moments()
pca_and_moments_args = {
    'k': 100,
    'num_moments': 10,
    'compute_loadings': True,
    'q_iterations': 10,
    'oversampling_param': 10,
    'moment_samples': 100
}

# Flags indicate which normalization args to use in _make_tsm_from_call()
parser = argparse.ArgumentParser(description='To run PCA and spectral moments estimation with whitening.')
parser.add_argument('--mean_center', action='store_true', help='Flag to pass mean_center=True to _make_tsm_from_call(). Otherwise, mean_center=False.')
parser.add_argument('--hwe_normalize', action='store_true', help='Flag to pass hwe_normalize=True to _make_tsm_from_call(). Otherwise, hwe_normalize=False.')
parser.add_argument('--normalize_after_whiten', action='store_true', help='Flag to pass normalize_after_whiten=True to _make_tsm_from_call(). Otherwise, normalize_after_whiten=False.')
args = parser.parse_args()

# Add normalization args to existing args dict
make_tsm_from_call_args = dict(make_tsm_from_call_args, **vars(args))
whiten_config = _get_whiten_config(args)
print(f'Whitening configuration 0{whiten_config}.')
print(f'Args passed to _make_tsm_from_call():\n{make_tsm_from_call_args}')
print(f'Args passed to _pca_and_moments():\n{pca_and_moments_args}')
print(f'Running PCA/SM on full UKB, {samples}...')

# Set full GCS path for writing out tsm block_table, scores table, and loadings table
k_pcs = pca_and_moments_args["k"]
config_folder = f'pca-sm-whitened-0{whiten_config}'
block_table_fname = f'full-block_table-ws{whiten_ws}'
block_table_ht = f'{gcs_prefix}/{config_folder}/{block_table_fname}.ht'
scores_fname = f'full-scores-ws{whiten_ws}-k{k_pcs}'
scores_ht = f'{gcs_prefix}/{config_folder}/{scores_fname}.ht'
loadings_fname = f'full-loadings-ws{whiten_ws}-k{k_pcs}'
loadings_ht = f'{gcs_prefix}/{config_folder}/{loadings_fname}.ht'

# Load full MatrixTable and create whitened TSM
mt = hl.read_matrix_table(f'{gcs_prefix}/{mt_name}')
m_variants, n_samples = mt.count()
tsm = _make_tsm_from_call(mt.GT, **make_tsm_from_call_args)
tsm.block_table.checkpoint(block_table_ht, overwrite=True, _read_if_exists=False)

# Run PCA/SM on whitened TSM
eigvals, scores, loadings, moments, stderrs = _pca_and_moments(tsm, **pca_and_moments_args)

# Set the 0th spectral moment = n_samples, and the 0th standard error = missing
eigvals = list(eigvals)
moments = [n_samples] + list(moments)
stderrs = hl.literal([None] + list(stderrs), 'array<float64>')

scores = scores.annotate_globals(
    name=scores_ht,
    eigenvalues=eigvals,
    spectral_moments=moments,
    standard_errors=stderrs,
    m_variants=m_variants,
    n_samples=n_samples
)
scores = scores.naive_coalesce(n_parts_scores)
scores.write(scores_ht, overwrite=overwrite)

loadings = loadings.annotate_globals(
    name=loadings_ht,
    eigenvalues=eigvals,
    spectral_moments=moments,
    standard_errors=stderrs,
    m_variants=m_variants,
    n_samples=n_samples
)
loadings = loadings.naive_coalesce(n_parts_loadings)
loadings.write(loadings_ht, overwrite=overwrite)
print()

hl.stop()
