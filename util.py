import torch


def acf(samples, max_lag=None):
    """Compute autocorrelation function (ACF) for each sample; result is size (max_lag, samples.size()[1:])
    """
    n, shape = samples.size(0), samples.size()[1:]
    max_lag = max_lag if max_lag is not None else n//2
    z_samples = (samples - samples.mean(dim=0, keepdims=True)) / samples.std(dim=0, keepdims=True)
    acf = torch.zeros((max_lag,) + shape)
    acf[0, :] = 1.
    for t in range(1, max_lag):
        acf[t, :] = torch.mean(z_samples[:-t, ...] * z_samples[t:, ...], dim=0)
    return acf


def ess(samples, max_lag=None):
    """Compute effective sample size (ESS) for each column of given set of samples
    """
    autocorrelation = acf(samples, max_lag)
    # Sums of adjacent elements should never be negative unless due to sample noise. Truncate
    # ACF as soon as we find a negative pair. Appending a row of -1 to pair_sum to ensure at
    # least one negative value is found.
    # See Geyer (1992) or tensorflow.org/probability/api_docs/python/tfp/mcmc/effective_sample_size
    pair_sum = torch.cat([autocorrelation[:-1, ...] + autocorrelation[1:, ...],
                          -torch.ones(1, samples.size(1))], dim=0)
    first_neg_idx = [torch.where(pair_sum[:, i] < 0)[0][0] for i in range(samples.size(1))]
    sum_rho = torch.tensor([autocorrelation[1:idx, i].sum() for i, idx in enumerate(first_neg_idx)])
    return samples.size(0) / (1 + 2 * sum_rho)


def is_positive_definite(m):
    try:
        _ = torch.linalg.cholesky(m)
        return True
    except RuntimeError:
        return False
