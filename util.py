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
    # TODO - add a more standard option dynamically choosing max_lag instead of clipping the ACF
    autocorrelation = torch.clip(acf(samples, max_lag), 0., 1.)
    return samples.size(0) / (1 + 2 * autocorrelation[1:, ...].sum(dim=0))
