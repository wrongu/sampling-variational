import torch

def create_cross_log_probs_mat_(qs, samples):
    """Given q distributions and samples from them, returns a matrix
    rows corresponding to the log probability of the ith sample
    under the jth q distribution

    Inputs:
    qs: A single torch.Distributions instance representing a batch of K kernel distributions. e.g.
                    `D.Normal(torch.zeros(5),torch.ones(5))` is a batch of 5 1d standard normals
                    (NOT a single 5d standard normal!)
    samples: A tensor represnting a batch of K samples from the above K distributions. Yes, we could
                resample them here but you probably already sampled them for the KL term, so just feed those!

    Returns:
    A K x K matrix of the log prob of each sample under each other kernel
                    """
    # we want to hand each distribution each other sample
    K = len(samples)
    expanded_samples = samples.repeat(K, 1)
    for i in range(K):
        expanded_samples[i] = expanded_samples[i].roll(i)

    cross_log_probs = qs.log_prob(expanded_samples)
    return cross_log_probs

def NCE_MI_approx(qs, samples, device):
    """Returns an NCE estimate of the MI between the theta ensemble and the sampled xs.
    Inputs:
    qs: A single torch.Distributions instance representing a batch of K kernel distributions. e.g.
                    `D.Normal(torch.zeros(5),torch.ones(5))` is a batch of 5 1d standard normals
                    (NOT a single 5d standard normal!)
    samples: A tensor represnting a batch of K samples from the above K distributions. Yes, we could
                resample them here but you probably already sampled them for the KL term, so just feed those!
    """
    device = samples.device
    cross_log_probs = create_cross_log_probs_mat_(qs, samples)
    K = float(len(samples))
    self_terms = torch.sum(cross_log_probs[0])
    denom_terms = torch.logsumexp(cross_log_probs, dim=0)-torch.log(torch.ones(1,device=device)*K)
    # ^ d=0 is over distributions
    total = self_terms - torch.sum(denom_terms)
    return total/K
