import torch
from samplers.base import Sampler


class LaplaceImportance(Sampler):
    def sample(self, init_x, n_samples=10000, cov_scale=4.):
        # Start with a Laplace approximation to create the proposal, scaling the
        # covariance up (precision down) by a factor of cov_scale for extra coverage
        map_x = self.map(init_x.flatten())
        _, _, hessian = self._log_p_helper(map_x, grads=2)
        laplace_q = torch.distributions.MultivariateNormal(loc=map_x, precision_matrix=-hessian/cov_scale)

        # Draw samples from the proposal distribution 'q'
        samples = laplace_q.sample((n_samples,))

        # Compute log p, log q, and unnormalized log weights
        log_p_values = torch.tensor([self.log_p(x) for x in samples])
        log_q_values = laplace_q.log_prob(samples)
        log_weights = (log_p_values - log_q_values).flatten()

        # Compute self-normalized importance weights
        weights = (log_weights - torch.logsumexp(log_weights, dim=0)).exp()

        return {'laplace_mu': laplace_q.loc,
                'laplace_cov': laplace_q.covariance_matrix,
                'samples': samples,
                'weights': weights,
                'log_p': log_p_values,
                'log_q': log_q_values,
                'ess': (weights.sum()*weights.sum()) / (weights*weights).sum()}

