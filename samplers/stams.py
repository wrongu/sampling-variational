import torch
from math import sqrt


def stams_mvn_langevin(log_p, lam_kl, q_init, n_samples=1000, burn_in=100, n_kl_samples=100, dt=.01):
    """Run Metropolis-adjusted Langevin dynamics over q parameters (theta), using bound on MI inspired
    by Stam's inequality.

    That is, log_psi = 1/2 log(det(FIM(theta))) - lambda_kl * KL(q(x|theta)||p(x)) + constants, where FIM
    is the Fisher Information Matrix.

    KL(q||p) is evaluated by monte-carlo sampling from q
    """
    samples = torch.zeros(n_samples + burn_in, q_init.n_params)
    samples[0, :] = q_init.theta

    log_psi = torch.zeros(n_samples + burn_in)
    kl = -q_init.entropy() - q_init.monte_carlo_ev(log_p, n_kl_samples)
    log_psi[0] = 0.5*q_init.log_det_fisher() - lam_kl*kl

    # Pre-sample accept/reject thresholds for Metropolis adjustments
    u = torch.rand(n_samples + burn_in).log()
    accept = torch.ones(n_samples + burn_in)

    q = q_init.clone()
    for t in range(1, n_samples + burn_in):
        # Use autograd + monte carlo to estimate gradient of kl(q||p) with respect to theta
        q.theta.requires_grad_(True)
        kl = -q.entropy() - q.monte_carlo_ev(log_p, n_kl_samples)
        grad_kl = torch.autograd.grad(kl, q.theta)[0]
        q.theta.requires_grad_(False)

        # Compute grad log psi
        grad_log_psi = 0.5*q.grad_log_det_fisher() - lam_kl * grad_kl

        # Proposed Langevin step
        new_theta = q.theta + dt * (grad_log_psi + sqrt(2/dt)*torch.randn(q.n_params))
        tmp_theta, q.theta = q.theta, new_theta
        new_kl = -q.entropy() - q.monte_carlo_ev(log_p, n_kl_samples)
        new_log_psi = 0.5*q.log_det_fisher() - lam_kl*new_kl

        # Accept or reject
        if new_log_psi - log_psi[t-1] > u[t]:
            accept[t] = 1.
            log_psi[t] = new_log_psi
        else:
            accept[t] = 0.
            q.theta = tmp_theta
            log_psi[t] = log_psi[t-1]

        # Store sample
        samples[t, :] = q.theta

    # Return a dict containing samples plus other useful metadata
    return {
        'samples': samples[burn_in:, ...],
        'accept': accept[burn_in:].mean(),
        'log_psi': log_psi[burn_in:],
        'lam_kl': lam_kl,
        'burn_samples': samples[:burn_in, ...],
        'burn_accept': accept[:burn_in].mean(),
        'burn_log_psi': log_psi[:burn_in],
    }
