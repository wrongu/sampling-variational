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


    def _log_psi_helper(q):
        q.theta.requires_grad_(True)
        _kl = -q.entropy() - q.monte_carlo_ev(log_p, n_kl_samples)
        _grad_kl = torch.autograd.grad(_kl, q.theta)[0]
        q.theta.requires_grad_(False)
        _log_psi = 0.5*q.log_det_fisher() - lam_kl*_kl.detach()
        _grad_log_psi = 0.5*q.grad_log_det_fisher() - lam_kl*_grad_kl
        return _log_psi, _grad_log_psi


    log_psi = torch.zeros(n_samples + burn_in)
    log_psi[0], grad_log_psi = _log_psi_helper(q_init)

    # Pre-sample accept/reject thresholds for Metropolis adjustments
    u = torch.rand(n_samples + burn_in).log()
    accept = torch.ones(n_samples + burn_in)

    q = q_init.clone()
    for t in range(1, n_samples + burn_in):
        # Proposed Langevin step
        new_theta = q.theta + dt * grad_log_psi + sqrt(2*dt)*torch.randn(q.n_params)

        # Provisionally assign new_theta to q and evaluate the new point
        tmp_theta, q.theta = q.theta, new_theta
        new_log_psi, new_grad_log_psi = _log_psi_helper(q)

        # Compute (log) Metropolis ratio, log[p(x')q(x|x')/p(x)q(x'|x)]
        log_q_forward = -1 / (4*dt) * torch.sum((new_theta - (tmp_theta + dt*grad_log_psi))**2)
        log_q_reverse = -1 / (4*dt) * torch.sum((tmp_theta - (new_theta + dt*new_grad_log_psi))**2)
        log_metropolis_ratio =  new_log_psi - log_psi[t-1] + log_q_reverse - log_q_forward

        # Accept or reject
        if log_metropolis_ratio > u[t]:
            accept[t] = 1.
            log_psi[t] = new_log_psi
            grad_log_psi = new_grad_log_psi
        else:
            accept[t] = 0.
            q.theta = tmp_theta  # Undo provisional update to q from above
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
