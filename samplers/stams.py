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


    q = q_init.clone()
    kl_eps = torch.randn(q.d, n_kl_samples)
    def _log_psi_helper(theta):
        q.theta.copy_(theta)
        q.theta.requires_grad_(True)
        _kl = -q.entropy() - q.monte_carlo_ev(log_p, eps=kl_eps)
        _grad_kl = torch.autograd.grad(_kl, q.theta)[0]
        q.theta.requires_grad_(False)
        _log_psi = 0.5*q.log_det_fisher() - lam_kl*_kl.detach()
        _grad_log_psi = 0.5*q.grad_log_det_fisher() - lam_kl*_grad_kl
        return _log_psi, _grad_log_psi


    log_psi = torch.zeros(n_samples + burn_in)
    log_psi[0], grad_log_psi = _log_psi_helper(samples[0, :])

    # Pre-sample accept/reject thresholds for Metropolis adjustments
    u = torch.rand(n_samples + burn_in).log()
    accept = torch.ones(n_samples + burn_in)

    for t in range(1, n_samples + burn_in):
        # Proposed Langevin step
        theta, step_theta = samples[t-1, :], dt * grad_log_psi + sqrt(2*dt)*torch.randn(q_init.n_params)

        # Evaluate the new point
        new_log_psi, new_grad_log_psi = _log_psi_helper(theta + step_theta)

        # Compute (log) Metropolis ratio, log[p(x')q(x|x')/p(x)q(x'|x)]
        log_q_forward = -1 / (4*dt) * torch.sum((step_theta - dt*grad_log_psi)**2)
        log_q_reverse = -1 / (4*dt) * torch.sum((step_theta + dt*new_grad_log_psi)**2)
        log_metropolis_ratio = new_log_psi - log_psi[t-1] + log_q_reverse - log_q_forward

        # Accept or reject
        if log_metropolis_ratio > u[t]:
            accept[t] = 1.
            # Upon accept, resample kl_eps and recompute log_psi so that the next iteration's 
            # metropolis_ratio is comparing the same kl_eps values
            kl_eps.copy_(torch.randn(q.d, n_kl_samples))
            log_psi[t], grad_log_psi = _log_psi_helper(th)
            samples[t, :] = theta + step_theta
        else:
            accept[t] = 0.
            log_psi[t] = log_psi[t-1]
            samples[t, :] = theta

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


def stams_mvn_hmc(log_p, lam_kl, q_init, n_samples=1000, burn_in=100, n_leapfrog=50, n_kl_quad=10, dt=.01):
    """Run Hamiltonian Monte Carlo dynamics over q parameters (theta), using bound on MI inspired
    by Stam's inequality.

    That is, log_psi = 1/2 log(det(FIM(theta))) - lambda_kl * KL(q(x|theta)||p(x)) + constants, where FIM
    is the Fisher Information Matrix.

    KL(q||p) is evaluated using Gauss-Hermite quadrature with 10 points per dimension
    """


    q = q_init.clone()
    def _log_psi_helper(theta):
        q.theta.copy_(theta)
        _kl = -q.entropy() - q.quadrature_ev(log_p, n=n_kl_quad)
        return 0.5*q.log_det_fisher() - lam_kl*_kl

    def _grad_log_psi_helper(theta):
        q.theta.copy_(theta)
        q.theta.requires_grad_(True)
        _kl = -q.entropy() - q.quadrature_ev(log_p, n=n_kl_quad)
        _grad_kl = torch.autograd.grad(_kl, q.theta)[0]
        q.theta.requires_grad_(False)
        return 0.5*q.grad_log_det_fisher() - lam_kl*_grad_kl

    def _mass_helper(theta, lower_bound=0.001, upper_bound=1000):
        q.theta.copy_(theta)
        return torch.clip(1/q.fisher().diag(), lower_bound, upper_bound)


    samples = torch.zeros(n_samples + burn_in, q_init.n_params)
    samples[0, :] = q_init.theta

    masses = torch.zeros(n_samples + burn_in, q_init.n_params)
    masses[0, :] = _mass_helper(q_init.theta)

    log_psi = torch.zeros(n_samples + burn_in)
    log_psi[0] = _log_psi_helper(samples[0, :])

    # Pre-sample accept/reject thresholds for Metropolis adjustments
    u = torch.rand(n_samples + burn_in).log()
    accept = torch.ones(n_samples + burn_in)

    # Pre-sample momentum values (not yet adjusted by mass)
    momentum_z = torch.randn(n_samples + burn_in, q_init.n_params)

    for t in range(1, n_samples + burn_in):
        th = samples[t-1, :]
        # Pick mass for this trajectory based on current theta but keep it constant across
        # leapfrog steps so we don't have to do fixed-point leapfrog steps
        mass = _mass_helper(th)
        masses[t, :] = mass
        # Run leapfrog dynamics
        p0, g = momentum_z[t, :] * torch.sqrt(mass), _grad_log_psi_helper(samples[t-1, :])
        # First half-step.. this places p on the 'half time' schedule
        p = p0 + (dt/2) * g
        for l in range(n_leapfrog):
            th = th + dt * p / mass
            g = _grad_log_psi_helper(th)
            p = p + dt * g
        # Undo extra half-step of momentum from final loop
        p = p - (dt/2) * g

        # Evaluate the new point
        new_log_psi = _log_psi_helper(th)

        # Compute (log) Metropolis ratio, log[p(x')q(x|x')/p(x)q(x'|x)]
        new_hamiltonian = new_log_psi - torch.sum(p*p/mass)/2
        old_hamiltonian = log_psi[t-1] - torch.sum(p0*p0/mass)/2
        log_metropolis_ratio = new_hamiltonian - old_hamiltonian

        # Accept or reject
        if log_metropolis_ratio > u[t]:
            accept[t] = 1.
            log_psi[t] = new_log_psi
            samples[t, :] = th
        else:
            accept[t] = 0.
            log_psi[t] = log_psi[t-1]
            samples[t, :] = samples[t-1, :]

    # Return a dict containing samples plus other useful metadata
    return {
        'samples': samples[burn_in:, ...],
        'accept': accept[burn_in:].mean(),
        'log_psi': log_psi[burn_in:],
        'lam_kl': lam_kl,
        'masses': masses[burn_in:,:],
        'burn_samples': samples[:burn_in, ...],
        'burn_accept': accept[:burn_in].mean(),
        'burn_log_psi': log_psi[:burn_in],
        'burn_masses': masses[:burn_in,:]
    }


def stams_importance_sampling(log_p, lam_kl, q_init, n_samples, n_kl_samples=100):
    """Generate a set of samples of theta~Q and importance-sampling weights, psi/Q

    Strategy is based on a Laplace approximation: we navigate to the MAP value of theta, then halve the curvature there
    to get a gaussian proposal distribution in theta-space.
    """


    q = q_init.clone()
    def _log_psi_helper(theta):
        q.theta.copy_(theta)
        _kl = -q.entropy() - q.monte_carlo_ev(log_p, n_kl_samples)
        return 0.5*q.log_det_fisher() - lam_kl*_kl

    def _grad_log_psi_helper(theta):
        q.theta.copy_(theta)
        q.theta.requires_grad_(True)
        _kl = -q.entropy() - q.monte_carlo_ev(log_p, n_kl_samples)
        _grad_kl = torch.autograd.grad(_kl, q.theta)[0]
        q.theta.requires_grad_(False)
        return 0.5*q.grad_log_det_fisher() - lam_kl*_grad_kl

    def _hess_log_psi_helper(theta):
        q.theta.copy_(theta)
        q.theta.requires_grad_(True)
        _kl = -q.entropy() - q.monte_carlo_ev(log_p, n_kl_samples)
        _tmp_log_psi = 0.5*q.log_det_fisher() - lam_kl*_kl
        _grad_log_psi = torch.autograd.grad(_tmp_log_psi, q.theta, create_graph=True)[0]
        curv = torch.zeros(q.n_params, q.n_params)
        for i in range(q.n_params):
            curv[i, :] = torch.autograd.grad(_grad_log_psi[i], q.theta, retain_graph=True)[0]
        q.theta.requires_grad_(False)
        return curv


    # First part: optimize theta towards mode of psi
    th = q_init.theta
    th_optim = torch.zeros(401, q_init.n_params)
    th_optim[0, :] = th
    # Warm-up with 100 gradient-ascent steps
    for t in range(100):
        # Get gradient
        g, lr = _grad_log_psi_helper(th), .1 / (1 + t // 5)
        # Take a gradient ascent step
        th = th + g * lr
        # Record optimization trajectory
        th_optim[t+1, :] = th

    # Rapidly find the max using newton's method with an additional bit of learning rate decay to ensure
    # convergence despite stochastic estimation of KL term
    for t in range(300):
        # Get gradient
        g, h, lr = _grad_log_psi_helper(th), _hess_log_psi_helper(th), 1 / (1 + t // 5)
        th = th - torch.linalg.solve(h, g) * lr
        # Record optimization trajectory
        th_optim[t+101, :] = th

    # Second part: construct a multivariate normal over theta values,
    # tripling the covariance to ensure it's relatively wide
    cov = -_hess_log_psi_helper(th).inverse() * 3
    prop = torch.distributions.MultivariateNormal(loc=th, covariance_matrix=cov)

    # Draw samples
    theta_samples = prop.sample((n_samples,))

    # Compute log proposal and log psi on each sample
    log_psi_values = torch.tensor([_log_psi_helper(th) for th in theta_samples])
    log_prop_values = prop.log_prob(theta_samples)

    # Compute importance weights
    weights = (log_psi_values - log_prop_values).exp()

    # Return a dict containing samples plus other useful metadata
    return {
        'th_optim': th_optim,
        'proposal_mean': th,
        'proposal_cov': cov,
        'samples': theta_samples,
        'weights': weights,
        'log_psi': log_psi_values,
        'log_prop': log_prop_values,
        'ess': (weights.sum()*weights.sum()) / (weights*weights).sum()
    }