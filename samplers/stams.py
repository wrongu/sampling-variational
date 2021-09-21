import torch
from math import sqrt


def log_psi_stams(log_p, lam_kl, q, n_kl_quad=10, include_grad=False, include_hess=False):
    if include_hess and not include_grad:
        raise ValueError("Set include_grad=True in order to get the Hessian")
    
    q.theta.requires_grad_(include_grad)
    kl_q_p = -q.entropy() - q.quadrature_ev(log_p, n_kl_quad)
    log_psi = 0.5*q.log_det_fisher() - lam_kl * kl_q_p
    if include_grad:
        g = torch.autograd.grad(log_psi, q.theta, create_graph=include_hess)[0]
        if include_hess:
            h = q.theta.new_zeros(q.n_params, q.n_params)
            for i in range(q.n_params):
                h[i, :] = torch.autograd.grad(g[i], q.theta, retain_graph=True)[0]
    q.theta.requires_grad_(False)
    if include_hess:
        return log_psi.detach(), g.detach(), h
    elif include_grad:
        return log_psi.detach(), g.detach()
    else:
        return log_psi.detach()


def stams_mvn_langevin(log_p, lam_kl, q_init, n_samples=1000, burn_in=100, n_kl_quad=10, dt=.01):
    """Run Metropolis-adjusted Langevin dynamics over q parameters (theta), using bound on MI inspired
    by Stam's inequality.

    That is, log_psi = 1/2 log(det(FIM(theta))) - lambda_kl * KL(q(x|theta)||p(x)) + constants, where FIM
    is the Fisher Information Matrix.

    KL(q||p) is evaluated using Gauss-Hermite quadrature
    """
    # Make a local copy of q_init that we will manipulate
    q = q_init.clone()

    samples = torch.zeros(n_samples + burn_in, q.n_params)
    samples[0, :] = q.theta

    log_psi = torch.zeros(n_samples + burn_in)
    log_psi[0], grad_log_psi = log_psi_stams(log_p, lam_kl, q, include_grad=True)

    # Pre-sample accept/reject thresholds for Metropolis adjustments
    u = torch.rand(n_samples + burn_in).log()
    accept = torch.ones(n_samples + burn_in)

    for t in range(1, n_samples + burn_in):
        # Proposed Langevin step
        theta, step_theta = samples[t-1, :], dt * grad_log_psi + sqrt(2*dt)*torch.randn(q.n_params)

        # Evaluate the new point
        new_log_psi, new_grad_log_psi = log_psi_stams(log_p, lam_kl, q.set_theta(theta + step_theta), include_grad=True)

        # Compute (log) Metropolis ratio, log[p(x')q(x|x')/p(x)q(x'|x)]
        log_q_forward = -1 / (4*dt) * torch.sum((step_theta - dt*grad_log_psi)**2)
        log_q_reverse = -1 / (4*dt) * torch.sum((step_theta + dt*new_grad_log_psi)**2)
        log_metropolis_ratio = new_log_psi - log_psi[t-1] + log_q_reverse - log_q_forward

        # Accept or reject
        if log_metropolis_ratio > u[t]:
            accept[t] = 1.
            log_psi[t] = new_log_psi
            grad_log_psi = new_grad_log_psi
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

    KL(q||p) is evaluated using Gauss-Hermite quadrature
    """


    def _mass_helper(q, lower_bound=0.001, upper_bound=1000):
        return torch.clip(1/q.fisher().diag(), lower_bound, upper_bound)


    # Make a local copy of q_init that we will manipulate
    q = q_init.clone()

    samples = torch.zeros(n_samples + burn_in, q.n_params)
    samples[0, :] = q.theta

    masses = torch.zeros(n_samples + burn_in, q.n_params)
    masses[0, :] = _mass_helper(q.theta)

    log_psi = torch.zeros(n_samples + burn_in)
    log_psi[0] = _log_psi_helper(samples[0, :])

    # Pre-sample accept/reject thresholds for Metropolis adjustments
    u = torch.rand(n_samples + burn_in).log()
    accept = torch.ones(n_samples + burn_in)

    # Pre-sample momentum values (not yet adjusted by mass)
    momentum_z = torch.randn(n_samples + burn_in, q.n_params)

    for t in range(1, n_samples + burn_in):
        th = samples[t-1, :]
        # Pick mass for this trajectory based on current theta but keep it constant across
        # leapfrog steps so we don't have to do fixed-point leapfrog steps
        mass = _mass_helper(q.set_theta(th))
        masses[t, :] = mass
        # Run leapfrog dynamics
        p0 = momentum_z[t, :] * torch.sqrt(mass)
        _, g = log_psi_stams(log_p, lam_kl, q.set_theta(th), include_grad=True)
        # First half-step.. this places p on the 'half time' schedule
        p = p0 + (dt/2) * g
        for l in range(n_leapfrog):
            th = th + dt * p / mass
            new_log_psi, g = log_psi_stams(log_p, lam_kl, q.set_theta(th), include_grad=True)
            p = p + dt * g
        # Undo extra half-step of momentum from final loop
        p = p - (dt/2) * g

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


def stams_importance_sampling(log_p, lam_kl, q_init, n_samples, n_kl_quad=10):
    """Generate a set of samples of theta~Q and importance-sampling weights, psi/Q

    Strategy is based on a Laplace approximation: we navigate to the MAP value of theta, then halve the curvature there
    to get a gaussian proposal distribution in theta-space.
    """
    # Make a local copy of q_init that we will manipulate
    q = q_init.clone()

    # First part: optimize theta towards mode of psi
    th = q.theta
    th_optim = torch.zeros(201, q.n_params)
    th_optim[0, :] = th
    # Warm-up with 100 gradient-ascent steps
    for t in range(100):
        # Get gradient
        _, g = log_psi_stams(log_p, lam_kl, q.set_theta(th), include_grad=True)
        lr =  .1 / (1 + t // 5) / torch.sum(g[:,None] * g[None,:] * q.fisher()).sqrt()
        # Take a gradient ascent step
        th = th + g * lr
        # Record optimization trajectory
        th_optim[t+1, :] = th

    # Rapidly find the max using newton's method for another 100 steps
    for t in range(100):
        # Get gradient and hessian
        _, g, h = log_psi_stams(log_p, lam_kl, q.set_theta(th), include_grad=True, include_hess=True)
        th = th - torch.linalg.solve(h, g)
        # Record optimization trajectory
        th_optim[t+101, :] = th

    # Second part: construct a multivariate normal over theta values,
    # tripling the covariance to ensure it's relatively wide
    cov = -h.inverse() * 3
    prop = torch.distributions.MultivariateNormal(loc=th, covariance_matrix=cov)

    # Draw samples
    theta_samples = prop.sample((n_samples,))

    # Compute log proposal and log psi on each sample
    log_psi_values = torch.tensor([log_psi_stams(log_p, lam_kl, q.set_theta(th)) for th in theta_samples])
    log_prop_values = prop.log_prob(theta_samples)

    # Compute normalized importance weights (since we only know log_psi up to additive constants)
    log_weights = (log_psi_values - log_prop_values).flatten()
    weights = (log_weights - torch.logsumexp(log_weights, dim=0)).exp()

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