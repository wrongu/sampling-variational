import torch
from math import sqrt


def hmc(log_p, init_x, n_samples, n_burnin=100, n_leapfrog=50, dt=.01, mass=1.):
    mass = mass * torch.ones(init_x.size())
    x_samples = torch.zeros((n_samples+n_burnin,)+init_x.size())
    log_probs = torch.zeros(n_samples + n_burnin)
    u = torch.rand(n_samples + n_burnin).log()
    momenta = torch.randn((n_samples+n_burnin,)+init_x.size())*torch.sqrt(mass)


    def _log_p_helper(x):
        x.requires_grad_(True)
        lp = log_p(x)
        g = torch.autograd.grad(lp, x)[0]
        x.requires_grad_(False)
        return lp.detach(), g

    x_samples[0, ...] = init_x
    log_probs[0], _ = _log_p_helper(init_x)
    accept = 0
    for s in range(1, n_samples + n_burnin):
        p0, x = momenta[s, ...], x_samples[s-1, ...]
        _, grad_log_prob = _log_p_helper(x)
        p = p0 + (dt/2) * grad_log_prob
        for l in range(n_leapfrog):
            x = x + dt * p / mass
            new_log_prob, grad_log_prob = _log_p_helper(x)
            p = p + dt * grad_log_prob
        # Undo extra half-step of momentum from final loop
        p = p - (dt/2) * grad_log_prob

        new_hamiltonian = new_log_prob - torch.sum(p*p/mass)/2
        old_hamiltonian = log_probs[s-1] - torch.sum(p0*p0/mass)/2
        log_metropolis_ratio = new_hamiltonian - old_hamiltonian

        # Accept or reject
        if log_metropolis_ratio > u[s]:
            log_probs[s] = new_log_prob
            x_samples[s, :] = x
            accept += 1 if s >= n_burnin else 0
        else:
            log_probs[s] = log_probs[s-1]
            x_samples[s, :] = x_samples[s-1, :]

    return {
        'samples': x_samples,
        'log_prob': log_probs,
        'accept': accept / n_samples
    }