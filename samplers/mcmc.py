import torch
import numpy as np
from samplers.base import Sampler


class DualAveragingStepSize(object):
    """This class copied from https://colindcarroll.com/2019/04/21/step-size-adaptation-in-hamiltonian-monte-carlo/
    """
    def __init__(self, initial_step_size, target_accept=0.65, gamma=0.05, t0=10.0, kappa=0.75):
        self.mu = np.log(10 * initial_step_size)  # proposals are biased upwards to stay away from 0.
        self.target_accept = target_accept
        self.gamma = gamma
        self.t = t0
        self.kappa = kappa
        self.error_sum = 0
        self.log_averaged_step = 0

    def update(self, p_accept, average=False):
        # Running tally of absolute error. Can be positive or negative. Want to be 0.
        self.error_sum += self.target_accept - p_accept

        # This is the next proposed (log) step size. Note it is biased towards mu.
        log_step = self.mu - self.error_sum / (np.sqrt(self.t) * self.gamma)

        # Forgetting rate. As `t` gets bigger, `eta` gets smaller.
        eta = self.t ** -self.kappa

        # Smoothed average step size
        self.log_averaged_step = eta * log_step + (1 - eta) * self.log_averaged_step

        # This is a stateful update, so t keeps updating
        self.t += 1

        # Return either the noisy step size or the smoothed step size depending on 'average' flag
        if average:
            return np.exp(self.log_averaged_step)
        else:
            return np.exp(log_step)


class HMC(Sampler):
    def __init__(self, log_p, **kwargs):
        super().__init__(log_p)
        self.tuned = False
        # Populate instance variables from kwargs with defaults
        self.dt = kwargs.get('dt', 0.01)
        self.leapfrog_t = kwargs.get('leapfrog_t', 0.5)
        self.mass = kwargs.get('mass', torch.ones(()))

    def leapfrog_propose(self, x0, mass, dt):
        lp0, g = self._log_p_helper(x0, grads=1)
        # Sample initial momentum
        p0 = torch.randn(len(x0)) * torch.sqrt(mass)
        # Start with p on the half-time schedule for leapfrogging
        x, p = x0, p0 + (dt/2) * g
        for l in range(int(np.ceil(self.leapfrog_t / dt))):
            x = x + dt * p / mass
            lp, g = self._log_p_helper(x, grads=1)
            p = p + dt * g
        # Undo extra half-step of momentum from final loop
        p = p - (dt/2) * g
        # Compute hamiltonian at start and end points
        ham_old = lp0 - torch.sum(p0*p0/mass)/2
        ham_new = lp - torch.sum(p*p/mass)/2
        # Return proposed x along with the log metropolis ratio and new log prob
        return x, ham_new - ham_old, lp

    def tune(self, init_x, n_leapfrog_tries=200, target_accept=0.8, tune_mass=True):
        # Start with a Laplace approximation
        map_x = self.map(init_x.flatten())
        _, _, hessian = self._log_p_helper(map_x, grads=2)
        u, s, _ = torch.svd(-hessian)
        # r is lower factor of covariance, i.e. cov = r.T@r
        r = u @ torch.diag(1/s.sqrt())

        # Draw starting points for leapfrog trajectories from the Laplace Gaussian
        leapfrog_start_x = map_x.view(1, -1) + torch.randn(n_leapfrog_tries, init_x.numel()) @ r.T

        # Set mass based on average curvature across sampled points
        avg_hess = sum(self._log_p_helper(start_x, grads=2)[2] for start_x in leapfrog_start_x) / n_leapfrog_tries
        mass = -avg_hess.diag()

        # Use dual-averaging method with a desired target acceptance rate to select dt
        dt, dual_avg = self.dt, DualAveragingStepSize(self.dt, target_accept=target_accept)
        for start_x in leapfrog_start_x:
            _, log_mh_ratio, _ = self.leapfrog_propose(start_x, mass, dt)
            # print("DEBUG TUNING", "dt", dt, "p(accept)", log_mh_ratio.exp().item(), "exp(log_avg_dt)", np.exp(dual_avg.log_averaged_step))
            dt = dual_avg.update(log_mh_ratio.exp().clip(max=1.).item(), average=False)
        dt = dual_avg.update(log_mh_ratio.exp().clip(max=1.).item(), average=True)

        # Save results as instance variables
        self.mass = mass
        self.dt = dt
        self.tuned = True

        return mass, dt

    def sample(self, init_x, n_samples=1000, n_burnin=100):
        if not self.tuned:
            raise RuntimeWarning("Running HMC.sample() without calling HMC.tune() first!")

        accept = torch.ones(n_samples + n_burnin)
        
        samples = torch.zeros(n_samples + n_burnin, init_x.numel())
        samples[0, :] = init_x.flatten()

        log_prob = torch.zeros(n_samples + n_burnin)
        log_prob[0] = self._log_p_helper(init_x)

        u = torch.rand(n_samples + n_burnin).log()

        for t in range(1, n_samples + n_burnin):
            prop_x, log_mh_ratio, prop_log_prob = self.leapfrog_propose(samples[t-1, :], self.mass, self.dt)

            if log_mh_ratio > u[t]:
                accept[t] = 1.
                samples[t, :] = prop_x
                log_prob[t] = prop_log_prob
            else:
                accept[t] = 0.
                samples[t, :] = samples[t-1, :]
                log_prob[t] = log_prob[t-1]

        return {'samples': samples[n_burnin:, :],
                'log_prob': log_prob[n_burnin:],
                'accept': accept[n_burnin:].mean(),
                'burn_samples': samples[:n_burnin, :],
                'burn_log_prob': log_prob[:n_burnin],
                'burn_accept': accept[:n_burnin].mean()}
