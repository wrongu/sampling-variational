import torch
import numpy as np
from util import is_positive_definite
from samplers.base import Sampler
from tqdm import tqdm


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

    def tune(self, init_xs, map_steps=50, target_accept=0.8, tune_mass=True, min_mass=1e-3, verbose=False):
        n_init, dim_x = init_xs.size()

        # Take each init_xs and nudge it up the gradient of log prob a few steps. Inside self.map,
        # the step size (lr) is updated if we overshoot a lot. The final value of the lr is returned
        # and re-used (after inflating by 10x) for the next iteration
        map_kwargs = {'return_lr': True}
        for i in range(n_init):
            tmp = init_xs[i, :].clone()
            init_xs[i, :], last_lr = self.map(tmp, steps=map_steps, **map_kwargs)
            map_kwargs['grad_lr'] = min(0.01, 10*last_lr[0])
            map_kwargs['newt_lr'] = min(1.0, 10*last_lr[1])
        init_xs = init_xs[~torch.any(torch.isnan(init_xs), dim=1), :]

        # Set mass based on average curvature across local maxima after iterating
        # a few steps from the sampled points. Pass result through softplus to ensure
        # no negative mass (convex dimensions are given small but nonnegative mass).
        # Include a single 'eye' matrix in the average to regularize, and exclude any nan
        # hessians
        hessians = torch.stack([torch.eye(dim_x)] + [self._log_p_helper(start_x, grads=2)[2] for start_x in init_xs], dim=0)
        avg_hess = hessians.nansum(dim=0) / torch.sum(~torch.isnan(hessians), dim=0);
        mass = min_mass + torch.nn.functional.softplus(-avg_hess.diag())

        # print("AVG HESS", avg_hess, "MASS", mass)

        # Use dual-averaging method with a desired target acceptance rate to select dt
        dt, dual_avg = self.dt, DualAveragingStepSize(self.dt, gamma=0.5, target_accept=target_accept)
        for start_x in init_xs:
            _, log_mh_ratio, _ = self.leapfrog_propose(start_x, mass, dt)
            # print(start_x.numpy(), dt, log_mh_ratio.item(), log_mh_ratio.exp().clip(max=1.).item())
            dt = dual_avg.update(log_mh_ratio.exp().clip(max=1.).item(), average=False)
        dt = dual_avg.update(log_mh_ratio.exp().clip(max=1.).item(), average=True)

        # Save results as instance variables
        self.mass = mass
        self.dt = dt
        self.tuned = True

        return mass, dt

    def sample(self, init_x, warmup_steps=100, n_samples=1000, n_burnin=100, progbar=False):
        if not self.tuned:
            raise RuntimeWarning("Running HMC.sample() without calling HMC.tune() first!")

        accept = torch.ones(n_samples + n_burnin)

        # 'Warmup' by taking a moderate number of steps towards the MAP point of the distribution.
        # This makes it less likely that initial points are poorly behaved and immediately rejected
        init_x = self.map(init_x.clone(), warmup_steps)
        
        samples = torch.zeros(n_samples + n_burnin, init_x.numel())
        samples[0, :] = init_x.flatten()

        log_prob = torch.zeros(n_samples + n_burnin)
        log_prob[0] = self._log_p_helper(init_x)

        u = torch.rand(n_samples + n_burnin).log()

        trange = range(1, n_samples + n_burnin)
        if progbar:
            trange = tqdm(trange, total=n_samples+n_burnin, leave=False, desc='HMC.sample [00.0]')

        for t in trange:
            prop_x, log_mh_ratio, prop_log_prob = self.leapfrog_propose(samples[t-1, :], self.mass, self.dt)

            if log_mh_ratio > u[t]:
                accept[t] = 1.
                samples[t, :] = prop_x
                log_prob[t] = prop_log_prob
            else:
                accept[t] = 0.
                samples[t, :] = samples[t-1, :]
                log_prob[t] = log_prob[t-1]

            if progbar:
                acc = accept[:t].mean().item() if t < n_burnin else accept[n_burnin:t].mean().item()
                trange.set_description(f'HMC.sample [{100*acc:.1f}]')

        return {'samples': samples[n_burnin:, :],
                'log_p': log_prob[n_burnin:],
                'accept': accept[n_burnin:].mean(),
                'burn_samples': samples[:n_burnin, :],
                'burn_log_p': log_prob[:n_burnin],
                'burn_accept': accept[:n_burnin].mean(),
                'mass': self.mass,
                'dt': self.dt,
                'leapfrog_t': self.leapfrog_t}
