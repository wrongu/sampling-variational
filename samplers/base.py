import torch
from warnings import warn
from util import is_positive_definite


class Sampler(object):
    def __init__(self, log_p):
        self.log_p = log_p

    def _log_p_helper(self, x, grads=0):
        x.requires_grad_(grads>0)
        lp = self.log_p(x)
        if grads > 0:
            g = torch.autograd.grad(lp, x, create_graph=grads>1)[0]
            if grads > 1:
                h = x.new_zeros((len(x), len(x)))
                for i in range(len(x)):
                    h[i, :] = torch.autograd.grad(g[i], x, retain_graph=True)[0]
        x.requires_grad_(False)
        if grads == 2:
            return (lp.detach(), g.detach(), h)
        elif grads == 1:
            return (lp.detach(), g.detach())
        elif grads == 0:
            return lp.detach()

    def map(self, init_x, steps=500):
        x = init_x.detach()
        newton, lr = False, 0.01

        # Warm-up with gradient ascent steps until the hessian is well-behaved, then switch to
        # Newton's method updates
        for i in range(steps):
            logp, g, h = self._log_p_helper(x, grads=2)

            # Upon finding a convex region, switch to newton and reset LR
            if not newton and is_positive_definite(-h):
                newton, lr = True, 1.0
            elif newton and not is_positive_definite(-h):
                warn("Encountered non-convex region after switching to newton!")
                newton, lr = False, 0.01

            with torch.no_grad():
                # Get dx direction for either newton or gradient update
                dx = -torch.linalg.solve(h, g) if newton else +g

                # Search along [x, x+dx] line, halving the learning rate any time we overshoot
                while self._log_p_helper(x + lr * dx) < logp:
                    lr = lr / 2
                x = x + lr * dx
        return x

    def sample(self, init_x, n_samples=1000, n_burnin=100):
        raise NotImplementedError("Sampler.sample() implemented by subclasses")