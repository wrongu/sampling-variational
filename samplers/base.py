import torch


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

    def map(self, init_x, grad_steps=100, newton_steps=100, lr=0.001):
        x = init_x

        # Warm-up with gradient ascent steps
        for _ in range(grad_steps):
            x = x + lr * self._log_p_helper(x, grads=1)[1]
        
        # Rapidly find optimum with Newton's method steps
        for _ in range(grad_steps):
            _, g, h = self._log_p_helper(x, grads=2)
            x = x - torch.linalg.solve(h, g)

        return x

    def sample(self, init_x, n_samples=1000, n_burnin=100):
        raise NotImplementedError("Sampler.sample() implemented by subclasses")