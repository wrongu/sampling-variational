import torch
from math import sqrt, log, pi
from torch.distributions import MultivariateNormal


LOG2PI = log(2*pi)


class MVN(object):
    """Base class for custom multivariate normals (MVNs), subclassed below by various specializations on covariance.

    In all cases, the primary 'parameters' are stored in self.theta.
    """
    def __init__(self, loc=None, d=None):
        if d is None and loc is None:
            raise ValueError("Must specificy one of loc or dimension d!")
        self.d = d or len(loc)

    def to(self, device: torch.device) -> None:
        self.theta = self.theta.to(device)

    def requires_grad_(self, flag) -> None:
        self.theta.requires_grad_(flag)

    def clone(self):
        new_mvn = self.__class__(d=self.d)
        new_mvn.theta = self.theta.clone()
        return new_mvn

    def loc(self) -> torch.Tensor:
        raise NotImplementedError("To be implemented by a subclass")

    def scale_tril(self) -> torch.Tensor:
        raise NotImplementedError("To be implemented by a subclass")

    def covariance(self) -> torch.Tensor:
        raise NotImplementedError("To be implemented by a subclass")

    def precision(self) -> torch.Tensor:
        raise NotImplementedError("To be implemented by a subclass")

    def log_det_cov(self) -> torch.Tensor:
        raise NotImplementedError("To be implemented by a subclass")

    def fisher(self) -> torch.Tensor:
        raise NotImplementedError("To be implemented by a subclass")

    def log_det_fisher(self) -> torch.Tensor:
        raise NotImplementedError("To be implemented by a subclass")

    def grad_log_det_fisher(self) -> torch.Tensor:
        raise NotImplementedError("To be implemented by a subclass")

    def to_torch_mvn(self) -> MultivariateNormal:
        """Get a torch.distributions.MultivariateNormal instance, which can be used for sampling, log prob, etc"""
        scale_tril = self.scale_tril()
        # Built-in MVN will complain if the diagonal has negatives
        scale_tril = scale_tril @ (scale_tril.detach().diag().sign().diag())
        return MultivariateNormal(loc=self.loc(), scale_tril=scale_tril)

    def entropy(self):
        """Shannon differential entropy of a multivariate normal
        """
        return (self.d*LOG2PI + self.d + self.log_det_cov())/2

    def kl(self, other):
        """Compute KL divergence from this MVN to another MVN
        """
        if self.d != other.d:
            raise ValueError("MVNs must have same d!")
        delta_mu = self.loc() - other.loc()
        cov1 = self.covariance()
        prec2 = other.precision()
        # Compute quadratic term: delta_mu.T * cov2.inverse() * delta_mu
        mC2im = delta_mu @ prec2 @ delta_mu
        # Trace of A.T @ B is simply sum(A*B)
        trC1C2i = torch.sum(cov1 * prec2)
        # Compute difference in logdets
        delta_logdet = other.log_det_cov() - self.log_det_cov()
        return 0.5*(mC2im + trC1C2i - delta_mu.size()[-1] + delta_logdet)

    def taylor_ev(self, fn):
        """Estimate the expected value (ev) of a given function using autograd to estimate curvature of the function.

        This is equivalent to truncating the taylor expansion of fn at mu to two terms and computing the expected value
        of the resulting quadratic.
        """
        mu, cov = self.loc(), self.covariance()
        mu.requires_grad_(True)
        ev = fn(mu)
        # First order gradient
        g = torch.autograd.grad(ev, mu, create_graph=True)[0]
        # Second order hessian
        H = torch.zeros(mu.size()*2)
        for i in range(mu.size()[0]):
            H[i, :] = torch.autograd.grad(g[i], mu, retain_graph=True)[0]
        # Add in 2nd order term (using trace(A.T @ B) == sum(A*B) trick)
        ev += 0.5*torch.sum(cov*H)
        return ev

    def monte_carlo_ev(self, fn, n_samples=1000, include_mcse=False):
        """Estimate the expected value (ev) of a given function using monte carlo samples from this normal.

        If include_mcse is set to True, there is a second return value: the estimated monte carlo standard error

        fn() must handle vectorization by accepting input of size (d, n_samples)
        """
        samples = self.loc()[:, None] + self.scale_tril() @ torch.randn(self.d, n_samples, device=self.theta.device)
        values = fn(samples)
        if include_mcse:
            return values.mean(), values.std() / sqrt(n_samples)
        else:
            return values.mean()

    def ellipse(self, nsigma=1.):
        """Get x, y coordinates defining the ellipse at nsigma standard deviations.

        Only works in 2D, i.e. requires d==2
        """
        assert self.d == 2, "ellipse() requires d==2"
        t = torch.linspace(0, 2*pi, 100)
        with torch.no_grad():
            mu, scale_tril = self.loc(), self.scale_tril()
            xy = mu.view(2, 1) + nsigma * scale_tril @ torch.stack([torch.cos(t), torch.sin(t)], dim=0)
        return xy[0], xy[1]


class MVNFull(MVN):
    """A custom Multivariate Normal (MVN) class with (loc, scale_tril) parameterization. Total
    number of parameters is d for the location plus d(d+1)/2 for the lower-triangular Cholesky
    factorization of the covariance.

    We allow scale_tril to have negative standard deviations, since L@L' will be positive definite
    regardless
    """

    def __init__(self, loc=None, scale_tril=None, d=None, theta=None):
        if theta is not None:
            # If x has dimension d, theta has dimension d+d(d+1)/2. The following expression inverts
            # this to get d from a given theta.
            super().__init__(d=int(sqrt(9+8*len(theta))-3)//2)
        else:
            super().__init__(loc, d)
        loc = loc if loc is not None else torch.zeros(self.d)
        scale_tril = scale_tril if scale_tril is not None else torch.eye(self.d)
        self._chol_ij = torch.tril_indices(self.d, self.d)
        self.theta = torch.cat([loc, scale_tril[self._chol_ij[0], self._chol_ij[1]]])
        self.n_params = self.d + self.d*(self.d+1)//2
        # If theta is given, it overrides all of the above default values
        if theta is not None:
            self.theta[...] = theta
        if len(self.theta) != self.n_params:
            raise ValueError(f"Dimension problem! Expected |theta| to be {self.n_params} but was {len(self.theta)}")

    def loc(self) -> torch.Tensor:
        """Get the location (mean) part of parameters"""
        return self.theta[:self.d]

    def scale_tril(self) -> torch.Tensor:
        """Get the Cholesky(covariance) lower-triangular matrix part of parameters"""
        out = self.theta.new_zeros(self.d, self.d)
        out[self._chol_ij[0], self._chol_ij[1]] = self.theta[self.d:]
        return out

    def covariance(self) -> torch.Tensor:
        scale_tril = self.scale_tril()
        return scale_tril @ scale_tril.T

    def precision(self) -> torch.Tensor:
        return torch.cholesky_inverse(self.scale_tril())

    def log_det_cov(self):
        return 2*torch.sum(self.scale_tril().diag().abs().log())

    def fisher(self) -> torch.Tensor:
        """Get the fisher information matrix (FIM) with respect to self.theta parameters"""
        scale_tril = self.scale_tril()
        precision = torch.cholesky_inverse(scale_tril)
        iL = scale_tril.inverse()  # TODO - can we avoid a call to inverse?
        fim_tril_part = torch.zeros(self.n_params - self.d, self.n_params - self.d)
        for m, (i_m, j_m) in enumerate(zip(*self._chol_ij)):
            for n, (i_n, j_n) in enumerate(zip(*self._chol_ij)):
                iLmiLn = iL[:, i_m] * iL[:, i_n] if j_m == j_n else torch.zeros(())
                iLmTiLn = iL[j_n, i_m] * iL[j_m, i_n] if j_n <= i_m and j_m <= i_n else torch.zeros(())
                fim_tril_part[m, n] = iLmTiLn + iLmiLn.sum()
        return torch.block_diag(precision, fim_tril_part)

    def log_det_fisher(self) -> torch.Tensor:
        # TODO - analytic version
        return torch.logdet(self.fisher())

    def grad_log_det_fisher(self) -> torch.Tensor:
        # TODO - analytic version
        tmp = self.theta.requires_grad
        self.theta.requires_grad_(True)
        log_det_fim = self.log_det_fisher()
        grad_log_det_fim = torch.autograd.grad(log_det_fim, self.theta)[0]
        self.theta.requires_grad_(tmp)
        return grad_log_det_fim

    @staticmethod
    def new_random(d, device=torch.device('cpu')):
        return MVNFull(loc=torch.randn(d, device=device),
                       scale_tril=torch.tril(torch.randn(d, d, device=device)))


class MVNDiag(MVN):
    """A custom Multivariate Normal (MVN) class with theta=(loc, sqrt(diag(cov))) parameterization.
    Total number of parameters is 2d: d for the location and d standard deviations.
    """

    def __init__(self, loc=None, scale=None, d=None, theta=None):
        if theta is not None:
            super().__init__(d=len(theta)//2)
        else:
            super().__init__(loc, d)
        loc = loc if loc is not None else torch.zeros(self.d)
        scale = scale if scale is not None else torch.ones(self.d)
        self.theta = torch.cat([loc, torch.log(torch.abs(scale))])
        self.n_params = 2 * self.d
        # If theta is given, it overrides all of the above default values
        if theta is not None:
            self.theta[...] = theta
        if len(self.theta) != self.n_params:
            raise ValueError(f"Dimension problem! Expected |theta| to be {self.n_params} but was {len(self.theta)}")

    def loc(self) -> torch.Tensor:
        """Get the location (mean) part of parameters"""
        return self.theta[:self.d]

    def scale_tril(self) -> torch.Tensor:
        """Get the Cholesky(covariance) lower-triangular matrix part of parameters"""
        return torch.diag(torch.exp(self.theta[self.d:]))

    def covariance(self) -> torch.Tensor:
        return torch.diag(torch.exp(self.theta[self.d:]*2))

    def precision(self) -> torch.Tensor:
        return torch.diag(torch.exp(self.theta[self.d:]*-2))

    def log_det_cov(self):
        return 2*torch.sum(self.theta[self.d:])

    def fisher(self) -> torch.Tensor:
        """Get the fisher information matrix (FIM) with respect to self.theta parameters"""
        fim_mu_part = self.precision()
        fim_scale_part = torch.diag(2 * self.theta.new_ones(self.d))
        return torch.block_diag(fim_mu_part, fim_scale_part)

    def log_det_fisher(self) -> torch.Tensor:
        return -2 * torch.sum(self.theta[self.d:]) + self.d * log(2)

    def grad_log_det_fisher(self) -> torch.Tensor:
        out = self.theta.new_zeros(self.n_params)
        out[self.d:] = -2.
        return out

    @staticmethod
    def new_random(d, device=torch.device('cpu')):
        return MVNDiag(loc=torch.randn(d, device=device),
                         scale=torch.randn(d, device=device).abs())


class MVNIso(MVN):
    """A custom Multivariate Normal (MVN) class with theta=(loc, log(sigma)) parameterization. Total
    number of parameters is d+1
    """

    def __init__(self, loc=None, scale=None, d=None, theta=None):
        if theta is not None:
            super().__init__(d=len(theta)-1)
        else:
            super().__init__(loc, d)
        loc = loc if loc is not None else torch.zeros(self.d)
        scale = scale if scale is not None else torch.ones(1)
        self.theta = torch.cat([loc, torch.log(torch.abs(scale))])
        self.n_params = self.d + 1
        # If theta is given, it overrides all of the above default values
        if theta is not None:
            self.theta[...] = theta
        if len(self.theta) != self.n_params:
            raise ValueError(f"Dimension problem! Expected |theta| to be {self.n_params} but was {len(self.theta)}")

    def loc(self) -> torch.Tensor:
        """Get the location (mean) part of parameters"""
        return self.theta[:self.d]

    def scale_tril(self) -> torch.Tensor:
        """Get the Cholesky(covariance) lower-triangular matrix part of parameters"""
        return torch.eye(self.d, self.d, device=self.theta.device) * torch.exp(self.theta[-1])

    def covariance(self) -> torch.Tensor:
        return torch.eye(self.d, self.d, device=self.theta.device) * torch.exp(self.theta[-1]*2)

    def precision(self) -> torch.Tensor:
        return torch.eye(self.d, self.d, device=self.theta.device) * torch.exp(self.theta[-1]*-2)

    def log_det_cov(self):
        return 2 * self.d * self.theta[-1]

    def fisher(self) -> torch.Tensor:
        """Get the fisher information matrix (FIM) with respect to self.theta parameters"""
        fim_mu_part = self.precision()
        fim_scale_part = torch.ones(1) * self.d * 2
        return torch.block_diag(fim_mu_part, fim_scale_part)

    def log_det_fisher(self) -> torch.Tensor:
        return -2 * self.d * self.theta[-1] + log(self.d * 2)

    def grad_log_det_fisher(self) -> torch.Tensor:
        out = self.theta.new_zeros(self.n_params)
        out[-1] = -2 * self.d
        return out

    @staticmethod
    def new_random(d, device=torch.device('cpu')):
        return MVNIso(loc=torch.randn(d, device=device),
                      scale=torch.randn(1, device=device).abs())


def create_mog(mvns):
    """Given an iterable set of MVN instances, construct a torch.distributions MixtureSameFamily
    instance
    """
    all_means, all_trils = zip(*[(mvn.loc().clone(), mvn.scale_tril().clone()) for mvn in mvns])
    weights = torch.distributions.Categorical(torch.ones(len(all_means)))
    return torch.distributions.MixtureSameFamily(weights, MultivariateNormal(loc=torch.stack(all_means),
                                                                             scale_tril=torch.stack(all_trils)))


def _run_cov_test(cls, dim):
    mvn = cls.new_random(dim)
    scale_tril = mvn.scale_tril()
    cov = mvn.covariance()
    prec = mvn.precision()

    ij_upper = torch.triu_indices(dim, dim, offset=1)

    print(f"=== Cov/precision test for {cls.__name__}[{dim}] ===")
    assert torch.all(scale_tril[ij_upper[0], ij_upper[1]] == 0.)
    assert torch.all(scale_tril.diag() != 0.)
    assert torch.allclose(cov, scale_tril @ scale_tril.T)
    assert torch.allclose(cov, prec.inverse())
    assert torch.allclose(torch.logdet(cov), mvn.log_det_cov())

    th_mvn = mvn.to_torch_mvn()
    assert torch.allclose(cov, th_mvn.covariance_matrix)


def _run_ev_test(cls):
    def quad_f(x):
        x, y = x[0], x[1]
        return x**2 +3*x*y - y**2/2

    def quart_f(x):
        x, y = x[0], x[1]
        return -(y-x**2)**2-x**2

    def test_ev(mvn, f):
        mc_ev = mvn.monte_carlo_ev(f, 10000)
        taylor_ev = mvn.taylor_ev(f)
        return mc_ev, taylor_ev

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)

    runs = 100
    mc_vals = torch.zeros(runs)
    tay_vals = torch.zeros(runs)
    for i in range(runs):
        mc_vals[i], tay_vals[i] = test_ev(cls.new_random(2), quad_f)

    plt.scatter(mc_vals, tay_vals.detach(), marker='.')
    plt.axis('equal')
    plt.grid()

    plt.subplot(1, 2, 2)
    mc_vals = torch.zeros(runs)
    tay_vals = torch.zeros(runs)
    for i in range(runs):
        mc_vals[i], tay_vals[i] = test_ev(cls.new_random(2), quart_f)

    plt.scatter(mc_vals, tay_vals.detach(), marker='.')
    plt.axis('equal')
    plt.grid()
    plt.show()


def _init_from_theta_test(cls, dim):
    q1 = cls.new_random(d=dim)
    q2 = cls(theta=q1.theta)
    assert torch.allclose(q1.loc(), q2.loc())
    assert torch.allclose(q1.scale_tril(), q2.scale_tril())
    # Override q1.theta -- should not affect q2, which should have separate memory
    q1.theta[0] = 0.
    assert q2.theta[0] != 0.


def _run_fisher_test(cls, dim):
    mvn = cls.new_random(dim)
    fim = mvn.fisher()

    assert fim.size() == (mvn.n_params, mvn.n_params)

    mvn2 = mvn.clone()
    mvn.requires_grad_(True)
    kl = mvn.kl(mvn2)
    fim_num = torch.zeros(fim.size())
    g = torch.autograd.grad(kl, mvn.theta, create_graph=True)[0]
    for i in range(mvn.n_params):
        fim_num[i, :] = torch.autograd.grad(g[i], mvn.theta, retain_graph=True)[0]

    print(f"=== FIM test for {cls.__name__}[{dim}] ===: ", torch.allclose(fim, fim_num))
    if not torch.allclose(fim, fim_num):
        abs_err = (fim - fim_num).abs()
        rel_err = (fim / fim_num)
        rel_err = rel_err[~torch.isnan(rel_err)]
        print(f"\taerr={abs_err.max().item()}\trerr={rel_err.min().item(), rel_err.max().item()}")


def _run_logdet_fisher_test(cls, dim):
    mvn = cls.new_random(dim)
    mvn.requires_grad_(True)
    fim = mvn.fisher()
    th_logdet_fim = torch.logdet(fim)
    th_grad_logdet_fim = torch.autograd.grad(th_logdet_fim, mvn.theta)[0]
    mvn.requires_grad_(False)

    ana_logdet_fim = mvn.log_det_fisher()
    ana_grad_logdet_fim = mvn.grad_log_det_fisher()

    print(f"=== logdet(FIM) test for {cls.__name__}[{dim}] === ")
    assert torch.allclose(th_logdet_fim.detach(), ana_logdet_fim)
    assert torch.allclose(th_grad_logdet_fim, ana_grad_logdet_fim)


def _run_kl_test(cls, dim, n=100000):
    mvn1, mvn2 = cls.new_random(dim), cls.new_random(dim)
    kl = mvn1.kl(mvn2)
    th_mvn1, th_mvn2 = mvn1.to_torch_mvn(), mvn2.to_torch_mvn()
    x1 = th_mvn1.sample((n,))
    diff_lp = th_mvn1.log_prob(x1) - th_mvn2.log_prob(x1)
    mc_kl = diff_lp.mean()
    mc_kl_mcse = diff_lp.std() / sqrt(n)

    n_sigma = (kl - mc_kl) / mc_kl_mcse
    print(f"=== KL test for {cls.__name__}[{dim}] ===: err={kl - mc_kl}\tsigma={n_sigma}")


def _run_entropy_test(cls, dim, n=100000):
    mvn = cls.new_random(dim)
    h = mvn.entropy()
    th_mvn = mvn.to_torch_mvn()
    th_h = th_mvn.entropy()
    x = th_mvn.sample((n,))
    lp = th_mvn.log_prob(x)
    mc_h = (-lp).mean()
    mc_h_mcse = lp.std() / sqrt(n)

    n_sigma = (h - mc_h) / mc_h_mcse
    print(f"=== Entropy test for {cls.__name__}[{dim}] ===: {h-th_h}\terr={h - mc_h}\tsigma={n_sigma}")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    CLASSES = [MVNIso, MVNDiag, MVNFull]
    for cls in CLASSES:
        for d in [2, 3, 4]:
            _run_cov_test(cls, dim=d)
            _run_fisher_test(cls, dim=d)
            _run_logdet_fisher_test(cls, dim=d)
            _run_kl_test(cls, dim=d)
            _run_entropy_test(cls, dim=d)
            _init_from_theta_test(cls, dim=d)

        # _run_ev_test(cls)