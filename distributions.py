import torch


def log_prob_banana(xy):
    x, y = xy[0], xy[1]
    # Easier to read version: -(y-(x/2)^2)^2 - (x/2)^2
    return -x*x*x*x/8 + y*x*x - 2*y*y - x*x/4


def log_prob_cigar(xy, c=.99):
    x, y = xy[0], xy[1]
    # Compute precision terms for covariance matrix [[1,c],[c,1]], where precision will be [[a,b],[b,a]]
    a = 1./(1. - c*c)
    b = -c/(1. - c*c)
    return -0.5*(a*x*x + a*y*y + 2*b*x*y)


def log_prob_mix_laplace(x, means=(-1.5, 1.5), weights=(0.4, 0.6), scales=(0.75, 0.75)):
    means, weights, scales = torch.tensor(means), torch.tensor(weights), torch.tensor(scales)
    mode_log_probs = -torch.abs((x.view(-1, 1) - means.view(1, -1)) / scales.view(1, -1)) - torch.log(2 * scales)
    log_probs = torch.logsumexp(torch.log(weights.view(1, -1)) + mode_log_probs, dim=1)
    return log_probs.reshape(x.size())


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = torch.linspace(-4, 4, 100)
    xx, yy = torch.meshgrid(x, x)
    xy = torch.stack([xx.flatten(), yy.flatten()], dim=0)

    def plot_helper_1d(log_prob_fn):
        p = log_prob_fn(x).exp()
        plt.plot(x, p)
        plt.xlim([min(x), max(x)])

    def plot_helper_2d(log_prob_fn):
        lp = log_prob_fn(xy)
        p = (lp - lp.max()).exp().reshape(xx.size())
        plt.contourf(xx, yy, p, origin='lower')
        plt.xlim([min(x), max(x)])
        plt.ylim([min(x), max(x)])

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    plot_helper_2d(log_prob_banana)
    plt.title('banana')
    plt.subplot(2, 2, 2)
    plot_helper_2d(log_prob_cigar)
    plt.title('cigar')
    plt.subplot(2, 2, 3)
    plot_helper_1d(log_prob_mix_laplace)
    plt.title('mix laplace')
    plt.show()