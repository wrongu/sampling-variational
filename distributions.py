import torch


def log_prob_banana(xy):
    if xy.size(0) == 2:
        x, y = xy[0], xy[1]
    else:
        x, y = xy[:, 0], xy[:, 1]
    # Easier to read version: -(y-(x/2)^2)^2 - (x/2)^2
    return -x*x*x*x/8 + y*x*x - 2*y*y - x*x/4


def log_prob_cigar(xy, c=.99):
    if xy.size(0) == 2:
        x, y = xy[0], xy[1]
    else:
        x, y = xy[:, 0], xy[:, 1]
    # Compute precision terms for covariance matrix [[1,c],[c,1]], where precision will be [[a,b],[b,a]]
    a = 1./(1. - c*c)
    b = -c/(1. - c*c)
    return -0.5*(a*x*x + a*y*y + 2*b*x*y)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = torch.linspace(-4, 4)
    xx, yy = torch.meshgrid(x, x)
    xy = torch.stack([xx.flatten(), yy.flatten()], dim=0)

    def plot_helper(log_prob_fn):
        lp = log_prob_fn(xy)
        p = (lp - lp.max()).exp().reshape(xx.size())
        plt.contourf(xx, yy, p, origin='lower')
        plt.xlim([min(x), max(x)])
        plt.ylim([min(x), max(x)])

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plot_helper(log_prob_banana)
    plt.title('banana')
    plt.subplot(1, 2, 2)
    plot_helper(log_prob_cigar)
    plt.title('cigar')
    plt.show()