import torch
import argparse
from pathlib import Path
from samplers.mcmc import HMC
from mvn import MVNIso, MVNDiag, MVNFull
from distributions import log_prob_banana, log_prob_cigar, log_prob_mix_laplace
from sys import exit


def log_psi(log_p, lam, q, n_kl_quad):
    kl_qp = -q.entropy() - q.quadrature_ev(log_p, n_kl_quad)
    return 0.5*q.log_det_fisher() - lam*kl_qp

def run_sampler(log_p, lam, q, n_samples, n_warmup, n_kl_quad):
    q = q.clone()
    hmc = HMC(lambda th: log_psi(log_p, lam, q.set_theta(th), n_kl_quad), leapfrog_t=2.)
    hmc.tune(torch.randn(200, q.n_params), target_accept=0.8)
    return hmc.sample(torch.randn(200, q.n_params),
                      n_samples=n_samples,
                      n_burnin=n_samples//10,
                      n_warmup=n_warmup,
                      progbar=True)

def extend_sampler(old_results, log_p, lam, q, n_total_samples, n_kl_quad):
    q = q.clone()
    hmc = HMC(lambda th: log_psi(log_p, lam, q.set_theta(th), n_kl_quad), leapfrog_t=old_results['leapfrog_t'])
    hmc.dt, hmc.mass, hmc.tuned = old_results['dt'], old_results['mass'], True
    n_prev_samples = old_results['samples'].size(0)
    n_new_samples = n_total_samples - n_prev_samples
    new_results = hmc.sample(old_results['samples'][-1,:],
                             n_samples=n_new_samples,
                             n_burnin=0,
                             n_warmup=0,
                             progbar=True)
    old_results.update({
        'samples': torch.cat([old_results['samples'], new_results['samples']], dim=0),
        'log_p': torch.cat([old_results['log_p'], new_results['log_p']], dim=0),
        'accept': (old_results['accept']*n_prev_samples + new_results['accept']*n_new_samples) / n_total_samples
    })
    return old_results



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--distrib', metavar='p', required=True, choices=['banana', 'cigar', 'laplace'])
    parser.add_argument('--component', metavar='q', default='Iso', choices=['MVNIso', 'MVNDiag', 'MVNFull'])
    parser.add_argument('--lam', metavar='λ', required=True, type=float)
    parser.add_argument('--save-dir', metavar='DIR', required=True, type=Path)
    parser.add_argument('--samples', metavar='N', default=10000, type=int)
    parser.add_argument('--warmup', metavar='W', default=100, type=int)
    parser.add_argument('--chain', metavar='C', default=0, type=int)
    parser.add_argument('--recompute', action='store_true', default=False)
    args = parser.parse_args()

    distrib_lookup = {
        'banana': {'log_p': log_prob_banana, 'dim': 2, 'quad_n': 5},
        'cigar': {'log_p': log_prob_cigar, 'dim': 2, 'quad_n': 5},
        'laplace': {'log_p': log_prob_mix_laplace, 'dim': 1, 'quad_n': 25},
    }

    component_lookup = {
        'MVNIso': MVNIso,
        'MVNDiag': MVNDiag,
        'MVNFull': MVNFull,
    }
    
    p = distrib_lookup[args.distrib]
    q = component_lookup[args.component](d=p['dim'])

    save_file = args.save_dir / f"hmc_{args.distrib}_{args.component}_{args.lam:.3f}_{args.chain:03d}.dat"
    if not save_file.exists() or args.recompute:
        result = run_sampler(p['log_p'], args.lam, q, args.samples, args.warmup, p['quad_n'])
        torch.save(result, save_file)
    elif save_file.exists():
        prev_result = torch.load(save_file)
        if prev_result['samples'].size(0) < args.samples:
            save_file = args.save_dir / f"hmc_{args.distrib}_{args.component}_{args.lam:.3f}_{args.chain:03d}_ext{args.samples}.dat"
            result = extend_sampler(prev_result, p['log_p'], args.lam, q, args.samples, p['quad_n'])
            torch.save(result, save_file)

