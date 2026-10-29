"""
Empirical Identifiability Validation
=====================================
Validates Theorems 3 and 5 from the theoretical section:
  1. Temporal mean diagnostic: ||mean(E_t)||_F << ||C0||_F
  2. Bootstrap stability: CV of C0 across bootstrap samples
  3. Sensitivity to hyperparameters
"""

import torch
import torch.optim as optim
import pandas as pd
import os
import json
import numpy as np
import time
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import re
# ============================================================================
# REUSE CORE FUNCTIONS FROM MAIN SCRIPT
# ============================================================================
# (Copy the following functions from your main script)
# - load_from_npy
# - build_space
# - uot_log
# - gauge
# - compute_loss
# - init_C0
# - optimize

# I'll import them here assuming they're in the main file
# If not, you can copy-paste them into this script

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = torch.cuda.is_available()
SINKHORN_ITERS = 100
SINKHORN_TOL = 1e-7
BATCH_SIZE = 8
GRAD_CLIP = 1.0

# ============================================================================
# CORE FUNCTIONS (paste from main script if needed)
# ============================================================================

def gauge(C):
    """Remove row and column means from cost matrix."""
    C = C - C.mean(1, keepdim=True)
    C = C - C.mean(0, keepdim=True)
    return C

def uot_log(C, T_ref, eps, lambda_):
    """Unbalanced OT via log-stabilized Sinkhorn."""
    n = C.shape[0]
    mu = T_ref.sum(1).clamp(min=1e-10)
    nu = T_ref.sum(0).clamp(min=1e-10)
    
    log_mu = torch.log(mu)
    log_nu = torch.log(nu)
    log_K = -C / eps
    tau = lambda_ / (lambda_ + eps)
    
    f = torch.zeros(n, device=DEVICE)
    g = torch.zeros(n, device=DEVICE)
    
    for it in range(SINKHORN_ITERS):
        f_prev = f.clone()
        f = tau * (log_mu - torch.logsumexp(log_K + g[None, :], dim=1))
        g = tau * (log_nu - torch.logsumexp(log_K + f[:, None], dim=0))
        if torch.max(torch.abs(f - f_prev)).item() < SINKHORN_TOL:
            break
    
    return torch.exp(f[:, None] + log_K + g[None, :])

def compute_loss(C0, E, T_list, M_list, indices,
                 eps, lambda_, reg_c0, reg_e, reg_temp,
                 return_per_t=False):
    """Compute total loss."""
    loss = torch.tensor(0.0, device=DEVICE)
    per_t_losses = {}
    
    for t in indices:
        C_t = C0 + E[t]
        T_hat = uot_log(C_t, T_list[t], eps, lambda_)
        recon = torch.norm(M_list[t] * (T_hat - T_list[t]), p='fro') ** 2
        reg_e_term = reg_e * torch.norm(E[t]) ** 2
        
        t_loss = recon + reg_e_term
        loss = loss + t_loss
        
        if return_per_t:
            per_t_losses[int(t)] = {
                'recon': recon.item(),
                'reg_e': reg_e_term.item(),
                'total': t_loss.item()
            }
    
    loss = loss * (len(T_list) / len(indices))
    reg_c0_term = reg_c0 * torch.norm(C0) ** 2
    loss = loss + reg_c0_term
    
    for t in range(1, len(E)):
        loss = loss + reg_temp * torch.norm(E[t] - E[t - 1]) ** 2
    
    if return_per_t:
        return loss, per_t_losses
    return loss

def init_C0(T_list):
    """Initialize C0 from time-averaged transport."""
    avg_T = sum(T_list) / len(T_list)
    total = avg_T.sum()
    
    if total < 1e-8:
        return torch.randn(avg_T.shape, device=DEVICE) * 0.1
    
    avg_T = avg_T / total
    C0 = -torch.log(avg_T + 1e-8)
    C0 = torch.clamp(C0, 0, 20)
    C0 = gauge(C0)
    
    return C0.to(DEVICE)

def optimize(T_list, M_list, N, eps, lambda_, lr, reg_c0, reg_e, reg_temp,
             n_iter, verbose=False, print_every=500):
    """Optimize (C0, E) decomposition."""
    T = len(T_list)
    
    C0 = init_C0(T_list).clone().detach().requires_grad_(True)
    E = [torch.zeros(N, N, device=DEVICE, requires_grad=True) for _ in range(T)]
    
    opt = optim.Adam([C0] + E, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=200, min_lr=lr * 1e-3
    )
    scaler = GradScaler() if USE_AMP else None
    
    best_loss = float('inf')
    best_state = None
    
    for it in range(n_iter):
        opt.zero_grad()
        idx = torch.randperm(T)[:min(BATCH_SIZE, T)]
        
        if USE_AMP:
            with autocast():
                batch_loss = compute_loss(
                    C0, E, T_list, M_list, idx,
                    eps, lambda_, reg_c0, reg_e, reg_temp
                )
            scaler.scale(batch_loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([C0] + E, GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
        else:
            batch_loss = compute_loss(
                C0, E, T_list, M_list, idx,
                eps, lambda_, reg_c0, reg_e, reg_temp
            )
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_([C0] + E, GRAD_CLIP)
            opt.step()
        
        # Gauge fixing
        with torch.no_grad():
            C0.copy_(gauge(C0))
            for i in range(T):
                E[i].copy_(gauge(E[i]))
        
        # Track best
        with torch.no_grad():
            if (it % 50 == 0 or it == n_iter - 1) and not torch.isnan(batch_loss):
                full_loss = compute_loss(
                    C0, E, T_list, M_list, range(T),
                    eps, lambda_, reg_c0, reg_e, reg_temp
                )
                full_loss_val = full_loss.item()
                scheduler.step(full_loss_val)
                
                if full_loss_val < best_loss:
                    best_loss = full_loss_val
                    best_state = {
                        'C0': C0.detach().cpu().clone(),
                        'E': [e.detach().cpu().clone() for e in E],
                        'iter': it
                    }
        
        if verbose and it % print_every == 0:
            print(f"  Iter {it:4d}/{n_iter} | Loss: {best_loss:.6f}")
    
    return best_state['C0'], best_state['E'], best_loss

def convert_to_native_types(obj):
    """
    Recursively convert numpy/torch types to native Python types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    else:
        return obj


# ============================================================================
# IDENTIFIABILITY DIAGNOSTICS
# ============================================================================

def temporal_mean_diagnostic(C0, E_list):
    """
    Theorem 3 diagnostic: Verify that mean(E_t) ≈ 0.
    
    Returns:
        ratio: ||mean(E_t)||_F / ||C0||_F  (should be << 1)
    """
    E_mean = sum(E_list) / len(E_list)
    
    norm_E_mean = torch.norm(E_mean).item()
    norm_C0 = torch.norm(C0).item()
    ratio = norm_E_mean / norm_C0
    
    return {
        'norm_E_mean': norm_E_mean,
        'norm_C0': norm_C0,
        'ratio': ratio,
        'E_mean': E_mean.numpy()
    }


def bootstrap_stability(T_list, M_list, N, hyperparams, 
                       n_bootstrap=100, n_iter=2000, seed=42):
    """
    Theorem 5 diagnostic: Bootstrap stability of C0.
    
    Resample years with replacement, re-estimate (C0, E), compute CV.
    
    Returns:
        results: dict with C0 statistics across bootstrap samples
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    T = len(T_list)
    C0_samples = []
    E_norms = []
    losses = []
    
    print(f"\n{'='*60}")
    print(f"Bootstrap Stability Analysis: {n_bootstrap} samples")
    print(f"{'='*60}")
    
    for b in tqdm(range(n_bootstrap), desc="Bootstrap"):
        # Resample years with replacement
        boot_idx = np.random.choice(T, size=T, replace=True)
        
        T_boot = [T_list[i] for i in boot_idx]
        M_boot = [M_list[i] for i in boot_idx]
        
        # Re-estimate
        C0_b, E_b, loss_b = optimize(
            T_boot, M_boot, N,
            eps=hyperparams['EPS'],
            lambda_=hyperparams['LAMBDA'],
            lr=hyperparams['LR'],
            reg_c0=hyperparams['REG_C0'],
            reg_e=hyperparams['REG_E'],
            reg_temp=hyperparams['REG_TEMP'],
            n_iter=n_iter,
            verbose=False
        )
        
        C0_samples.append(C0_b.numpy())
        E_norms.append([torch.norm(e).item() for e in E_b])
        losses.append(loss_b)
    
    # Stack and compute statistics
    C0_stack = np.stack(C0_samples)  # (n_bootstrap, N, N)
    
    C0_mean = C0_stack.mean(axis=0)
    C0_std = C0_stack.std(axis=0)
    
    # Element-wise coefficient of variation
    # Avoid division by zero
    C0_cv = np.divide(
        C0_std, 
        np.abs(C0_mean) + 1e-8,
        out=np.zeros_like(C0_std),
        where=np.abs(C0_mean) > 1e-8
    )
    
    # Summary statistics
    cv_mean = C0_cv.mean()
    cv_median = np.median(C0_cv)
    cv_95 = np.percentile(C0_cv, 95)
    
    # Frobenius norm stability
    C0_norms = [np.linalg.norm(C0_samples[i]) for i in range(n_bootstrap)]
    norm_mean = np.mean(C0_norms)
    norm_std = np.std(C0_norms)
    norm_cv = norm_std / norm_mean
    
    return {
        'C0_samples': C0_stack,
        'C0_mean': C0_mean,
        'C0_std': C0_std,
        'C0_cv': C0_cv,
        'cv_summary': {
            'mean': cv_mean,
            'median': cv_median,
            'p95': cv_95
        },
        'norm_stability': {
            'mean': norm_mean,
            'std': norm_std,
            'cv': norm_cv
        },
        'E_norms': E_norms,
        'losses': losses
    }


def hyperparameter_sensitivity(T_list, M_list, N, base_hyperparams,
                               param_name='REG_C0', n_iter=2000):
    """
    Sensitivity analysis: vary one hyperparameter, measure C0 stability.
    
    Returns:
        results: dict mapping param values to (C0, E, loss)
    """
    # Define ranges for each hyperparameter
    ranges = {
        'REG_C0': np.logspace(-4, -2, 7),    # 0.0001 to 0.01
        'REG_E': np.logspace(-4, -1.5, 7),   # 0.0001 to 0.03
        'REG_TEMP': np.logspace(-4.5, -2, 7),
        'EPS': np.logspace(-2, -0.5, 7),
        'LAMBDA': np.logspace(0.3, 1.7, 7)
    }
    
    values = ranges[param_name]
    results = []
    
    print(f"\n{'='*60}")
    print(f"Sensitivity Analysis: {param_name}")
    print(f"{'='*60}")
    
    for val in tqdm(values, desc=param_name):
        hp = base_hyperparams.copy()
        hp[param_name] = val
        
        C0, E, loss = optimize(
            T_list, M_list, N,
            eps=hp['EPS'],
            lambda_=hp['LAMBDA'],
            lr=hp['LR'],
            reg_c0=hp['REG_C0'],
            reg_e=hp['REG_E'],
            reg_temp=hp['REG_TEMP'],
            n_iter=n_iter,
            verbose=False
        )
        
        results.append({
            'value': val,
            'C0': C0.numpy(),
            'E_norms': [torch.norm(e).item() for e in E],
            'loss': loss,
            'C0_norm': torch.norm(C0).item()
        })
    
    return {
        'param_name': param_name,
        'results': results
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_temporal_mean(E_mean, concepts, save_path):
    """Heatmap of mean(E_t) - should be close to zero."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap
    im = ax.imshow(E_mean, cmap='RdBu_r', aspect='auto',
                   vmin=-np.abs(E_mean).max(), 
                   vmax=np.abs(E_mean).max())
    
    ax.set_title(r'Temporal Mean of Deviations: $\frac{1}{T}\sum_t E_t$', 
                 fontsize=14, pad=20)
    ax.set_xlabel('Target Concept', fontsize=12)
    ax.set_ylabel('Source Concept', fontsize=12)
    
    plt.colorbar(im, ax=ax, label='Mean Deviation')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")


def plot_bootstrap_stability(boot_results, save_folder):
    """Visualize bootstrap stability."""
    
    # 1. CV distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    cv_flat = boot_results['C0_cv'].flatten()
    axes[0].hist(cv_flat, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(boot_results['cv_summary']['mean'], 
                    color='red', linestyle='--', 
                    label=f"Mean: {boot_results['cv_summary']['mean']:.3f}")
    axes[0].set_xlabel('Coefficient of Variation')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('C0 Element-wise CV Distribution')
    axes[0].legend()
    axes[0].set_yscale('log')
    
    # 2. Norm stability
    norms = [np.linalg.norm(C0) for C0 in boot_results['C0_samples']]
    axes[1].hist(norms, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(boot_results['norm_stability']['mean'],
                    color='red', linestyle='--',
                    label=f"CV: {boot_results['norm_stability']['cv']:.3f}")
    axes[1].set_xlabel(r'$||C_0||_F$')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Bootstrap Distribution of C0 Norm')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_folder}/bootstrap_stability.png", dpi=300)
    plt.close()
    
    print(f"  Saved: {save_folder}/bootstrap_stability.png")
    
    # 3. Heatmap of C0 std
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(boot_results['C0_std'], cmap='viridis', aspect='auto')
    ax.set_title('Bootstrap Standard Deviation of C0', fontsize=14)
    plt.colorbar(im, ax=ax, label='Std Dev')
    plt.tight_layout()
    plt.savefig(f"{save_folder}/bootstrap_std_heatmap.png", dpi=300)
    plt.close()
    
    print(f"  Saved: {save_folder}/bootstrap_std_heatmap.png")


def plot_sensitivity(sens_results, save_path):
    """Plot hyperparameter sensitivity."""
    results = sens_results['results']
    param_name = sens_results['param_name']
    
    values = [r['value'] for r in results]
    C0_norms = [r['C0_norm'] for r in results]
    losses = [r['loss'] for r in results]
    mean_E_norms = [np.mean(r['E_norms']) for r in results]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # C0 norm
    axes[0].plot(values, C0_norms, 'o-', linewidth=2, markersize=8)
    axes[0].set_xscale('log')
    axes[0].set_xlabel(param_name, fontsize=12)
    axes[0].set_ylabel(r'$||C_0||_F$', fontsize=12)
    axes[0].set_title('Baseline Cost Magnitude')
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(values, losses, 'o-', linewidth=2, markersize=8, color='orange')
    axes[1].set_xscale('log')
    axes[1].set_xlabel(param_name, fontsize=12)
    axes[1].set_ylabel('Reconstruction Loss', fontsize=12)
    axes[1].set_title('Model Fit')
    axes[1].grid(True, alpha=0.3)
    
    # E norm
    axes[2].plot(values, mean_E_norms, 'o-', linewidth=2, markersize=8, color='green')
    axes[2].set_xscale('log')
    axes[2].set_xlabel(param_name, fontsize=12)
    axes[2].set_ylabel(r'Mean $||E_t||_F$', fontsize=12)
    axes[2].set_title('Deviation Magnitude')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"  Saved: {save_path}")


# ============================================================================
# MAIN VALIDATION PIPELINE
# ============================================================================

def run_identifiability_validation(
    tensor_path="ot_filtered_tensor.npy",
    concepts_path="concept_list.csv",
    hyperparams_path="results_bayesopt/best_hyperparameters.txt",
    results_path="results_bayesopt/results.npz",
    output_folder="identifiability_validation",
    n_bootstrap=1,
    n_iter_boot=1,
    n_iter_sens=2000,
    run_sensitivity=True
):
    """
    Complete identifiability validation pipeline.
    """
    print("\n" + "="*70)
    print("EMPIRICAL IDENTIFIABILITY VALIDATION")
    print("="*70)
    
    os.makedirs(output_folder, exist_ok=True)
    
    # ────────────────────────────────────────────────────────────────
    # 1. Load data and hyperparameters
    # ────────────────────────────────────────────────────────────────
    print("\n[1/5] Loading data and hyperparameters...")
    
    # Load data (assuming you have load_from_npy and build_space functions)
    from mainBO import load_from_npy, build_space
    Ts, concepts, norm_factor, year_labels = load_from_npy(tensor_path, concepts_path)
    T_list, M_list, N = build_space(Ts)
    
    # Load optimal hyperparameters
    def extract_float(s):
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if match:
            return float(match.group())
        else:
            raise ValueError(f"No numeric value in: {s}")
    
    hyperparams = {}

    with open(hyperparams_path, 'r') as f:
        for line in f:
            if ':' in line and not line.startswith('Best'):
                key, val = line.strip().split(':', 1)
                key = key.strip()
                val = extract_float(val)
                hyperparams[key] = val
    
    print(f"  Loaded {len(T_list)} timepoints, {N}×{N} concepts")
    print(f"  Hyperparameters: {hyperparams}")
    
    # Load estimated C0 and E from main results
    data = np.load(results_path)
    C0_main = torch.from_numpy(data['C0'])
    E_main = [torch.from_numpy(data['E'][t]) for t in range(len(T_list))]
    
    # ────────────────────────────────────────────────────────────────
    # 2. Temporal Mean Diagnostic (Theorem 3)
    # ────────────────────────────────────────────────────────────────
    print("\n[2/5] Temporal mean diagnostic (Theorem 3)...")
    
    tm_result = temporal_mean_diagnostic(C0_main, E_main)
    
    print(f"  ||C0||_F          = {tm_result['norm_C0']:.4f}")
    print(f"  ||mean(E_t)||_F   = {tm_result['norm_E_mean']:.4f}")
    print(f"  Ratio             = {tm_result['ratio']:.4f}")
    print(f"  ✓ PASS" if tm_result['ratio'] < 0.1 else "  ⚠ WARNING: ratio > 0.1")
    
    # Save results
    # Save results
    with open(f"{output_folder}/temporal_mean_diagnostic.json", 'w') as f:
        save_data = {k: v for k, v in tm_result.items() if k != 'E_mean'}
        json.dump(convert_to_native_types(save_data), f, indent=2)
    
    # Plot
    plot_temporal_mean(tm_result['E_mean'], concepts,
                      f"{output_folder}/temporal_mean_heatmap.png")
    
    # ────────────────────────────────────────────────────────────────
    # 3. Bootstrap Stability (Theorem 5)
    # ────────────────────────────────────────────────────────────────
    print(f"\n[3/5] Bootstrap stability ({n_bootstrap} samples, {n_iter_boot} iters each)...")
    
    boot_results = bootstrap_stability(
        T_list, M_list, N, hyperparams,
        n_bootstrap=n_bootstrap,
        n_iter=n_iter_boot
    )
    
    print(f"\n  Bootstrap Summary:")
    print(f"    CV (element-wise):   mean={boot_results['cv_summary']['mean']:.4f}, "
          f"median={boot_results['cv_summary']['median']:.4f}, "
          f"95th={boot_results['cv_summary']['p95']:.4f}")
    print(f"    CV (Frobenius norm): {boot_results['norm_stability']['cv']:.4f}")
    print(f"    ✓ PASS" if boot_results['norm_stability']['cv'] < 0.15 else 
          "    ⚠ WARNING: norm CV > 0.15")
    
    # Save
    np.savez(f"{output_folder}/bootstrap_results.npz",
             C0_samples=boot_results['C0_samples'],
             C0_mean=boot_results['C0_mean'],
             C0_std=boot_results['C0_std'],
             C0_cv=boot_results['C0_cv'])
    
    with open(f"{output_folder}/bootstrap_summary.json", 'w') as f:
        summary_data = {
            'cv_summary': boot_results['cv_summary'],
            'norm_stability': boot_results['norm_stability'],
            'n_bootstrap': n_bootstrap,
            'n_iter': n_iter_boot
        }
        json.dump(convert_to_native_types(summary_data), f, indent=2)
    
    # Plot
    plot_bootstrap_stability(boot_results, output_folder)
    
    # ────────────────────────────────────────────────────────────────
    # 4. Hyperparameter Sensitivity
    # ────────────────────────────────────────────────────────────────
    if run_sensitivity:
        print(f"\n[4/5] Hyperparameter sensitivity analysis...")
        
        for param in ['REG_C0', 'REG_E', 'REG_TEMP']:
            print(f"\n  Varying {param}...")
            sens_result = hyperparameter_sensitivity(
                T_list, M_list, N, hyperparams,
                param_name=param,
                n_iter=n_iter_sens
            )
            
            # Save
            np.savez(f"{output_folder}/sensitivity_{param}.npz",
                     **{f"val_{i}": r['C0'] for i, r in enumerate(sens_result['results'])})
            
            with open(f"{output_folder}/sensitivity_{param}.json", 'w') as f:
                sens_data = {
                    'param': param,
                    'values': [r['value'] for r in sens_result['results']],
                    'losses': [r['loss'] for r in sens_result['results']],
                    'C0_norms': [r['C0_norm'] for r in sens_result['results']]
                }
                json.dump(convert_to_native_types(sens_data), f, indent=2)
            
            # Plot
            plot_sensitivity(sens_result, 
                           f"{output_folder}/sensitivity_{param}.png")
    
    # ────────────────────────────────────────────────────────────────
    # 5. Generate LaTeX Summary Table
    # ────────────────────────────────────────────────────────────────
    print(f"\n[5/5] Generating summary report...")
    
    latex_table = f"""
\\begin{{table}}[ht]
\\centering
\\caption{{Empirical Identifiability Diagnostics}}
\\label{{tab:identifiability}}
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Diagnostic}} & \\textbf{{Value}} & \\textbf{{Status}} \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textit{{Temporal Mean (Theorem 3)}}}} \\\\
$||C_0||_F$ & {tm_result['norm_C0']:.4f} & --- \\\\
$||\frac{{1}}{{T}}\sum_t E_t||_F$ & {tm_result['norm_E_mean']:.4f} & --- \\\\
Ratio & {tm_result['ratio']:.4f} & {'✓' if tm_result['ratio'] < 0.1 else '⚠'} \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textit{{Bootstrap Stability (Theorem 5)}}}} \\\\
CV (element-wise, mean) & {boot_results['cv_summary']['mean']:.4f} & --- \\\\
CV (Frobenius norm) & {boot_results['norm_stability']['cv']:.4f} & {'✓' if boot_results['norm_stability']['cv'] < 0.15 else '⚠'} \\\\
Bootstrap samples & {n_bootstrap} & --- \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    
    with open(f"{output_folder}/identifiability_table.tex", 'w') as f:
        f.write(latex_table)
    
    # Summary text report
    report = f"""
EMPIRICAL IDENTIFIABILITY VALIDATION REPORT
{'='*70}

1. TEMPORAL MEAN DIAGNOSTIC (Assumption A2, Theorem 3)
   - Baseline norm:     ||C0||_F = {tm_result['norm_C0']:.6f}
   - Mean deviation:    ||mean(E_t)||_F = {tm_result['norm_E_mean']:.6f}
   - Ratio:             {tm_result['ratio']:.6f}
   - Required:          ratio << 1 (typically < 0.1)
   - Status:            {'PASS ✓' if tm_result['ratio'] < 0.1 else 'WARNING ⚠'}

2. BOOTSTRAP STABILITY (Theorem 5)
   - Samples:           {n_bootstrap}
   - Iterations/sample: {n_iter_boot}
   - Element-wise CV:   mean={boot_results['cv_summary']['mean']:.4f}, median={boot_results['cv_summary']['median']:.4f}
   - Norm CV:           {boot_results['norm_stability']['cv']:.4f}
   - Required:          CV < 0.15 for robust identification
   - Status:            {'PASS ✓' if boot_results['norm_stability']['cv'] < 0.15 else 'WARNING ⚠'}

3. INTERPRETATION
   {'Low ratio and CV indicate that C0 is stably identified and separates from E_t.' if tm_result['ratio'] < 0.1 and boot_results['norm_stability']['cv'] < 0.15 else 'Higher values suggest weaker identification; consider longer time series or stronger regularization.'}

4. FILES GENERATED
   - temporal_mean_heatmap.png       : Heatmap of mean(E_t)
   - bootstrap_stability.png         : CV and norm distributions
   - bootstrap_std_heatmap.png       : Spatial pattern of uncertainty
   - sensitivity_*.png               : Hyperparameter robustness
   - identifiability_table.tex       : LaTeX table for paper
   - All numerical results in .npz and .json files

{'='*70}
"""
    
    with open(f"{output_folder}/REPORT.txt", 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\n✓ All results saved to: {output_folder}/")
    
    return {
        'temporal_mean': tm_result,
        'bootstrap': boot_results,
        'output_folder': output_folder
    }


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    # Adjust paths to match your setup
    results = run_identifiability_validation(
        tensor_path="ot_filtered_tensor.npy",
        concepts_path="concept_list.csv",
        hyperparams_path="results_bayesopt/best_hyperparameters.txt",
        results_path="results_bayesopt/results.npz",
        output_folder="identifiability_validation",
        n_bootstrap=1,          # Use 100+ for publication
        n_iter_boot=1,         # Shorter than main (2000 vs 5000)
        n_iter_sens=2000,
        run_sensitivity=True      # Set False to skip sensitivity
    )
