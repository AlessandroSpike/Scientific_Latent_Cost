import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import json
import time
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fixed hyperparameters (from your original script)
EPS = 0.2
LAMBDA = 31.6
LR = 5.01e-3
REG_C0 = 3.16e-4
REG_E = 3.16e-4
REG_TEMP = 1.00e-4

N_ITER_MAIN = 3000      # Main optimization iterations
N_ITER_BOOT = 100      # Bootstrap iterations (faster)
SINKHORN_ITERS = 100
SINKHORN_TOL = 1e-7
BATCH_SIZE = 8
USE_AMP = torch.cuda.is_available()
GRAD_CLIP = 1.0

print(f"Device:                  {DEVICE}")
print(f"Main iterations:         {N_ITER_MAIN}")
print(f"Bootstrap iterations:    {N_ITER_BOOT}")
print(f"Sinkhorn iters:          {SINKHORN_ITERS}")
print(f"Batch size:              {BATCH_SIZE}")
print(f"AMP enabled:             {USE_AMP}")
print(f"Gradient clipping:       {GRAD_CLIP}")


# =========================
# LOAD FROM .NPY + .CSV
# =========================
def load_from_npy(tensor_path="ot_filtered_tensor.npy",
                  concepts_path="concept_list.csv"):
    """
    Load OT tensor and concept list from OpenAlex pipeline outputs.
    """
    print(f"\nLoading tensor from: {tensor_path}")
    print(f"Loading concepts from: {concepts_path}")

    # Load numpy tensor
    data_3d = np.load(tensor_path)
    print(f"Raw tensor shape:    {data_3d.shape}")

    if data_3d.ndim != 3:
        raise ValueError(f"Expected 3D array, got {data_3d.ndim}D")

    N1, N2, T = data_3d.shape

    if N1 != N2:
        raise ValueError(f"Non-square concept matrices: {N1} x {N2}")

    N = N1
    print(f"Concepts (N):        {N}")
    print(f"Timepoints (T):      {T}  →  years 2000–{2000+T-1}")

    # Load concept names
    concept_df = pd.read_csv(concepts_path)
    if "concept" not in concept_df.columns:
        raise ValueError(f"CSV must have column 'concept'. Found: {list(concept_df.columns)}")
    concepts = concept_df["concept"].tolist()

    if len(concepts) != N:
        raise ValueError(f"Concept list length ({len(concepts)}) != tensor dim ({N})")
    
    print(f"Concepts loaded:     {len(concepts)}")

    # Diagnostics
    print(f"\nRaw data statistics:")
    print(f"  Min:      {data_3d.min():.6e}")
    print(f"  Max:      {data_3d.max():.6e}")
    print(f"  Mean:     {data_3d.mean():.6e}")
    print(f"  Sparsity: {(data_3d == 0).mean():.2%} zeros")

    # Normalize by global max
    global_max = data_3d.max()
    if global_max > 0:
        data_3d = data_3d / global_max
        print(f"\nNormalized by global max: {global_max:.6e}")
    else:
        print("⚠ Warning: global max is 0, skipping normalization")

    normalization_factor = float(global_max)

    # Build list of tensors
    Ts = []
    for t in range(T):
        mat = data_3d[:, :, t]
        tensor = torch.tensor(mat, dtype=torch.float32)
        Ts.append(tensor)

    year_labels = list(range(2000, 2000 + T))

    print(f"\n✓ Loaded {len(Ts)} timepoints, each shape {Ts[0].shape}")
    return Ts, concepts, normalization_factor, year_labels


# =========================
# BUILD SPACE
# =========================
def build_space(Ts):
    N = Ts[0].shape[0]
    T_list, M_list = [], []

    for T_mat in Ts:
        T_list.append(T_mat.to(DEVICE))
        M_list.append(torch.ones(N, N, device=DEVICE))

    print(f"Space built: {N}×{N}")
    return T_list, M_list, N


# =========================
# UOT SOLVER
# =========================
def uot_log(C, T_ref, eps, lambda_):
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


# =========================
# GAUGE FIXING
# =========================
def gauge(C):
    C = C - C.mean(1, keepdim=True)
    C = C - C.mean(0, keepdim=True)
    return C


# =========================
# LOSS
# =========================
def compute_loss(C0, E, T_list, M_list, indices,
                 eps, lambda_, reg_c0, reg_e, reg_temp):
    loss = torch.tensor(0.0, device=DEVICE)

    for t in indices:
        C_t = C0 + E[t]
        T_hat = uot_log(C_t, T_list[t], eps, lambda_)
        recon = torch.norm(M_list[t] * (T_hat - T_list[t]), p='fro') ** 2
        reg_e_term = reg_e * torch.norm(E[t]) ** 2
        loss = loss + recon + reg_e_term

    # Scale to full dataset
    loss = loss * (len(T_list) / len(indices))
    
    # Global regularization
    loss = loss + reg_c0 * torch.norm(C0) ** 2
    
    # Temporal smoothness
    for t in range(1, len(E)):
        loss = loss + reg_temp * torch.norm(E[t] - E[t-1]) ** 2

    return loss


# =========================
# INIT C0
# =========================
def init_C0(T_list):
    avg_T = sum(T_list) / len(T_list)
    total = avg_T.sum()

    if total < 1e-8:
        print("⚠ Near-zero input, using random init")
        return torch.randn(avg_T.shape, device=DEVICE) * 0.1

    avg_T = avg_T / total
    C0 = -torch.log(avg_T + 1e-8)
    C0 = torch.clamp(C0, 0, 20)
    C0 = gauge(C0)

    return C0.to(DEVICE)


# =========================
# OPTIMIZE (MAIN)
# =========================
def optimize(T_list, M_list, N, n_iter, verbose=True, print_every=100):
    T = len(T_list)

    C0 = init_C0(T_list).clone().detach().requires_grad_(True)
    E = [torch.zeros(N, N, device=DEVICE, requires_grad=True) for _ in range(T)]

    opt = optim.Adam([C0] + E, lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=200, min_lr=LR * 1e-3
    )
    scaler = GradScaler() if USE_AMP else None

    start_time = time.time()
    best_loss = float('inf')
    best_state = None

    for it in range(n_iter):
        opt.zero_grad()
        idx = torch.randperm(T)[:min(BATCH_SIZE, T)]

        if USE_AMP:
            with autocast():
                batch_loss = compute_loss(C0, E, T_list, M_list, idx,
                                         EPS, LAMBDA, REG_C0, REG_E, REG_TEMP)
            scaler.scale(batch_loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([C0] + E, GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
        else:
            batch_loss = compute_loss(C0, E, T_list, M_list, idx,
                                     EPS, LAMBDA, REG_C0, REG_E, REG_TEMP)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_([C0] + E, GRAD_CLIP)
            opt.step()

        # Gauge fixing
        with torch.no_grad():
            C0.copy_(gauge(C0))
            for i in range(T):
                E[i].copy_(gauge(E[i]))

        # Full loss computation
        with torch.no_grad():
            if it % print_every == 0:
                full_loss = compute_loss(C0, E, T_list, M_list, range(T),
                                        EPS, LAMBDA, REG_C0, REG_E, REG_TEMP)
                full_loss_val = full_loss.item()
                scheduler.step(full_loss_val)
                
                if full_loss_val < best_loss:
                    best_loss = full_loss_val
                    best_state = {
                        'C0': C0.detach().cpu().clone(),
                        'E': [e.detach().cpu().clone() for e in E],
                        'iter': it
                    }
                
                if verbose:
                    elapsed = time.time() - start_time
                    print(f"Iter {it:5d}/{n_iter} | "
                          f"Batch: {batch_loss.item():.6f} | "
                          f"Full: {full_loss_val:.6f} | "
                          f"Best: {best_loss:.6f} | "
                          f"Time: {elapsed:.0f}s")

    total_time = time.time() - start_time
    if verbose:
        print(f"\n{'='*60}")
        print(f"Optimization done in {total_time:.1f}s ({total_time/60:.2f} min)")
        print(f"Best loss: {best_loss:.6f} (iter {best_state['iter']})")
        print(f"{'='*60}\n")

    return best_state['C0'], best_state['E'], best_loss


# =========================
# BOOTSTRAP ANALYSIS
# =========================
def temporal_bootstrap(T_list, M_list, N, n_boot=100, seed=42):
    """Temporal bootstrap: resample years with replacement"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    n_time = len(T_list)
    baseline_shares = []
    C0_norms = []
    E_mean_norms = []
    
    print(f"\n{'='*60}")
    print(f"TEMPORAL BOOTSTRAP: {n_boot} iterations")
    print(f"{'='*60}")
    
    for b in tqdm(range(n_boot), desc="Bootstrap"):
        # Resample time indices with replacement
        boot_idx = np.random.choice(n_time, size=n_time, replace=True)
        
        # Create bootstrapped dataset
        T_boot = [T_list[i] for i in boot_idx]
        M_boot = [M_list[i] for i in boot_idx]
        
        # Optimize (fewer iterations for speed)
        C0_boot, E_boot, _ = optimize(T_boot, M_boot, N, 
                                      n_iter=N_ITER_BOOT, 
                                      verbose=False)
        
        # Compute metrics
        C0_norm = torch.norm(C0_boot).item()
        E_mean_norm = np.mean([torch.norm(e).item() for e in E_boot])
        baseline_share = (C0_norm**2) / (C0_norm**2 + E_mean_norm**2)
        
        baseline_shares.append(baseline_share)
        C0_norms.append(C0_norm)
        E_mean_norms.append(E_mean_norm)
    
    results = {
        'baseline_shares': np.array(baseline_shares),
        'C0_norms': np.array(C0_norms),
        'E_mean_norms': np.array(E_mean_norms),
        'mean_baseline_share': np.mean(baseline_shares),
        'std_baseline_share': np.std(baseline_shares),
        'ci_95_baseline': np.percentile(baseline_shares, [2.5, 97.5]),
        'cv_C0_norm': np.std(C0_norms) / np.mean(C0_norms)
    }
    
    return results


def permutation_test(T_list, M_list, N, n_perm=1000, seed=42):
    """Temporal permutation test: shuffle year labels"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    n_time = len(T_list)
    baseline_shares_perm = []
    
    print(f"\n{'='*60}")
    print(f"TEMPORAL PERMUTATION TEST: {n_perm} iterations")
    print(f"{'='*60}")
    
    for p in tqdm(range(n_perm), desc="Permutation"):
        # Randomly permute time indices
        perm_idx = np.random.permutation(n_time)
        
        # Create permuted dataset
        T_perm = [T_list[i] for i in perm_idx]
        M_perm = [M_list[i] for i in perm_idx]
        
        # Optimize
        C0_perm, E_perm, _ = optimize(T_perm, M_perm, N,
                                      n_iter=N_ITER_BOOT,
                                      verbose=False)
        
        # Compute baseline share
        C0_norm = torch.norm(C0_perm).item()
        E_mean_norm = np.mean([torch.norm(e).item() for e in E_perm])
        baseline_share = (C0_norm**2) / (C0_norm**2 + E_mean_norm**2)
        
        baseline_shares_perm.append(baseline_share)
    
    results = {
        'baseline_shares_perm': np.array(baseline_shares_perm),
        'mean_baseline_share_perm': np.mean(baseline_shares_perm),
        'std_baseline_share_perm': np.std(baseline_shares_perm),
        'ci_95_perm': np.percentile(baseline_shares_perm, [2.5, 97.5])
    }
    
    return results


# =========================
# SAVE FUNCTIONS
# =========================
def save_results(C0, E_list, folder, concepts=None, year_labels=None):
    os.makedirs(folder, exist_ok=True)

    idx = concepts if concepts else None

    # Save C0
    df_C0 = pd.DataFrame(C0.numpy(), index=idx, columns=idx)
    df_C0.to_csv(f"{folder}/C0_baseline.csv")

    # Save E and C per year
    for t, e in enumerate(E_list):
        label = str(year_labels[t]) if year_labels else f"{t:02d}"
        df_E = pd.DataFrame(e.numpy(), index=idx, columns=idx)
        df_E.to_csv(f"{folder}/E_{label}.csv")
        
        df_C = pd.DataFrame((C0 + e).numpy(), index=idx, columns=idx)
        df_C.to_csv(f"{folder}/C_{label}.csv")

    # Save npz
    np.savez(f"{folder}/results.npz",
             C0=C0.numpy(),
             E=np.stack([e.numpy() for e in E_list]),
             C_full=np.stack([(C0 + e).numpy() for e in E_list]),
             concepts=np.array(concepts if concepts else []),
             years=np.array(year_labels if year_labels else []))

    print(f"✓ Saved results to {folder}/")


def save_bootstrap_results(boot_results, perm_results, original_share, folder):
    os.makedirs(folder, exist_ok=True)
    
    # Save text summary
    with open(f"{folder}/bootstrap_summary.txt", 'w') as f:
        f.write("="*60 + "\n")
        f.write("BOOTSTRAP & PERMUTATION TEST RESULTS\n")
        f.write("="*60 + "\n\n")
        
        f.write("ORIGINAL MODEL:\n")
        f.write(f"  Baseline share: {original_share:.4f} ({original_share*100:.2f}%)\n\n")
        
        f.write("TEMPORAL BOOTSTRAP (resampling years with replacement):\n")
        f.write(f"  Mean baseline share: {boot_results['mean_baseline_share']:.4f}\n")
        f.write(f"  Std baseline share:  {boot_results['std_baseline_share']:.4f}\n")
        f.write(f"  95% CI: [{boot_results['ci_95_baseline'][0]:.4f}, {boot_results['ci_95_baseline'][1]:.4f}]\n")
        f.write(f"  CV of ||C0||: {boot_results['cv_C0_norm']:.4f}\n\n")
        
        f.write("TEMPORAL PERMUTATION (shuffling year labels):\n")
        f.write(f"  Mean baseline share: {perm_results['mean_baseline_share_perm']:.4f}\n")
        f.write(f"  Std baseline share:  {perm_results['std_baseline_share_perm']:.4f}\n")
        f.write(f"  95% CI: [{perm_results['ci_95_perm'][0]:.4f}, {perm_results['ci_95_perm'][1]:.4f}]\n\n")
        
        f.write("INTERPRETATION:\n")
        f.write(f"  Bootstrap CI width: {boot_results['ci_95_baseline'][1] - boot_results['ci_95_baseline'][0]:.4f}\n")
        f.write(f"  Permutation collapse: {original_share:.4f} → {perm_results['mean_baseline_share_perm']:.4f}\n")
        f.write(f"  Relative reduction: {(1 - perm_results['mean_baseline_share_perm']/original_share)*100:.1f}%\n")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: Bootstrap
    ax = axes[0]
    ax.hist(boot_results['baseline_shares'], bins=30, alpha=0.7,
            color='steelblue', edgecolor='black')
    ax.axvline(original_share, color='red', linestyle='--', linewidth=2,
               label=f'Original: {original_share:.3f}')
    ax.axvline(boot_results['mean_baseline_share'], color='blue',
               linestyle='--', linewidth=2,
               label=f'Bootstrap mean: {boot_results["mean_baseline_share"]:.3f}')
    ax.axvline(boot_results['ci_95_baseline'][0], color='gray',
               linestyle=':', alpha=0.7)
    ax.axvline(boot_results['ci_95_baseline'][1], color='gray',
               linestyle=':', alpha=0.7, label='95% CI')
    ax.set_xlabel('Baseline Share (||C⁰||² / Total)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('A. Temporal Bootstrap Distribution\n(Resampling Years)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    
    # Panel B: Permutation
    ax = axes[1]
    ax.hist(perm_results['baseline_shares_perm'], bins=30, alpha=0.7,
            color='coral', edgecolor='black')
    ax.axvline(original_share, color='red', linestyle='--', linewidth=2,
               label=f'Original: {original_share:.3f}')
    ax.axvline(perm_results['mean_baseline_share_perm'], color='orange',
               linestyle='--', linewidth=2,
               label=f'Permuted mean: {perm_results["mean_baseline_share_perm"]:.3f}')
    ax.set_xlabel('Baseline Share (||C⁰||² / Total)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('B. Temporal Permutation Distribution\n(Shuffling Year Labels)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{folder}/bootstrap_permutation.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{folder}/bootstrap_permutation.pdf", bbox_inches='tight')
    plt.close()
    
    # Save raw data
    np.savez(f"{folder}/bootstrap_data.npz",
             baseline_shares_boot=boot_results['baseline_shares'],
             baseline_shares_perm=perm_results['baseline_shares_perm'],
             original_share=original_share)
    
    print(f"\n✓ Bootstrap results saved to {folder}/")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    overall_start = time.time()

    TENSOR_PATH = "ot_filtered_tensor.npy"
    CONCEPTS_PATH = "concept_list.csv"
    OUTPUT_FOLDER = "results"

    # Load data
    try:
        Ts, concepts, norm_factor, year_labels = load_from_npy(
            TENSOR_PATH, CONCEPTS_PATH
        )
    except Exception as e:
        print(f"❌ Load error: {e}")
        raise

    # Build space
    T_list, M_list, N = build_space(Ts)

    # ══════════════════════════════════════════════════════════════
    # MAIN OPTIMIZATION
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("MAIN OPTIMIZATION")
    print("="*60)

    C0, E, final_loss = optimize(T_list, M_list, N, 
                                 n_iter=N_ITER_MAIN, 
                                 verbose=True)

    # Compute original baseline share
    C0_norm = torch.norm(C0).item()
    E_mean_norm = np.mean([torch.norm(e).item() for e in E])
    original_share = (C0_norm**2) / (C0_norm**2 + E_mean_norm**2)

    print(f"\n{'='*60}")
    print(f"ORIGINAL MODEL RESULTS:")
    print(f"  ||C0|| = {C0_norm:.4f}")
    print(f"  Mean ||E(t)|| = {E_mean_norm:.4f}")
    print(f"  Baseline share = {original_share:.4f} ({original_share*100:.2f}%)")
    print(f"{'='*60}")

    # Save main results
    save_results(C0, E, OUTPUT_FOLDER, concepts=concepts, year_labels=year_labels)

    # ══════════════════════════════════════════════════════════════
    # BOOTSTRAP ANALYSIS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("BOOTSTRAP ANALYSIS OPTIONS")
    print("="*60)
    print("1. Skip bootstrap (save main results only)")
    print("2. Run bootstrap only (n=100)")
    print("3. Run permutation only (n=1000)")
    print("4. Run both bootstrap + permutation (RECOMMENDED)")

    choice = input("\nEnter choice [1-4]: ").strip()

    if choice == '1':
        print("\n✓ Skipping bootstrap analysis")

    elif choice == '2':
        boot_results = temporal_bootstrap(T_list, M_list, N, n_boot=100, seed=42)
        perm_results = {
            'baseline_shares_perm': np.array([original_share]),
            'mean_baseline_share_perm': original_share,
            'std_baseline_share_perm': 0.0,
            'ci_95_perm': [original_share, original_share]
        }
        save_bootstrap_results(boot_results, perm_results, original_share,
                              f"{OUTPUT_FOLDER}/bootstrap")

    elif choice == '3':
        perm_results = permutation_test(T_list, M_list, N, n_perm=100, seed=42)
        boot_results = {
            'baseline_shares': np.array([original_share]),
            'mean_baseline_share': original_share,
            'std_baseline_share': 0.0,
            'ci_95_baseline': [original_share, original_share],
            'cv_C0_norm': 0.0
        }
        save_bootstrap_results(boot_results, perm_results, original_share,
                              f"{OUTPUT_FOLDER}/bootstrap")

    elif choice == '4':
        # Full analysis
        boot_results = temporal_bootstrap(T_list, M_list, N, n_boot=100, seed=42)
        perm_results = permutation_test(T_list, M_list, N, n_perm=100, seed=42)
        save_bootstrap_results(boot_results, perm_results, original_share,
                              f"{OUTPUT_FOLDER}/bootstrap")

        # Print summary
        print("\n" + "="*60)
        print("FINAL SUMMARY:")
        print(f"  Original baseline share: {original_share:.4f}")
        print(f"  Bootstrap mean ± std: {boot_results['mean_baseline_share']:.4f} ± {boot_results['std_baseline_share']:.4f}")
        print(f"  Bootstrap 95% CI: [{boot_results['ci_95_baseline'][0]:.4f}, {boot_results['ci_95_baseline'][1]:.4f}]")
        print(f"  CV of ||C0||: {boot_results['cv_C0_norm']:.4f}")
        print(f"\n  Permuted mean ± std: {perm_results['mean_baseline_share_perm']:.4f} ± {perm_results['std_baseline_share_perm']:.4f}")
        print(f"  Reduction from permutation: {((original_share - perm_results['mean_baseline_share_perm'])/original_share)*100:.1f}%")
        print("="*60)

    else:
        print("Invalid choice. Skipping bootstrap.")

    total_time = time.time() - overall_start
    print(f"\n✓ DONE | Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
