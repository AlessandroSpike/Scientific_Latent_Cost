import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import os
import json
import numpy as np
import time
from torch.cuda.amp import autocast, GradScaler
from bayes_opt import BayesianOptimization, UtilityFunction

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fixed hyperparameters
N_ITER_BO     = 1000   # Iterations per BO evaluation
N_ITER_FINAL  = 5000   # Final optimization (increased from 3000)
SINKHORN_ITERS = 100   # Increased from 80 for better convergence
SINKHORN_TOL  = 1e-7   # Tighter tolerance
BATCH_SIZE    = 8      # Increased from 5 (24 timepoints, 8 is a good batch)
USE_AMP       = torch.cuda.is_available()
GRAD_CLIP     = 1.0    # NEW: gradient clipping threshold

print(f"Device:                  {DEVICE}")
print(f"Tensor expected shape:   (300, 300, 24)")
print(f"BO eval iterations:      {N_ITER_BO}")
print(f"Final optim iterations:  {N_ITER_FINAL}")
print(f"Sinkhorn iters:          {SINKHORN_ITERS}")
print(f"Batch size:              {BATCH_SIZE}")
print(f"AMP enabled:             {USE_AMP}")
print(f"Gradient clipping:       {GRAD_CLIP}")


# =========================
# LOAD FROM .NPY + .CSV
# (replaces load_from_mat)
# =========================
def load_from_npy(tensor_path="ot_filtered_tensor.npy",
                  concepts_path="concept_list.csv"):
    """
    Load OT tensor and concept list from OpenAlex pipeline outputs.
    
    Expected:
        tensor_path : shape (N_concepts, N_concepts, T_years)
                      e.g. (300, 300, 24)
        concepts_path: CSV with column 'concept', 300 rows
    
    Returns:
        Ts               : list of T tensors, each (N, N)
        concepts         : list of concept names (length N)
        normalization_factor : float used to normalize
        year_labels      : list of ints [2000, 2001, ..., 2023]
    """
    print(f"\nLoading tensor from: {tensor_path}")
    print(f"Loading concepts from: {concepts_path}")

    # ── Load numpy tensor ──────────────────────────────────────────
    data_3d = np.load(tensor_path)           # shape (N, N, T)
    print(f"Raw tensor shape:    {data_3d.shape}")
    print(f"Raw dtype:           {data_3d.dtype}")

    if data_3d.ndim != 3:
        raise ValueError(f"Expected 3D array, got {data_3d.ndim}D")

    N1, N2, T = data_3d.shape

    if N1 != N2:
        raise ValueError(f"Non-square concept matrices: {N1} x {N2}")

    N = N1
    print(f"Concepts (N):        {N}")
    print(f"Timepoints (T):      {T}  →  years 2000–{2000+T-1}")

    # ── Load concept names ─────────────────────────────────────────
    concept_df = pd.read_csv(concepts_path)
    if "concept" not in concept_df.columns:
        raise ValueError(f"CSV must have column 'concept'. "
                         f"Found: {list(concept_df.columns)}")
    concepts = concept_df["concept"].tolist()

    if len(concepts) != N:
        raise ValueError(
            f"Concept list length ({len(concepts)}) != "
            f"tensor dim ({N})"
        )
    print(f"Concepts loaded:     {len(concepts)}")
    print(f"First 5:             {concepts[:5]}")

    # ── Diagnostics ────────────────────────────────────────────────
    print(f"\nRaw data statistics:")
    print(f"  Min:      {data_3d.min():.6e}")
    print(f"  Max:      {data_3d.max():.6e}")
    print(f"  Mean:     {data_3d.mean():.6e}")
    print(f"  Std:      {data_3d.std():.6e}")
    sparsity = (data_3d == 0).mean()
    print(f"  Sparsity: {sparsity:.2%} zeros")

    # Per-year total mass (useful diagnostic for the OT tensor)
    print(f"\n  Per-year total mass:")
    for t in range(T):
        mass = data_3d[:, :, t].sum()
        year = 2000 + t
        print(f"    {year}: {mass:.4e}")

    # ── Normalize by global max ────────────────────────────────────
    # Preserves relative dynamics across years
    global_max = data_3d.max()
    if global_max > 0:
        data_3d = data_3d / global_max
        print(f"\nNormalized by global max: {global_max:.6e}")
        print(f"Post-norm range: [{data_3d.min():.4f}, {data_3d.max():.4f}]")
    else:
        print("⚠ Warning: global max is 0, skipping normalization")

    normalization_factor = float(global_max)

    # ── Build list of tensors ──────────────────────────────────────
    Ts = []
    for t in range(T):
        mat = data_3d[:, :, t]                          # (N, N) numpy
        tensor = torch.tensor(mat, dtype=torch.float32) # (N, N) torch
        Ts.append(tensor)

    year_labels = list(range(2000, 2000 + T))  # [2000, 2001, ..., 2023]

    print(f"\n✓ Loaded {len(Ts)} timepoints, each shape {Ts[0].shape}")
    return Ts, concepts, normalization_factor, year_labels


# =========================
# BUILD SPACE
# (same logic, cleaner docs)
# =========================
def build_space(Ts):
    N = Ts[0].shape[0]
    T_list, M_list = [], []

    for T_mat in Ts:
        T_list.append(T_mat.to(DEVICE))
        # M is just ones: no masking needed
        M_list.append(torch.ones(N, N, device=DEVICE))

    print(f"Space built: {N}×{N}")
    print(f"Mask: DISABLED (all concepts active, density was 99.42%)")
    print(f"Effective constraint: cost matrix C only")

    return T_list, M_list, N


# =========================
# UOT SOLVER (log-stabilized)
# =========================
def uot_log(C, T_ref, eps, lambda_):
    """
    Unbalanced Optimal Transport via Sinkhorn in log-domain.
    
    C      : (N,N) cost matrix
    T_ref  : (N,N) reference transport (target)
    eps    : entropic regularization
    lambda_: marginal relaxation

    Returns:
        T_hat : (N,N) transport plan
    """
    n = C.shape[0]

    # Marginals from reference
    mu = T_ref.sum(1).clamp(min=1e-10)   # source marginal (row sums)
    nu = T_ref.sum(0).clamp(min=1e-10)   # target marginal (col sums)

    log_mu  = torch.log(mu)
    log_nu  = torch.log(nu)
    log_K   = -C / eps                   # Gibbs kernel in log space
    tau     = lambda_ / (lambda_ + eps)  # softening factor

    f = torch.zeros(n, device=DEVICE)
    g = torch.zeros(n, device=DEVICE)

    for it in range(SINKHORN_ITERS):
        f_prev = f.clone()

        # Sinkhorn updates in log domain
        f = tau * (log_mu - torch.logsumexp(log_K + g[None, :], dim=1))
        g = tau * (log_nu - torch.logsumexp(log_K + f[:, None], dim=0))

        # Early stopping
        if torch.max(torch.abs(f - f_prev)).item() < SINKHORN_TOL:
            break

    return torch.exp(f[:, None] + log_K + g[None, :])


# =========================
# GAUGE FIXING
# =========================
def gauge(C):
    """
    Remove row and column means from cost matrix.
    Removes translational degrees of freedom so the
    optimization is better conditioned.
    """
    C = C - C.mean(1, keepdim=True)
    C = C - C.mean(0, keepdim=True)
    return C


# =========================
# LOSS
# (added per-timepoint tracking)
# =========================
def compute_loss(C0, E, T_list, M_list, indices,
                 eps, lambda_, reg_c0, reg_e, reg_temp,
                 return_per_t=False):
    """
    Total loss = reconstruction + regularization

    Reconstruction: ||M_t ⊙ (T_hat_t - T_t)||_F^2
    Reg E:          reg_e  * ||E_t||^2        (per timepoint)
    Reg C0:         reg_c0 * ||C0||^2         (global)
    Temporal:       reg_temp * ||E_t - E_{t-1}||^2

    Args:
        return_per_t : if True, also return dict of per-timepoint losses
    """
    loss = torch.tensor(0.0, device=DEVICE)
    per_t_losses = {}

    for t in indices:
        C_t   = C0 + E[t]
        T_hat = uot_log(C_t, T_list[t], eps, lambda_)

        recon      = torch.norm(M_list[t] * (T_hat - T_list[t]), p='fro') ** 2
        reg_e_term = reg_e * torch.norm(E[t]) ** 2

        t_loss = recon + reg_e_term
        loss   = loss + t_loss

        if return_per_t:
            per_t_losses[int(t)] = {
                'recon':  recon.item(),
                'reg_e':  reg_e_term.item(),
                'total':  t_loss.item()
            }

    # Scale batch → full dataset
    loss = loss * (len(T_list) / len(indices))

    # Global regularization on C0
    reg_c0_term = reg_c0 * torch.norm(C0) ** 2
    loss = loss + reg_c0_term

    # Temporal smoothness on E
    for t in range(1, len(E)):
        loss = loss + reg_temp * torch.norm(E[t] - E[t - 1]) ** 2

    if return_per_t:
        return loss, per_t_losses
    return loss


# =========================
# INIT C0
# =========================
def init_C0(T_list):
    """
    Initialize cost matrix from average transport plan.
    Uses -log(avg_T) as a natural cost initialization,
    clipped and gauge-fixed.
    """
    avg_T = sum(T_list) / len(T_list)
    total = avg_T.sum()

    if total < 1e-8:
        print("⚠ Near-zero input, using random init")
        return torch.randn(avg_T.shape, device=DEVICE) * 0.1

    avg_T = avg_T / total
    C0    = -torch.log(avg_T + 1e-8)
    C0    = torch.clamp(C0, 0, 20)
    C0    = gauge(C0)

    return C0.to(DEVICE)


# =========================
# OPTIMIZE
# (major changes: full loss every iter, scheduler, grad clip)
# =========================
def optimize(T_list, M_list, N,
             eps, lambda_, lr, reg_c0, reg_e, reg_temp,
             n_iter, verbose=True, print_every=100,
             save_every_iter=True):

    T = len(T_list)

    C0 = init_C0(T_list).clone().detach().requires_grad_(True)
    E  = [torch.zeros(N, N, device=DEVICE, requires_grad=True)
          for _ in range(T)]

    opt = optim.Adam([C0] + E, lr=lr)

    # ── Scheduler: verbose kwarg removed (deprecated in PyTorch 2.2+) ──
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode='min',
        factor=0.5,
        patience=200,
        min_lr=lr * 1e-3
        # verbose=verbose  <-- REMOVED
    )

    scaler = GradScaler() if USE_AMP else None

    start_time   = time.time()
    loss_history = []
    best_loss    = float('inf')
    best_state   = None
    prev_lr      = lr  # track LR changes manually

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

        # Full loss computation
        with torch.no_grad():
            if save_every_iter or (it % print_every == 0):
                full_loss, per_t = compute_loss(
                    C0, E, T_list, M_list, range(T),
                    eps, lambda_, reg_c0, reg_e, reg_temp,
                    return_per_t=True
                )
                full_loss_val = full_loss.item()
            else:
                full_loss_val = None
                per_t = {}

        # Scheduler step
        if full_loss_val is not None:
            scheduler.step(full_loss_val)

        # Best state tracking
        if full_loss_val is not None and full_loss_val < best_loss:
            best_loss  = full_loss_val
            best_state = {
                'C0':  C0.detach().cpu().clone(),
                'E':   [e.detach().cpu().clone() for e in E],
                'iter': it
            }

        # Record history
        elapsed    = time.time() - start_time
        current_lr = opt.param_groups[0]['lr']

        entry = {
            'iteration':  it,
            'batch_loss': batch_loss.item(),
            'full_loss':  full_loss_val,
            'lr':         current_lr,
            'time_s':     elapsed,
        }
        if it % print_every == 0 and per_t:
            entry['per_t_recon'] = {
                str(t): v['recon'] for t, v in per_t.items()
            }
        loss_history.append(entry)

        # Console output
        if verbose and it % print_every == 0:
            iter_per_sec = (it + 1) / elapsed if elapsed > 0 else 0
            eta          = (n_iter - it - 1) / iter_per_sec if iter_per_sec > 0 else 0

            # Manual LR change notification
            lr_str = f"LR: {current_lr:.2e}"
            if current_lr < prev_lr:
                lr_str += f" ↓ (was {prev_lr:.2e})"
                prev_lr = current_lr

            print(f"Iter {it:5d}/{n_iter} | "
                  f"Batch: {batch_loss.item():.6f} | "
                  f"Full: {full_loss_val:.6f} | "
                  f"{lr_str} | "
                  f"Best: {best_loss:.6f} | "
                  f"Speed: {iter_per_sec:.1f} it/s | "
                  f"ETA: {eta:.0f}s")

    # Final loss
    with torch.no_grad():
        final_loss_tensor, _ = compute_loss(
            C0, E, T_list, M_list, range(T),
            eps, lambda_, reg_c0, reg_e, reg_temp,
            return_per_t=True
        )
    final_loss = final_loss_tensor.item()
    total_time = time.time() - start_time

    if verbose:
        print(f"\n{'='*65}")
        print(f"Optimization done in {total_time:.1f}s ({total_time/60:.2f} min)")
        print(f"Final loss:  {final_loss:.6f}")
        print(f"Best loss:   {best_loss:.6f}  (iter {best_state['iter']})")
        print(f"{'='*65}\n")

    return (best_state['C0'],
            best_state['E'],
            best_loss,
            loss_history)

# =========================
# SAVE
# (now concept-aware)
# =========================
def save_results(C0, E_list, folder,
                 concepts=None,
                 year_labels=None,
                 hyperparams=None,
                 loss_history=None,
                 normalization_factor=None):
    """
    Save all results.  Now includes:
      - concept-indexed DataFrames (rows/cols = concept names)
      - full loss history CSV (one row per iteration)
      - concept-level flow analysis
    """
    os.makedirs(folder, exist_ok=True)

    idx   = concepts if concepts else None
    T_val = len(E_list)

    # ── C0 baseline ───────────────────────────────────────────────
    df_C0 = pd.DataFrame(C0.numpy(), index=idx, columns=idx)
    df_C0.to_csv(f"{folder}/C0_baseline.csv")
    print(f"  Saved C0_baseline.csv  shape {df_C0.shape}")

    # ── Per-timepoint E and C matrices ────────────────────────────
    C_stack = []
    E_stack = []
    for t, e in enumerate(E_list):
        label = str(year_labels[t]) if year_labels else f"{t:02d}"

        df_E = pd.DataFrame(e.numpy(), index=idx, columns=idx)
        df_E.to_csv(f"{folder}/E_{label}.csv")

        df_C = pd.DataFrame((C0 + e).numpy(), index=idx, columns=idx)
        df_C.to_csv(f"{folder}/C_{label}.csv")

        C_stack.append((C0 + e).numpy())
        E_stack.append(e.numpy())

    print(f"  Saved E_YEAR.csv and C_YEAR.csv for {T_val} years")

    # ── NumPy archive (efficient storage) ─────────────────────────
    np.savez(f"{folder}/results.npz",
             C0=C0.numpy(),
             E=np.stack(E_stack),          # (T, N, N)
             C_full=np.stack(C_stack),     # (T, N, N)
             concepts=np.array(concepts if concepts else []),
             years=np.array(year_labels if year_labels else []))
    print(f"  Saved results.npz")

    # ── Loss history ──────────────────────────────────────────────
    if loss_history:
        # Flatten per_t_recon dict columns for CSV
        flat_history = []
        for entry in loss_history:
            row = {k: v for k, v in entry.items()
                   if k != 'per_t_recon'}
            per_t = entry.get('per_t_recon', {})
            for t_key, recon_val in per_t.items():
                year = (year_labels[int(t_key)]
                        if year_labels else int(t_key))
                row[f'recon_{year}'] = recon_val
            flat_history.append(row)

        df_loss = pd.DataFrame(flat_history)
        df_loss.to_csv(f"{folder}/loss_history.csv", index=False)

        with open(f"{folder}/loss_history.json", 'w') as f:
            json.dump(loss_history, f, indent=2)

        print(f"  Saved loss_history.csv  ({len(df_loss)} rows = 1 per iter)")

    # ── Concept-level flow analysis ───────────────────────────────
    if concepts:
        _save_concept_analysis(
            C0, E_list, concepts, year_labels, folder
        )

    # ── Metadata ──────────────────────────────────────────────────
    with open(f"{folder}/meta.txt", 'w') as f:
        f.write(f"Shape: {C0.shape}\n")
        f.write(f"Timepoints: {T_val}\n")
        if year_labels:
            f.write(f"Years: {year_labels[0]}–{year_labels[-1]}\n")
        if normalization_factor is not None:
            f.write(f"Normalization factor: {normalization_factor:.6e}\n")
        if hyperparams:
            f.write(f"\nOptimized Hyperparameters:\n")
            for k, v in hyperparams.items():
                f.write(f"  {k}: {v:.6e}\n")
        f.write(f"\nC0 norm: {torch.norm(C0).item():.4f}\n")
        f.write(f"\nE norms per year:\n")
        for t, e in enumerate(E_list):
            label = str(year_labels[t]) if year_labels else f"t={t:02d}"
            f.write(f"  {label}: {torch.norm(e).item():.4f}\n")

    print(f"\n✓ All results saved to {folder}/")


def _save_concept_analysis(C0, E_list, concepts, year_labels, folder):
    """
    Derive interpretable concept-level statistics and save.

    Saved files:
      concept_outflow.csv   : how much each concept "sends" per year
      concept_inflow.csv    : how much each concept "receives" per year
      concept_volatility.csv: std of E[concept, :, t] across years
      top_concept_pairs.csv : top 50 concept pairs by mean |C_t|
    """
    T    = len(E_list)
    N    = C0.shape[0]
    years = year_labels if year_labels else list(range(T))

    # C_full[t] = C0 + E[t],  shape (N, N)
    C_full = np.stack([(C0 + e).numpy() for e in E_list])   # (T, N, N)
    # Convert cost to "affinity" (low cost = high flow)
    A_full = -C_full   # (T, N, N)  higher = more flow

    # ── Per-year outflow (row sums of affinity) ───────────────────
    outflow = A_full.sum(axis=2)   # (T, N)
    df_out  = pd.DataFrame(outflow.T, index=concepts, columns=years)
    df_out.to_csv(f"{folder}/concept_outflow.csv")

    # ── Per-year inflow (col sums of affinity) ────────────────────
    inflow  = A_full.sum(axis=1)   # (T, N)
    df_in   = pd.DataFrame(inflow.T, index=concepts, columns=years)
    df_in.to_csv(f"{folder}/concept_inflow.csv")

    # ── Concept volatility across time ────────────────────────────
    E_stack = np.stack([e.numpy() for e in E_list])   # (T, N, N)
    # Row-wise std: how variable is each concept's outflow deviation?
    row_std = E_stack.std(axis=0).mean(axis=1)        # (N,)
    df_vol  = pd.DataFrame({
        'concept':    concepts,
        'E_row_std':  row_std,
        'rank':       pd.Series(row_std).rank(ascending=False).values
    }).sort_values('E_row_std', ascending=False)
    df_vol.to_csv(f"{folder}/concept_volatility.csv", index=False)

    # ── Top concept pairs by mean affinity ────────────────────────
    mean_A = A_full.mean(axis=0)   # (N, N)
    pairs  = []
    for i in range(N):
        for j in range(N):
            if i != j:
                pairs.append({
                    'source':     concepts[i],
                    'target':     concepts[j],
                    'mean_affinity': mean_A[i, j],
                    'abs_affinity':  abs(mean_A[i, j])
                })
    df_pairs = (pd.DataFrame(pairs)
                  .sort_values('abs_affinity', ascending=False)
                  .head(100))
    df_pairs.to_csv(f"{folder}/top_concept_pairs.csv", index=False)

    print(f"  Saved concept_outflow.csv, concept_inflow.csv, "
          f"concept_volatility.csv, top_concept_pairs.csv")


# =========================
# CUSTOM LOGGER (BO)
# =========================
class CustomLogger:
    def __init__(self, filepath):
        self.filepath  = filepath
        self.data      = []
        self.start_time = time.time()

    def log(self, iteration, params, target):
        entry = {
            'iteration':    iteration,
            'params':       params,
            'target':       target,
            'loss':         -target,
            'time_elapsed': time.time() - self.start_time
        }
        self.data.append(entry)
        with open(self.filepath, 'w') as f:
            json.dump(self.data, f, indent=2)


# =========================
# BAYESIAN OPTIMIZATION
# =========================
bo_iteration_counter = 0
bo_start_time        = None
T_list_global        = None
M_list_global        = None
N_global             = None


def objective_function(log_eps, log_lambda, log_lr,
                       log_reg_c0, log_reg_e, log_reg_temp):
    global bo_iteration_counter, bo_start_time

    if bo_start_time is None:
        bo_start_time = time.time()

    bo_iteration_counter += 1

    eps      = 10 ** log_eps
    lambda_  = 10 ** log_lambda
    lr       = 10 ** log_lr
    reg_c0   = 10 ** log_reg_c0
    reg_e    = 10 ** log_reg_e
    reg_temp = 10 ** log_reg_temp

    elapsed = time.time() - bo_start_time
    print(f"\n{'='*60}")
    print(f"BO Iter {bo_iteration_counter} | Elapsed: {elapsed:.0f}s")
    print(f"  eps={eps:.3e} lambda={lambda_:.3e} lr={lr:.3e}")
    print(f"  reg_c0={reg_c0:.3e} reg_e={reg_e:.3e} reg_temp={reg_temp:.3e}")

    try:
        _, _, final_loss, _ = optimize(
            T_list_global, M_list_global, N_global,
            eps, lambda_, lr, reg_c0, reg_e, reg_temp,
            n_iter=N_ITER_BO,
            verbose=True,
            print_every=250,
            save_every_iter=False   # speed up BO: don't save every iter
        )
        print(f"  → loss: {final_loss:.6f}")
        return -final_loss

    except Exception as e:
        print(f"❌ BO eval failed: {e}")
        import traceback; traceback.print_exc()
        return -1e10


def run_bayesian_optimization(T_list, M_list, N,
                               n_iter=25, init_points=5):
    global T_list_global, M_list_global, N_global
    global bo_iteration_counter, bo_start_time

    T_list_global        = T_list
    M_list_global        = M_list
    N_global             = N
    bo_iteration_counter = 0
    bo_start_time        = time.time()

    pbounds = {
        'log_eps':      (-1.7, -0.7),   # EPS:      0.02 – 0.20
        'log_lambda':   ( 0.5,  1.5),   # LAMBDA:   3.16 – 31.6
        'log_lr':       (-3.3, -2.3),   # LR:       0.0005 – 0.005
        'log_reg_c0':   (-3.5, -2.5),   # REG_C0:   3e-4 – 3e-3
        'log_reg_e':    (-3.5, -2.0),   # REG_E:    3e-4 – 0.01
        'log_reg_temp': (-4.0, -2.5),   # REG_TEMP: 1e-4 – 3e-3
    }

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
    logger  = CustomLogger("./bo_logs.json")

    print(f"\n{'='*60}")
    print(f"Bayesian Optimization:  {init_points} init + {n_iter} BO iters")
    print(f"{'='*60}")

    iteration = 0

    # Phase 1: random init
    print(f"\n{'#'*60}")
    print(f"PHASE 1: Random Initialization ({init_points} evals)")
    print(f"{'#'*60}")
    for i in range(init_points):
        print(f"\n>>> Random init {i+1}/{init_points}")
        next_point = {k: np.random.uniform(v[0], v[1])
                      for k, v in pbounds.items()}
        optimizer.probe(params=next_point, lazy=False)
        logger.log(iteration, next_point, optimizer.space.target[-1])
        print(f"Best so far: {-optimizer.max['target']:.6f}")
        iteration += 1

    # Phase 2: BO
    print(f"\n{'#'*60}")
    print(f"PHASE 2: Bayesian Optimization ({n_iter} evals)")
    print(f"{'#'*60}")
    for i in range(n_iter):
        print(f"\n>>> BO iter {i+1}/{n_iter}")
        next_point = optimizer.suggest(utility)
        optimizer.probe(params=next_point, lazy=False)
        logger.log(iteration, next_point, optimizer.space.target[-1])
        print(f"Best so far: {-optimizer.max['target']:.6f}")
        iteration += 1

    best_params = optimizer.max['params']
    best_loss   = -optimizer.max['target']

    best_hyperparams = {
        'EPS':      10 ** best_params['log_eps'],
        'LAMBDA':   10 ** best_params['log_lambda'],
        'LR':       10 ** best_params['log_lr'],
        'REG_C0':   10 ** best_params['log_reg_c0'],
        'REG_E':    10 ** best_params['log_reg_e'],
        'REG_TEMP': 10 ** best_params['log_reg_temp'],
    }

    total_bo_time = time.time() - bo_start_time
    print(f"\n{'#'*60}")
    print(f"BO complete in {total_bo_time:.0f}s ({total_bo_time/60:.1f} min)")
    for k, v in best_hyperparams.items():
        print(f"  {k}: {v:.6e}")
    print(f"Best loss: {best_loss:.6f}")
    print(f"{'#'*60}\n")

    return best_hyperparams, best_loss


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    overall_start = time.time()

    TENSOR_PATH   = "ot_filtered_tensor.npy"
    CONCEPTS_PATH = "concept_list.csv"
    OUTPUT_FOLDER = "results_bayesopt"

    # ── Load data ─────────────────────────────────────────────────
    try:
        Ts, concepts, norm_factor, year_labels = load_from_npy(
            TENSOR_PATH, CONCEPTS_PATH
        )
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ Load error: {e}")
        raise

    # ── Build space ───────────────────────────────────────────────
    T_list, M_list, N = build_space(Ts)

    # ── Bayesian Optimization ─────────────────────────────────────
    print("\n" + "="*60)
    print("Bayesian Hyperparameter Search")
    print("="*60)

    best_hyperparams, best_bo_loss = run_bayesian_optimization(
        T_list, M_list, N,
        n_iter=35,
        init_points=10
    )

    # ── Final Optimization ────────────────────────────────────────
    print("\n" + "#"*60)
    print("FINAL OPTIMIZATION WITH BEST HYPERPARAMETERS")
    print(f"Running {N_ITER_FINAL} iterations, print every 50")
    print("#"*60)

    final_start = time.time()

    C0, E, final_loss, loss_history = optimize(
        T_list, M_list, N,
        eps      = best_hyperparams['EPS'],
        lambda_  = best_hyperparams['LAMBDA'],
        lr       = best_hyperparams['LR'],
        reg_c0   = best_hyperparams['REG_C0'],
        reg_e    = best_hyperparams['REG_E'],
        reg_temp = best_hyperparams['REG_TEMP'],
        n_iter   = N_ITER_FINAL,
        verbose  = True,
        print_every   = 50,
        save_every_iter = True   # record full loss every iteration
    )

    final_time = time.time() - final_start
    total_time = time.time() - overall_start

    # ── Save everything ───────────────────────────────────────────
    save_results(
        C0, E, OUTPUT_FOLDER,
        concepts             = concepts,
        year_labels          = year_labels,
        hyperparams          = best_hyperparams,
        loss_history         = loss_history,
        normalization_factor = norm_factor
    )

    # Save best hyperparameters summary
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    with open(f"{OUTPUT_FOLDER}/best_hyperparameters.txt", 'w') as f:
        f.write("Best Hyperparameters from Bayesian Optimization\n")
        f.write("="*50 + "\n\n")
        for k, v in best_hyperparams.items():
            f.write(f"{k}: {v:.6e}\n")
        f.write(f"\nBest BO loss:              {best_bo_loss:.6f}\n")
        f.write(f"Final optimization loss:   {final_loss:.6f}\n")
        f.write(f"Final optimization time:   {final_time:.1f}s\n")
        f.write(f"Total time:                {total_time:.1f}s\n")
        f.write(f"Normalization factor:      {norm_factor:.6e}\n")

    # ── Final summary ─────────────────────────────────────────────
    print("\n" + "#"*60)
    print("✓  ALL DONE")
    print("#"*60)
    print(f"Total time:  {total_time:.0f}s  "
          f"({total_time/60:.1f} min, {total_time/3600:.2f} h)")
    print(f"\nOutput files in  {OUTPUT_FOLDER}/:")
    print("  C0_baseline.csv          — baseline cost matrix (concepts × concepts)")
    print("  C_YEAR.csv               — cost matrix per year (24 files)")
    print("  E_YEAR.csv               — deviation per year  (24 files)")
    print("  results.npz              — all arrays in one file")
    print("  loss_history.csv         — loss at every iteration")
    print("  concept_outflow.csv      — per-concept outflow over time")
    print("  concept_inflow.csv       — per-concept inflow over time")
    print("  concept_volatility.csv   — which concepts changed most")
    print("  top_concept_pairs.csv    — strongest concept→concept flows")
    print("  best_hyperparameters.txt — BO result")
    print("  bo_logs.json             — full BO log")
    print("#"*60)
