import torch
import torch.optim as optim
import scipy.io
import pandas as pd
import os
from torch.cuda.amp import autocast, GradScaler

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPS = 0.01
LAMBDA = 20.0

LR = 5e-3
N_ITER = 5000

REG_C0 = 1e-3
REG_E = 1e-3
REG_TEMP = 5e-3

SINKHORN_ITERS = 80
SINKHORN_TOL = 1e-6
BATCH_SIZE = 5

USE_AMP = torch.cuda.is_available()

print(f"Device: {DEVICE}")


# =========================
# LOAD FROM .MAT
# =========================
def load_from_mat(filepath, var_name='Data'):
    """
    Load 3D array from .mat file
    
    Args:
        filepath: path to .mat file (e.g., 'Data.mat')
        var_name: variable name in .mat file (default: 'Data')
    
    Returns:
        List of 2D torch tensors, one per timepoint
    """
    print(f"Loading {filepath}...")
    
    # Load .mat file
    mat_contents = scipy.io.loadmat(filepath)
    
    # Extract the 3D array
    if var_name not in mat_contents:
        print(f"Available variables: {list(mat_contents.keys())}")
        raise ValueError(f"Variable '{var_name}' not found in .mat file")
    
    data_3d = mat_contents[var_name]
    
    print(f"Loaded array shape: {data_3d.shape}")
    
    # Convert to list of 2D tensors
    # Assuming shape is (N, N, T) where T=16
    if len(data_3d.shape) != 3:
        raise ValueError(f"Expected 3D array, got shape {data_3d.shape}")
    
    N1, N2, T = data_3d.shape
    
    if N1 != N2:
        print(f"⚠ Warning: Non-square matrices ({N1}x{N2})")
    
    # Extract each timepoint as separate matrix
    Ts = []
    for t in range(T):
        matrix = data_3d[:, :, t]
        tensor = torch.tensor(matrix, dtype=torch.float32)
        Ts.append(tensor)
    
    print(f"✓ Extracted {len(Ts)} timepoints of shape {Ts[0].shape}")
    
    # Print some statistics
    print(f"  Data range: [{data_3d.min():.4f}, {data_3d.max():.4f}]")
    print(f"  Mean: {data_3d.mean():.4f}, Std: {data_3d.std():.4f}")
    print(f"  Sparsity: {(data_3d == 0).mean():.2%} zeros")
    
    return Ts


# =========================
# BUILD SPACE
# =========================
def build_space(Ts):
    """
    Since all matrices are already 207x207, no padding needed!
    Just create masks and move to device.
    """
    N = Ts[0].shape[0]
    
    # Verify all same size
    for i, T in enumerate(Ts):
        if T.shape != (N, N):
            raise ValueError(f"Timepoint {i} has inconsistent shape {T.shape}")
    
    T_list, M_list = [], []

    for T in Ts:
        # Active node mask
        row_active = (T.sum(1) > 0).float()
        col_active = (T.sum(0) > 0).float()
        active = torch.clamp(row_active + col_active, 0, 1)
        
        M = active[:, None] * active[None, :]

        T_list.append(T.to(DEVICE))
        M_list.append(M.to(DEVICE))

    density = sum(m.mean() for m in M_list) / len(M_list)
    print(f"Common dim: {N}x{N}, avg mask density: {density:.2%}")
    
    return T_list, M_list, N


# =========================
# UOT SOLVER
# =========================
def uot_log(C, T_ref):
    n = C.shape[0]
    mu = T_ref.sum(1).clamp(min=1e-10)
    nu = T_ref.sum(0).clamp(min=1e-10)

    log_mu = torch.log(mu)
    log_nu = torch.log(nu)
    log_K = -C / EPS
    tau = LAMBDA / (LAMBDA + EPS)

    f = torch.zeros(n, device=DEVICE)
    g = torch.zeros(n, device=DEVICE)

    for it in range(SINKHORN_ITERS):
        f_prev = f.clone()
        f = tau * (log_mu - torch.logsumexp(log_K + g[None, :], dim=1))
        g = tau * (log_nu - torch.logsumexp(log_K + f[:, None], dim=0))
        
        if torch.max(torch.abs(f - f_prev)) < SINKHORN_TOL:
            break

    return torch.exp(f[:, None] + log_K + g[None, :])


# =========================
# GAUGE
# =========================
def gauge(C):
    C = C - C.mean(1, keepdim=True)
    C = C - C.mean(0, keepdim=True)
    return C


# =========================
# LOSS
# =========================
def compute_loss(C0, E, T_list, M_list, indices):
    loss = 0

    for t in indices:
        C_t = C0 + E[t]
        T_hat = uot_log(C_t, T_list[t])
        
        recon = torch.norm(M_list[t] * (T_hat - T_list[t]), p='fro')**2
        reg_e = REG_E * torch.norm(E[t])**2
        
        loss += recon + reg_e

    # Scale to full dataset
    loss = loss * (len(T_list) / len(indices))
    
    # Global regularization
    loss += REG_C0 * torch.norm(C0)**2
    
    # Temporal smoothness
    for t in range(1, len(E)):
        loss += REG_TEMP * torch.norm(E[t] - E[t-1])**2

    return loss


# =========================
# INIT
# =========================
def init_C0(T_list):
    avg_T = sum(T_list) / len(T_list)
    total = avg_T.sum()
    
    if total < 1e-8:
        print("⚠ Input matrices near-zero, using random init")
        return torch.randn(avg_T.shape, device=DEVICE) * 0.1
    
    avg_T = avg_T / total
    C0 = -torch.log(avg_T + 1e-8)
    C0 = torch.clamp(C0, 0, 20)
    C0 = gauge(C0)
    
    return C0.to(DEVICE)


# =========================
# OPTIMIZE
# =========================
def optimize(T_list, M_list, N):
    T = len(T_list)

    C0 = init_C0(T_list).clone().detach().requires_grad_(True)
    E = [torch.zeros(N, N, device=DEVICE, requires_grad=True) for _ in range(T)]

    opt = optim.Adam([C0] + E, lr=LR)
    scaler = GradScaler() if USE_AMP else None

    for it in range(N_ITER):
        opt.zero_grad()
        
        idx = torch.randperm(T)[:min(BATCH_SIZE, T)]

        if USE_AMP:
            with autocast():
                loss = compute_loss(C0, E, T_list, M_list, idx)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss = compute_loss(C0, E, T_list, M_list, idx)
            loss.backward()
            opt.step()

        # Gauge fixing
        with torch.no_grad():
            C0.copy_(gauge(C0))
            for i in range(T):
                E[i].copy_(gauge(E[i]))

        # Logging
        if it % 100 == 0:
            with torch.no_grad():
                full_loss = compute_loss(C0, E, T_list, M_list, range(T))
            print(f"Iter {it:4d} | Batch: {loss.item():.4f} | Full: {full_loss.item():.4f}")

    return C0.detach().cpu(), [e.detach().cpu() for e in E]


# =========================
# SAVE
# =========================
def save_results(C0, E_list, folder):
    os.makedirs(folder, exist_ok=True)

    # Save as CSV
    pd.DataFrame(C0.numpy()).to_csv(f"{folder}/C0_baseline.csv", index=False, header=False)

    for t, e in enumerate(E_list):
        pd.DataFrame(e.numpy()).to_csv(f"{folder}/E_{t:02d}.csv", index=False, header=False)
        pd.DataFrame((C0 + e).numpy()).to_csv(f"{folder}/C_{t:02d}.csv", index=False, header=False)

    # Also save as .mat for MATLAB compatibility
    mat_dict = {
        'C0': C0.numpy(),
        'E': torch.stack(E_list).numpy(),  # Shape: (16, 207, 207)
        'C_full': torch.stack([C0 + e for e in E_list]).numpy()
    }
    scipy.io.savemat(f"{folder}/results.mat", mat_dict)

    # Metadata
    with open(f"{folder}/meta.txt", 'w') as f:
        f.write(f"Shape: {C0.shape}\n")
        f.write(f"Timepoints: {len(E_list)}\n")
        f.write(f"Params: EPS={EPS}, LAMBDA={LAMBDA}, REG_C0={REG_C0}, REG_E={REG_E}, REG_TEMP={REG_TEMP}\n")
        f.write(f"C0 norm: {torch.norm(C0).item():.4f}\n")
        f.write(f"E norms per timepoint:\n")
        for t, e in enumerate(E_list):
            f.write(f"  t={t:02d}: {torch.norm(e).item():.4f}\n")

    print(f"✓ Saved to {folder}/")
    print(f"  - CSV files: C0_baseline.csv, E_*.csv, C_*.csv")
    print(f"  - MATLAB file: results.mat")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    
    # Load from .mat file
    mat_file = "Data.mat"
    var_name = "Data"  # Change if your variable has a different name
    
    try:
        Ts = load_from_mat(mat_file, var_name)
    except FileNotFoundError:
        print(f"❌ File '{mat_file}' not found")
        exit(1)
    except Exception as e:
        print(f"❌ Error loading .mat file: {e}")
        exit(1)

    # Build common space
    T_list, M_list, N = build_space(Ts)
    
    # Optimize
    print("\n" + "="*50)
    print("Starting optimization...")
    print("="*50)
    C0, E = optimize(T_list, M_list, N)
    
    # Save results
    save_results(C0, E, "results")

    print("\n✓ Done")
