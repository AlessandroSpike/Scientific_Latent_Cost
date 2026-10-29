import numpy as np
import pandas as pd
from scipy.io import savemat

# =========================
# LOAD FROM .NPY + .CSV
# =========================
def load_from_npy(
        tensor_path="ot_filtered_tensor.npy",
        concepts_path="concept_list.csv"):

    print(f"\nLoading tensor from: {tensor_path}")
    print(f"Loading concepts from: {concepts_path}")

    # Load tensor
    data_3d = np.load(tensor_path)

    print(f"Raw tensor shape: {data_3d.shape}")
    print(f"Raw dtype:        {data_3d.dtype}")

    if data_3d.ndim != 3:
        raise ValueError(
            f"Expected 3D tensor, got {data_3d.ndim}D"
        )

    N1, N2, T = data_3d.shape

    if N1 != N2:
        raise ValueError(
            f"Tensor must be square. Got {N1} x {N2}"
        )

    # Load concepts
    concept_df = pd.read_csv(concepts_path)

    if "concept" not in concept_df.columns:
        raise ValueError(
            f"CSV must contain column 'concept'"
        )

    concepts = concept_df["concept"].tolist()

    if len(concepts) != N1:
        raise ValueError(
            f"{len(concepts)} concepts but tensor dimension is {N1}"
        )

    # Create list of yearly matrices
    Ts = [data_3d[:, :, t] for t in range(T)]

    # Metadata
    normalization_factor = 1.0
    year_labels = np.arange(2000, 2000 + T)

    return (
        data_3d,
        Ts,
        concepts,
        normalization_factor,
        year_labels
    )


# =========================
# SAVE MATLAB FILE
# =========================
def save_to_mat(
        filename,
        data_3d,
        Ts,
        concepts,
        normalization_factor,
        year_labels):

    mat_dict = {
        "data_3d": data_3d,
        "Ts": np.array(Ts, dtype=object),  # MATLAB cell array
        "concepts": np.array(concepts, dtype=object),
        "normalization_factor": normalization_factor,
        "year_labels": year_labels
    }

    savemat(filename, mat_dict)

    print(f"Saved: {filename}")


# =========================
# MAIN
# =========================
data_3d, Ts, concepts, normalization_factor, year_labels = load_from_npy(
    "ot_filtered_tensor.npy",
    "concept_list.csv"
)

save_to_mat(
    "openalex_tensor.mat",
    data_3d,
    Ts,
    concepts,
    normalization_factor,
    year_labels
)
