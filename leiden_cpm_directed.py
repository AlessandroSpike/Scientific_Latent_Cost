import sys
import numpy as np
import pandas as pd

import igraph as ig
import leidenalg as la

from pathlib import Path


def main():

    adjacency_file = sys.argv[1]
    gamma_file = sys.argv[2]
    output_directory = Path(sys.argv[3])

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # LOAD DIRECTED ADJACENCY
    # ------------------------------------------------------------

    A = np.loadtxt(
        adjacency_file,
        delimiter=","
    )

    gamma_values = np.loadtxt(
        gamma_file,
        delimiter=",",
        ndmin=1
    )

    N = A.shape[0]

    np.fill_diagonal(A, 0)

    # ------------------------------------------------------------
    # DIRECTED IGRAPH NETWORK
    # ------------------------------------------------------------

    graph = ig.Graph.Weighted_Adjacency(
        A.tolist(),
        mode="directed",
        attr="weight",
        loops=False
    )

    print("Directed graph created.")

    print("Nodes:", graph.vcount())

    print("Edges:", graph.ecount())

    print("Directed:", graph.is_directed())

    # ------------------------------------------------------------
    # LEIDEN CPM RESOLUTION SWEEP
    # ------------------------------------------------------------

    summaries = []

    partitions = []

    for gamma in gamma_values:

        best_quality = -np.inf

        best_membership = None

        # Repeated independent optimizations
        for run in range(100):

            partition = la.find_partition(
                graph,
                la.CPMVertexPartition,
                weights="weight",
                resolution_parameter=float(gamma),
                n_iterations=-1,
                seed=42 + run
            )

            quality = partition.quality()

            if quality > best_quality:

                best_quality = quality

                best_membership = np.array(
                    partition.membership
                )

        # --------------------------------------------------------
        # COMMUNITY STATISTICS
        # --------------------------------------------------------

        labels, sizes = np.unique(
            best_membership,
            return_counts=True
        )

        K = len(labels)

        singletons = np.sum(sizes == 1)

        singleton_percentage = (
            100 * singletons / N
        )

        print(
            f"Gamma={gamma:.3f} | "
            f"K={K} | "
            f"Singletons={singletons} | "
            f"Singleton%={singleton_percentage:.2f}"
        )

        summaries.append({

            "Gamma": gamma,

            "K": K,

            "Quality": best_quality,

            "Singletons": singletons,

            "SingletonPct": singleton_percentage,

            "MinSize": np.min(sizes),

            "MaxSize": np.max(sizes),

            "MedianSize": np.median(sizes)

        })

        partitions.append(
            best_membership
        )

    # ------------------------------------------------------------
    # EXPORT RESULTS
    # ------------------------------------------------------------

    summary = pd.DataFrame(
        summaries
    )

    summary.to_csv(
        output_directory / "leiden_summary.csv",
        index=False
    )

    partition_matrix = np.column_stack(
        partitions
    )

    np.savetxt(
        output_directory / "leiden_partitions.csv",
        partition_matrix,
        delimiter=",",
        fmt="%d"
    )

    print("\nLeiden-CPM completed successfully.")


if __name__ == "__main__":
    main()
