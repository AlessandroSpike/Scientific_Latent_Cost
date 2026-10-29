%% ═══════════════════════════════════════════════════════════════════════
%  INCREMENTAL VALUE ANALYSIS: C0 BASELINE vs. RAW CITATION MATRIX (P_bar)
%
%  Addresses Reviewer Concern:
%  1. Quantifies structural divergence between raw P_bar and latent C0/A0
%  2. Tests removal of publication volume biases (Scale-Invariance)
%  3. Compares community modularity and cluster purity
%  4. Identifies representative concept pairs (Volume Artifacts vs. Hidden Affinities)
%% ═══════════════════════════════════════════════════════════════════════

clearvars -except config C0 E C_full concepts N outflow_table inflow_table;
clc; close all;

%% ─── 0. AUTO-INITIALIZE CONFIG & LOAD WORKSPACE VARIABLES IF MISSING ───
if ~exist('config', 'var') || ~isfield(config, 'T')
    config = struct();
    config.results_folder = 'results_bayesopt';
    config.output_folder  = 'matlab_analysis';
    config.figures_folder = fullfile(config.output_folder, 'figures');
    config.tables_folder  = fullfile(config.output_folder, 'tables');
    config.years = 2000:2023;
    config.T = length(config.years);
end

if ~exist(config.output_folder, 'dir'), mkdir(config.output_folder); end
if ~exist(config.tables_folder, 'dir'), mkdir(config.tables_folder); end

% Load C0 and concept names if missing
if ~exist('C0', 'var') || ~exist('concepts', 'var')
    fprintf('Loading C0_baseline.csv...\n');
    C0_table = readtable(fullfile(config.results_folder, 'C0_baseline.csv'));
    concepts = C0_table.Properties.VariableNames(2:end);
    N = length(concepts);
    C0 = table2array(C0_table(:, 2:end));
else
    N = length(concepts);
end

% Load C_full if missing
if ~exist('C_full', 'var')
    fprintf('Loading temporal cost matrices C_t...\n');
    C_full = zeros(N, N, config.T);
    for t = 1:config.T
        fname_C = sprintf('C_%d.csv', config.years(t));
        C_table = readtable(fullfile(config.results_folder, fname_C));
        C_full(:, :, t) = table2array(C_table(:, 2:end));
    end
end

fprintf('\n╔════════════════════════════════════════════════════════╗\n');
fprintf('║   INCREMENTAL VALUE: C0 BASELINE vs. RAW CITATION MATRIX ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n\n');

%% ─── 1. Construct Historical Raw Citation Matrix (P_bar) ─────────────
fprintf('═══ STEP 1: CONSTRUCTING HISTORICAL RAW CITATION MATRIX ═══\n');

load openalex_tensor.mat

P_bar = mean(data_3d, 3);
P_bar(1:N+1:end) = 0;  % Zero diagonal

% Compute baseline Gibbs Affinity matrix A0
epsilon_opt = 1;
A0 = exp(-C0 / epsilon_opt);
A0(1:N+1:end) = 0;  % Zero diagonal

% Normalized versions for scale-free comparison
P_bar_norm = P_bar / sum(P_bar(:));
A0_norm    = A0    / sum(A0(:));

fprintf('  Raw citation matrix P_bar constructed (%d x %d)\n', N, N);
fprintf('  Gibbs affinity A0 = exp(-C0/%.2f) computed\n\n', epsilon_opt);

%% ─── 2. Global Structural & Topological Divergence ─────────────────
fprintf('═══ STEP 2: GLOBAL STRUCTURAL DIVERGENCE ═══\n');

% Vectorize off-diagonal entries only for fair comparison
mask = ~eye(N, 'logical');
p_vec = P_bar(mask);
a_vec = A0(mask);

spearman_rho = corr(p_vec, a_vec, 'type', 'Spearman');
pearson_r    = corr(p_vec, a_vec, 'type', 'Pearson');
frob_dist    = norm(A0_norm - P_bar_norm, 'fro');

% Subspace alignment via symmetrized eigendecomposition (top 5 axes)
K = 5;
[V_raw, ~] = eigs((P_bar ), K, 'largestreal');
[V_uot, ~] = eigs((A0 ), K, 'largestreal');

subspace_angles = zeros(K, 1);
for k = 1:K
    cosval = abs(V_raw(:,k)' * V_uot(:,k));
    cosval = min(1, cosval);  % numerical safety
    subspace_angles(k) = acosd(cosval);
end

fprintf('  Spearman Rank Correlation (P_bar vs A0):     %.4f\n', spearman_rho);
fprintf('  Pearson Linear Correlation (P_bar vs A0):    %.4f\n', pearson_r);
fprintf('  Normalized Frobenius Distance (A0 vs P_bar): %.4f\n', frob_dist);
fprintf('  PC1 Axis Divergence Angle:                   %.2f degrees\n', subspace_angles(1));
fprintf('  Mean Subspace Angle (Top %d PCs):             %.2f degrees\n\n', K, mean(subspace_angles));

%% ─── 3. Volume Bias Removal Test (Scale-Invariance) ────────────────
fprintf('═══ STEP 3: TESTING VOLUME BIAS REMOVAL ═══\n');

% Total field volume proxy: sum of all incoming + outgoing raw citations per concept
concept_total_vol    = sum(P_bar, 2) + sum(P_bar, 1)';
raw_node_strength    = sum(P_bar, 2);
latent_node_strength = sum(A0,    2);

r2_raw_vol    = corr(raw_node_strength,    concept_total_vol)^2;
r2_latent_vol = corr(latent_node_strength, concept_total_vol)^2;

% Volume bias reduction: how much R2 drops when using latent vs raw
bias_reduction_pct = max(0, (1 - r2_latent_vol / r2_raw_vol)) * 100;

fprintf('  R2 (Raw Node Strength vs Total Field Volume):    %.4f\n', r2_raw_vol);
fprintf('  R2 (Latent Strength vs Total Field Volume):      %.4f\n', r2_latent_vol);
fprintf('  Volume Bias Reduction:                           %.2f%%\n\n', bias_reduction_pct);

%% ─── 4. Community Modularity & Cluster Purity Comparison ────────────
fprintf('═══ STEP 4: COMMUNITY MODULARITY & CLUSTER PURITY ═══\n');

% Symmetrize for undirected community detection
W_raw = (P_bar );
W_uot = (A0  );

% Normalize rows to avoid scale effects in spectral embedding
W_raw_norm = W_raw ./ (sum(W_raw, 2) + 1e-12);
W_uot_norm = W_uot ./ (sum(W_uot, 2) + 1e-12);

% Use BCT community_louvain if available, else spectral clustering fallback
use_bct = exist('community_louvain', 'file') == 2;

if use_bct
    % BCT: run Louvain multiple times and take best
    n_runs = 20;
    Q_raw_best = -Inf; Q_uot_best = -Inf;
    comm_raw_best = ones(N,1); comm_uot_best = ones(N,1);
    for r = 1:n_runs
        [cr, Qr] = community_louvain(W_raw, 1);
        [cu, Qu] = community_louvain(W_uot, 1);
        if Qr > Q_raw_best, Q_raw_best = Qr; comm_raw_best = cr; end
        if Qu > Q_uot_best, Q_uot_best = Qu; comm_uot_best = cu; end
    end
    comm_raw = comm_raw_best;
    comm_uot = comm_uot_best;
    Q_raw    = Q_raw_best;
    Q_uot    = Q_uot_best;
    fprintf('  Community detection: BCT Louvain (%d runs)\n', n_runs);
else
    % Spectral clustering fallback (k=5 clusters, matching paper)
    fprintf('  BCT not found. Using spectral clustering fallback (k=5).\n');
    k_clust = 5;
    comm_raw = spectral_cluster(W_raw_norm, k_clust);
    comm_uot = spectral_cluster(W_uot_norm, k_clust);
    Q_raw    = compute_modularity(W_raw, comm_raw);
    Q_uot    = compute_modularity(W_uot, comm_uot);
end

ari_value = adjusted_rand_index(comm_raw, comm_uot);

% Number of communities detected
n_comm_raw = length(unique(comm_raw));
n_comm_uot = length(unique(comm_uot));

fprintf('  Modularity Q (Raw P_bar):                    %.4f  (%d communities)\n', Q_raw, n_comm_raw);
fprintf('  Modularity Q (Latent A0):                    %.4f  (%d communities)\n', Q_uot, n_comm_uot);
fprintf('  Adjusted Rand Index (Raw vs Latent):         %.4f\n\n', ari_value);

%% ─── 5. Micro-Level Evidence: Representative Concept Pairs ──────────
fprintf('═══ STEP 5: REPRESENTATIVE CONCEPT PAIRS ANALYSIS ═══\n');

% Z-score both matrices (off-diagonal only)
p_all = P_bar(:); a_all = A0(:);
Z_raw_full = (P_bar - mean(p_all(mask))) / std(p_all(mask));
Z_uot_full = (A0   - mean(a_all(mask))) / std(a_all(mask));

% Specificity gap: how much more (or less) the latent geometry values this pair
D_spec = Z_uot_full - Z_raw_full;
D_spec(~mask) = NaN;  % ignore diagonal

% --- Type A: Volume Artifacts (High Raw, Low Latent Affinity) ---
D_vec = D_spec(:);
[~, idx_typeA] = sort(D_vec, 'ascend', 'MissingPlacement', 'last');

TypeA_Source = cell(10,1); TypeA_Target = cell(10,1);
TypeA_RawFlow = zeros(10,1); TypeA_LatentA = zeros(10,1);
TypeA_ZRaw = zeros(10,1);   TypeA_ZUOT  = zeros(10,1);

for k = 1:10
    [r, c] = ind2sub([N, N], idx_typeA(k));
    TypeA_Source{k}  = concepts{r};
    TypeA_Target{k}  = concepts{c};
    TypeA_RawFlow(k) = P_bar(r,c);
    TypeA_LatentA(k) = A0(r,c);
    TypeA_ZRaw(k)    = Z_raw_full(r,c);
    TypeA_ZUOT(k)    = Z_uot_full(r,c);
end

table_typeA = table(TypeA_Source, TypeA_Target, TypeA_RawFlow, TypeA_LatentA, ...
    TypeA_ZRaw, TypeA_ZUOT, ...
    'VariableNames', {'Source','Target','Raw_Citation_Flow','Latent_Affinity_A0',...
                      'Z_Raw','Z_UOT'});

% --- Type B: Hidden Structural Bridges (Low Raw, High Latent Affinity) ---
[~, idx_typeB] = sort(D_vec, 'descend', 'MissingPlacement', 'last');

TypeB_Source = cell(10,1); TypeB_Target = cell(10,1);
TypeB_RawFlow = zeros(10,1); TypeB_LatentA = zeros(10,1);
TypeB_ZRaw = zeros(10,1);   TypeB_ZUOT  = zeros(10,1);

for k = 1:10
    [r, c] = ind2sub([N, N], idx_typeB(k));
    TypeB_Source{k}  = concepts{r};
    TypeB_Target{k}  = concepts{c};
    TypeB_RawFlow(k) = P_bar(r,c);
    TypeB_LatentA(k) = A0(r,c);
    TypeB_ZRaw(k)    = Z_raw_full(r,c);
    TypeB_ZUOT(k)    = Z_uot_full(r,c);
end

table_typeB = table(TypeB_Source, TypeB_Target, TypeB_RawFlow, TypeB_LatentA, ...
    TypeB_ZRaw, TypeB_ZUOT, ...
    'VariableNames', {'Source','Target','Raw_Citation_Flow','Latent_Affinity_A0',...
                      'Z_Raw','Z_UOT'});

fprintf('\n  Top 10 Volume Artifacts (High Raw Flow, Low Latent Affinity):\n');
fprintf('  [These pairs appear prominent in raw citations but are discounted by the latent geometry]\n');
disp(table_typeA);

fprintf('\n  Top 10 Hidden Structural Bridges (Low Raw Flow, High Latent Affinity):\n');
fprintf('  [These pairs are underrepresented in raw counts but structurally central in latent space]\n');
disp(table_typeB);

writetable(table_typeA, fullfile(config.tables_folder, 'REVIEWER_Volume_Artifacts_TypeA.csv'));
writetable(table_typeB, fullfile(config.tables_folder, 'REVIEWER_Hidden_Bridges_TypeB.csv'));

%% ─── 6. Summary Diagnostics Table ──────────────────────────────────
fprintf('═══ STEP 6: SUMMARY DIAGNOSTICS TABLE ═══\n\n');

Metric = {...
    'Spearman Correlation (P_bar vs A0)'; ...
    'Pearson Correlation (P_bar vs A0)'; ...
    'Normalized Frobenius Distance'; ...
    'PC1 Axis Divergence Angle (degrees)'; ...
    'Mean Subspace Angle Top-5 PCs (degrees)'; ...
    'R2: Raw Node Strength vs Field Volume'; ...
    'R2: Latent Strength vs Field Volume'; ...
    'Volume Bias Reduction (%)'; ...
    'Modularity Q (Raw P_bar)'; ...
    'Modularity Q (Latent A0)'; ...
    'Adjusted Rand Index (Raw vs Latent Clusters)'};

Value = [...
    spearman_rho; ...
    pearson_r; ...
    frob_dist; ...
    subspace_angles(1); ...
    mean(subspace_angles); ...
    r2_raw_vol; ...
    r2_latent_vol; ...
    bias_reduction_pct; ...
    Q_raw; ...
    Q_uot; ...
    ari_value];

Interpretation = {...
    'Rank-order agreement between raw and latent structures'; ...
    'Linear agreement (inflated by volume effects in raw)'; ...
    'Normalized distance between distributions'; ...
    'Angular separation of dominant structural axis'; ...
    'Average angular separation across top-5 axes'; ...
    'Volume dependency of raw centrality (high = biased)'; ...
    'Volume dependency of latent centrality (low = scale-free)'; ...
    'Percentage of volume bias removed by latent geometry'; ...
    'Mesoscopic structure quality in raw citation space'; ...
    'Mesoscopic structure quality in latent transport space'; ...
    'Overlap between raw and latent community partitions'};

reviewer_summary = table(Metric, Value, Interpretation);

disp(reviewer_summary);

writetable(reviewer_summary, fullfile(config.tables_folder, 'REVIEWER_summary_diagnostics.csv'));

fprintf('\n═══ INCREMENTAL VALUE ANALYSIS COMPLETE ═══\n\n');

%% ═══════════════════════════════════════════════════════════════════════
%  LOCAL HELPER FUNCTIONS
%% ═══════════════════════════════════════════════════════════════════════

function labels = spectral_cluster(W, k)
% Spectral clustering via normalized Laplacian eigenvectors
    N = size(W, 1);
    d = sum(W, 2);
    d(d == 0) = 1e-12;
    D_inv_sqrt = diag(1 ./ sqrt(d));
    L_sym = eye(N) - D_inv_sqrt * W * D_inv_sqrt;
    L_sym = (L_sym + L_sym') / 2;  % enforce symmetry
    [V, ~] = eigs(L_sym, k, 'smallestreal');
    % Normalize rows
    row_norms = sqrt(sum(V.^2, 2)) + 1e-12;
    V_norm = V ./ row_norms;
    % k-means on spectral embedding
    rng(42);
    labels = kmeans(V_norm, k, 'Replicates', 10, 'MaxIter', 500);
end

function Q = compute_modularity(W, labels)
% Newman-Girvan modularity for weighted undirected graph
    m = sum(W(:)) / 2;
    if m == 0, Q = 0; return; end
    k = sum(W, 2);
    communities = unique(labels);
    Q = 0;
    for c = 1:length(communities)
        idx = labels == communities(c);
        e_c  = sum(sum(W(idx, idx))) / (2 * m);
        a_c  = sum(k(idx)) / (2 * m);
        Q = Q + (e_c - a_c^2);
    end
end

function ari = adjusted_rand_index(labels1, labels2)
% Adjusted Rand Index between two label vectors
    labels1 = labels1(:);
    labels2 = labels2(:);
    N = length(labels1);
    if N ~= length(labels2)
        error('Input vectors must have the same length.');
    end

    u = unique(labels1);
    v = unique(labels2);
    n_u = length(u);
    n_v = length(v);

    contingency = zeros(n_u, n_v);
    for i = 1:n_u
        for j = 1:n_v
            contingency(i,j) = sum(labels1 == u(i) & labels2 == v(j));
        end
    end

    row_sums = sum(contingency, 2);
    col_sums = sum(contingency, 1);

    total_pairs    = N * (N - 1) / 2;
    sum_row_comb   = sum(row_sums .* (row_sums - 1) / 2);
    sum_col_comb   = sum(col_sums .* (col_sums - 1) / 2);
    sum_cell_comb  = sum(contingency(:) .* (contingency(:) - 1) / 2);

    expected_index = (sum_row_comb * sum_col_comb) / total_pairs;
    max_index      = (sum_row_comb + sum_col_comb) / 2;
    denominator    = max_index - expected_index;

    if denominator == 0
        ari = 1;
    else
        ari = (sum_cell_comb - expected_index) / denominator;
    end
end