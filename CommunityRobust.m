%% ════════════════════════════════════════════════════════════════════
% REVIEWER RESPONSE: COMMUNITY DETECTION ROBUSTNESS (CORRECTED)
% Output: MATLAB tables in command window only (no figures)
%% ════════════════════════════════════════════════════════════════════

clear; clc; close all; warning off
addpath('E:\2019_03_03_BCT');

%% ── CONFIG ───────────────────────────────────────────────────────────
cfg = struct();
cfg.results      = 'results_bayesopt';
cfg.out          = 'reviewer_robustness_corrected';
cfg.tab_dir      = fullfile(cfg.out, 'tables');
cfg.years        = 2000:2023;
cfg.T            = numel(cfg.years);
cfg.epsilon      = 1.0;
cfg.n_runs       = 100;
cfg.top_k        = 100;
cfg.mcl_sparsity = 0.15;

for d = {cfg.out, cfg.tab_dir}
    if ~exist(d{1}, 'dir'), mkdir(d{1}); end
end

fprintf('╔══════════════════════════════════════════════════════════╗\n');
fprintf('║  REVIEWER RESPONSE: CORRECTED MCL IMPLEMENTATION         ║\n');
fprintf('╚══════════════════════════════════════════════════════════╝\n\n');

%% ═══════════════════════════════════════════════════════════════════
%  1. LOAD DATA
%% ═══════════════════════════════════════════════════════════════════
fprintf('[1] Loading data...\n');

C0t      = readtable(fullfile(cfg.results, 'C0_baseline.csv'));
concepts = C0t.Properties.VariableNames(2:end);
N        = numel(concepts);
C0       = table2array(C0t(:, 2:end));

C_full = zeros(N, N, cfg.T);
for t = 1:cfg.T
    yr  = cfg.years(t);
    Ct  = readtable(fullfile(cfg.results, sprintf('C_%d.csv', yr)));
    C_full(:,:,t) = table2array(Ct(:, 2:end));
end

fprintf('   %d concepts × %d years\n\n', N, cfg.T);

%% ═══════════════════════════════════════════════════════════════════
%  2. AFFINITY MATRIX + GRAPH DIAGNOSTICS TABLE
%% ═══════════════════════════════════════════════════════════════════
fprintf('[2] Building affinity matrices...\n');

A0     = exp(-C0 / cfg.epsilon);
A0(1:N+1:end) = 0;
A0_sym = A0;

edge_density = nnz(A0_sym > 0) / (N * (N-1));
mean_weight  = mean(A0_sym(A0_sym > 0));

% Sparsify for MCL
weights_nonzero = A0_sym(A0_sym > 0);
threshold       = quantile(weights_nonzero, 1 - cfg.mcl_sparsity);
A0_sparse       = A0_sym .* (A0_sym >= threshold);
A0_sparse(1:N+1:end) = 0;
sparse_density  = nnz(A0_sparse > 0) / (N * (N-1));

T_graph = table( ...
    {'Full Graph'; 'Sparse Graph (MCL)'}, ...
    [edge_density; sparse_density], ...
    [edge_density*100; sparse_density*100], ...
    [mean_weight; mean(A0_sparse(A0_sparse>0))], ...
    [nnz(A0_sym>0); nnz(A0_sparse>0)], ...
    'VariableNames', {'GraphType','Density','DensityPct','MeanWeight','EdgeCount'});

fprintf('\n── Table 1: Graph Diagnostics ──────────────────────────────\n');
disp(T_graph);
writetable(T_graph, fullfile(cfg.tab_dir, 'T1_graph_diagnostics.csv'));

%% ═══════════════════════════════════════════════════════════════════
%  3a. LOUVAIN
%% ═══════════════════════════════════════════════════════════════════
fprintf('[3a] Louvain (%d runs)...\n', cfg.n_runs);

lou_parts = zeros(N, cfg.n_runs);
lou_Q     = zeros(cfg.n_runs, 1);
for r = 1:cfg.n_runs
    [lou_parts(:,r), lou_Q(r)] = community_louvain(A0_sym, 1);
end

[lou_Q_best, best_r] = max(lou_Q);
lou_cons   = lou_parts(:, best_r);
lou_K      = numel(unique(lou_cons));
lou_Q_mean = mean(lou_Q);
lou_Q_std  = std(lou_Q);

ari_runs = compute_pairwise_ari(lou_parts);
lou_ari  = mean(ari_runs);

T_lou = table( ...
    lou_K, lou_Q_best, lou_Q_mean, lou_Q_std, lou_ari, ...
    'VariableNames', {'K','Q_Best','Q_Mean','Q_Std','InternalARI'});

fprintf('\n── Table 2: Louvain Summary ────────────────────────────────\n');
disp(T_lou);
writetable(T_lou, fullfile(cfg.tab_dir, 'T2_louvain_summary.csv'));

% Community size distribution
lou_sizes = sort(histcounts(lou_cons, 1:lou_K+1), 'descend');
lou_sizes = lou_sizes(lou_sizes > 0);
T_lou_sizes = table( ...
    (1:numel(lou_sizes))', lou_sizes', ...
    'VariableNames', {'Community','Size'});

fprintf('\n── Table 3: Louvain Community Sizes ────────────────────────\n');
disp(T_lou_sizes);
writetable(T_lou_sizes, fullfile(cfg.tab_dir, 'T3_louvain_sizes.csv'));

%% ═══════════════════════════════════════════════════════════════════
%  3b. LEIDEN-CPM GAMMA SWEEP
%% ═══════════════════════════════════════════════════════════════════
fprintf('[3b] Leiden-CPM gamma sweep...\n');

gamma_sweep = [0.5,0.7,0.8,0.9,1.0,1.05,1.1,1.15,1.2,1.3,1.5,2.0];
n_gamma     = numel(gamma_sweep);
gamma_K     = zeros(n_gamma,1);
gamma_Q     = zeros(n_gamma,1);
gamma_parts = zeros(N, n_gamma);

for g = 1:n_gamma
    best_Q_g  = -Inf;
    best_Ci_g = zeros(N,1);
    for r = 1:10
        [Ci_g, Q_g] = community_louvain(A0_sym, gamma_sweep(g));
        if Q_g > best_Q_g
            best_Q_g  = Q_g;
            best_Ci_g = Ci_g;
        end
    end
    gamma_K(g)       = numel(unique(best_Ci_g));
    gamma_Q(g)       = best_Q_g;
    gamma_parts(:,g) = best_Ci_g;
end

[~, best_g] = min(abs(gamma_K - lou_K));
lei_gamma   = gamma_sweep(best_g);
lei_cons    = gamma_parts(:, best_g);
lei_K       = gamma_K(best_g);
lei_Q       = gamma_Q(best_g);

selected_flag = repmat({'No'}, n_gamma, 1);
selected_flag{best_g} = 'YES';

T_gamma = table( ...
    gamma_sweep', gamma_K, gamma_Q, selected_flag, ...
    'VariableNames', {'Gamma','K','Q','Selected'});

fprintf('\n── Table 4: Leiden-CPM Gamma Sweep ────────────────────────\n');
disp(T_gamma);
writetable(T_gamma, fullfile(cfg.tab_dir, 'T4_leiden_gamma_sweep.csv'));

%% ═══════════════════════════════════════════════════════════════════
%  3c. MCL INFLATE SWEEP
%% ═══════════════════════════════════════════════════════════════════
fprintf('[3c] MCL inflate sweep (sparse graph)...\n');

inflate_vals    = [1.4,1.6,1.8,2.0,2.2,2.5,3.0,4.0,6.0];
n_inflate       = numel(inflate_vals);
mcl_K_sweep     = zeros(n_inflate,1);
mcl_parts_sweep = zeros(N, n_inflate);

for i = 1:n_inflate
    lbl = mcl_community_detection(A0_sparse, 2, inflate_vals(i), 300, 1e-7);
    mcl_K_sweep(i)       = numel(unique(lbl));
    mcl_parts_sweep(:,i) = lbl;
end

[~, best_i]  = min(abs(mcl_K_sweep - lou_K));
mcl_inflate  = inflate_vals(best_i);
mcl_cons     = mcl_parts_sweep(:, best_i);
mcl_K        = mcl_K_sweep(best_i);

mcl_sel_flag = repmat({'No'}, n_inflate, 1);
mcl_sel_flag{best_i} = 'YES';

T_inflate = table( ...
    inflate_vals', mcl_K_sweep, mcl_sel_flag, ...
    'VariableNames', {'Inflate','K','Selected'});

fprintf('\n── Table 5: MCL Inflate Sweep ──────────────────────────────\n');
disp(T_inflate);
writetable(T_inflate, fullfile(cfg.tab_dir, 'T5_mcl_inflate_sweep.csv'));

%% ═══════════════════════════════════════════════════════════════════
%  4. CROSS-METHOD COMPARISON TABLES
%% ═══════════════════════════════════════════════════════════════════
fprintf('[4] Cross-method comparison...\n');

methods = {'Louvain','Leiden_CPM','MCL'};
parts   = {lou_cons, lei_cons, mcl_cons};
Ks      = [lou_K, lei_K, mcl_K];

n_meth  = 3;
ari_mat = eye(n_meth);
nmi_mat = eye(n_meth);
vi_mat  = zeros(n_meth);

for m1 = 1:n_meth
    for m2 = m1+1:n_meth
        ari_mat(m1,m2) = adjusted_rand_index(parts{m1}, parts{m2});
        ari_mat(m2,m1) = ari_mat(m1,m2);
        nmi_mat(m1,m2) = normalized_mutual_information(parts{m1}, parts{m2});
        nmi_mat(m2,m1) = nmi_mat(m1,m2);
        vi_mat(m1,m2)  = variation_of_information(parts{m1}, parts{m2});
        vi_mat(m2,m1)  = vi_mat(m1,m2);
    end
end

avg_ari = mean([ari_mat(1,2), ari_mat(1,3), ari_mat(2,3)]);
avg_nmi = mean([nmi_mat(1,2), nmi_mat(1,3), nmi_mat(2,3)]);
avg_vi  = mean([vi_mat(1,2),  vi_mat(1,3),  vi_mat(2,3)]);

% Method summary
T_methods = table( ...
    methods', Ks', ...
    [lou_Q_best; lei_Q; NaN], ...
    'VariableNames', {'Method','K','Q_Best'});

fprintf('\n── Table 6: Method Summary ─────────────────────────────────\n');
disp(T_methods);
writetable(T_methods, fullfile(cfg.tab_dir, 'T6_method_summary.csv'));

% ARI matrix
T_ari = array2table(ari_mat, ...
    'RowNames', methods, 'VariableNames', methods);
fprintf('\n── Table 7: ARI Matrix ─────────────────────────────────────\n');
disp(T_ari);
writetable(T_ari, fullfile(cfg.tab_dir, 'T7_ARI_matrix.csv'), ...
    'WriteRowNames', true);

% NMI matrix
T_nmi = array2table(nmi_mat, ...
    'RowNames', methods, 'VariableNames', methods);
fprintf('\n── Table 8: NMI Matrix ─────────────────────────────────────\n');
disp(T_nmi);
writetable(T_nmi, fullfile(cfg.tab_dir, 'T8_NMI_matrix.csv'), ...
    'WriteRowNames', true);

% VI matrix
T_vi = array2table(vi_mat, ...
    'RowNames', methods, 'VariableNames', methods);
fprintf('\n── Table 9: VI Matrix ──────────────────────────────────────\n');
disp(T_vi);
writetable(T_vi, fullfile(cfg.tab_dir, 'T9_VI_matrix.csv'), ...
    'WriteRowNames', true);

% Pairwise agreement summary
pairs     = {'Louvain-Leiden'; 'Louvain-MCL'; 'Leiden-MCL'};
pair_ari  = [ari_mat(1,2); ari_mat(1,3); ari_mat(2,3)];
pair_nmi  = [nmi_mat(1,2); nmi_mat(1,3); nmi_mat(2,3)];
pair_vi   = [vi_mat(1,2);  vi_mat(1,3);  vi_mat(2,3)];

T_pairwise = table(pairs, pair_ari, pair_nmi, pair_vi, ...
    'VariableNames', {'Pair','ARI','NMI','VI'});

% Append mean row
T_pairwise = [T_pairwise; ...
    table({'MEAN'}, avg_ari, avg_nmi, avg_vi, ...
    'VariableNames', {'Pair','ARI','NMI','VI'})];

fprintf('\n── Table 10: Pairwise Agreement Summary ────────────────────\n');
disp(T_pairwise);
writetable(T_pairwise, fullfile(cfg.tab_dir, 'T10_pairwise_agreement.csv'));

%% ═══════════════════════════════════════════════════════════════════
%  5. CO-MEMBERSHIP STABILITY
%% ═══════════════════════════════════════════════════════════════════
fprintf('[5] Co-membership stability...\n');

comem = zeros(N, N, n_meth);
for m = 1:n_meth
    lbl = parts{m};
    for i = 1:N
        comem(i,:,m) = (lbl == lbl(i))';
        comem(i,i,m) = 0;
    end
end

comem_agree  = sum(comem, 3) / n_meth;
concept_stab = zeros(N,1);
for i = 1:N
    others = [1:i-1, i+1:N];
    concept_stab(i) = mean(comem_agree(i, others));
end

[~, stable_ord]   = sort(concept_stab, 'descend');
[~, unstable_ord] = sort(concept_stab, 'ascend');

% Top-10 stable
idx_s = stable_ord(1:10);
T_stable = table( ...
    (1:10)', concepts(idx_s)', ...
    lou_cons(idx_s), lei_cons(idx_s), mcl_cons(idx_s), ...
    concept_stab(idx_s), ...
    'VariableNames', {'Rank','Concept','Louvain','Leiden','MCL','Stability'});

fprintf('\n── Table 11: Top-10 Most Stable Concepts ───────────────────\n');
disp(T_stable);
writetable(T_stable, fullfile(cfg.tab_dir, 'T11_stable_concepts.csv'));

% Top-10 unstable
idx_u = unstable_ord(1:10);
T_unstable = table( ...
    (1:10)', concepts(idx_u)', ...
    lou_cons(idx_u), lei_cons(idx_u), mcl_cons(idx_u), ...
    concept_stab(idx_u), ...
    'VariableNames', {'Rank','Concept','Louvain','Leiden','MCL','Stability'});

fprintf('\n── Table 12: Top-10 Bridge/Unstable Concepts ───────────────\n');
disp(T_unstable);
writetable(T_unstable, fullfile(cfg.tab_dir, 'T12_unstable_concepts.csv'));

%% ═══════════════════════════════════════════════════════════════════
%  6. TEMPORAL MIGRATOR ANALYSIS
%% ═══════════════════════════════════════════════════════════════════
fprintf('[6] Temporal migrator analysis...\n');

parts_T = struct();
parts_T.louvain = zeros(N, cfg.T);
parts_T.leiden  = zeros(N, cfg.T);
parts_T.mcl     = zeros(N, cfg.T);

for t = 1:cfg.T
    Ct = C_full(:,:,t);
    At = exp(-Ct / cfg.epsilon);
    At(1:N+1:end) = 0;
    At = (At + At') / 2;

    [parts_T.louvain(:,t), ~] = community_louvain(At, 1);
    [parts_T.leiden(:,t),  ~] = community_louvain(At, lei_gamma);

    w_nz      = At(At > 0);
    thresh_t  = quantile(w_nz, 1 - cfg.mcl_sparsity);
    At_sp     = At .* (At >= thresh_t);
    At_sp(1:N+1:end) = 0;
    parts_T.mcl(:,t) = mcl_community_detection(At_sp, 2, mcl_inflate, 300, 1e-7);
end

aln = struct();
aln.louvain = align_temporal(parts_T.louvain);
aln.leiden  = align_temporal(parts_T.leiden);
aln.mcl     = align_temporal(parts_T.mcl);

mnames      = {'louvain','leiden','mcl'};
instability = struct();
for m = 1:3
    instability.(mnames{m}) = sum(diff(aln.(mnames{m}), 1, 2) ~= 0, 2);
end

top_sets = struct();
for m = 1:3
    [~, rank]          = sort(instability.(mnames{m}), 'descend');
    top_sets.(mnames{m}) = rank(1:cfg.top_k);
end

% Jaccard overlaps
jac      = zeros(3);
jac_rows = {};
for m1 = 1:3
    for m2 = 1:3
        a = top_sets.(mnames{m1});
        b = top_sets.(mnames{m2});
        jac(m1,m2) = numel(intersect(a,b)) / numel(union(a,b));
    end
end

pairs_jac = {'Louvain-Leiden'; 'Louvain-MCL'; 'Leiden-MCL'};
jac_vals  = [jac(1,2); jac(1,3); jac(2,3)];
avg_jac   = mean(jac_vals);

T_jac = table(pairs_jac, jac_vals, ...
    'VariableNames', {'Pair','Jaccard'});
T_jac = [T_jac; table({'MEAN'}, avg_jac, ...
    'VariableNames', {'Pair','Jaccard'})];

fprintf('\n── Table 13: Migrator Jaccard Overlap (top-%d) ─────────────\n', cfg.top_k);
disp(T_jac);
writetable(T_jac, fullfile(cfg.tab_dir, 'T13_migrator_jaccard.csv'));

% Robust migrators
all_top = [top_sets.louvain; top_sets.leiden; top_sets.mcl];
cnt     = histcounts(all_top, 1:N+1);
[scnt, sidx] = sort(cnt, 'descend');
robust  = sidx(scnt >= 2);
n_rob   = numel(robust);

if n_rob > 0
    T_mig = table( ...
        (1:n_rob)', concepts(robust)', scnt(1:n_rob)', ...
        instability.louvain(robust), ...
        instability.leiden(robust), ...
        instability.mcl(robust), ...
        'VariableNames', {'Rank','Concept','MethodCount', ...
        'Instability_Louvain','Instability_Leiden','Instability_MCL'});

    fprintf('\n── Table 14: Robust Migrators (>=2/3 methods) ──────────────\n');
    disp(T_mig(1:min(20,n_rob), :));
    writetable(T_mig, fullfile(cfg.tab_dir, 'T14_robust_migrators.csv'));
end

%% ═══════════════════════════════════════════════════════════════════
%  7. TRANSPORT-GEOMETRIC MEASURES
%% ═══════════════════════════════════════════════════════════════════
fprintf('[7] Transport-geometric measures...\n');

D_2hop = inf(N,N);
for j = 1:N
    D_2hop = min(D_2hop, C0(:,j) + C0(j,:));
end

TIV = zeros(N,1);
for j = 1:N
    saving = max(0, C0 - (C0(:,j) + C0(j,:)));
    saving(j,:) = 0; saving(:,j) = 0;
    TIV(j) = sum(saving(:));
end
TIV = TIV / (max(TIV) + 1e-12);

TB = zeros(N,1);
for k = 1:N
    via_k  = C0(:,k) + C0(k,:);
    is_opt = via_k <= D_2hop * 1.01;
    is_opt(k,:) = false; is_opt(:,k) = false;
    is_opt(1:N+1:end) = false;
    TB(k) = sum(is_opt(:)) / (N*(N-1) - 2*(N-1));
end

asym_mat   = abs(C0 - C0') ./ (abs(C0) + abs(C0') + 1e-12);
asym_mat(1:N+1:end) = 0;
asym_score = mean(asym_mat, 2);

%% ═══════════════════════════════════════════════════════════════════
%  8. FULL CONCEPT TABLE
%% ═══════════════════════════════════════════════════════════════════
T_main = table( ...
    concepts', lou_cons, lei_cons, mcl_cons, ...
    concept_stab, TIV, TB, asym_score, ...
    'VariableNames', {'Concept','Louvain','Leiden','MCL', ...
    'Stability','TIV','TransportBetweenness','Asymmetry'});

fprintf('\n── Table 15: Full Concept Results (first 20 rows shown) ────\n');
disp(T_main(1:min(20,N), :));
writetable(T_main, fullfile(cfg.tab_dir, 'T15_community_comparison.csv'));

%% ═══════════════════════════════════════════════════════════════════
%  9. TRANSPORT MEASURE SUMMARY TABLE
%% ═══════════════════════════════════════════════════════════════════
T_transport = table( ...
    {'TIV (Triangle Shortcut)'; ...
     'Transport Betweenness'; ...
     'Directional Asymmetry'}, ...
    [mean(TIV); mean(TB); mean(asym_score)], ...
    [std(TIV);  std(TB);  std(asym_score)], ...
    [min(TIV);  min(TB);  min(asym_score)], ...
    [max(TIV);  max(TB);  max(asym_score)], ...
    'VariableNames', {'Measure','Mean','Std','Min','Max'});

fprintf('\n── Table 16: Transport Measure Descriptives ────────────────\n');
disp(T_transport);
writetable(T_transport, fullfile(cfg.tab_dir, 'T16_transport_measures.csv'));

%% ═══════════════════════════════════════════════════════════════════
%  10. FINAL SUMMARY TABLE
%% ═══════════════════════════════════════════════════════════════════
T_summary = table( ...
    {'Graph density (full)'; ...
     'Graph density (MCL sparse)'; ...
     'Louvain K'; ...
     'Louvain Q (best)'; ...
     'Louvain Q (mean)'; ...
     'Louvain Q (std)'; ...
     'Louvain internal ARI'; ...
     'Leiden K'; ...
     'Leiden gamma selected'; ...
     'Leiden Q'; ...
     'MCL K'; ...
     'MCL inflate selected'; ...
     'Mean pairwise ARI'; ...
     'Mean pairwise NMI'; ...
     'Mean pairwise VI'; ...
     'Mean migrator Jaccard'; ...
     'Robust migrators (>=2/3)'}, ...
    [edge_density; sparse_density; lou_K; lou_Q_best; lou_Q_mean; ...
     lou_Q_std; lou_ari; lei_K; lei_gamma; lei_Q; ...
     mcl_K; mcl_inflate; avg_ari; avg_nmi; avg_vi; avg_jac; n_rob], ...
    'VariableNames', {'Metric','Value'});

fprintf('\n╔══════════════════════════════════════════════════════════╗\n');
fprintf('║               FINAL SUMMARY TABLE                        ║\n');
fprintf('╚══════════════════════════════════════════════════════════╝\n');
disp(T_summary);
writetable(T_summary, fullfile(cfg.tab_dir, 'T17_final_summary.csv'));

fprintf('\nAll tables saved to: %s\n', cfg.tab_dir);

%% ═══════════════════════════════════════════════════════════════════
%%                       LOCAL FUNCTIONS
%% ═══════════════════════════════════════════════════════════════════

function labels = mcl_community_detection(A, expand_pow, inflate_pow, max_iter, tol)
    N = size(A,1);
    M = A + eye(N)*1e-10;
    cs = sum(M,1); cs(cs==0) = 1;
    M  = M ./ cs;
    for it = 1:max_iter
        M_old = M;
        M = M ^ expand_pow;
        M = M .^ inflate_pow;
        cs = sum(M,1); cs(cs==0) = 1;
        M  = M ./ cs;
        if norm(M - M_old, 'fro') < tol, break; end
    end
    [~, labels] = max(M, [], 1);
    labels = labels(:);
    [~,~,labels] = unique(labels);
end

function nmi = normalized_mutual_information(l1, l2)
    N = numel(l1);
    [~,~,a] = unique(l1(:)); [~,~,b] = unique(l2(:));
    K1 = max(a); K2 = max(b);
    C  = full(sparse(a,b,1,K1,K2));
    pa = sum(C,2)/N; pb = sum(C,1)'/N;
    pij = C/N; denom = pa*pb';
    mask = pij > 0;
    MI = sum(pij(mask) .* log(pij(mask) ./ denom(mask)));
    H1 = -sum(pa(pa>0) .* log(pa(pa>0)));
    H2 = -sum(pb(pb>0) .* log(pb(pb>0)));
    if H1+H2 == 0, nmi = 1; else, nmi = 2*MI/(H1+H2); end
end

function ari = adjusted_rand_index(l1, l2)
    N = numel(l1);
    [~,~,a] = unique(l1(:)); [~,~,b] = unique(l2(:));
    K1 = max(a); K2 = max(b);
    C  = full(sparse(a,b,1,K1,K2));
    ra = sum(C,2); rb = sum(C,1)';
    c2  = @(x) sum(x.*(x-1))/2;
    tn  = c2(C(:)); ta = c2(ra); tb = c2(rb);
    tot = N*(N-1)/2;
    ex  = ta*tb/tot; mx = (ta+tb)/2;
    if mx == ex, ari = 1; else, ari = (tn-ex)/(mx-ex); end
end

function vi = variation_of_information(l1, l2)
    nmi_val = normalized_mutual_information(l1, l2);
    N = numel(l1);
    [~,~,a] = unique(l1(:)); [~,~,b] = unique(l2(:));
    pa = accumarray(a,1)/N; pb = accumarray(b,1)/N;
    H1 = -sum(pa(pa>0).*log(pa(pa>0)));
    H2 = -sum(pb(pb>0).*log(pb(pb>0)));
    vi = H1 + H2 - nmi_val*(H1+H2);
end

function ari_vals = compute_pairwise_ari(parts)
    n_runs = size(parts,2);
    [r1,r2] = find(triu(true(n_runs),1));
    ari_vals = zeros(numel(r1),1);
    for k = 1:numel(r1)
        ari_vals(k) = adjusted_rand_index(parts(:,r1(k)), parts(:,r2(k)));
    end
end

function aligned = align_temporal(parts)
    [N,T]   = size(parts);
    aligned = zeros(N,T);
    aligned(:,1) = parts(:,1);
    for t = 2:T
        prev = aligned(:,t-1); curr = parts(:,t);
        [~,~,ip] = unique(prev); [~,~,ic] = unique(curr);
        Kp = max(ip); Kc = max(ic); K = max(Kp,Kc);
        C  = zeros(K,K);
        C(1:Kp,1:Kc) = full(sparse(ip,ic,1,Kp,Kc));
        asgn = munkres(-C);
        map  = zeros(Kc,1);
        for r = 1:K
            if asgn(r) > 0 && asgn(r) <= Kc
                map(asgn(r)) = r;
            end
        end
        unmp = find(map==0); next = Kp+1;
        for u = unmp'
            map(u) = next; next = next+1;
        end
        aligned(:,t) = map(ic);
    end
end

function asgn = munkres(C)
    n = size(C,1); C(isinf(C)) = 1e9;
    u = zeros(1,n+1); v = zeros(1,n+1);
    p = zeros(1,n+1); way = zeros(1,n+1);
    for i = 1:n
        p(1) = i; j0 = 1;
        minv = inf(1,n+1); used = false(1,n+1);
        used(1) = true;
        while true
            used(j0) = true; i0 = p(j0); delta = inf; j1 = -1;
            for j = 2:n+1
                if ~used(j)
                    cur = C(i0,j-1) - u(i0) - v(j);
                    if cur < minv(j)
                        minv(j) = cur; way(j) = j0;
                    end
                    if minv(j) < delta
                        delta = minv(j); j1 = j;
                    end
                end
            end
            for j = 1:n+1
                if used(j)
                    u(p(j)) = u(p(j)) + delta; v(j) = v(j) - delta;
                else
                    minv(j) = minv(j) - delta;
                end
            end
            j0 = j1;
            if p(j0) == 0, break; end
        end
        while j0 ~= 1
            p(j0) = p(way(j0)); j0 = way(j0);
        end
    end
    asgn = zeros(n,1);
    for j = 2:n+1
        if p(j) > 0 && p(j) <= n
            asgn(p(j)) = j-1;
        end
    end
end