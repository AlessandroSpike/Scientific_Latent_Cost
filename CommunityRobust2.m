%% ================================================================
% REVIEWER #1 RESPONSE
% GENUINE LEIDEN-CPM ROBUSTNESS ANALYSIS
%
% IMPORTANT:
% 1. Gibbs affinity epsilon = 1.0
% 2. Directed adjacency matrix (NO symmetrization)
% 3. Genuine Leiden-CPM through Python
% 4. Resolution sensitivity analysis
% 5. Singleton community diagnostics
% 6. Independent comparison against Louvain
%
% OUTPUT:
% MATLAB command-window tables + CSV files
%% ================================================================

clear; clc; close all;

addpath('E:\2019_03_03_BCT');

%% ================================================================
% CONFIGURATION
%% ================================================================

cfg = struct();

cfg.results = 'results_bayesopt';

cfg.out = 'reviewer_leiden_corrected';

cfg.tab_dir = fullfile(cfg.out,'tables');

cfg.epsilon = 1.0;

cfg.n_runs = 100;

cfg.gamma_values = ...
    0.90:0.005:1.10;

if ~exist(cfg.out,'dir')
    mkdir(cfg.out);
end

if ~exist(cfg.tab_dir,'dir')
    mkdir(cfg.tab_dir);
end

rng(42);

fprintf('\n=============================================\n');
fprintf('REVIEWER #1: LEIDEN-CPM ROBUSTNESS\n');
fprintf('=============================================\n');

%% ================================================================
% 1. LOAD BASELINE COST MATRIX
%% ================================================================

fprintf('\n[1] Loading baseline cost matrix...\n');

C0t = readtable( ...
    fullfile(cfg.results,'C0_baseline.csv'));

concepts = C0t.Properties.VariableNames(2:end);

C0 = table2array(C0t(:,2:end));

N = size(C0,1);

fprintf('Concepts: %d\n',N);

%% ================================================================
% 2. CONSTRUCT DIRECTED GIBBS AFFINITY
%% ================================================================

fprintf('\n[2] Constructing directed affinity...\n');

A0 = exp(-C0/cfg.epsilon);

% Remove self loops
A0(1:N+1:end) = 0;

% IMPORTANT:
% Keep directed network.
% Do NOT symmetrize.

A_directed = A0;

asymmetry = ...
    norm(A_directed-A_directed','fro') / ...
    max(norm(A_directed,'fro'),eps);

fprintf('Directed asymmetry: %.6f\n',asymmetry);

weights = A_directed(A_directed>0);

fprintf('Mean edge weight: %.6f\n',mean(weights));
fprintf('Median edge weight: %.6f\n',median(weights));
fprintf('Minimum edge weight: %.6f\n',min(weights));
fprintf('Maximum edge weight: %.6f\n',max(weights));

T_graph = table( ...
    N, ...
    nnz(A_directed), ...
    nnz(A_directed)/(N*(N-1)), ...
    mean(weights), ...
    median(weights), ...
    min(weights), ...
    max(weights), ...
    asymmetry, ...
    'VariableNames', ...
    {'Nodes','Edges','Density','MeanWeight', ...
     'MedianWeight','MinWeight','MaxWeight', ...
     'DirectedAsymmetry'});

disp(T_graph);

writetable(T_graph, ...
    fullfile(cfg.tab_dir,'T1_graph_diagnostics.csv'));

%% ================================================================
% 3. LOUVAIN BASELINE
%% ================================================================

fprintf('\n[3] Running Louvain baseline...\n');

lou_parts = zeros(N,cfg.n_runs);

lou_Q = zeros(cfg.n_runs,1);

for r = 1:cfg.n_runs

    [Ci,Q] = community_louvain( ...
        A_directed,1);

    lou_parts(:,r) = Ci;

    lou_Q(r) = Q;

end

[best_Q,best_run] = max(lou_Q);

lou_labels = lou_parts(:,best_run);

lou_K = numel(unique(lou_labels));

fprintf('Louvain communities: %d\n',lou_K);

fprintf('Louvain modularity: %.6f\n',best_Q);

%% ================================================================
% 4. EXPORT DIRECTED NETWORK TO PYTHON
%% ================================================================

fprintf('\n[4] Exporting directed adjacency...\n');

adj_file = fullfile(cfg.out,'directed_affinity.csv');

writematrix(A_directed,adj_file);

%% ================================================================
% 5. CALL GENUINE LEIDEN-CPM
%% ================================================================

fprintf('\n[5] Running genuine Leiden-CPM...\n');

gamma_file = fullfile(cfg.out,'gamma_values.csv');

writematrix(cfg.gamma_values(:),gamma_file);

python_script = 'leiden_cpm_directed.py';

command = sprintf( ...
    'python "%s" "%s" "%s" "%s"', ...
    python_script, ...
    adj_file, ...
    gamma_file, ...
    cfg.out);

[status,cmdout] = system(command);

fprintf('%s\n',cmdout);

if status ~= 0
    error('Python Leiden execution failed.');
end

%% ================================================================
% 6. LOAD LEIDEN RESULTS
%% ================================================================

fprintf('\n[6] Loading Leiden results...\n');

leiden_summary = readtable( ...
    fullfile(cfg.out,'leiden_summary.csv'));

leiden_parts = readmatrix( ...
    fullfile(cfg.out,'leiden_partitions.csv'));

fprintf('\nLEIDEN RESOLUTION SENSITIVITY\n');

disp(leiden_summary);

writetable(leiden_summary, ...
    fullfile(cfg.tab_dir, ...
    'T2_leiden_resolution_sensitivity.csv'));

%% ================================================================
% 7. COMMUNITY SIZE DISTRIBUTIONS
%% ================================================================

fprintf('\n[7] Computing community sizes...\n');

n_gamma = numel(cfg.gamma_values);

size_rows = [];

for g = 1:n_gamma

    labels = leiden_parts(:,g);

    [~,~,labels] = unique(labels);

    sizes = accumarray(labels,1);

    gamma = cfg.gamma_values(g);

    for k = 1:numel(sizes)

        size_rows = [size_rows; ...
            gamma,k,sizes(k)];

    end

end

T_sizes = array2table( ...
    size_rows, ...
    'VariableNames', ...
    {'Gamma','Community','Size'});

fprintf('\nLEIDEN COMMUNITY SIZE DISTRIBUTION\n');

disp(T_sizes);

writetable(T_sizes, ...
    fullfile(cfg.tab_dir, ...
    'T3_leiden_community_sizes.csv'));

%% ================================================================
% 8. SINGLETON DIAGNOSTICS
%% ================================================================

fprintf('\n[8] Singleton diagnostics...\n');

singleton_rows = zeros(n_gamma,6);

for g = 1:n_gamma

    labels = leiden_parts(:,g);

    [~,~,labels] = unique(labels);

    sizes = accumarray(labels,1);

    n_singletons = sum(sizes==1);

    singleton_rows(g,:) = [ ...
        cfg.gamma_values(g), ...
        numel(sizes), ...
        n_singletons, ...
        100*n_singletons/N, ...
        min(sizes), ...
        max(sizes)];

end

T_singletons = array2table( ...
    singleton_rows, ...
    'VariableNames', ...
    {'Gamma','K','Singletons', ...
     'SingletonPct','MinSize','MaxSize'});

fprintf('\nSINGLETON ROBUSTNESS TABLE\n');

disp(T_singletons);

writetable(T_singletons, ...
    fullfile(cfg.tab_dir, ...
    'T4_singleton_diagnostics.csv'));

%% ================================================================
% 9. AGREEMENT WITH LOUVAIN
%% ================================================================

fprintf('\n[9] Computing agreement with Louvain...\n');

ARI = zeros(n_gamma,1);

NMI = zeros(n_gamma,1);

VI = zeros(n_gamma,1);

for g = 1:n_gamma

    labels = leiden_parts(:,g);

    ARI(g) = adjusted_rand_index( ...
        lou_labels,labels);

    NMI(g) = normalized_mutual_information( ...
        lou_labels,labels);

    VI(g) = variation_of_information( ...
        lou_labels,labels);

end

T_agreement = table( ...
    cfg.gamma_values(:), ...
    leiden_summary.K, ...
    ARI,NMI,VI, ...
    'VariableNames', ...
    {'Gamma','K','ARI','NMI','VI'});

fprintf('\nLEIDEN-LOUVAIN AGREEMENT\n');

disp(T_agreement);

writetable(T_agreement, ...
    fullfile(cfg.tab_dir, ...
    'T5_leiden_louvain_agreement.csv'));

%% ================================================================
% 10. NEIGHBORING RESOLUTION STABILITY
%% ================================================================

fprintf('\n[10] Computing neighboring-resolution stability...\n');

neighbor_ARI = NaN(n_gamma,1);

for g = 2:n_gamma

    neighbor_ARI(g) = adjusted_rand_index( ...
        leiden_parts(:,g-1), ...
        leiden_parts(:,g));

end

T_stability = table( ...
    cfg.gamma_values(:), ...
    leiden_summary.K, ...
    neighbor_ARI, ...
    'VariableNames', ...
    {'Gamma','K','ARI_PreviousGamma'});

disp(T_stability);

writetable(T_stability, ...
    fullfile(cfg.tab_dir, ...
    'T6_resolution_stability.csv'));

%% ================================================================
% 11. GAMMA = 1.0: DIRECT REVIEWER RESPONSE
%% ================================================================

fprintf('\n=============================================\n');
fprintf('REVIEWER QUESTION: GAMMA = 1.0\n');
fprintf('=============================================\n');

idx = find(abs(cfg.gamma_values-1)<1e-10);

if ~isempty(idx)

    labels = leiden_parts(:,idx);

    [~,~,labels] = unique(labels);

    sizes = sort(accumarray(labels,1),'descend');

    fprintf('Gamma: 1.0\n');

    fprintf('Communities: %d\n',numel(sizes));

    fprintf('Singleton communities: %d\n', ...
        sum(sizes==1));

    fprintf('Singleton percentage: %.2f%%\n', ...
        100*sum(sizes==1)/N);

    fprintf('Community sizes:\n');

    disp(sizes');

    fprintf('ARI with Louvain: %.6f\n',ARI(idx));

    fprintf('NMI with Louvain: %.6f\n',NMI(idx));

end

fprintf('\nAnalysis completed.\n');


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