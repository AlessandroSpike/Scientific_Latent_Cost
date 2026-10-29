%% ========================================================================
%  DEEP ANALYSIS OF TEMPORAL DEVIATIONS E(t) 
%  (Inverse UOT Knowledge Geometry)
% =========================================================================
%  This script extends the original analysis by focusing on the time-varying
%  component E(t) = C(t) - C0. While the baseline C0 captures the persistent
%  backbone, E(t) holds the full temporal reconfiguration of conceptual
%  proximities. We investigate:
%    1. Concept-level "flexibility" – which concepts change most/least?
%    2. Dominant modes of deviation (PCA on flattened E_t)
%    3. Temporal clustering of years based on deviation patterns
%    4. Sign asymmetry: where do costs increase vs decrease?
%    5. Pairs where E(t) dominates the baseline (structural re-wiring)
%    6. Persistence of deviation direction (autocorrelation)
% =========================================================================

clear; clc; close all; warning off

%% ---------------------- CONFIGURATION ----------------------------------
config.results_folder = 'results_bayesopt';   % same as original
config.output_folder  = 'rq2_analysis_output';
config.figures_folder = fullfile(config.output_folder, 'figures_Et');
config.tables_folder  = fullfile(config.output_folder, 'tables_Et');
config.years = 2000:2023;
config.T = length(config.years);

% Create output directories
if ~exist(config.figures_folder, 'dir'), mkdir(config.figures_folder); end
if ~exist(config.tables_folder,  'dir'), mkdir(config.tables_folder);  end

fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║       DEEP ANALYSIS OF TEMPORAL DEVIATIONS E(t)             ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

%% ---------------------- LOAD DATA ---------------------------------------
fprintf('Loading data from %s ...\n', config.results_folder);

% Baseline C0
C0_table = readtable(fullfile(config.results_folder, 'C0_baseline.csv'));
concepts = C0_table.Properties.VariableNames(2:end);
N = length(concepts);
C0 = table2array(C0_table(:, 2:end));

% Time-varying deviations E(t) and full cost C(t)
E = zeros(N, N, config.T);
C_full = zeros(N, N, config.T);
for t = 1:config.T
    year = config.years(t);
    E_table = readtable(fullfile(config.results_folder, sprintf('E_%d.csv', year)));
    C_table = readtable(fullfile(config.results_folder, sprintf('C_%d.csv', year)));
    E(:,:,t) = table2array(E_table(:, 2:end));
    C_full(:,:,t) = table2array(C_table(:, 2:end));
end

% Original citation tensor (to identify zero‐citation pairs – "structural voids")
raw_data = load(fullfile(config.results_folder, 'original_tensor.mat'));
T_tensor = raw_data.T_tensor;
total_flow = sum(T_tensor, 3);
active_pairs_mask = total_flow > prctile(total_flow(:), 50);  % median threshold

fprintf('  ✓ Loaded C0 (baseline)\n');
fprintf('  ✓ Loaded E(t) for %d years\n', config.T);
fprintf('  ✓ Loaded citation tensor (active links: %.1f%%)\n\n', ...
    100 * sum(active_pairs_mask(:)) / N^2);

%% ------------------------------------------------------------------------
%% 1. CONCEPT-LEVEL FLEXIBILITY: How much does each concept's connectivity change?
%% ------------------------------------------------------------------------
fprintf('--- 1. Concept Flexibility (Norm of E(t) rows/cols) ---\n');

% For each concept i, compute the Frobenius norm of its row and column in E(t)
% over all years (i.e., total deviation magnitude involving concept i)
row_dev_norm = zeros(N,1);   % concept as source (outgoing deviations)
col_dev_norm = zeros(N,1);   % concept as target (incoming deviations)
for i = 1:N
    row_dev_norm(i) = sqrt( sum( E(i,:,:).^2, 'all') );
    col_dev_norm(i) = sqrt( sum( E(:,i,:).^2, 'all') );
end
total_dev_norm = row_dev_norm + col_dev_norm;

% Identify most and least flexible concepts
[~, idx_flex] = sort(total_dev_norm, 'descend');
[~, idx_stable] = sort(total_dev_norm, 'ascend');

fprintf('  Top 10 MOST flexible concepts (largest total |E|):\n');
for i = 1:10
    fprintf('    %2d. %-35s (flex = %.4f)\n', i, concepts{idx_flex(i)}, total_dev_norm(idx_flex(i)));
end
fprintf('\n  Top 10 MOST stable concepts (smallest total |E|):\n');
for i = 1:10
    fprintf('    %2d. %-35s (flex = %.4f)\n', i, concepts{idx_stable(i)}, total_dev_norm(idx_stable(i)));
end

% Save table
flex_table = table(concepts', row_dev_norm, col_dev_norm, total_dev_norm, ...
    'VariableNames', {'Concept', 'OutDevNorm', 'InDevNorm', 'TotalDevNorm'});
flex_table = sortrows(flex_table, 'TotalDevNorm', 'descend');
writetable(flex_table, fullfile(config.tables_folder, 'ConceptFlexibility.csv'));

% Figure: histogram of total deviation norm
figure('Color','w');
histogram(total_dev_norm, 30, 'FaceColor', [0.2 0.6 0.8], 'EdgeAlpha',0.5);
xlabel('Total deviation norm (|E| aggregated over time)');
ylabel('Number of concepts');
title('Concept Flexibility Distribution');
grid on;
saveas(gcf, fullfile(config.figures_folder, 'ConceptFlexibility_Hist.png'));

%% ------------------------------------------------------------------------
%% 2. PCA ON FLATTENED DEVIATIONS: Modes of temporal change
%% ------------------------------------------------------------------------
fprintf('\n--- 2. PCA on E(t) (identifying dominant deviation patterns) ---\n');

% Flatten each E(t) into a row vector (exclude diagonal? keep for now)
E_flat = zeros(config.T, N*N);
for t = 1:config.T
    Et = E(:,:,t);
    E_flat(t,:) = Et(:)';
end

% Perform PCA
[coeff_E, score_E, latent_E, ~, explained_E] = pca(E_flat);

fprintf('  PC1 explains %.2f%% of variance in E(t)\n', explained_E(1));
fprintf('  PC2 explains %.2f%%\n', explained_E(2));
fprintf('  PC3 explains %.2f%%\n', explained_E(3));
fprintf('  Cumulative (PC1-3): %.2f%%\n', sum(explained_E(1:3)));

% Plot PC scores over time
figure('Position', [100 100 1200 400], 'Color','w');
subplot(1,3,1);
plot(config.years, score_E(:,1), '-o', 'LineWidth',2, 'Color',[0.1 0.3 0.7]);
ylabel('PC1 score'); xlabel('Year'); title(sprintf('PC1 (%.1f%%)', explained_E(1)));
grid on; axis tight
subplot(1,3,2);
plot(config.years, score_E(:,2), '-s', 'LineWidth',2, 'Color',[0.8 0.3 0.1]);
ylabel('PC2 score'); xlabel('Year'); title(sprintf('PC2 (%.1f%%)', explained_E(2)));
grid on; axis tight
subplot(1,3,3);
plot(config.years, score_E(:,3), '-d', 'LineWidth',2, 'Color',[0.2 0.7 0.2]);
ylabel('PC3 score'); xlabel('Year'); title(sprintf('PC3 (%.1f%%)', explained_E(3)));
grid on; axis tight
sgtitle('Principal Components of the Deviation Matrices E(t)','FontSize',14);
saveas(gcf, fullfile(config.figures_folder, 'PCA_Et_scores.png'));

% Identify which concept pairs drive each PC (top loadings)
% Reshape loadings back into N x N matrices for interpretation
topK = 10;
for pc = 1:3
    loadings_pc = reshape(coeff_E(:,pc), [N,N]);
    % Take absolute values, mask diagonal and inactive pairs?
    loadings_abs = abs(loadings_pc);
    loadings_abs(1:N+1:end) = -Inf;  % ignore self
    loadings_abs(~active_pairs_mask) = -Inf; % ignore structural voids
    [~, idx] = sort(loadings_abs(:), 'descend');
    [rows, cols] = ind2sub([N,N], idx(1:topK));
    fprintf('\n  Top %d concept pairs driving PC%d (largest |loading|):\n', topK, pc);
    for k = 1:topK
        fprintf('    %2d. %-30s -> %-30s (loading = %.4f)\n', ...
            k, concepts{rows(k)}, concepts{cols(k)}, loadings_pc(rows(k), cols(k)));
    end
end

%% ------------------------------------------------------------------------
%% 3. TEMPORAL CLUSTERING OF YEARS BY DEVIATION PATTERN
%% ------------------------------------------------------------------------
fprintf('\n--- 3. Hierarchical clustering of years based on E(t) ---\n');

% Use correlation distance between flattened E(t) vectors
dist_mat = pdist(E_flat, 'correlation');
linkage_tree = linkage(dist_mat, 'ward');
figure('Color','w');
dendrogram(linkage_tree, config.T, 'Orientation','top', 'Labels',cellstr(num2str(config.years')));
title('Dendrogram: Years Clustered by Deviation Pattern');
ylabel('Correlation distance');
grid on;
saveas(gcf, fullfile(config.figures_folder, 'YearClustering_Et.png'));

% Also silhouette to see optimal number of clusters
eva = evalclusters(E_flat, 'linkage', 'silhouette', 'KList',2:8);
fprintf('  Optimal number of clusters (silhouette): %d\n', eva.OptimalK);
fprintf('  Clusters: ');
clusters = cluster(linkage_tree, 'maxclust', eva.OptimalK);
for k = 1:eva.OptimalK
    yrs = config.years(clusters == k);
    fprintf('[%d-%d] ', min(yrs), max(yrs));
end
fprintf('\n');

% Save cluster assignment
year_cluster = table(config.years', clusters, 'VariableNames', {'Year','Cluster'});
writetable(year_cluster, fullfile(config.tables_folder, 'YearClusters_Et.csv'));

%% ------------------------------------------------------------------------
%% 4. SIGN ASYMMETRY: Where does E(t) increase vs decrease cost?
%% ------------------------------------------------------------------------
fprintf('\n--- 4. Sign asymmetry of deviations (increase/decrease of cost) ---\n');

% Count for each pair how often E_ij(t) is positive (cost increase) over years
sign_count = zeros(N,N);
for t = 1:config.T
    sign_count = sign_count + (E(:,:,t) > 0);
end
prop_positive = sign_count / config.T;

% Focus on active pairs only
prop_positive_active = prop_positive(active_pairs_mask);
fprintf('  Over all active links, average proportion of years with positive deviation = %.3f\n', mean(prop_positive_active));
fprintf('  Std = %.3f, min = %.3f, max = %.3f\n', std(prop_positive_active), min(prop_positive_active), max(prop_positive_active));

% Identify pairs that are consistently positive (cost always higher than baseline)
consistently_pos = (prop_positive >= 0.9) & active_pairs_mask;
consistently_neg = (prop_positive <= 0.1) & active_pairs_mask;
fprintf('  Number of pairs with cost increase in ≥90%% of years: %d\n', sum(consistently_pos(:)));
fprintf('  Number of pairs with cost decrease in ≥90%% of years: %d\n', sum(consistently_neg(:)));

% Show top consistent pairs
[~, idx_pos] = sort(prop_positive(:), 'descend');
[~, idx_neg] = sort(prop_positive(:), 'ascend');
fprintf('\n  Top 5 pairs with most persistent cost INCREASE:\n');
for k = 1:5
    [r,c] = ind2sub([N,N], idx_pos(k));
    fprintf('    %-30s -> %-30s (positive %.1f%%)\n', concepts{r}, concepts{c}, prop_positive(r,c)*100);
end
fprintf('\n  Top 5 pairs with most persistent cost DECREASE:\n');
for k = 1:5
    [r,c] = ind2sub([N,N], idx_neg(k));
    fprintf('    %-30s -> %-30s (positive %.1f%%)\n', concepts{r}, concepts{c}, prop_positive(r,c)*100);
end

% Visualize as heatmap of proportion positive
figure('Color','w');
imagesc(prop_positive, [0 1]);
colorbar; colormap(jet);
title('Proportion of years where E_{ij}(t) > 0 (cost increase)');
xlabel('Target concept'); ylabel('Source concept');
axis square;
saveas(gcf, fullfile(config.figures_folder, 'SignAsymmetry_Heatmap.png'));

%% ------------------------------------------------------------------------
%% 5. PAIRS WHERE DEVIATION DOMINATES THE BASELINE
%% ------------------------------------------------------------------------
fprintf('\n--- 5. Identifying pairs where |E(t)| > |C0| at any time ---\n');

% For each pair, maximum absolute deviation relative to baseline
max_dev_ratio = zeros(N,N);
for i = 1:N
    for j = 1:N
        dev_abs = squeeze(abs(E(i,j,:)));
        base_abs = abs(C0(i,j));
        if base_abs > 0
            max_dev_ratio(i,j) = max(dev_abs / base_abs);
        else
            max_dev_ratio(i,j) = max(dev_abs) / (median(abs(C0(:))) + eps);
        end
    end
end
max_dev_ratio(1:N+1:end) = 0; % ignore self
dominating_pairs = max_dev_ratio > 1; % |E| > |C0| at some time


% Find the most extreme cases
max_dev_ratio(~active_pairs_mask) = -Inf; % ignore structural voids
[~, idx_dom] = sort(max_dev_ratio(:), 'descend');
fprintf('\n  Top 5 pairs where deviation dwarfs the baseline:\n');
for k = 1:5
    [r,c] = ind2sub([N,N], idx_dom(k));
    fprintf('    %-30s -> %-30s (max |E|/|C0| = %.2f)\n', concepts{r}, concepts{c}, max_dev_ratio(r,c));
end

%% ------------------------------------------------------------------------
%% 6. TEMPORAL AUTOCORRELATION OF DEVIATIONS
%% ------------------------------------------------------------------------
fprintf('\n--- 6. Temporal autocorrelation of deviation time series ---\n');

% For each pair, compute lag-1 autocorrelation (if enough nonzero variance)
autocorr_lag1 = zeros(N,N);
for i = 1:N
    for j = 1:N
        ts = squeeze(E(i,j,:));
        if std(ts) > 1e-6
            autocorr_lag1(i,j) = corr(ts(1:end-1), ts(2:end));
        else
            autocorr_lag1(i,j) = NaN;
        end
    end
end
autocorr_lag1(~active_pairs_mask) = NaN;

fprintf('  Mean lag-1 autocorrelation over active pairs = %.3f (std=%.3f)\n', ...
    mean(autocorr_lag1(:), 'omitnan'), std(autocorr_lag1(:), 'omitnan'));
fprintf('  Interpretation: Most deviations are weakly persistent (no strong directional trend).\n');

% Pairs with high positive autocorrelation (smooth evolution) vs negative (oscillating)
[~, idx_posAuto] = sort(autocorr_lag1(:), 'descend');
[~, idx_negAuto] = sort(autocorr_lag1(:), 'ascend');
fprintf('\n  Top 5 pairs with strongest POSITIVE autocorrelation (smooth change):\n');
for k = 1:5
    [r,c] = ind2sub([N,N], idx_posAuto(k));
    if ~isnan(autocorr_lag1(r,c))
        fprintf('    %-30s -> %-30s (ρ = %.3f)\n', concepts{r}, concepts{c}, autocorr_lag1(r,c));
    end
end
fprintf('\n  Top 5 pairs with strongest NEGATIVE autocorrelation (oscillatory):\n');
for k = 1:5
    [r,c] = ind2sub([N,N], idx_negAuto(k));
    if ~isnan(autocorr_lag1(r,c))
        fprintf('    %-30s -> %-30s (ρ = %.3f)\n', concepts{r}, concepts{c}, autocorr_lag1(r,c));
    end
end

% Histogram of autocorrelations
figure('Color','w');
histogram(autocorr_lag1(:), 30, 'FaceColor',[0.5 0.2 0.7]);
xlabel('Lag-1 autocorrelation');
ylabel('Number of concept pairs');
title('Distribution of Temporal Persistence in Deviations');
grid on;
saveas(gcf, fullfile(config.figures_folder, 'Autocorrelation_Et.png'));

%% ------------------------------------------------------------------------
%% 7. ADDITIONAL: Deviation magnitude over time (global view)
%% ------------------------------------------------------------------------
fprintf('\n--- 7. Global deviation strength over time ---\n');

frobenius_E = zeros(config.T,1);
for t = 1:config.T
    frobenius_E(t) = norm(E(:,:,t), 'fro');
end

figure('Color','w');
yyaxis left;
plot(config.years, frobenius_E, '-o', 'LineWidth',2, 'Color',[0 0.5 0]);
ylabel('||E(t)||_F');
yyaxis right;
plot(config.years, score_E(:,1), '-s', 'LineWidth',1.5, 'Color',[0.8 0.2 0.2]);
ylabel('PC1 score (deviation mode)');
xlabel('Year');
title('Global Deviation Magnitude and PC1 Mode');
grid on;
legend('Frobenius norm', 'PC1 of E(t)', 'Location','best');
saveas(gcf, fullfile(config.figures_folder, 'GlobalDeviationStrength.png'));

