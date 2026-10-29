
clear; clc; close all; warning off

%% ══════════════════════════════════════════════════════════════════

% CONFIGURATION & SETUP

%% ══════════════════════════════════════════════════════════════════

config.results_folder = 'results_bayesopt'; % Python Inverse UOT output folder
config.output_folder = 'rq2_analysis_output';
config.figures_folder = fullfile(config.output_folder, 'figures');
config.tables_folder = fullfile(config.output_folder, 'tables');
config.years = 2000:2023;
config.T = length(config.years);
% Create output directories
if ~exist(config.figures_folder, 'dir'), mkdir(config.figures_folder); end
if ~exist(config.tables_folder, 'dir'), mkdir(config.tables_folder); end

fprintf('╔════════════════════════════════════════════════════════╗\n');
fprintf('║ EVOLUTION OF THE UOT KNOWLEDGE GEOMETRY ║\n');
fprintf('╚════════════════════════════════════════════════════════╝\n\n');
%% ══════════════════════════════════════════════════════════════════
%  1. LOAD DATA (N = 300 Concepts, T = 24 Years)
%% ══════════════════════════════════════════════════════════════════

fprintf('Loading Inverse UOT parameters & Original Data...\n');

% Load time-invariant baseline C0
C0_table = readtable(fullfile(config.results_folder, 'C0_baseline.csv'));
concepts = C0_table.Properties.VariableNames(2:end);
N = length(concepts);
C0 = table2array(C0_table(:, 2:end));

% Load time-specific deviations E(t) and full cost C(t)
E = zeros(N, N, config.T);
C_full = zeros(N, N, config.T);

for t = 1:config.T
    year = config.years(t);
    E_table = readtable(fullfile(config.results_folder, sprintf('E_%d.csv', year)));
    C_table = readtable(fullfile(config.results_folder, sprintf('C_%d.csv', year)));
    E(:, :, t) = table2array(E_table(:, 2:end));
    C_full(:, :, t) = table2array(C_table(:, 2:end));
end

% LOAD ORIGINAL CITATION TENSOR to identify structural voids
raw_data = load(fullfile(config.results_folder, 'original_tensor.mat'));
T_tensor = raw_data.T_tensor; % Shape: 300 x 300 x 24
total_flow = sum(T_tensor, 3); % Total citations over 24 years

% Create a mask: Pair must have at least 1 citation to be considered "active"
flow_threshold = prctile(total_flow(:), 50); % The median flow value
active_pairs_mask = total_flow > flow_threshold;

fprintf('  ✓ Loaded OT Parameters.\n');
fprintf('  ✓ Loaded Original Tensor. Found %.2f%% active links, %.2f%% structural voids.\n\n', ...
    (sum(active_pairs_mask(:)) / N^2)*100, (1 - sum(active_pairs_mask(:)) / N^2)*100);

%% ══════════════════════════════════════════════════════════════════
%  2. VARIANCE DECOMPOSITION: BASELINE VS. STRUCTURAL SHOCKS
%% ══════════════════════════════════════════════════════════════════
fprintf('--- 1. Variance Decomposition (C0 vs E_t) ---\n');

% Global Magnitude Ratio
norm_C0 = norm(C0, 'fro');
norm_E_t = zeros(config.T, 1);
for t = 1:config.T
    norm_E_t(t) = norm(E(:,:,t), 'fro');
end
avg_shock_magnitude = mean(norm_E_t / norm_C0) * 100;

fprintf('  Global Metric (Relative Magnitude):\n');
fprintf('    On average, time-specific shocks E(t) are %.2f%% the size of the baseline C0.\n', avg_shock_magnitude);
fprintf('    This implies the latent geometry is %.2f%% stable baseline and %.2f%% temporal fluctuation.\n\n', ...
    100 - avg_shock_magnitude, avg_shock_magnitude);

% -------------------------------------------------------------------
% Stabilized Relative Volatility
% CV = std(E_t) / (|C0| + median(|C0|))
% -------------------------------------------------------------------
E_std = std(E, 0, 3);
C0_abs = abs(C0);
stabilizer = median(C0_abs(:)); % Prevents division by zero
CV_matrix = E_std ./ (C0_abs + stabilizer);

% Safely mask the diagonal (self-connections)
CV_descend = CV_matrix;
CV_descend(1:N+1:end) = -Inf; % Pushes diagonal to bottom for max sort

% For finding "Persistent" links, we mask BOTH the diagonal AND structural voids
CV_ascend = CV_matrix;
CV_ascend(1:N+1:end) = Inf;         % Ignore self-connections
CV_ascend(~active_pairs_mask) = Inf; % IGNORE STRUCTURAL VOIDS (No zero-citation links)

% --- Extract Top TRANSIENT Pairs ---
[CV_sorted_desc, idx_desc] = sort(CV_descend(:), 'descend');
[row_vol, col_vol] = ind2sub([N, N], idx_desc(1:10));

volatile_pairs = table(concepts(row_vol)', concepts(col_vol)', ...
    C0(idx_desc(1:10)), E_std(idx_desc(1:10)), CV_sorted_desc(1:10), ...
    'VariableNames', {'Source', 'Target', 'Baseline_Cost_C0', 'Shock_Std_E', 'Relative_Volatility'});

% --- Extract Top PERSISTENT Pairs (ACTIVE LINKS ONLY) ---
[CV_sorted_asc, idx_asc] = sort(CV_ascend(:), 'ascend');
[row_stab, col_stab] = ind2sub([N, N], idx_asc(1:10));

stable_pairs = table(concepts(row_stab)', concepts(col_stab)', ...
    C0(idx_asc(1:10)), E_std(idx_asc(1:10)), CV_sorted_asc(1:10), ...
    'VariableNames', {'Source', 'Target', 'Baseline_Cost_C0', 'Shock_Std_E', 'Relative_Volatility'});

% --- Print Output ---
disp('======================================================================================');
disp('   TOP 10 MOST TRANSIENT CONCEPT LINKS (Highest Relative Shocks)');
disp('======================================================================================');
disp(volatile_pairs);

disp('======================================================================================');
disp('   TOP 10 MOST PERSISTENT ACTIVE LINKS (The True Core of Science)');
disp('======================================================================================');
disp(stable_pairs);

writetable(volatile_pairs, fullfile(config.tables_folder, 'RQ2_Transient_Links_CV.csv'));
writetable(stable_pairs, fullfile(config.tables_folder, 'RQ2_Persistent_Links_CV.csv'));
%% ══════════════════════════════════════════════════════════════════
%  3. PCA: EXTRACTING LATENT EVOLUTIONARY TRENDS
%% ══════════════════════════════════════════════════════════════════
fprintf('--- 2. Principal Component Analysis (Temporal Dynamics) ---\n');

% Flatten matrices to observe temporal structural shifts
C_flat = reshape(C_full, [N^2, config.T])'; % (T x N^2)

% Run PCA
[coeff, score, latent, tsquared, explained] = pca(C_flat);

fprintf('  PC1 (Dominant stationary structure): %.2f%% variance\n', explained(1));
fprintf('  PC2 (Primary trend/shock): %.2f%% variance\n', explained(2));
fprintf('  PC3 (Secondary shock/realignment): %.2f%% variance\n', explained(3));
fprintf('  Cumulative Variance (PC1-3): %.2f%%\n', sum(explained(1:3)));

% -------------------------------------------------------------------
% PUBLICATION-READY 2x2 PCA PLOT
% -------------------------------------------------------------------
figure('Position', [100, 100, 1000, 800], 'Color', 'w'); 

% 1. Variance Explained (Pareto)
subplot(2, 2, 1);
pareto(explained(1:10)); 
title('Variance Explained by Components', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Principal Component'); ylabel('Variance Explained (%)');
grid on;
axis tight
% 2. PC1 Evolution
subplot(2, 2, 2);
plot(config.years, score(:, 1), '-o', 'LineWidth', 2.5, 'Color', [0 0.4470 0.7410], 'MarkerFaceColor', [0 0.4470 0.7410]);
yline(0, 'k--', 'LineWidth', 1);
title(sprintf('PC1: The Secular Trend (%.1f%%)', explained(1)), 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Year'); ylabel('PC1 Score');
grid on;
axis tight
% 3. PC2 Evolution
subplot(2, 2, 3);
plot(config.years, score(:, 2), '-o', 'LineWidth', 2.5, 'Color', [0.8500 0.3250 0.0980], 'MarkerFaceColor', [0.8500 0.3250 0.0980]);
yline(0, 'k--', 'LineWidth', 1);
title(sprintf('PC2: Structural Realignment (%.1f%%)', explained(2)), 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Year'); ylabel('PC2 Score');
grid on;
axis tight
% 4. PC3 Evolution (COVID-19 Highlight)
subplot(2, 2, 4);
plot(config.years, score(:, 3), '-o', 'LineWidth', 2.5, 'Color', [0.4660 0.6740 0.1880], 'MarkerFaceColor', [0.4660 0.6740 0.1880]);
yline(0, 'k--', 'LineWidth', 1);
xline(2020, 'r:', 'LineWidth', 2); % Highlight the COVID-19 year
text(2020.5, max(score(:,3))*0.9, 'COVID-19 Shock', 'Color', 'r', 'FontWeight', 'bold');
title(sprintf('PC3: Exogenous Shocks (%.1f%%)', explained(3)), 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Year'); ylabel('PC3 Score');
grid on;
axis tight
sgtitle('Temporal PCA of Latent UOT Cost Matrices', 'FontSize', 16, 'FontWeight', 'bold');
saveas(gcf, fullfile(config.figures_folder, 'RQ2_PCA_Evolution_2x2.png'));


%% ══════════════════════════════════════════════════════════════════
%  3b. RELATIVE SPECIFICITY: HARD PARTITION OF PCA CONCEPTS
%% ══════════════════════════════════════════════════════════════════
fprintf('\n--- 3b. Concept Specificity (Hard Partitioning) ---\n');

% Calculate Absolute Impact for first 3 PCs
impact_matrix = zeros(N, 3);
for pc = 1:3
    W_pc = reshape(coeff(:, pc), [N, N]);
    impact_matrix(:, pc) = sum(abs(W_pc), 2) + sum(abs(W_pc), 1)'; 
end

% 1. Calculate Relative Specificity (Row normalization)
% What % of a concept's total PC1-3 impact is dedicated to each specific PC?
total_impact = sum(impact_matrix, 2);
relative_specificity = impact_matrix ./ (total_impact + eps);

% 2. Hard Partition: Assign concept to the PC where it is most concentrated
[~, assigned_pc] = max(relative_specificity, [], 2);

% Print Top Concepts explicitly partitioned into their dominating PC
top_k = 10;
for pc = 1:3
    fprintf('  Top %d Concepts UNIQUELY driving PC%d:\n', top_k, pc);
    
    % Filter only concepts that were assigned to THIS PC
    valid_idx = find(assigned_pc == pc);
    
    if isempty(valid_idx)
        fprintf('    [None assigned strictly to this PC]\n\n');
        continue;
    end
    
    % Of those assigned to this PC, sort by their Absolute Magnitude
    % (We want the concepts that are both specifically in this PC, and very large)
    pc_magnitudes = impact_matrix(valid_idx, pc);
    [~, sort_idx] = sort(pc_magnitudes, 'descend');
    sorted_valid_idx = valid_idx(sort_idx);
    
    for i = 1:min(top_k, length(sorted_valid_idx))
        c_idx = sorted_valid_idx(i);
        fprintf('    %2d. %-30s (Impact: %.4f, Specificity: %.1f%%)\n', ...
            i, concepts{c_idx}, impact_matrix(c_idx, pc), relative_specificity(c_idx, pc)*100);
    end
    fprintf('\n');
end

%% ══════════════════════════════════════════════════════════════════
%  4. STATIC TOPOLOGY: CENTRALITY & MODULARITY (C0)
%% ══════════════════════════════════════════════════════════════════
fprintf('\n--- 4. Topological Analysis of the Baseline Geometry (C0) ---\n');

% Ensure Brain Connectivity Toolbox is loaded
addpath('E:\2019_03_03_BCT');

% 1. DIRECTED AFFINITY MATRIX (A0)
% Shift C0 to be non-negative, convert to Affinity via Gibbs Kernel
C0_shifted = C0 - min(C0(:));
epsilon_0 = median(C0_shifted(:));
A0 = exp(-C0_shifted); % Directed Gibbs Kernel
A0(1:N+1:end) = 0; % Remove self-loops

% -------------------------------------------------------------------
%  4a. CENTRALITY ANALYSIS (In-Strength & Out-Strength)
% -------------------------------------------------------------------
fprintf('  Computing Directed Centrality Metrics...\n');

% In-Strength: Incoming affinities (col sums). How heavily is this cited?
in_strength = sum(A0, 1)';

% Out-Strength: Outgoing affinities (row sums). How broadly does this cite?
out_strength = sum(A0, 2);

% Net Flow (Exporters vs Importers)
net_flow = out_strength - in_strength;

% Sort to identify top concepts
[sorted_in, idx_in] = sort(in_strength, 'descend');
[sorted_out, idx_out] = sort(out_strength, 'descend');

% Print Top Concepts to Console
fprintf('\n  Top 10 Knowledge Sinks (Highest In-Strength / Target Affinity):\n');
for i = 1:10
    fprintf('    %2d. %-30s (In-Strength: %.4f)\n', i, concepts{idx_in(i)}, sorted_in(i));
end

fprintf('\n  Top 10 Knowledge Sources (Highest Out-Strength / Source Affinity):\n');
for i = 1:10
    fprintf('    %2d. %-30s (Out-Strength: %.4f)\n', i, concepts{idx_out(i)}, sorted_out(i));
end

% Save Centrality Metrics to CSV
centrality_table = table(concepts', in_strength, out_strength, net_flow, ...
    'VariableNames', {'Concept', 'In_Strength', 'Out_Strength', 'Net_Flow'});
writetable(centrality_table, fullfile(config.tables_folder, 'RQ2_Baseline_Centrality.csv'));

% --- FIGURE 1: ENHANCED CENTRALITY DISTRIBUTIONS & TOP CONCEPTS ---
figure('Position', [100, 100, 1200, 800], 'Color', 'w');
tlo1 = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Tile 1: Scatter (In vs Out Strength)
nexttile;
% Color by Net Flow to emphasize Exporters (Red/Positive) vs Importers (Blue/Negative)
scatter(in_strength, out_strength, 60, net_flow, 'filled', 'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.8);
colormap(gca, jet); c = colorbar; c.Label.String = 'Net Flow (Out - In)';
hold on;
max_val = max([in_strength; out_strength]);
min_val = min([in_strength; out_strength]);

plot([min_val, max_val], [min_val, max_val], 'k--', 'LineWidth', 1.5); % Unity Line

% Annotate Top 3 Net Exporters and Importers
[~, idx_export] = sort(net_flow, 'descend');
[~, idx_import] = sort(net_flow, 'ascend');
for i = 1:3
    text(in_strength(idx_export(i)), out_strength(idx_export(i)), ...
        [' \leftarrow ' strrep(concepts{idx_export(i)}, '_', ' ')], 'FontSize', 9, 'Color', [0.85 0.32 0.09], 'FontWeight', 'bold');
    text(in_strength(idx_import(i)), out_strength(idx_import(i)), ...
        [' \leftarrow ' strrep(concepts{idx_import(i)}, '_', ' ')], 'FontSize', 9, 'Color', [0 0.44 0.74], 'FontWeight', 'bold');
end
hold off;
xlabel('In-Strength (Target Affinity)'); ylabel('Out-Strength (Source Affinity)');
title('Directional Imbalance', 'FontWeight', 'bold');
grid on;
axis tight
% Tile 2: Overlaid Distributions
nexttile;
histogram(in_strength, 25, 'FaceColor', [0 0.44 0.74], 'EdgeColor', 'w', 'FaceAlpha', 0.6);
hold on;
histogram(out_strength, 25, 'FaceColor', [0.85 0.32 0.09], 'EdgeColor', 'w', 'FaceAlpha', 0.6);
hold off;
xlabel('Affinity Strength'); ylabel('Frequency');
legend('In-Strength (Targets)', 'Out-Strength (Sources)', 'Location', 'northeast');
title('Strength Distributions', 'FontWeight', 'bold');
grid on;
axis tight
% Tile 3: Top 10 In-Strength Bar Chart
nexttile;
barh(1:10, sorted_in(1:10), 'FaceColor', [0 0.44 0.74], 'EdgeColor', 'k');
yticks(1:10);
% Clean up underscores for display
clean_in_labels = strrep(concepts(idx_in(1:10)), '_', ' '); 
yticklabels(clean_in_labels);
set(gca, 'YDir', 'reverse'); % Put #1 at the top
xlabel('In-Strength');
title('Top 10 Most Targeted', 'FontWeight', 'bold');
grid on;
axis tight
% Tile 4: Top 10 Out-Strength Bar Chart
nexttile;
barh(1:10, sorted_out(1:10), 'FaceColor', [0.85 0.32 0.09], 'EdgeColor', 'k');
yticks(1:10);
clean_out_labels = strrep(concepts(idx_out(1:10)), '_', ' ');
yticklabels(clean_out_labels);
set(gca, 'YDir', 'reverse'); % Put #1 at the top
xlabel('Out-Strength');
title('Top 10 Broadest Citers', 'FontWeight', 'bold');
grid on;
axis tight
title(tlo1, 'Baseline Topology (C^0): Centrality & Directed Flow', 'FontSize', 16, 'FontWeight', 'bold');
saveas(gcf, fullfile(config.figures_folder, 'RQ2_C0_Centrality_Distributions.png'));

% -------------------------------------------------------------------
%  4b. COMMUNITY DETECTION & WORD CLOUDS (BCT Louvain)
% -------------------------------------------------------------------
fprintf('\n  Detecting Mega-Disciplines using BCT Louvain Modularity...\n');

gamma_res = 1.0; 
[c0_communities, Q_modularity] = community_louvain(A0, gamma_res);
num_communities = length(unique(c0_communities));

fprintf('  Found %d Mega-Disciplines (Modularity Q = %.4f)\n\n', num_communities, Q_modularity);

% --- FIGURE 2: MODULARITY MATRIX & Tiled WORD CLOUDS ---
figure('Position', [50, 50, 1600, 850], 'Color', 'w');

% Define a dynamic layout grid. Max 8 word clouds.
max_clouds = min(num_communities, 8);
grid_rows = ceil(max_clouds / 2);
tlo2 = tiledlayout(grid_rows, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

% 1. Plot Modularity Matrix (Spans all rows, and the first 2 columns)
ax_mat = nexttile(1, [grid_rows, 2]);
[~, sort_comm_idx] = sort(c0_communities);
A0_sorted = A0(sort_comm_idx, sort_comm_idx);

imagesc(ax_mat, A0_sorted);
colormap(ax_mat, flipud(hot)); 
cbar = colorbar(ax_mat); cbar.Label.String = 'Directed Affinity';
%title(ax_mat, sprintf('Modular Backbone of Science (Q = %.2f)', Q_modularity), 'FontWeight', 'bold', 'FontSize', 14);
xlabel(ax_mat, 'Target Concepts (Sorted by Community)');
ylabel(ax_mat, 'Source Concepts (Sorted by Community)');
axis(ax_mat, 'square');

% Draw Community Boundaries on the matrix
hold(ax_mat, 'on');
current_idx = 0;
for k = 1:num_communities
    comm_size = sum(c0_communities == k);
    rectangle(ax_mat, 'Position', [current_idx+0.5, current_idx+0.5, comm_size, comm_size], ...
              'EdgeColor', 'b', 'LineWidth', 2.0);
    text(ax_mat, current_idx + comm_size/2, current_idx + comm_size/2, sprintf('C%d', k), ...
        'Color', 'c', 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    current_idx = current_idx + comm_size;
end
hold(ax_mat, 'off');

% 2. Plot Word Clouds in the remaining tiles (columns 3 and 4)
tile_indices = [];
for r = 1:grid_rows
    tile_indices = [tile_indices, (r-1)*4 + 3, (r-1)*4 + 4];
end

for k = 1:max_clouds
    % Select the specific tile for this word cloud
    nexttile(tile_indices(k));
    
    comm_idx = find(c0_communities == k);
    clean_words = strrep(concepts(comm_idx)', '_', ' '); % Clean underscores
    sizes = in_strength(comm_idx) + out_strength(comm_idx);
    
    wordcloud(clean_words, sizes, 'HighlightColor', [0 0.4470 0.7410]);
    title(sprintf('C%d (%d concepts)', k, length(comm_idx)));
end

title(tlo2, 'Network Modularity & Semantic Communities of the Baseline', 'FontSize', 16, 'FontWeight', 'bold');
saveas(gcf, fullfile(config.figures_folder, 'RQ2_Baseline_Modularity_WordClouds.png'));

fprintf('  ✓ Static Network analysis complete. Figures saved.\n\n');
%% ══════════════════════════════════════════════════════════════════
%  5. DYNAMIC TOPOLOGY: INFORMATION FLOW & COMMUNITY MIGRATION
%% ══════════════════════════════════════════════════════════════════
fprintf('\n--- 5. Dynamic Topological Evolution (2000-2023) ---\n');

% Preallocate tracking metrics
modularity_t = zeros(config.T, 1);
communities_t = zeros(N, config.T);
avg_path_t = zeros(config.T, 1);
diameter_t = zeros(config.T, 1);
clust_wei_t = zeros(config.T, 1);
clust_bin_t = zeros(config.T, 1);
assort_out_in = zeros(config.T, 1);
assort_in_out = zeros(config.T, 1);
assort_out_out = zeros(config.T, 1);
assort_in_in = zeros(config.T, 1);

fprintf('  Computing BCT Metrics (Path Length, Clustering, Assortativity)...\n');
for t = 1:config.T
    % 1. Convert C_t to Directed Affinity
    Ct = C_full(:,:,t);
    Ct_shifted = Ct - min(C0(:)); 
    At = exp(-Ct_shifted); % Assuming epsilon=1 or previously defined
    At(1:N+1:end) = 0; % No self-loops
    
    % 2. Distance Matrix (for Path Length and Diameter)
    Lt = 1 ./ At;
    Lt(isinf(Lt)) = 0;
    Lt(1:N+1:end) = 0;
    D = distance_wei(Lt);
    
    % Extract valid (reachable) off-diagonal distances
    D_off = D(~eye(N));
    D_valid = D_off(~isinf(D_off) & D_off > 0);
    avg_path_t(t) = mean(D_valid);
    diameter_t(t) = max(D_valid);
    
    % 3. Clustering Coefficient (Weighted & Binary)
    clust_wei_t(t) = mean(clustering_coef_wd(At));
    
    % Create Binary matrix (Threshold at mean affinity to define clear edges)
    At_bin = At > mean(At(:));
    clust_bin_t(t) = mean(clustering_coef_bd(At_bin));
    
    % 4. Assortativity (4 Cases for Directed Networks)
    assort_out_in(t)  = assortativity_wei(At, 1); % Out-degree vs In-degree
    assort_in_out(t)  = assortativity_wei(At, 2); % In-degree vs Out-degree
    assort_out_out(t) = assortativity_wei(At, 3); % Out-degree vs Out-degree
    assort_in_in(t)   = assortativity_wei(At, 4); % In-degree vs In-degree
    
    % 5. Community Detection
    [communities_t(:, t), modularity_t(t)] = community_louvain(At, 1.0);
end

% -------------------------------------------------------------------
%  5a. FIGURE 1: INFORMATION FLOW & MICRO-TOPOLOGY
% -------------------------------------------------------------------
figure('Position', [50, 100, 1500, 450], 'Color', 'w');
tlo1 = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel 1: Path Length & Diameter (Double Y-Axis)
nexttile;
yyaxis left;
plot(config.years, avg_path_t, '-o', 'LineWidth', 2.5, 'Color', [0 0.4470 0.7410]);
ylabel('Avg Path Length', 'Color', [0 0.4470 0.7410]);
set(gca, 'YColor', [0 0.4470 0.7410]);
axis tight
yyaxis right;
plot(config.years, diameter_t, '-s', 'LineWidth', 2.5, 'Color', [0.8500 0.3250 0.0980]);
ylabel('Network Diameter', 'Color', [0.8500 0.3250 0.0980]);
set(gca, 'YColor', [0.8500 0.3250 0.0980]);
axis tight
xlabel('Year'); title('A. Knowledge Distance', 'FontWeight', 'bold');
grid on;

% Panel 2: Clustering Coefficient
nexttile;
yyaxis left;
plot(config.years, clust_wei_t, '-o', 'LineWidth', 2.5, 'Color', [0 0.4470 0.7410]); hold on;
ylabel('Weighted Clustering Coefficient', 'Color', [0 0.4470 0.7410]);
yyaxis right;
plot(config.years, clust_bin_t, '-^', 'LineWidth', 2.5, 'Color', [0.8500 0.3250 0.0980]); hold off;
ylabel('Binary', 'Color', [0.8500 0.3250 0.0980]);
axis tight
xlabel('Year'); ylabel('Binary Clustering Coefficient');
legend('Weighted (Intensity)', 'Binary (Structure)', 'Location', 'best');
title('Local Neighborhood Density (Clustering)', 'FontWeight', 'bold');
grid on;
axis tight
% Panel 3: Assortativity
nexttile;
plot(config.years, assort_out_in, '-o', 'LineWidth', 2, 'DisplayName', 'Out \rightarrow In'); hold on;
plot(config.years, assort_in_out, '-s', 'LineWidth', 2, 'DisplayName', 'In \rightarrow Out');
plot(config.years, assort_out_out, '-d', 'LineWidth', 2, 'DisplayName', 'Out \rightarrow Out');
plot(config.years, assort_in_in, '-^', 'LineWidth', 2, 'DisplayName', 'In \rightarrow In');
xlabel('Year'); ylabel('Assortativity (r)');
legend('Location', 'best');
title('Assortativity', 'FontWeight', 'bold');
grid on;
axis tight
title(tlo1, 'Dynamic Topology: Information Flow Architecture', 'FontSize', 16, 'FontWeight', 'bold');
saveas(gcf, fullfile(config.figures_folder, 'RQ2_Dynamic_InformationFlow.png'));

% -------------------------------------------------------------------
%  5b. PARTITION DISTANCE & MIGRATOR IDENTIFICATION
% -------------------------------------------------------------------
fprintf('  Computing Variation of Information (Partition Distance)...\n');

VI_consecutive = zeros(config.T, 1); % Volatility from previous year
VI_baseline    = zeros(config.T, 1); % Drift from Year 2000

for t = 2:config.T
    % Variation of Information (VI) between t-1 and t
    [VI_cons, MI_cons] = partition_distance(communities_t(:, t-1), communities_t(:, t));
    VI_consecutive(t) = MI_cons;
    
    % Variation of Information (VI) between 2000 and t
    [VI_base, MI_base] = partition_distance(communities_t(:, 1), communities_t(:, t));
    VI_baseline(t) = MI_base;
end

% Align community labels over time for the Heatmap (so colors stay consistent)
aligned_comm = communities_t;
for t = 2:config.T
    old_c = aligned_comm(:, t-1);
    new_c = communities_t(:, t);
    mapped_c = zeros(N,1);
    unq_new = unique(new_c);
    
    for i = 1:length(unq_new)
        idx_new = find(new_c == unq_new(i));
        best_match = mode(old_c(idx_new)); % Find most common past label
        mapped_c(idx_new) = best_match;
    end
    aligned_comm(:, t) = mapped_c;
end

% Identify Top 20 Migrators (Boundary Spanners) based on aligned labels
instability_score = zeros(N, 1);
for t = 2:config.T
    for i = 1:N
        prev_mates = aligned_comm(:, t-1) == aligned_comm(i, t-1);
        curr_mates = aligned_comm(:, t) == aligned_comm(i, t);
        intersection = sum(prev_mates & curr_mates);
        union_mates = sum(prev_mates | curr_mates);
        instability_score(i) = instability_score(i) + (1 - intersection/union_mates);
    end
end

[~, idx_migrators] = sort(instability_score, 'descend');
num_migrators = 20;
top_migrator_idx = idx_migrators(1:num_migrators);
disp(strrep(concepts(top_migrator_idx), '_', ' '));
%% -------------------------------------------------------------------
%  5c. FIGURE 2 + TABLE: DYNAMIC COMMUNITY ANALYSIS
% -------------------------------------------------------------------

% ============================================================
% FIGURE: ONLY TWO PANELS
% ============================================================

figure();
tlo2 = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% ------------------------------------------------------------
% Panel 1: Modularity Evolution
% ------------------------------------------------------------
nexttile;

plot(config.years, modularity_t, '-o', ...
    'LineWidth', 2.5, ...
    'Color', [0.4940 0.1840 0.5560], ...
    'MarkerFaceColor', [0.4940 0.1840 0.5560]);

xlabel('Year');
ylabel('Modularity (Q)');
axis tight
title('Modularity Evolution of the Knowledge System', ...
    'FontWeight', 'bold');

grid on;
box on;

% ------------------------------------------------------------
% Panel 2: Structural Reorganization
% ------------------------------------------------------------
nexttile;

hold on;

plot(config.years(2:end), VI_consecutive(2:end), '-o', ...
    'LineWidth', 2.5, ...
    'Color', [0.8500 0.3250 0.0980], ...
    'MarkerFaceColor', [0.8500 0.3250 0.0980], ...
    'DisplayName', 'Local Volatility (t-1 \rightarrow t)');
    axis tight
plot(config.years(2:end), VI_baseline(2:end), '-s', ...
    'LineWidth', 2.5, ...
    'Color', [0 0.4470 0.7410], ...
    'MarkerFaceColor', [0 0.4470 0.7410], ...
    'DisplayName', 'Cumulative Drift (2000 \rightarrow t)');
    axis tight
hold off;

xlabel('Year');
ylabel('Mutual Information (MI)');

title('Structural Reorganization of Scientific Communities', ...
    'FontWeight', 'bold');

legend('Location', 'northwest');
grid on;
box on;

title(tlo2, ...
    'Dynamic Evolution of Scientific Communities', ...
    'FontSize', 16, ...
    'FontWeight', 'bold');

saveas(gcf, ...
    fullfile(config.figures_folder, ...
    'RQ2_Dynamic_Communities.png'));

%% ============================================================
% TOP MIGRATORS TABLE
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('TOP MIGRATING CONCEPTS (BOUNDARY SPANNERS)\n');
fprintf('============================================================\n');

num_migrators = 20;

top_idx = idx_migrators(1:num_migrators);

% Preallocate
migration_count = zeros(num_migrators,1);
migration_index = zeros(num_migrators,1);

first_migration_year = NaN(num_migrators,1);
last_migration_year  = NaN(num_migrators,1);

for k = 1:num_migrators

    idx = top_idx(k);

    % Community trajectory through time
    traj = aligned_comm(idx,:);

    % Detect changes
    changes = find(diff(traj) ~= 0);

    % Number of migrations
    migration_count(k) = length(changes);

    % Relative migration index
    migration_index(k) = instability_score(idx) / (config.T - 1);

    % Migration years
    if ~isempty(changes)

        first_migration_year(k) = config.years(changes(1)+1);

        last_migration_year(k) = config.years(changes(end)+1);

    end

end

% Clean concept names
clean_names = strrep(concepts(top_idx), '_', ' ');

% Build table
migrator_table = table( ...
    (1:num_migrators)', ...
    clean_names', ...
    instability_score(top_idx), ...
    migration_index, ...
    migration_count, ...
    first_migration_year, ...
    last_migration_year, ...
    'VariableNames', ...
    {'Rank', ...
     'Concept', ...
     'InstabilityScore', ...
     'RelativeMigrationIndex', ...
     'NumMigrations', ...
     'FirstMigrationYear', ...
     'LastMigrationYear'});

% Sort by migration index
migrator_table = sortrows(migrator_table, ...
    'RelativeMigrationIndex', 'descend');

% Display
disp(migrator_table);

% Save CSV
writetable(migrator_table, ...
    fullfile(config.tables_folder, ...
    'RQ2_Top_Migrators.csv'));

fprintf('✓ Dynamic community analysis completed.\n');
fprintf('✓ Figure and migrator table saved.\n\n');