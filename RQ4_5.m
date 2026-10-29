%% ═══════════════════════════════════════════════════════════════════════
%  PREDICTIVE POWER & STRUCTURAL SHOCK DETECTION
%  
%% ═══════════════════════════════════════════════════════════════════════

clear; clc; close all;

%% ══════════════════════════════════════════════════════════════════
%  0. LOAD DATA
%% ══════════════════════════════════════════════════════════════════
config.results_folder = 'results_bayesopt';  
config.output_folder  = 'rq3_rq4_analysis_output';
config.figures_folder = fullfile(config.output_folder, 'figures');
config.tables_folder  = fullfile(config.output_folder, 'tables');
config.years          = 2000:2023;
config.T              = length(config.years);

if ~exist(config.figures_folder, 'dir'), mkdir(config.figures_folder); end
if ~exist(config.tables_folder, 'dir'), mkdir(config.tables_folder); end

fprintf('Loading Latent OT Parameters & Empirical Citation Tensor...\n');

C0_table = readtable(fullfile(config.results_folder, 'C0_baseline.csv'));
concepts = strrep(C0_table.Properties.VariableNames(2:end), '_', ' '); % Clean names
N = length(concepts);
C0 = table2array(C0_table(:, 2:end));

E = zeros(N, N, config.T);
C_full = zeros(N, N, config.T);
for t = 1:config.T
    E_table = readtable(fullfile(config.results_folder, sprintf('E_%d.csv', config.years(t))));
    C_table = readtable(fullfile(config.results_folder, sprintf('C_%d.csv', config.years(t))));
    E(:, :, t) = table2array(E_table(:, 2:end));
    C_full(:, :, t) = table2array(C_table(:, 2:end));
end

raw_data = load(fullfile(config.results_folder, 'original_tensor.mat'));
T_flow = raw_data.T_tensor; 

fprintf('  ✓ Data loaded.\n\n');

%% ══════════════════════════════════════════════════════════════════
%  PART 1: PREDICTING CITATION SURGES (WITH UNCERTAINTY BOUNDS)
%% ══════════════════════════════════════════════════════════════════
fprintf('--- Part 1: Predicting Scientific Momentum (Citation Surges) ---\n');

delta_cost_all = [];
surge_labels_all = [];
yearly_corr = zeros(config.T-2, 1);
corr_upper_95 = zeros(config.T-2, 1);
corr_lower_95 = zeros(config.T-2, 1);

for t = 2:config.T-1
    % 1. Calculate Cost Velocity at t
    delta_cost_t = exp(-C_full(:,:,t)) - exp(-C_full(:,:,t-1));
    
    % 2. Calculate Citation Growth at t+1
    flow_t   = T_flow(:,:,t);
    flow_tp1 = T_flow(:,:,t+1);
    
    stabilizer = median(flow_t(flow_t > 0));
    growth_tp1 = (flow_tp1 - flow_t) ./ (flow_t + stabilizer);
    
    idx_off_diag = ~eye(N);
    dc_vec = delta_cost_t(idx_off_diag);
    g_vec = growth_tp1(idx_off_diag);
    
    median_growth = median(g_vec);
    surge_label = g_vec > median_growth;
    
    delta_cost_all = [delta_cost_all; dc_vec];
    surge_labels_all = [surge_labels_all; surge_label];
    
    % Spearman Correlation with 95% CI (Fisher Z-transform)
    r = corr(dc_vec, g_vec, 'Type', 'Spearman');
    yearly_corr(t-1) = r;
    
    n_pairs = length(dc_vec);
    z = 0.5 * log((1 + r) / (1 - r)); % Fisher Z
    se_z = 1 / sqrt(n_pairs - 3);     % Standard error
    corr_upper_95(t-1) = tanh(z + 1.96 * se_z); % Inverse Fisher back to r
    corr_lower_95(t-1) = tanh(z - 1.96 * se_z);
end

% 1. Global Decile Analysis with 95% Binomial CI
num_bins = 10;
quantiles = prctile(delta_cost_all, linspace(0, 100, num_bins+1));
prob_surge = zeros(num_bins, 1);
ci_surge = zeros(num_bins, 1);
base_rate = mean(surge_labels_all);

for b = 1:num_bins
    in_bin = (delta_cost_all >= quantiles(b)) & (delta_cost_all <= quantiles(b+1));
    p_bin = mean(surge_labels_all(in_bin));
    n_bin = sum(in_bin);
    
    prob_surge(b) = p_bin;
    % 95% Wald Confidence Interval for proportion
    ci_surge(b) = 1.96 * sqrt((p_bin * (1 - p_bin)) / n_bin);
end

fprintf('  Decile 1 (Cost Dropped Most):  %.2f%% probability of citation surge\n', prob_surge(1)*100);
fprintf('  Decile 10 (Cost Increased Most): %.2f%% probability of citation surge\n', prob_surge(10)*100);
fprintf('  Mean Year-over-Year Spearman Correlation: %.3f (Negative = Predictive)\n\n', mean(yearly_corr));

% --- PLOT 1: PREDICTING SURGES (WITH UNCERTAINTY) ---
figure('Position', [100, 100, 1200, 500], 'Color', 'w');
tlo1 = tiledlayout(1, 2, 'TileSpacing', 'compact');

% Panel A: Decile Bar Chart with Error Bars
nexttile;
hold on;
bar_colors = [linspace(0.466, 0.850, num_bins)', linspace(0.674, 0.325, num_bins)', linspace(0.188, 0.098, num_bins)'];
b_chart = bar(1:num_bins, prob_surge * 100, 'FaceColor', 'flat', 'EdgeColor', 'k');
b_chart.CData = bar_colors;

% Overlay 95% CI Error Bars
errorbar(1:num_bins, prob_surge * 100, ci_surge * 100, 'k', 'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 5);

yline(base_rate * 100, 'k--', 'LineWidth', 2);
text(1.5, base_rate*100 + 1.0, 'Random Base Rate', 'Color', 'k', 'FontWeight', 'bold');
hold off;

xlabel('Change in Latent Affinity Deciles');
xticks([1, 10]); xticklabels({'Affinity Dropped (Divergence)', 'Affinity Increased (Convergence)'});
ylabel('Probability of Citation Surge at Year t+1 (%)');
title('A. Predicting Citation Surges (95% CI)', 'FontWeight', 'bold');
grid on;

% Panel B: Time-Series Correlation with Shaded Confidence Band
nexttile;
hold on;
x_years = config.years(2:end-1);

% Draw Shaded 95% Confidence Interval Band
x_fill = [x_years, fliplr(x_years)];
y_fill = [corr_upper_95', fliplr(corr_lower_95')];
fill(x_fill, y_fill, [0 0.4470 0.7410], 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');

% Draw the main correlation line
plot(x_years, yearly_corr, '-o', 'LineWidth', 2.5, 'Color', [0 0.4470 0.7410], 'MarkerFaceColor', [0 0.4470 0.7410]);

yline(mean(yearly_corr), 'r--', 'LineWidth', 2);
yline(0, 'k-', 'LineWidth', 1);
hold off;

xlabel('Year (t)');
ylabel('Spearman Correlation (\DeltaAffinity_t vs Growth_{t+1})');
title(sprintf('B. Predictive Stability (Mean r = %.3f, 95%% CI)', mean(yearly_corr)), 'FontWeight', 'bold');
grid on;

title(tlo1, 'Predictive Inference: Latent Geometry Predicts Scientific Momentum', 'FontSize', 16, 'FontWeight', 'bold');
saveas(gcf, fullfile(config.figures_folder, 'RQ3_Momentum_Prediction.png'));
%% ══════════════════════════════════════════════════════════════════
%  PART 2: "BLACK SWAN" ANOMALIES (COVID-19 vs. AI REVOLUTION)
%% ══════════════════════════════════════════════════════════════════
fprintf('\n--- Part 2: Isolating the 2020 Pandemic and 2023 AI Shocks ---\n');

% Train on the pre-shock era (2000-2019) to establish the "Normal" bounds of volatility
train_idx = 1:20; % 2000 to 2019
E_hist = abs(E(:, :, train_idx));
mu_E = mean(E_hist, 3);
sigma_E = std(E_hist, 0, 3);

% Extract time indices
t_2019 = find(config.years == 2019); % Universal Pristine Control
t_2020 = find(config.years == 2020); % COVID Event
t_2023 = find(config.years == 2023); % AI Event

% Calculate Z-Scores
Z_2019 = (abs(E(:,:,t_2019)) - mu_E) ./ (sigma_E + eps);
Z_2020 = (abs(E(:,:,t_2020)) - mu_E) ./ (sigma_E + eps);
Z_2023 = (abs(E(:,:,t_2023)) - mu_E) ./ (sigma_E + eps);

% Ignore diagonals
Z_2019(1:N+1:end) = 0; 
Z_2020(1:N+1:end) = 0; 
Z_2023(1:N+1:end) = 0;

anomaly_threshold = 5.0;
clean_concepts = strrep(concepts, '_', ' '); % Clean underscores

% ===================================================================
% EVENT 1: THE COVID-19 SHOCK (2020)
% ===================================================================
anom_2019 = sum(Z_2019(:) > anomaly_threshold);
anom_2020 = sum(Z_2020(:) > anomaly_threshold);

fprintf('  Number of >5-Sigma Structural Anomalies (COVID-19):\n');
fprintf('    2019 (Universal Control): %d anomalies\n', anom_2019);
fprintf('    2020 (Pandemic Year): %d anomalies\n\n', anom_2020);

% Calculate Node Impact (Sum of >5 sigma edges)
Z_2020_thresh = Z_2020;
Z_2020_thresh(Z_2020_thresh <= anomaly_threshold) = 0;
impact_2020 = sum(Z_2020_thresh, 1)' + sum(Z_2020_thresh, 2);
[~, top_2020] = sort(impact_2020, 'descend');

fprintf('  The Structural Epicenter of the 2020 Shock (Top Nodes by Anomaly Volume):\n');
for i = 1:10
    if impact_2020(top_2020(i)) > 0
        fprintf('    %2d. %-50s (Total Z-Score Shock: %.1f)\n', i, clean_concepts{top_2020(i)}, impact_2020(top_2020(i)));
    end
end

% ===================================================================
% EVENT 2: THE AI / LLM REVOLUTION (2023)
% ===================================================================
fprintf('\n------------------------------------------------------------------\n\n');
anom_2023 = sum(Z_2023(:) > anomaly_threshold);

fprintf('  Number of >5-Sigma Structural Anomalies (AI Boom):\n');
fprintf('    2019 (Universal Control): %d anomalies\n', anom_2019);
fprintf('    2023 (LLM Boom Year): %d anomalies\n\n', anom_2023);

% Calculate Node Impact (Sum of >5 sigma edges)
Z_2023_thresh = Z_2023;
Z_2023_thresh(Z_2023_thresh <= anomaly_threshold) = 0;
impact_2023 = sum(Z_2023_thresh, 1)' + sum(Z_2023_thresh, 2);
[~, top_2023] = sort(impact_2023, 'descend');

fprintf('  The Structural Epicenter of the 2023 Shock (Top Nodes by Anomaly Volume):\n');
for i = 1:10
    if impact_2023(top_2023(i)) > 0
        fprintf('    %2d. %-50s (Total Z-Score Shock: %.1f)\n', i, clean_concepts{top_2023(i)}, impact_2023(top_2023(i)));
    end
end
if sum(impact_2023) == 0
    fprintf('    [No >5-sigma anomalies detected in 2023]\n');
end

% --- PLOT: 1x2 HISTOGRAM COMPARISON ---
figure('Position', [100, 100, 1200, 500], 'Color', 'w');
tlo2 = tiledlayout(1, 2, 'TileSpacing', 'compact');
bin_edges = -2:0.5:20;

% Panel A: COVID-19 (2019 vs 2020)
nexttile;
histogram(Z_2019(:), 'BinEdges', bin_edges, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', 'w', 'Normalization', 'probability');
hold on;
histogram(Z_2020(:), 'BinEdges', bin_edges, 'FaceColor', [0.8500 0.3250 0.0980], 'EdgeColor', 'w', 'Normalization', 'probability', 'FaceAlpha', 0.7);
xline(anomaly_threshold, 'r--', 'LineWidth', 2);
text(anomaly_threshold + 0.5, 0.1, '> 5\sigma Black Swans \rightarrow', 'Color', 'r', 'FontWeight', 'bold');
set(gca, 'YScale', 'log'); 
ylim([1e-6 1]); xlim([-2 20]);
xlabel('Z-Score of Latent Shock (vs 2000-2019 Baseline)'); ylabel('Probability (Log Scale)');
legend('2019 (Pristine Control)', '2020 (Pandemic Event)', 'Location', 'northeast');
title('A. The COVID-19 Exogenous Shock (2020)', 'FontWeight', 'bold');
grid on; hold off;

% Panel B: AI Revolution (2019 vs 2023)
nexttile;
histogram(Z_2019(:), 'BinEdges', bin_edges, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', 'w', 'Normalization', 'probability');
hold on;
histogram(Z_2023(:), 'BinEdges', bin_edges, 'FaceColor', [0 0.4470 0.7410], 'EdgeColor', 'w', 'Normalization', 'probability', 'FaceAlpha', 0.7);
xline(anomaly_threshold, 'r--', 'LineWidth', 2);
set(gca, 'YScale', 'log'); 
ylim([1e-6 1]); xlim([-2 20]);
xlabel('Z-Score of Latent Shock (vs 2000-2019 Baseline)'); ylabel('Probability (Log Scale)');
legend('2019 (Pristine Control)', '2023 (LLM Event)', 'Location', 'northeast');
title('B. The Artificial Intelligence Boom (2023)', 'FontWeight', 'bold');
grid on; hold off;

title(tlo2, 'RQ4: Exogenous Shocks vs Endogenous Evolution ("Black Swan" Detection)', 'FontSize', 16, 'FontWeight', 'bold');
saveas(gcf, fullfile(config.figures_folder, 'RQ4_COVID_vs_AI_Histograms.png'));

fprintf('\n  ✓ Anomaly analysis complete. Figure saved.\n')