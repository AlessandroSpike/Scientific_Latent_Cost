clc; close all; clearvars

%% Load loss
addpath('results_bayesopt')
lossData   = readtable('loss_history.csv');

iter       = lossData.iteration;
batch_loss = lossData.batch_loss;
tot_loss   = lossData.full_loss;

%% Avoid zero values (important for log scale)
batch_loss(batch_loss <= 0) = eps;
tot_loss(tot_loss <= 0)     = eps;

%% Plot
figure('Color','w','Position',[200 200 800 500])

semilogy(iter, batch_loss, 'LineWidth',1.5, ...
    'Color',[0 0.447 0.741])
hold on

semilogy(iter, tot_loss, '-', 'LineWidth',2, ...
    'Color',[0.850 0.325 0.098])

grid on
box on

xlabel('Iteration','FontSize',12,'FontWeight','bold')
ylabel('Loss (log scale)','FontSize',12,'FontWeight','bold')
title('Training Loss History','FontSize',14,'FontWeight','bold')

legend({'Batch Loss','Full Loss'}, ...
    'Location','northeast')

set(gca,'FontSize',11,'LineWidth',1.2)