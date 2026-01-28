% Purpose: Run CPM-PLSR analysis with cross-validation and performance evaluation

clear; close all; clc;

%% Environment Setup
addpath(genpath('PLSC_PCR_original'));  % Path to PLSC code, needs modification
addpath(genpath('Subfunction'));

path = 'example/2';
base_file = fullfile(path, 'Processing'); % Input path, needs modification
output_file = fullfile(path, 'Results_PLSR_F5'); % Output path, needs modification

mkdir(output_file)

%% Data Loading
Data = load(fullfile(base_file, 'data.mat'));  % Load data

x0 = Data.brain_data';  % n_edges x n_subj
y0 = Data.beh_data;  % n_subj x 8_behaviors

%% Generate Cross-Validation Parameters
fprintf('===== Generate Cross-Validation Parameters =====\n');

n_subjs = size(x0, 2);
n_permutations = 1000;
n_folds = size(y0, 1);

%% Generate perm_indices (shared across permutations)
fprintf('Generating permutation list...\n');
perm_idxs = zeros(n_permutations, n_subjs);
for np = 1:n_permutations
    perm_idxs(np, :) = randperm(n_subjs);
end

%% Generate fold_sizes and fold_bounds (shared across permutations)
fprintf('Generating cross-validation folds...\n');
fold_sizes = repmat(floor(n_subjs / n_folds), n_folds, 1);
fold_sizes(1:mod(n_subjs, n_folds)) = fold_sizes(1:mod(n_subjs, n_folds)) + 1;

fold_bounds = zeros(n_folds, 2);
start_idx = 1;
for nf = 1:n_folds
    end_idx = start_idx + fold_sizes(nf) - 1;
    fold_bounds(nf, :) = [start_idx, end_idx];
    start_idx = end_idx + 1;
end

%% 10-fold Cross-Validation with Real Labels
real_perm_idx = 1;  % Use real labels from the first permutation
n_selected_behav = size(y0, 2);  % Use all behavioral variables for prediction

%% Parameter Settings
bsr_k = 1;  % Feature selection parameter
bsr_thresh = [1.96 2.58 3.29];  % Bootstrap Ratio thresholds
n_comp = 6;  % Number of PLS components

%% Initialize Storage
for nbsr = 1:3
    for nc = 1:n_comp

        % Clear or reset variables before inner loop
        clear perf_results beta feature_mask
        y_test_norm = zeros(size(y0, 1), n_selected_behav);  % Reset normalized behavioral data
        y_test_pred = zeros(size(y0, 1), 1);  % Reset
        y_test_pca  = zeros(size(y0, 1), 1);  % Reset

        for nf = 1:n_folds
            fprintf('\tReal-label Cross-Validation - Fold %d/%d\n', nf, n_folds);

            %% Split Training/Testing Sets
            test_subj_idx = perm_idxs(real_perm_idx, fold_bounds(nf, 1):fold_bounds(nf, 2));
            train_subj_idx = setdiff(1:size(y0, 1), test_subj_idx);

            %% Get Connectivity Matrices
            x_train = x0(:, train_subj_idx);
            x_test = x0(:, test_subj_idx);

            %% Behavioral Data (No Z-score)
            y_train = y0(train_subj_idx, 1:n_selected_behav);
            y_test = y0(test_subj_idx, 1:n_selected_behav);

            %% Normalize Testing Data (Using Training Parameters)
            y_test_norm(test_subj_idx, :) = (y_test - mean(y_train)) ./ (std(y_train) + 1e-8);

            %% CPM Prediction
            [feature_mask, y_test_pred(test_subj_idx, :), perf_results.pca_loading_y{nf}, pca_mu, perf_results.beta{nf}] = cpm_lr_D1D2(x_train, x_test, y_train, [], bsr_k, nc, bsr_thresh(nbsr));

            %% PCA on Testing Set
            y_test_pca(test_subj_idx, :) = y_test_norm(test_subj_idx, :) * perf_results.pca_loading_y{nf}(:, 1);

            %% Store Feature Mask
            perf_results.mask{nf} = feature_mask;
        end

        %% Calculate Real Performance
        fprintf('===== Calculate Real Performance =====\n');

        for t1 = 1:1
            for t2 = t1 %1:n_selected_behav
                [perf_results.R(t1, t2), perf_results.P(t1, t2)] = corr(y_test_pca(:, t1), y_test_pred(:, t2));
                deno = y_test_pca(:, t1) - 0;
                nume = y_test_pca(:, t1) - y_test_pred(:, t2);
                perf_results.q2(t1, t2) = 1 - mean(nume.^2) / mean(deno.^2);
                % fprintf('Behavior Variable %d->%d: q2 = %.4f, r = %.4f, p = %.4f\n', t1, t2, perf_results.q2(t1, t2), perf_results.R(t1, t2), perf_results.P(t1, t2));
            end
        end

        %% Save Results
        fprintf('===== Save Results =====\n');

        save_name = sprintf('result_real_allBehav_lv%d_bsr%d_ncomp%d_fold%d.mat', bsr_k, bsr_thresh(nbsr), nc, n_folds);

        save(fullfile(output_file, save_name), 'perf_results', 'y_test_pred', 'y_test_norm', 'y_test_pca');
        fprintf('Results saved to: %s\n', fullfile(output_file, save_name));

    end
end