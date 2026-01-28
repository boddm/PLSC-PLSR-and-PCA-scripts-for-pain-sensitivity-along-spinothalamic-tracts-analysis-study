% Function: Perform CPM-PLSR analysis, create final prediction model (without cross-validation)

clear; close all; clc;

%% Environment setup
addpath(genpath('PLSC_PCR_original'));  % Path to PLSC programs, needs to be changed
addpath(genpath('Subfunction'));

path = 'example/3';
base_file = fullfile(path, 'Processing');  % Input path - needs to be changed
output_file = fullfile(path, 'Results_PLSR');  % Output path, needs to be changed

mkdir(output_file)

%% Data loading
Data = load(fullfile(base_file, 'data.mat'));  % Load data

x0 = Data.brain_data;  % n_edges x n_subj
y0 = Data.beh_data;  % n_subj x 8_behaviors

%% Final model parameter setup
fprintf('===== Creating final prediction model =====\n');

n_selected_behav = size(y0, 2);  % Use all behavioral variables for prediction

%% Parameter setup
bsr_k = 1;  % Feature selection parameter
bsr_thresh = [1.96 2.58 3.29];  % Bootstrap Ratio thresholds
n_comp = 6;  % Number of PLS components

%% Create final models with different parameter combinations
for nbsr = 1:3
    for nc = 1:n_comp
        fprintf('\tCreating final model - BSR threshold %.2f, PLS components %d\n', bsr_thresh(nbsr), nc);

        %% Use all data to train model
        x_train = x0;
        y_train = y0(:, 1:n_selected_behav);

        %% Calculate normalization parameters
        x_mean = mean(x_train);
        x_std = std(x_train) + 1e-8;
        y_mean = mean(y_train);
        y_std = std(y_train) + 1e-8;

        %% Data normalization
        x_train_norm = (x_train - x_mean) ./ x_std;
        y_train_norm = (y_train - y_mean) ./ y_std;

        %% CPM model training
        [feature_mask, pca_loading_y, pca_mu, reg_beta] = PLSC_PCRmask(x_train, y_train, bsr_k, nc, bsr_thresh(nbsr));

        %% Get non-zero feature indices
        selected_feature_indices = feature_mask ~= 0;

        %% Save model parameters
        model_params.feature_mask = feature_mask;
        model_params.selected_feature_indices = selected_feature_indices;
        model_params.pca_loading_y = pca_loading_y;
        model_params.pca_mu = pca_mu;
        model_params.reg_beta = reg_beta;
        model_params.bsr_k = bsr_k;
        model_params.bsr_thresh = bsr_thresh(nbsr);
        model_params.n_comp = nc;
        model_params.n_selected_behav = n_selected_behav;

        %% Save normalization parameters
        model_params.x_mean = x_mean;
        model_params.x_std = x_std;
        model_params.y_mean = y_mean;
        model_params.y_std = y_std;

        %% Calculate model performance on training set
        selected_train_features = x_train_norm(:, selected_feature_indices);
        y_train_pred = [ones(size(selected_train_features, 1), 1), selected_train_features] * reg_beta;

        y_train_pca = y_train_norm * pca_loading_y(:, 1);

        % Calculate performance metrics
        [model_params.R, model_params.P] = corr(y_train_pca, y_train_pred);
        deno = y_train_pca - 0;
        nume = y_train_pca - y_train_pred;
        model_params.q2 = 1 - mean(nume.^2) / mean(deno.^2);

        model_params.y_train_pred = y_train_pred;
        model_params.y_train_pca = y_train_pca;

        fprintf('Model performance: q2 = %.4f, r = %.4f, p = %.4f\n', model_params.q2, model_params.R, model_params.P);

        %% Save final model
        fprintf('===== Saving final model =====\n');

        save_name = sprintf('final_model_allBehav_lv%d_bsr%.2f_ncomp%d.mat', bsr_k, bsr_thresh(nbsr), nc);

        save(fullfile(output_file, save_name), 'model_params');
        fprintf('Final model saved to: %s\n', fullfile(output_file, save_name));

    end
end

fprintf('===== All final models created =====\n');