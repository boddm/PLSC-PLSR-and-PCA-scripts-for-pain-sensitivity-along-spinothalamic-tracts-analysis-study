% Function: Process single-column test data using PCA inverse transformation and mapping method

clear; close all; clc;

%% Environment setup
addpath(genpath('PLSC_PCR_original_noCV'));  % Path to PLSC program, needs to be changed
addpath(genpath('Subfunction'));

% Test data path
test_data_path = 'example/3';
test_base_file = fullfile(test_data_path, 'Processing_Test');  % Test data path, needs to be changed
model_path = fullfile(test_data_path, 'Results_PLSR');  % Model path
output_file = fullfile(test_data_path, 'Test_Predictions');  % Output path for prediction results

mkdir(output_file)

%% Load test data
fprintf('===== Loading test data =====\n');
test_data = load(fullfile(test_base_file, 'data.mat'));  % Test data
x_test = test_data.brain_data;  % n_edges x n_subj
y_test = test_data.beh_data;  % n_subj x 1_behaviors (original behavioral data, not standardized)

% Check the number of columns of behavioral data in test data
fprintf('Test data behavioral data dimensions: %d rows x %d columns\n', size(y_test, 1), size(y_test, 2));
fprintf('Test data behavioral data statistics: mean=%.4f, std=%.4f\n', mean(y_test), std(y_test));

%% Model parameter settings
bsr_k = 1;  % Feature selection parameter
bsr_thresh = [1.96 2.58 3.29];  % Bootstrap Ratio threshold
n_comp = 6;  % Number of PLS components

%% Make predictions for each model
for nbsr = 1:3
    for nc = 1:n_comp
        fprintf('\tPredicting with model - BSR threshold %.2f, PLS components %d\n', bsr_thresh(nbsr), nc);
        
        %% Load model
        model_name = sprintf('final_model_allBehav_lv%d_bsr%.2f_ncomp%d.mat', bsr_k, bsr_thresh(nbsr), nc);
        model_file = fullfile(model_path, model_name);
        
        if ~exist(model_file, 'file')
            fprintf('Warning: Model file does not exist: %s\n', model_file);
            continue;
        end
        
        load(model_file, 'model_params');
        
        %% Step 1: Use model to predict PC1
        PC1_test_pred = predict_with_final_model(x_test, model_params);
        
        %% Step 2: Use saved PCA parameters for inverse transformation
        % Construct score matrix with only first column having values, rest are 0
        score_test = zeros(size(PC1_test_pred, 1), size(model_params.pca_loading_y, 2));
        score_test(:, 1) = PC1_test_pred;  % Only put predicted PC1 values in the first column

        % Inverse PCA transformation and inverse standardization
        Y_test_std_pred_12 = score_test * model_params.pca_loading_y';
        y_test_std_pred_approx = Y_test_std_pred_12 .* model_params.y_std + model_params.y_mean;

        %% Step 3: Use PLSR to establish mapping relationship and optimize number of components
        X_cal = y_test_std_pred_approx;  % Predictor variables (using all columns)
        max_ncomp = size(X_cal, 2);  % Maximum number of components equals number of features

        % Initialize result storage
        n_behaviors = size(y_test, 2);
        R2_all = zeros(max_ncomp, n_behaviors);
        RMSE_all = zeros(max_ncomp, n_behaviors);
        R_all = zeros(max_ncomp, n_behaviors);
        P_all = zeros(max_ncomp, n_behaviors);

        % Loop through different numbers of PLSR components
        for ncomp = 1:max_ncomp
            % Perform regression using PLSR
            [~, ~, ~, ~, beta, ~, ~, ~] = plsregress(X_cal, y_test, ncomp);

            % Make predictions using PLSR model
            y_test_pred_mapped = [ones(size(X_cal, 1), 1), X_cal] * beta;

            % Calculate performance metrics (calculated by column)
            residual_sq = (y_test - y_test_pred_mapped).^2;
            total_var = sum((y_test - mean(y_test)).^2);

            % Calculate R² and RMSE by column
            R2_all(ncomp, :) = 1 - sum(residual_sq, 1) ./ total_var;
            RMSE_all(ncomp, :) = sqrt(mean(residual_sq, 1));
            % Calculate correlation coefficients (only take diagonal elements, i.e., correlation between corresponding columns)
            [R_matrix, P_matrix] = corr(y_test, y_test_pred_mapped);
            R_all(ncomp, :) = diag(R_matrix);
            P_all(ncomp, :) = diag(P_matrix);

            fprintf('PLSR components %d:\n', ncomp);
            for b = 1:n_behaviors
                fprintf('  Behavior %d: R² = %.4f, RMSE = %.4f, R = %.4f, P = %.4f\n', b, R2_all(ncomp, b), RMSE_all(ncomp, b), R_all(ncomp, b), P_all(ncomp, b));
            end
        end

        % Find optimal number of components (based on average R²)
        mean_R2 = mean(R2_all, 2);
        [~, best_ncomp] = max(mean_R2);
        fprintf('Optimal PLSR components: %d (average R² = %.4f)\n', best_ncomp, mean_R2(best_ncomp));

        % Retrain model using optimal number of components
        [XL, YL, XS, YS, beta, PCTVAR, MSE, stats] = plsregress(X_cal, y_test, best_ncomp);
        y_test_pred_mapped = [ones(size(X_cal, 1), 1), X_cal] * beta;

        % Calculate final correlation coefficients (only take diagonal elements)
        [R_matrix, P_matrix] = corr(y_test, y_test_pred_mapped);
        R_final = diag(R_matrix);
        P_final = diag(P_matrix);

        %% Step 4: Save prediction results
        prediction_results.y_pred = PC1_test_pred;
        prediction_results.y_pred_mapped = y_test_pred_mapped;  % New method results
        prediction_results.y_true = y_test;

        % New method evaluation metrics
        prediction_results.R2 = R2_all(best_ncomp, :);
        prediction_results.RMSE = RMSE_all(best_ncomp, :);
        prediction_results.R = R_final;
        prediction_results.P = P_final;
        prediction_results.beta = beta;

        % Save detailed PLSR model information
        prediction_results.XL = XL;
        prediction_results.YL = YL;
        prediction_results.XS = XS;
        prediction_results.YS = YS;
        prediction_results.PCTVAR = PCTVAR;
        prediction_results.MSE = MSE;
        prediction_results.stats = stats;
        prediction_results.best_ncomp = best_ncomp;

        prediction_results.model_params = model_params;

        save_name = sprintf('test_prediction_newmethod_simple_lv%d_bsr%.2f_ncomp%d.mat', bsr_k, bsr_thresh(nbsr), nc);
        save(fullfile(output_file, save_name), 'prediction_results');
        fprintf('Prediction results saved to: %s\n', fullfile(output_file, save_name));

        % Clear variables to prepare for the next model
        clear model_params prediction_results PC1_test_pred y_test_std_pred_approx y_test_pred_mapped
    end
end

fprintf('===== All model predictions completed =====\n');