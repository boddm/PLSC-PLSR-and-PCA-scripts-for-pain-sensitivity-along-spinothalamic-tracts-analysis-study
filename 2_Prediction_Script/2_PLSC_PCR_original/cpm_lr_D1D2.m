function [feature_mask, y_test_pred, pca_loading_y, pca_mu, reg_beta] = cpm_lr_D1D2(x_train, x_test, y_train, ~, bsr_k, n_comp, bsr_thresh)
% Purpose: Prediction using PLSC+PCR
% Inputs:
%   x_train - training feature matrix
%   x_test - test feature matrix
%   y_train - training target variable
%   ~ - placeholder (unused)
%   bsr_k - feature-selection parameter
%   n_comp - number of PLS components
%   bsr_thresh - Bootstrap Ratio threshold
% Outputs:
%   feature_mask - feature-selection mask
%   y_test_pred - test-set predictions
%   pca_loading_y - PCA loading matrix
%   pca_mu - PCA mean
%   reg_beta - regression coefficients

%% Transpose connectivity matrix (features×samples -> samples×features)
x_train = x_train';
x_test = x_test';

%% Obtain feature mask and regression coefficients
[feature_mask, pca_loading_y, pca_mu, reg_beta] = PLSC_PCRmask(x_train, y_train, bsr_k, n_comp, bsr_thresh);

%% Get indices of non-zero features
selected_feature_indices = feature_mask ~= 0;

%% Normalize test-set data
x_test_norm = (x_test - mean(x_train)) ./ (std(x_train) + 1e-8);

%% Predict using selected features
selected_test_features = x_test_norm(:, selected_feature_indices);
y_test_pred = [ones(size(selected_test_features, 1), 1), selected_test_features] * reg_beta;

end