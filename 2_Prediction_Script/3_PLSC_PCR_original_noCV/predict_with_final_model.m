function y_pred_norm = predict_with_final_model(x_test, model_params)
% Function: Make predictions using the trained final model
% Input:
%   x_test - Test set connectivity matrix (n_edges x n_subj)
%   model_params - Model parameter structure
% Output:
%   y_pred - Predicted behavioral data (n_subj x 1)

%% Normalize test set data (using training set parameters)
x_test_norm = (x_test - model_params.x_mean) ./ model_params.x_std;

%% Predict using selected features
selected_test_features = x_test_norm(:, model_params.selected_feature_indices);
y_pred_norm = [ones(size(selected_test_features, 1), 1), selected_test_features] * model_params.reg_beta;

end