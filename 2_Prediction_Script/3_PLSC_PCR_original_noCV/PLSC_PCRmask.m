function [feature_mask, pca_loading_y, pca_mu, plsr_beta] = PLSC_PCRmask(x_train, y_train, bsr_k, n_comp, bsr_thresh)
% Function: Perform feature selection and regression analysis using PLSC method
% Input:
%   x_train - Training feature matrix
%   y_train - Training target variable
%   bsr_k - Feature selection parameter
%   n_comp - Number of PLS components
%   bsr_thresh - Bootstrap Ratio threshold
% Output:
%   feature_mask - Feature selection mask
%   pca_loading_y - PCA loading matrix
%   pca_mu - PCA mean
%   plsr_beta - PLS regression coefficients

%% Parameter setup
n_bootstraps = 5000;
norm_type = 1;  % Normalization type
n_groups = 1;  % Number of groups
subj_groups = ones(size(x_train, 1), 1);  % Group labels

%% 1. Data normalization (only for connectivity matrix)
fprintf('Data normalization...\n');
x_train_norm = myPLS_norm(x_train, n_groups, subj_groups, norm_type);
y_train_norm = myPLS_norm(y_train, n_groups, subj_groups, norm_type);

%% 2. Calculate covariance matrix
fprintf('Calculating covariance matrix...\n');
cov_mat = myPLS_cov(x_train_norm, y_train_norm, n_groups, subj_groups);

%% 3. SVD decomposition
fprintf('Performing SVD decomposition...\n');
[left_sv, ~, right_sv] = svd(cov_mat, 'econ');

%% 4. Sign adjustment (ensure maximum value is positive)
for i = 1:size(right_sv, 2)
    [~, idx] = max(abs(right_sv(:, i)));
    if sign(right_sv(idx, i)) < 0
        right_sv(:, i) = -right_sv(:, i);
        left_sv(:, i) = -left_sv(:, i);
    end
end

%% 5. Bootstrap analysis
time_points = 1;  % Number of time points
boot_order_mat = bootstrap_order(subj_groups, time_points, n_bootstraps);
fprintf('Starting Bootstrap analysis...\n');

for iter = 1:n_bootstraps
    % Bootstrap sampling
    boot_indices = boot_order_mat(:, iter);
    x_boot = x_train(boot_indices, :);
    y_boot = y_train(boot_indices, :);

    % Normalization
    x_boot_norm = myPLS_norm(x_boot, n_groups, subj_groups, norm_type);
    y_boot_norm = myPLS_norm(y_boot, n_groups, subj_groups, norm_type);

    % Calculate covariance
    cov_mat_boot = myPLS_cov(x_boot_norm, y_boot_norm, n_groups, subj_groups);

    % SVD decomposition
    [left_sv_boot, ~, right_sv_boot] = svd(cov_mat_boot, 'econ');

    % Procrustas transform (to correct for axis rotation/reflection)
    rot_left = rri_bootprocrust(left_sv, left_sv_boot);
    rot_right = rri_bootprocrust(right_sv, right_sv_boot);
    right_sv_boot = right_sv_boot * rot_left;
    left_sv_boot = left_sv_boot * rot_right;

    % Online computing of mean and variance
    if iter == 1
        mean_right_sv = right_sv_boot;
        mean_left_sv = left_sv_boot;

        mean_sq_right_sv = right_sv_boot.^2;
        mean_sq_left_sv = left_sv_boot.^2;
    else
        mean_right_sv = mean_right_sv + right_sv_boot;
        mean_left_sv = mean_left_sv + left_sv_boot;

        mean_sq_right_sv = mean_sq_right_sv + right_sv_boot.^2;
        mean_sq_left_sv = mean_sq_left_sv + left_sv_boot.^2;
    end
end

%% 6. Calculate standard error
fprintf('Calculating standard error...\n');
mean_left_sv = mean_left_sv / n_bootstraps;
mean_sq_left_sv = mean_sq_left_sv / n_bootstraps;
mean_right_sv = mean_right_sv / n_bootstraps;
mean_sq_right_sv = mean_sq_right_sv / n_bootstraps;

std_left_sv = sqrt(mean_sq_left_sv - mean_left_sv.^2);
std_left_sv = real(std_left_sv);
std_right_sv = sqrt(mean_sq_right_sv - mean_right_sv.^2);
std_right_sv = real(std_right_sv);

%% 7. Calculate Bootstrap ratio
fprintf('Calculating Bootstrap ratio...\n');
left_bsr = left_sv ./ std_left_sv;
right_bsr = right_sv ./ std_right_sv;

%% 8. Handle infinite values
% (when std_left_sv/std_right_sv is close to 0)
inf_right_idx = find(~isfinite(right_bsr));
for iter_inf = 1:size(inf_right_idx, 1)
    right_bsr(inf_right_idx(iter_inf)) = right_sv(inf_right_idx(iter_inf));
end
inf_left_idx = find(~isfinite(left_bsr));
for iter_inf = 1:size(inf_left_idx, 1)
    left_bsr(inf_left_idx(iter_inf)) = left_sv(inf_left_idx(iter_inf));
end
clear inf_right_idx inf_left_idx iter_inf

%% 9. Feature selection
fprintf('Performing feature selection...\n');
feature_mask = zeros(size(right_bsr, 1), 1);
pos_feat_idx = right_bsr(:, bsr_k) > bsr_thresh;
neg_feat_idx = right_bsr(:, bsr_k) < -bsr_thresh;
feature_mask(pos_feat_idx) = 1;
feature_mask(neg_feat_idx) = -1;

selected_feat_idx = (feature_mask ~= 0);
selected_features = x_train_norm(:, selected_feat_idx);

%% 10. Feature dimensionality reduction
[pca_loading_y, pca_score_y, ~, ~, ~, pca_mu] = pca(y_train_norm, 'Centered', false);

%% 11. PLS regression prediction
fprintf('Performing PLS regression...\n');

[pls_loadings_x, pls_loadings_y, pls_scores_x, pls_scores_y, plsr_beta, pls_variance_explained, pls_mse, pls_stats] = plsregress(selected_features, pca_score_y(:, 1), n_comp); % PLSR

end