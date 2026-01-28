function X = myPLS_norm(X,num_groups,subj_grouping,mode)
% Purpose: Normalize data
% Inputs:
%   X : imaging matrix (subjects x imaging values)
%   num_groups : normalioze across all subjects (default=1) or within groups (2)
%   subj_grouping : matrix of ones (subjects x 1) -> change to grouping information if you want to normalize within each group
%   mode : normalization option 
%          (0=no normalization | 
%           1=zscore across all subjects | 
%           2=zscore within groups | 
%           3=std normalization across groups (no centering) | 
%           4=std normalization within groups (no centering)
% Output:
%   X : normalized X matrix

%% Normalize data
switch mode
    case 1  % Z-score normalize across all subjects
        X = zscore(X);
    case 2  % Z-score normalize within groups
        for iter_group = 1:num_groups
            idx = find(subj_grouping == iter_group);
            X(idx, :) = zscore(X(idx, :));
        end 
    case 3  % Standardize across all subjects
        X2 = sqrt(mean(X.^2,1));
        X = X ./ repmat(X2,[size(X,1) 1]);
    case 4  % Standardize within groups
        for iter_group = 1:num_groups
            idx = find(subj_grouping == iter_group);
            X2 = sqrt(mean(X(idx, :).^2, 1));
            X(idx, :) = X(idx, :) ./ repmat(X2,[size(X(idx, :), 1) 1]);
        end 
end