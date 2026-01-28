function R = myPLS_cov(X,Y,mode,subj_grouping)
% Purpose: Compute cross-covariance matrix
% Inputs:
%   X : imaging matrix (subjects x imaging values)
%   Y : behavior matrix (subjects x behavior values)
%   num_groups : compute R across all subjects (default=1) or within groups (2)
%   subj_grouping : matrix of ones (subjects x 1) -> change to grouping
% information if you want to normalize within each group
% Output:
%   R : cross-covariance matrix

switch mode
    case 1
        R = Y' * X;
    case 2
        num_groups = length(unique(subj_grouping));
        for iter_group = 1:num_groups
            Ysel = Y(find(subj_grouping == iter_group), :);
            Xsel = X(find(subj_grouping == iter_group), :);

            R0 = Ysel.' * Xsel;
            if ~exist('R')
                R = R0;
            else
                R = [R; R0];
            end
        end
    otherwise
        error('Invalid mode: must be 1 or 2');
end

end