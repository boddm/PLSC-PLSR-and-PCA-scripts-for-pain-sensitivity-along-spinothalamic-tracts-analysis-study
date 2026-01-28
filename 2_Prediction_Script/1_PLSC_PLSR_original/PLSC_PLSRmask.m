function [mask, beta] = PLSC_PLSRmask(inputConnMat, inputBehav, bsr_k, n_comp, bsr_thresh)
% Function: Feature selection and regression analysis using PLSC
% Inputs:
%   inputConnMat - connectivity matrix
%   inputBehav - behavioral data
%   bsrK - feature-selection parameter
%   nComp - number of PLS components
%   bsrThresh - Bootstrap Ratio threshold
% Outputs:
%   mask - feature-selection mask
%   beta - regression coefficients

%% Parameter settings
n_bootstraps = 5000;
normType = 1;  % normalization type
nGroups = 1;  % number of groups
subjGroups = ones(size(inputConnMat, 1), 1);  % group labels

%% 1. Data normalization (connectivity matrix only)
fprintf('Data normalization...\n');
x_norm = myPLS_norm(inputConnMat, nGroups, subjGroups, normType);
y_norm = myPLS_norm(inputBehav, nGroups, subjGroups, normType);

%% 2. Compute covariance matrix
fprintf('Computing covariance matrix...\n');
covMat = myPLS_cov(x_norm, y_norm, nGroups, subjGroups);

%% 3. Perform SVD decomposition
fprintf('Performing SVD decomposition...\n');
[leftSV, ~, rightSV] = svd(covMat, 'econ');

%% 4. Sign adjustment (ensure the maximum value is positive)
for i = 1:size(rightSV, 2)
    [~, idx] = max(abs(rightSV(:, i)));
    if sign(rightSV(idx, i)) < 0
        rightSV(:, i) = -rightSV(:, i);
        leftSV(:, i) = -leftSV(:, i);
    end
end

%% 5. Bootstrap analysis
time = 1;  % number of time points
bootOrderMat = bootstrap_order(subjGroups, time, n_bootstraps);
fprintf('Starting Bootstrap analysis...\n');

for iter = 1:n_bootstraps
    % Bootstrap sampling
    bootIndices = bootOrderMat(:, iter);
    connBoot = inputConnMat(bootIndices, :);
    behavBoot = inputBehav(bootIndices, :);

    % Normalization
    connBootNorm = myPLS_norm(connBoot, nGroups, subjGroups, normType);
    behavBootNorm = myPLS_norm(behavBoot, nGroups, subjGroups, normType);

    % Compute covariance
    covMatBoot = myPLS_cov(connBootNorm, behavBootNorm, nGroups, subjGroups);

    % SVD decomposition
    [leftSVBoot, ~, rightSVBoot] = svd(covMatBoot, 'econ');

    % Procrustes transform (to correct for axis rotation/reflection)
    rotLeft = rri_bootprocrust(leftSV, leftSVBoot);
    rotRight = rri_bootprocrust(rightSV, rightSVBoot);
    rightSVBoot = rightSVBoot * rotLeft;
    leftSVBoot = leftSVBoot * rotRight;

    % Online computation of mean and variance
    if iter == 1
        meanRightSV = rightSVBoot;
        meanLeftSV = leftSVBoot;

        meanSqRightSV = rightSVBoot.^2;
        meanSqLeftSV = leftSVBoot.^2;
    else
        meanRightSV = meanRightSV + rightSVBoot;
        meanLeftSV = meanLeftSV + leftSVBoot;

        meanSqRightSV = meanSqRightSV + rightSVBoot.^2;
        meanSqLeftSV = meanSqLeftSV + leftSVBoot.^2;
    end
end

%% 6. Compute standard errors
fprintf('Computing standard errors...\n');
meanLeftSV = meanLeftSV / n_bootstraps;
meanSqLeftSV = meanSqLeftSV / n_bootstraps;
meanRightSV = meanRightSV / n_bootstraps;
meanSqRightSV = meanSqRightSV / n_bootstraps;

stdLeftSV = sqrt(meanSqLeftSV - meanLeftSV.^2);
stdLeftSV = real(stdLeftSV);
stdRightSV = sqrt(meanSqRightSV - meanRightSV.^2);
stdRightSV = real(stdRightSV);

%% 7. Compute Bootstrap ratio
fprintf('Computing Bootstrap ratio...\n');
leftBSR = leftSV ./ stdLeftSV;
rightBSR = rightSV ./ stdRightSV;

%% 8. Handle infinite values
% (when stdLeftSV/stdRightSV is close to zero)
infRightIdx = find(~isfinite(rightBSR));
for iterInf = 1:size(infRightIdx, 1)
    rightBSR(infRightIdx(iterInf)) = rightSV(infRightIdx(iterInf));
end
infLeftIdx = find(~isfinite(leftBSR));
for iterInf = 1:size(infLeftIdx, 1)
    leftBSR(infLeftIdx(iterInf)) = leftSV(infLeftIdx(iterInf));
end
clear infRightIdx infLeftIdx iterInf

%% 9. Feature selection
fprintf('Performing feature selection...\n');
mask = zeros(size(rightBSR, 1), 1);
posFeatIdx = rightBSR(:, bsr_k) > bsr_thresh;
negFeatIdx = rightBSR(:, bsr_k) < -bsr_thresh;
mask(posFeatIdx) = 1;
mask(negFeatIdx) = -1;

%% 10. PLS regression prediction
fprintf('Performing PLS regression...\n');
selFeatIdx = mask ~= 0;

sel_Features = x_norm(:, selFeatIdx);
[xl1, yl1, xs1, ys1, beta, pctvar1, mse1, stats1] = plsregress(sel_Features, y_norm, n_comp);  % PLSR

end