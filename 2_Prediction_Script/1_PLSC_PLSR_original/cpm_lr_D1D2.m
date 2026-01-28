function [mask, predTestScores, beta] = cpm_lr_D1D2(connMatTrain, connMatTest, behavTrain, bsrK, nComp, bsrThresh)
% Function: Prediction using PLSC+PLSR method
% Inputs:
%   connMatTrain - training connectivity matrix
%   connMatTest - testing connectivity matrix
%   behavTrain - training behavioral data
%   bsrK - feature-selection parameter
%   nComp - number of PLS components
%   bsrThresh - Bootstrap Ratio threshold
% Outputs:
%   mask - feature-selection mask
%   predTestScores - predicted scores for the test set
%   beta - regression coefficients

%% Transpose connectivity matrices (features×subjects → subjects×features)
connMatTrain = connMatTrain';
connMatTest = connMatTest';

%% Obtain feature mask and regression coefficients
[mask, beta] = PLSC_PLSRmask(connMatTrain, behavTrain, bsrK, nComp, bsrThresh);

%% Get indices of non-zero features
selFeatIdx = mask ~= 0;

%% Normalize test-set data
connTestNorm = (connMatTest - mean(connMatTrain)) ./ (std(connMatTrain) + 1e-8);

%% Predict using selected features
selTestFeatures = connTestNorm(:, selFeatIdx);
predTestScores = [ones(size(selTestFeatures, 1), 1), selTestFeatures] * beta;
end