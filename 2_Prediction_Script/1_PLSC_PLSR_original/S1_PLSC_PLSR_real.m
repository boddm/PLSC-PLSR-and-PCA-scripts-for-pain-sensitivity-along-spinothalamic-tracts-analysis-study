% Purpose: Run CPM-PLSR analysis with cross-validation and performance evaluation
% Inputs:
%   None
% Outputs:
%   Save results to mat file

clear; close all; clc;

%% Environment setup
addpath(genpath('PLSC_PLSR_original'));  % Path to PLSC programs, needs modification
addpath(genpath('Subfunction'));

path = 'example/1';
basefile = fullfile(path, 'Processing');  % Input path, needs modification
outputFile = fullfile(path, 'Results_PLSR');  % Output path, needs modification

mkdir(outputFile)

%%
Data = load(fullfile(basefile, 'data.mat'));  % Load data

connMatAll = Data.brain_data';  % n_edges x n_subj
behavDataAll = Data.beh_data;  % n_subj x 8_behaviors

%% Generate cross-validation parameters
fprintf('===== Generate cross-validation parameters =====\n');

nSubjects = size(connMatAll, 2);
nPermutations = 1000;
nFolds = size(behavDataAll, 1);

%% Generate permIndices (shared across permutations)
fprintf('Generate permutation list...\n');
permIndices = zeros(nPermutations, nSubjects);
for np = 1:nPermutations
    permIndices(np, :) = randperm(nSubjects);
end

%% Generate foldSizes and foldBounds (shared across permutations)
fprintf('Generate cross-validation folds...\n');
foldSizes = repmat(floor(nSubjects / nFolds), nFolds, 1);
foldSizes(1:mod(nSubjects, nFolds)) = foldSizes(1:mod(nSubjects, nFolds)) + 1;

foldBounds = zeros(nFolds, 2);
startIdx = 1;
for nf = 1:nFolds
    endIdx = startIdx + foldSizes(nf) - 1;
    foldBounds(nf, :) = [startIdx, endIdx];
    startIdx = endIdx + 1;
end

%% 10-fold cross-validation with true labels
realPermIdx = 1;  % Use true labels from first permutation
nSelectedBehav = size(behavDataAll, 2);  % Use all behavior variables for prediction

%% Parameter settings
bsrK = 1;  % Feature selection parameter
bsrThresh = [1.96 2.58 3.29];  % Bootstrap Ratio thresholds
nComp = 6;  % Number of PLS components

%% Initialize storage
for nbsr = 1:3
    for nc = 1:nComp

        % Clear or reset variables before inner loop
        clear perfResults beta mask
        behavTestNorm = zeros(size(behavDataAll, 1), nSelectedBehav);  % Reset normalized behavior data
        predTestScores = zeros(size(behavDataAll, 1), nSelectedBehav);  % Reset predictions

        for nf = 1:nFolds
            fprintf('\tTrue-label cross-validation - Fold %d/%d\n', nf, nFolds);

            %% Split train/test sets
            testSubjIdx = permIndices(realPermIdx, foldBounds(nf, 1):foldBounds(nf, 2));
            trainSubjIdx = setdiff(1:size(behavDataAll, 1), testSubjIdx);

            %% Get connectivity matrices
            connMatTrain = connMatAll(:, trainSubjIdx);
            connMatTest = connMatAll(:, testSubjIdx);

            %% Behavior data (no Z-score)
            behavTrain = behavDataAll(trainSubjIdx, 1:nSelectedBehav);
            behavTest = behavDataAll(testSubjIdx, 1:nSelectedBehav);

            %% Normalize test data (using train parameters)
            behavTestNorm(testSubjIdx, :) = (behavTest - mean(behavTrain)) ./ (std(behavTrain) + 1e-8);

            %% CPM prediction
            [mask, predTestScores(testSubjIdx, :), perfResults.beta{nf}] = cpm_lr_D1D2(connMatTrain, connMatTest, behavTrain, bsrK, nc, bsrThresh(nbsr));

            %% Store feature mask
            perfResults.mask{nf} = mask;
        end

        %% Compute true performance
        fprintf('===== Compute true performance =====\n');

        for t1 = 1:nSelectedBehav
            for t2 = t1 %1:nSelectedBehav
                [perfResults.R(t1, t2), perfResults.P(t1, t2)] = corr(behavTestNorm(:, t1), predTestScores(:, t2));
                deno = behavTestNorm(:, t1) - 0;
                nume = behavTestNorm(:, t1) - predTestScores(:, t2);
                perfResults.q2(t1, t2) = 1 - mean(nume.^2) / mean(deno.^2);
                fprintf('Behavior variable %d->%d: q2 = %.4f, r = %.4f, p = %.4f\n', t1, t2, perfResults.q2(t1, t2), perfResults.R(t1, t2), perfResults.P(t1, t2));
            end
        end

        %% Save results
        fprintf('===== Save results =====\n');

        saveName = sprintf('result_real_allBehav_lv%d_bsr%d_ncomp%d_fold%d.mat', bsrK, bsrThresh(nbsr), nc, nFolds);

        save(fullfile(outputFile, saveName), 'perfResults', 'predTestScores', 'behavTestNorm');
        fprintf('Results saved to: %s\n', fullfile(outputFile, saveName));

    end
end