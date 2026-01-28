function [PRESS, R2] = CalculateR2(Y0, Y_pre)
% Purpose:
%   Calculate the R² coefficient of determination and PRESS statistic between predicted and actual values
% Inputs:
%   Y0 - Vector of actual values
%   Y_pre - Vector of predicted values
% Outputs:
%   PRESS - Predicted residual sum of squares
%   R2 - R² coefficient of determination
% Example:
%   [press, r2] = CalculateR2(y_true, y_pred)
% History:
%   1.0: Created function

%% Calculate basic statistics
Y_m = mean(Y0);
num_cols = size(Y_pre, 2);
num_rows = size(Y0, 1);

%% Initialize result arrays
PRESS = zeros(1, num_cols);
R2 = zeros(1, num_cols);

%% Calculate R² and PRESS for each variable 
for i = 1:num_cols
    %% Calculate total sum of squares and residual sum of squares
    TSS = 0;
    RSS = 0;

    for k = 1:num_rows
        TSS = TSS + (Y0(k, i) - Y_m(i))^2;
        RSS = RSS + (Y0(k, i) - Y_pre(k, i))^2;
    end

    %% Calculate R² and PRESS
    R2(i) = 1 - RSS / TSS;
    PRESS(i) = RSS;
end

end