function Disp_Saliences(X, Y, Y_std, explVarLVs, outputPath, NUM_GROUPS, mySignifLVs, Saliences_Mask, CONST_DIAGNOSIS, Y_label)
% Purpose: Plot bar charts of correlation coefficients between Y and LX or Y and LY
% Inputs:
%   X: names of behavioral variables
%   Y: correlation coefficients
%   Y_std: standard deviation (SD) or standard error of mean (SEM) of correlation coefficients [SEM = SD/sqrt(N)]
%   NUM_GROUPS: number of groups

%% Set default parameters
if nargin < 8
    Saliences_Mask = [];
end
if nargin < 7
    mySignifLVs = (1:size(X, 2))';
end
if nargin  < 6
    NUM_GROUPS = 1;
end
if nargin  < 5
    outputPath = [];
end
if nargin  < 4
    explVarLVs = [];
end
if nargin  < 3
    error('Missing required parameters!');
end
CONST_NUM_BEHAV = size(X, 2);
if nargin < 9
    for i = 1:size(NUM_GROUPS)
        CONST_DIAGNOSIS{1, i} = num2str(i);
    end
end
if nargin < 10
    Y_label = [];
end
if ~exist(outputPath)
    mkdir(outputPath);
end

%% Plot bar charts
for iter_lv = 1: size(mySignifLVs, 1)
    this_lv = mySignifLVs(iter_lv);

    figure;
    switch NUM_GROUPS
        case 1
            bar(1:CONST_NUM_BEHAV, Y(:,this_lv), 0.5, 'b');
            hold on
            errorbar(1:CONST_NUM_BEHAV, Y(:,this_lv), Y_std(:,this_lv), 'r.',...
                'LineWidth', 1, 'MarkerSize', 10, 'Color', 'black');
            axis([0 size(Y,2)+1 min(Y(:,this_lv))-0.1 max(Y(:,this_lv))+0.2]);
            hold on
            for i = 1:size(Y,1)
                if ~isempty(Saliences_Mask)
                    if Saliences_Mask(i, this_lv) ~= 0
                        if Y(i, this_lv) > 0
                            % plot(i,Y(i,this_lv)+Y_std(i,this_lv)+0.02,'*','color',[255 69 0]/255);
                            plot(i, Y(i,this_lv)+Y_std(i,this_lv)+0.05, '*', 'color', 'black');
                        else
                            % plot(i,Y(i,this_lv)-Y_std(i,this_lv)-0.02,'*','color',[255 69 0]/255);
                            plot(i, Y(i,this_lv)-Y_std(i,this_lv)-0.05, '*', 'color', 'black');
                        end
                    end
                end
            end
        case 2
            Y_med = Y(:,this_lv);
            Y_std_2 = Y_std(:,this_lv);
            for i = 1: CONST_NUM_BEHAV
                for j = 1: NUM_GROUPS
                    Y_fig(i,j) = Y_med(i + (j-1) * CONST_NUM_BEHAV);
                    Y_fig2(i,j) = Y_std_2(i + (j-1) * CONST_NUM_BEHAV);
                end
            end
            
            %% Plot bar charts
            x1 = 0.85:1:size(Y_fig,1)-0.15;
            x2 = 1.15:1:size(Y_fig,1)+0.15;
            bar(x1, Y_fig(:, 1), 0.3, 'FaceColor', [255 69 0]/255);
            hold on;
            bar(x2, Y_fig(:, 2), 0.3, 'FaceColor', [30 144 255]/255);
            legend(CONST_DIAGNOSIS, 'FontSize', 8, 'AutoUpdate', 'off');
            hold on;
            errorbar(x1, Y_fig(:,1), Y_fig2(:, 1), 'r.', 'LineWidth', 1, 'MarkerSize', 10, 'Color', 'black');
            hold on;
            errorbar(x2, Y_fig(:, 2), Y_fig2(:, 2), 'r.', 'LineWidth', 1, 'MarkerSize', 10, 'Color', 'black');
            axis([0 size(Y_fig, 1)+1 min(Y_fig(:))-0.1 max(Y_fig(:))+0.2]);
            if ~isempty(Saliences_Mask)
                hold on;
                for i = 1:size(Y_fig, 1)
                    if Saliences_Mask(i, this_lv) ~= 0
                        if Y_fig(i,1) > 0
                            plot(x1(i), Y_fig(i, 1)+Y_fig2(i, 1)+0.05, '*', 'color', 'black');
                        else
                            plot(x1(i), Y_fig(i, 1)-Y_fig2(i, 1)-0.05, '*', 'color', 'black');
                        end
                    end
                end
                hold on
                for i = 1:size(Y_fig, 1)
                    if Saliences_Mask(i+size(Y_fig,1), this_lv) ~= 0
                        if Y_fig(i,2) > 0
                            plot(x2(i), Y_fig(i, 2)+Y_fig2(i, 2)+0.05, '*', 'color', 'black');
                        else
                            plot(x2(i), Y_fig(i, 2)-Y_fig2(i, 2)-0.05, '*', 'color', 'black');
                        end
                    end
                end
            end
            clear Y_med Y_fig i j x1 x2
        otherwise
            error('Error, number of groups exceeds 2');
    end
    hold on;
    set(gca, 'XTickLabel', X, 'FontSize', 8)
    xlabel('Behavioral variables', 'FontSize', 12);
    ylabel([Y_label, ' correlation coefficient'], 'FontSize', 12);
    set(gca, 'xtick', 1:length(X), 'xtickLabel', X)
    if ~isempty(explVarLVs)
        title({['LV' num2str(this_lv) ' - ' num2str(100*explVarLVs(this_lv), '%.2f') '% of covariance']}, 'FontSize', 12);
    end
    ylim([-1, 1]);
    hold off
    if ~isempty(outputPath)
        saveas(gcf, fullfile(outputPath, ['LV' num2str(this_lv), ' ', Y_label]), 'png');
    end
end
end