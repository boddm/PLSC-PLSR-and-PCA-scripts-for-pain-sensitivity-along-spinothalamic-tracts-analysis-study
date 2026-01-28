function myScreePlot(SS, K, XAX, lam)
% Purpose: Plot scree plot for PLS-SVD
% Inputs:
%   SS: singular values values
%   K: cut-off
%   XAX: x axis limits
%   lam: null distribution lambda

%%
if nargin<3
    XAX=[1 length(SS)];
end

% pos=get(gcf,'position');
% set(gcf,'Position',[pos(1) pos(2) 1300 300])

title('Scree plot');
EV = cumsum(SS.^2/sum(SS.^2))*100;
a = find(EV>80);

[AX, H1, H2] = plotyy(1:length(SS),SS,1:length(SS),EV,'plot','plot');
hold on; 
stem(SS, 'color', 'k');

set(get(AX(1), 'Ylabel'), 'String', 'Singular value', 'fontsize', 16)
set(get(AX(2), 'Ylabel'),'String','Explained covariance','fontsize',16)
xlabel('Latent variable','fontsize',16);
axes(AX(1))
ylim([0 1.01*max(SS)]); xlim(XAX)
%set(H1,'LineStyle','-','Marker','o')
set(H1,'color','w');
set(H2,'color','b'); set(AX,{'ycolor'},{'k';'b'})
set(gca,'Box', 'off','TickDir', 'out','TickLength'  , [.01 .01] ,'XMinorTick'  , 'on'      , ...
    'YMinorTick'  , 'on'); %,'YTick' , 100:100:700);
if exist('lam','var')
    hold on;
    %errorbar(1:XAX(2), mean(lam(:,1:XAX(2))), std(lam(:,1:XAX(2))),'color','k');
    %     plot(prctile(lam(:,1:XAX(2)),95),'k.-');
    line(XAX,[1 1]*prctile(max(lam),95),'color','k')
    %     a=find(SS(1:size(lam,2))'>prctile(lam,95));
    %     plot(a(end),SS(a(end)),'r','Marker','o','MarkerFaceColor','r')
    %     disp(a(end))
    a=find(SS(1:length(lam))'>prctile(max(lam),95));
    plot(a(end),SS(a(end)),'r','Marker','o','MarkerFaceColor','r')
    disp(a(end))
end


axes(AX(2))
ylim([0 100]); 
xlim(XAX)

set(H2,'LineStyle', '-')
if exist('K', 'var')
    hold on; 
    line([K K], [0 100], 'Color', 'k');
end

set(gca, 'YGrid', 'on');
set(gca, 'Box', 'off', 'TickDir', 'out', 'TickLength', [.01 .01], 'XMinorTick', 'on', 'YMinorTick' , 'on');

end