function plot1=createfigure(X1, YMatrix1)
%CREATEFIGURE1(X1, YMatrix1)
%  X1:  x 数据的向量
%  YMATRIX1:  y 数据的矩阵

%  由 MATLAB 于 05-Feb-2022 22:20:06 自动生成

% 创建 figure
figure1 = figure('InvertHardcopy','off','Color',[1 1 1]);

% 创建 axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

% 使用 plot 的矩阵输入创建多行
plot1 = plot(X1,YMatrix1,'MarkerSize',1,'Marker','o','LineWidth',1.1,...
    'Parent',axes1);
set(plot1(1),'DisplayName','UPSVM','MarkerFaceColor',[1 1 1],...
    'MarkerEdgeColor',[0 1 0],...
    'Color',[0 1 0]);
set(plot1(2),'DisplayName','Pin-SVM','Color',[0 0 0]);
set(plot1(3),'DisplayName','FUPLDM','Color',[1 0 0]);

% 创建 ylabel
ylabel({'Acc'});

% 创建 xlabel
xlabel({'τ'});

box(axes1,'on');
% 创建 legend
legend1 = legend(axes1,'show');
set(legend1,...
    'Position',[0.678929373547355 0.13620725246618 0.196791864598115 0.182510508208716],...
    'LineWidth',1.1);

