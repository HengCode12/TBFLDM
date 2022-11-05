function plot1=createfigure(X1, YMatrix1)
%CREATEFIGURE1(X1, YMatrix1)
%  X1:  x ���ݵ�����
%  YMATRIX1:  y ���ݵľ���

%  �� MATLAB �� 05-Feb-2022 22:20:06 �Զ�����

% ���� figure
figure1 = figure('InvertHardcopy','off','Color',[1 1 1]);

% ���� axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

% ʹ�� plot �ľ������봴������
plot1 = plot(X1,YMatrix1,'MarkerSize',1,'Marker','o','LineWidth',1.1,...
    'Parent',axes1);
set(plot1(1),'DisplayName','UPSVM','MarkerFaceColor',[1 1 1],...
    'MarkerEdgeColor',[0 1 0],...
    'Color',[0 1 0]);
set(plot1(2),'DisplayName','Pin-SVM','Color',[0 0 0]);
set(plot1(3),'DisplayName','FUPLDM','Color',[1 0 0]);

% ���� ylabel
ylabel({'Acc'});

% ���� xlabel
xlabel({'��'});

box(axes1,'on');
% ���� legend
legend1 = legend(axes1,'show');
set(legend1,...
    'Position',[0.678929373547355 0.13620725246618 0.196791864598115 0.182510508208716],...
    'LineWidth',1.1);

