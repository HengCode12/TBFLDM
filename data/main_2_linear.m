%%
% 2022-9-20修改：1、s的计算移到了交叉验证循环之中，因为涉及到新的训练集（交叉验证训练集）
%                2、增加了K_flod变量、count变量（用于保存tune_tau中运行的步数，便于步数的计算）
%                3、增加了用于交叉验证的循环，循环开始前添加了各个结果的均值变量设初值为0
%                4、每次交叉验证循环结束后添加了计算均值的公式
%                5、打印结果函数全部改为了打印交叉验证结果的均值
%                6、更改了tune_tau文件中打印还有多少步的代码（函数的输入加入了之前循环已经运行的步数count，输出加入了最新的步数），因为添加了交叉验证步数增加
%%
tic
close all;
clear all;
clc
rand('state', 2015)
randn('state', 2015)
%%
d= {'Dataset', 'Accuracy','Time','p','C','tau'};
% xlswrite('1111.xlsx', d,'F1');
dd2 = zeros(20,48);
pp1 = zeros(20,1);

for index=2:20
    %%
    if(index==2)
        load('monks_2_train.txt');
        load('monks_2_test.txt');
        Ctrain= monks_2_train(:,2:end);
        dtrain=  monks_2_train(:,1);
        dtrain(find(dtrain==0))=-1;
        Ctest= monks_2_test(:,2:end);
        dtest=  monks_2_test(:,1);
        dtest(find(dtest==0))=-1;
        disp('Monk 2');
    end
    %%
    if (index==3)
        load('monks_3_train.txt');
        load('monks_3_test.txt');
        Ctrain= monks_3_train(:,2:end);
        dtrain=  monks_3_train(:,1);
        dtrain(find(dtrain==0))=-1;
        Ctest= monks_3_test(:,2:end);
        dtest=  monks_3_test(:,1);
        dtest(find(dtest==0))=-1;
        disp('Monk 3');
    end
    %%
    if (index==4)
        load('SPECT_train.txt')
        load('SPECT_test.txt');
        Ctrain = SPECT_train(:,2:end);
        dtrain=  SPECT_train(:,1);
        dtrain(find(dtrain==0))=-1;
        Ctest= SPECT_test(:,2:end);
        dtest=  SPECT_test(:,1);
        dtest(find(dtest==0))=-1;
        disp('Spect');
    end
    %%
    if (index==5)
        
        load('Haberman_dataset.mat')
        X=Haberman_data(:,1:end-1);
        Y= Haberman_data(:,end);
        Y(find(Y==2))=-1;
        Ctrain=X(1:150,:);
        dtrain= Y(1:150,:);
        Ctest= X(151:end,:);
        dtest= Y(151:end,:);
        disp ('Heberman');
    end
    
    %%
    if (index==6)
        load('heartdata.mat');
        X=heartdata(:,1:end-1);
        Y= heartdata(:,end);
        Y(find(Y==2))=-1;
        Ctrain=X(1:150,:);
        dtrain= Y(1:150,:);
        Ctest= X(151:end,:);
        dtest= Y(151:end,:);
        disp ('Statlog');
    end
    %%
    if (index==7)
        load('ionosphere_data.mat');
        X=data(:,2:end);
        Y= data(:,1);
        Y(find(Y==0))=-1;
        Ctrain=X(1:200,:);
        dtrain= Y(1:200,:);
        Ctest= X(201:end,:);
        dtest= Y(201:end,:);
        disp('Ionosphere')
    end
    %%
    if (index==8)
        data= xlsread('Pima-Indian.xlsx');
        X=data(:,2:end);
        Y= data(:,1);
        Y(find(Y==0))=-1;
        Ctrain=X(1:300,:);
        dtrain= Y(1:300,:);
        Ctest= X(301:end,:);
        dtest= Y(301:end,:);
        disp('Pima-Indian');
    end
    %%
    
    if (index==9)
        load('wdbc_data.mat')
        load('wdbc_label.mat')
        X=wdbc_data;
        Y= wdbc_label;
        Y(find(Y==2))=-1;
        Ctrain=X(1:400,:);
        dtrain= Y(1:400,:);
        Ctest= X(401:end,:);
        dtest= Y(401:end,:);
        disp('WDBC');
    end
    %%
    if (index==10)
        load('echocardiogram_data.mat');
        load('echocardiogram_label.mat');
        X=x;
        Y= y;
        Y(find(Y==0))=-1;
        Ctrain=X(1:80,:);
        dtrain= Y(1:80,:);
        Ctest= X(81:end,:);
        dtest= Y(81:end,:);
        disp('Echo');
    end
    
    %%
    if (index==11)
        
        load('german.txt');
        X=german(:,1:end-1);
        Y= german(:,end);
        Y(find(Y==2))=-1;
        Ctrain=X(1:500,:);
        dtrain= Y(1:500,:);
        Ctest= X(501:end,:);
        dtest= Y(501:end,:);
        disp('Germans');
        
    end
    %%
    
    if (index==12)
        
        data=xlsread('Australian.xlsx');
        X=data(:,2:end);
        Y= data(:,1);
        Y(find(Y==2))=-1;
        Ctrain=X(1:400,:);
        dtrain= Y(1:400,:);
        Ctest= X(401:end,:);
        dtest= Y(401:end,:);
        disp('Australian');
        
    end
    %%
    if (index==13)
        
        data=xlsread('Bupa-Liver.xlsx');
        X=data(:,2:end);
        Y= data(:,1);
        Y(find(Y==0))=-1;
        Ctrain=X(1:250,:);
        dtrain= Y(1:250,:);
        Ctest= X(251:end,:);
        dtest= Y(251:end,:);
        disp('Bupa');
        
    end
    
    %%
    if (index==14)
        load('votes.mat')
        X=votes(:,2:end);
        Y= votes(:,1);
        Y(find(Y==2))=-1;
        Ctrain=X(1:200,:);
        dtrain= Y(1:200,:);
        Ctest= X(201:end,:);
        dtest= Y(201:end,:);
        disp('Votes');
        
    end
    %%
    if (index==15)
        load('diabetes_data.mat')
        load('diabetes_label.mat')
        X= data1;
        Y= label;
        Y(find(Y==0))=-1;
        Ctrain=X(1:500,:);
        dtrain= Y(1:500,:);
        Ctest= X(501:end,:);
        dtest= Y(501:end,:);
        disp('Daibetes');
        
    end
    
    %%
    if(index==16)
        data=xlsread('fertility_Diagnosis.xlsx');
        X= data(:,1:end-1);
        Y= data(:,end);
        Y(find(Y==0))=-1;
        Ctrain=X(1:50,:);
        dtrain= Y(1:50,:);
        Ctest= X(51:end,:);
        dtest= Y(51:end,:);
        disp('Fertility');
        
    end
    %%
    if(index==17)
        data= xlsread('Sonar.xlsx');
        %         rng(2);
        X= data(:,2:end);
        Y= data(:,1);
        Y(find(Y==0))=-1;
        r1=randperm(size(X,1));
        X = X(r1,:);
        Y=Y(r1,:);
        Ctrain=X(1:100,:);
        dtrain= Y(1:100,:);
        Ctest= X(101:end,:);
        dtest= Y(101:end,:);
        disp('Sonar');
        
    end
    
    %%
    % %  rng(9);
    %  r1=randperm(size(Ctrain,1));
    %  Ctrain= Ctrain(r1,:);
    %  dtrain=dtrain(r1,:);
    %%
    if(index==18)
        load('ecoli_data.mat');
        %         rng(2);
        X= ecoli_data(:,1:end-1);
        Y= ecoli_data(:,end);
        Y(find(Y~=1))=-1;
        r1=randperm(size(X,1));
        X = X(r1,:);
        Y=Y(r1,:);
        Ctrain=X(1:200,:);
        dtrain= Y(1:200,:);
        Ctest= X(201:end,:);
        dtest= Y(201:end,:);
        disp('Ecoil');
        
    end
    %%
    if(index==19)
        load('plrx.txt');
        X=plrx(:,1:end-1);
        Y= plrx(:,end);
        Y(find(Y==2))=-1;
        Ctrain=X(1:100,:);
        dtrain= Y(1:100,:);
        Ctest= X(101:end,:);
        dtest= Y(101:end,:);
        disp('Prlx');
        
    end
    
    %%
    if (index==20)
        load('monks_1_train.txt');
        load('monks_1_test.txt');
        Ctrain= monks_1_train(:,2:end);
        dtrain=  monks_1_train(:,1);
        dtrain(find(dtrain==0))=-1;
        Ctest= monks_1_test(:,2:end);
        dtest=  monks_1_test(:,1);
        dtest(find(dtest==0))=-1;
        disp('Monk 1');
    end
    if(index==21)
        load('spambase_data.mat');
        X=spambase(:,1:end-1);
        Y= spambase(:,end);
        Y(find(Y==0))=-1;
        rng(1);
        r=randperm(length(Y));
        X= X(r,:);
        Y=Y(r,:);
        Ctrain=X(1:3000,:);
        dtrain= Y(1:3000,:);
        Ctest= X(3001:end,:);
        dtest= Y(3001:end,:);
        disp('Spambase');
        
        
    end
    if (index==1) %artificial
        mul=[0.5,-3]; % 均值
        S1=[0.2 0;0 3]; % 协方差
        data1=mvnrnd(mul, S1, 200); % 产生高斯分布数据
        data1(:,3)=1;
        % 第二组数据
        mu2=[-0.5,3];
        S2=[0.2 0;0 3];
        data2=mvnrnd(mu2,S2,200);
        data2(:,3)=-1;
        % noises of p
        mm1=60;
        %         mu3=[0,0.3];
        %         S3=[0.3 0;0 0.3];
        mu3=[0,0];
        S3=[1 -0.8;-0.8 1];
        data3=mvnrnd(mu3,S3,mm1);
        data3(:,3)=-1;
        % noises of n
        mm2=mm1;
        %         mu4=[0,-0.3];
        %         S4=[0.3 0.1;0.1 0.3];
        mu4=[0,0];
        S4=[1 -0.8;-0.8 1];
        data4=mvnrnd(mu4,S4,mm2);
        data4(:,3)=1;
        data_noise = [data3;data4];
        %% all
        data_train=[data1(1:100,:);data2(1:100,:);data_noise];
        Ctrain = data_train(:,1:end-1);
        dtrain = data_train(:,end);
        data_predict=[data1(101:end,:);data2(101:end,:)];
        Ctest = data_predict(:,1:end-1);
        dtest = data_predict(:,end);
        
    end
    
    
    %% Parameter setting
    Ctrain=awgn(Ctrain,0.05); % 高斯噪声
    Ctest=awgn(Ctest,0.05); % 高斯噪声
    Ctrain= svdatanorm(Ctrain,'svpline');
    Ctest= svdatanorm(Ctest,'scpline');
    %s = Fuzzy_MemberShip_FCM(Ctrain, dtrain);
    %s = Fuzzy_MemberShip(Ctrain, dtrain, kernel, 0.5,p1);
    clear C;
    kernel=1;
    C0= 2^0;
    tau=0;
    svm_tau=0;
    p1= 2^-2;
    lamb1 = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1];
    lamb2 = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1];
    p1val=[2^-6];
    c1val=[2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3];
    K_fold = 2;
    count = 0;
    %     p1val=2;
    %     c1val=2^-2;
    %% Prameter Tunning
    %     [acc_svm, opt_p1,opt_c1,t1]= tune_para_svm(Ctrain,dtrain,Ctest,dtest,kernel,c1val,p1val);
    %     p1= opt_p1;
    %     pp1(index,1)= p1;
    %     C0= opt_c1;
    %     lamb1 = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1];
    %     lamb2 = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1];
    %       lamb1 = 0;
    %       lamb2 = 0;
    %     if (kernel==2)
    %         fprintf('\n Optimal Accuracy = %3.2f with kernel parameter p1= %3.4f and C= %3.4f',acc_svm, opt_p1,opt_c1);
    %     end
    %     if (kernel==1)
    %         fprintf('\n Optimal Accuracy = %3.2f with  C= %3.4f',acc_svm,opt_c1);
    %     end
    %     fprintf('\n Time Elpased  in Tunning Paramter = %3.2f seconds',t1);
    
    %% K折交叉验证
    Indices = crossvalind('Kfold', length(dtrain), K_fold);
    Acc_SubPredict = zeros(K_fold, 1);
    acc_svm_ = 0;acc_upsvm_ = 0;acc_psvm_ = 0;acc_fupsvm_ = 0;acc_upldm_ = 0;acc_fupldm_ = 0;acc_ldm_ = 0;acc_fldm_ = 0;
    opt_C0_ = 0;opt_C1_ = 0;opt_C2_ = 0;opt_C3_ = 0;opt_C4_ = 0;opt_C5_ = 0;opt_C6_ = 0;opt_C7_ = 0;
    time0_ = 0;time1_ = 0;time2_ = 0;time3_ = 0;time4_ = 0;time5_ = 0;time6_ = 0;time7_ = 0;
    for repeat = 1:K_fold
        I_SubTrain = ~(Indices==repeat);
        Ctrain_ = Ctrain(I_SubTrain, :);
        dtrain_ = dtrain(I_SubTrain, :);
        I_SubPredict = ~ I_SubTrain;
        Ctest_ = Ctrain(I_SubPredict, :);
        dtest_ = dtrain(I_SubPredict, :);
        s = Fuzzy_MemberShip_FCM(Ctrain_, dtrain_);
        [opt_tau1,opt_tau2,opt_tau3,opt_tau4,opt_tau5,...
            acc_svm,acc_upsvm,acc_psvm,acc_fupsvm,acc_upldm,acc_fupldm,acc_ldm,acc_fldm,...
            time0,time1,time2,time3,time4,time5,time6,time7,...
            opt_C0,opt_C1,opt_C2,opt_C3,opt_C4,opt_C5,opt_C6,opt_C7,h,count_]= tune_tau(Ctrain_,dtrain_,Ctest_,dtest_,c1val,kernel,p1,s,lamb1,lamb2,K_fold,count);
        % 开始求平均
        acc_svm_ = acc_svm_ + acc_svm/K_fold;acc_upsvm_ = acc_upsvm_ + acc_upsvm/K_fold;acc_psvm_ = acc_psvm_ + acc_psvm/K_fold;
        acc_fupsvm_ = acc_fupsvm_ + acc_fupsvm/K_fold;acc_upldm_ = acc_upldm_ + acc_upldm/K_fold;acc_ldm_ = acc_ldm_ + acc_ldm/K_fold;
        acc_fldm_ = acc_fldm_ + acc_fldm/K_fold;acc_fupldm_ = acc_fupldm_ + acc_fupldm/K_fold;
        opt_C0_ = opt_C0_ + opt_C0/K_fold;opt_C1_ = opt_C1_ + opt_C1/K_fold;opt_C2_ = opt_C2_ + opt_C2/K_fold;opt_C3_ = opt_C3_ + opt_C3/K_fold;
        opt_C4_ = opt_C4_ + opt_C4/K_fold;opt_C5_ = opt_C5_ + opt_C5/K_fold;opt_C6_ = opt_C6_ + opt_C6/K_fold;opt_C7_ = opt_C7_ + opt_C7/K_fold;
        time0_ = time0_ + time0/K_fold;time1_ = time1_ + time1/K_fold;time2_ = time2_ + time2/K_fold;time3_ = time3_ + time3/K_fold;
        time4_ = time4_ + time4/K_fold;time5_ = time5_ + time5/K_fold;time6_ = time6_ + time6/K_fold;time7_ = time7_ + time7/K_fold;
        count = count_;
    end
    %% SVM
    fprintf('\n SVM Accuracy_MEAN=%3.2f,time_MEAN = %3.2f,C_MEAN = %3.2f',acc_svm_,time0_,opt_C0_);
    d0= [index, acc_svm_,time0_,p1,opt_C0_,0];
    
    %% UPSVM
    fprintf('\n UPSVM Accuracy_MEAN=%3.2f,time_MEAN = %3.2f,tau = %3.2f,C_MEAN = %3.2f',acc_upsvm_,time1_,opt_tau1,opt_C1_);
    d1= [index, acc_upsvm_,time1_,p1,opt_C1_,opt_tau1];
    
    %% PSVM
    fprintf('\n PSVM Accuracy_MEAN=%3.2f,time_MEAN = %3.2f,tau = %3.2f,C_MEAN = %3.2f',acc_psvm_,time2_,opt_tau2,opt_C2_);
    d2= [index, acc_psvm_,time2_,p1,opt_C2_,opt_tau2];
    
    
    %% FUPSVM
    fprintf('\n FUPSVM Accuracy_MEAN=%3.2f,time_MEAN = %3.2f,tau = %3.2f,C_MEAN = %3.2f',acc_fupsvm_,time3_,opt_tau3,opt_C3_);
    d3= [index, acc_fupsvm_,time3_,p1,opt_C3_,opt_tau3];
    
    %% UPLDM
    fprintf('\n UPLDM Accuracy_MEAN=%3.2f,time_MEAN = %3.2f,tau = %3.2f,C_MEAN = %3.2f',acc_upldm_,time4_,opt_tau4,opt_C4_);
    d4= [index, acc_upldm_,time4_,p1,opt_C4_,opt_tau4];
    
    %% FUPLDM
    fprintf('\n FUPLDM Accuracy_MEAN=%3.2f,time_MEAN = %3.2f,tau = %3.2f,C_MEAN = %3.2f',acc_fupldm_,time5_,opt_tau5,opt_C5_);
    d5= [index, acc_fupldm_,time5_,p1,opt_C5_,opt_tau5];
    
    %% LDM
    fprintf('\n LDM_MEAN Accuracy=%3.2f,time_MEAN = %3.2f,C_MEAN = %3.2f',acc_ldm_,time6_,opt_C6_);
    d6= [index, acc_ldm_,time6_,p1,opt_C6_,0];
    
    %% F-LDM
    fprintf('\n F-LDM Accuracy_MEAN=%3.2f,time_MEAN = %3.2f,C_MEAN = %3.2f',acc_fldm_,time7_,opt_C7_);
    d7= [index, acc_fldm_,time7_,p1,opt_C7_,0];
    
    %%
    dd2(index,:)= [d0,d1,d2,d3,d4,d5,d6,d7];
    
    %   saveas(gcf,sprintf('Dataset%d.fig',index))
end
% [m,n] = size(dd2);
% A1=[];
% A2=[];
% A3=[];
% A4=[];
% A5=[];
% A6=[];
%
% A1= dd2(:,1:6);
% A2= dd2(:,7:12);
% A3= dd2(:,13:18);
% A4= dd2(:,19:24);
% A5= dd2(:,25:30);
% A6= dd2(:,31:36);
%
% final = zeros( m*6,6);
% j=1;
% for i=1:m
%     final(j,:) = A1(i,:);
%     final(j+1,:) = A2(i,:);
%     final(j+2,:) = A3(i,:);
%     final(j+3,:) = A4(i,:);
%     final(j+4,:) = A5(i,:);
%     final(j+5,:) = A6(i,:);
%     j=j+6;
% end
%
% xlswrite('1111.xlsx', final);
