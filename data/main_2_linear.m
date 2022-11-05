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

for index=1
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
        data_name = 'monks_2';
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
        data_name = 'monks_3';
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
        data_name = 'SPECT';
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
        data_name = 'Haberman_dataset';
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
        data_name = 'heartdata';
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
        data_name = 'ionosphere_data';
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
        data_name = 'Pima-Indian';
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
        data_name = 'wdbc_data';
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
        data_name = 'echocardiogram_data';
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
        data_name = 'german';
        
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
        data_name = 'Australian';
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
        data_name = 'Bupa-Live';
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
        data_name = 'votes';
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
        data_name = 'diabetes_data';
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
        data_name = 'fertility_Diagnosis';
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
        data_name = 'Sonar';
    end
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
        data_name = 'ecoli_data';
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
        data_name = 'plrx';
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
        data_name = 'monks_1';
    end
    if(index==21)
        load('spambase_data.mat');
        X=spambase(:,1:end-1);
        Y= spambase(:,end);
        Y(find(Y==0))=-1;
%         rng(1);
%         r=randperm(length(Y));
%         X= X(r,:);
%         Y=Y(r,:);
        Ctrain=X(1:3000,:);
        dtrain= Y(1:3000,:);
        Ctest= X(3001:end,:);
        dtest= Y(3001:end,:);
        disp('Spambase');
        data_name = 'spambase_data';
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
        data_name = 'artificial_data';
    end
    %% Parameter setting
    Ctrain=awgn(Ctrain,0.05); % 高斯噪声
    Ctest=awgn(Ctest,0.05); % 高斯噪声
    Ctrain= svdatanorm(Ctrain,'svpline');
    Ctest= svdatanorm(Ctest,'scpline');
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
    %% 开始训练
    for N_times = 1:3
        tune_tau(Ctrain,dtrain,Ctest,dtest,c1val,kernel,p1,lamb1,lamb2,5,data_name,N_times);
    end
end