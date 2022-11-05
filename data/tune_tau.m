function tune_tau(Ctrain,dtrain,Ctest,dtest,C,kernel,p1,lamb1,lamb2,K_fold,data_name,N_times)
tauval= -1:0.05:1;
%% 为画图做数据准备
acc_k1 = zeros(1, length(C));acc_k2 = zeros(1, length(C));acc_k5 = zeros(length(C), length(lamb1), length(lamb2));
acc_1 = zeros(1, length(tauval));acc_2 = zeros(1, length(tauval));acc_5 = zeros(1, length(tauval));
%% 迭代训练找出最优超参数
Best_Acc_0 = 0;Best_Acc_1 = 0;Best_Acc_2 = 0;Best_Acc_3 = 0;
Best_Acc_4 = 0;Best_Acc_5 = 0;Best_Acc_6 = 0;Best_Acc_7 = 0;
stop_num = (length(tauval)*length(lamb1)*length(lamb2)*K_fold + length(tauval)*K_fold)*length(C);
for jj=1:length(tauval)
    for i=1:length(C)
        c=C(i);
        Indices = crossvalind('Kfold', length(dtrain), K_fold);
        Acc_SubPredict_0 = zeros(K_fold, 1);
        Acc_SubPredict_1 = zeros(K_fold, 1);
        Acc_SubPredict_2 = zeros(K_fold, 1);
        Acc_SubPredict_3 = zeros(K_fold, 1);
        for repeat = 1:K_fold
            time0=0;time1=0;time2=0;time3=0;
            I_SubTrain = ~(Indices==repeat);
            Ctrain_ = Ctrain(I_SubTrain, :);
            dtrain_ = dtrain(I_SubTrain, :);
            I_SubPredict = ~ I_SubTrain;
            Ctest_ = Ctrain(I_SubPredict, :);
            dtest_ = dtrain(I_SubPredict, :);
            s = Fuzzy_MemberShip_FCM(Ctrain_, dtrain_);
            %fprintf('%3.0f steps remaining...\n',length(tauval)-count);
            tau= tauval(jj);
            tic
            [acc0,C0] = Unified_pin_svm(Ctrain_, dtrain_, Ctest_,dtest_, kernel, 0,c,p1);  %SVM
            time0 = time0 + toc;
            tic
            [acc1,C1] = Unified_pin_svm(Ctrain_, dtrain_, Ctest_,dtest_, kernel, tau,c,p1);  %UPSVM
            time1 = time1 + toc;
            tic
            [acc2,C2] = pin_svm(Ctrain_, dtrain_, Ctest_,dtest_, kernel, tau,c,p1); %PSVM
            time2 = time2 + toc;
            tic
            [acc3,C3] = Unified_pin_fsvm(Ctrain_, dtrain_, Ctest_,dtest_, kernel, tau,c,p1,s); %FUPSVM
            time3 = time3 + toc;
            Acc_SubPredict_0(repeat) = acc0;
            Acc_SubPredict_1(repeat) = acc1;
            Acc_SubPredict_2(repeat) = acc2;
            Acc_SubPredict_3(repeat) = acc3;
            stop_num = stop_num - 1;
            fprintf('%d step(s) remaining.\n',stop_num);
        end
        Index_Acc_0 = mean(Acc_SubPredict_0);
        Index_Acc_1 = mean(Acc_SubPredict_1);
        Index_Acc_2 = mean(Acc_SubPredict_2);
        Index_Acc_3 = mean(Acc_SubPredict_3);
        acc_k1(i) = Index_Acc_1;
        acc_k2(i) = Index_Acc_2;
        if Index_Acc_0>Best_Acc_0
            Best_Acc_0 = Index_Acc_0;
            Best_C0 = C0;
            Time_0 = time0;
        end
        if Index_Acc_1>Best_Acc_1
            Best_Acc_1 = Index_Acc_1;
            Best_C1 = C1;
            Best_tauval_1 = tau;
            Time_1 = time1;
        end
        if Index_Acc_2>Best_Acc_2
            Best_Acc_2 = Index_Acc_2;
            Best_C2 = C2;
            Best_tauval_2 = tau;
            Time_2 = time2;
        end
        if Index_Acc_3>Best_Acc_3
            Best_Acc_3 = Index_Acc_3;
            Best_C3 = C3;
            Best_tauval_3 = tau;
            Time_3 = time3;
        end
        for k1 = 1:length(lamb1)
            for k2 = 1:length(lamb2)
                Indices = crossvalind('Kfold', length(dtrain), K_fold);
                Acc_SubPredict_4 = zeros(K_fold, 1);
                Acc_SubPredict_5 = zeros(K_fold, 1);
                Acc_SubPredict_6 = zeros(K_fold, 1);
                Acc_SubPredict_7 = zeros(K_fold, 1);
                for repeat = 1:K_fold
                    time4=0;time5=0;time6=0;time7=0;
                    I_SubTrain = ~(Indices==repeat);
                    Ctrain_ = Ctrain(I_SubTrain, :);
                    dtrain_ = dtrain(I_SubTrain, :);
                    I_SubPredict = ~ I_SubTrain;
                    Ctest_ = Ctrain(I_SubPredict, :);
                    dtest_ = dtrain(I_SubPredict, :);
                    tic
                    s = Fuzzy_MemberShip_FCM(Ctrain_, dtrain_);
                    [acc4,C4] = Unified_pin_ldm(Ctrain_, dtrain_, Ctest_,dtest_, kernel, tau,c,p1,lamb1(k1),lamb2(k2));  %UPLDM
                    time4 = time4 + toc;
                    tic
                    [acc5,C5] = Unified_pin_fldm(Ctrain_, dtrain_, Ctest_,dtest_, kernel, tau,c,p1,s,lamb1(k1),lamb2(k2));  %FUPLDM
                    time5 = time5 + toc;
                    tic
                    [acc6,C6] = Unified_pin_ldm(Ctrain_, dtrain_, Ctest_,dtest_, kernel, 0,c,p1,lamb1(k1),lamb2(k2));  %LDM
                    time6 = time6 + toc;
                    tic
                    [acc7,C7] = Unified_pin_fldm(Ctrain_, dtrain_, Ctest_,dtest_, kernel, 0,c,p1,s,lamb1(k1),lamb2(k2));  %FLDM
                    time7 = time7 + toc;
                    Acc_SubPredict_4(repeat) = acc4;
                    Acc_SubPredict_5(repeat) = acc5;
                    Acc_SubPredict_6(repeat) = acc6;
                    Acc_SubPredict_7(repeat) = acc7;
                    stop_num = stop_num - 1;
                    fprintf('%d step(s) remaining.\n',stop_num);
                end
                Index_Acc_4 = mean(Acc_SubPredict_4);
                Index_Acc_5 = mean(Acc_SubPredict_5);
                Index_Acc_6 = mean(Acc_SubPredict_6);
                Index_Acc_7 = mean(Acc_SubPredict_7);
                acc_k5(i, k1, k2) = Index_Acc_5;
                if Index_Acc_4>Best_Acc_4
                    Best_Acc_4 = Index_Acc_4;
                    Best_C4 = C4;
                    Best_tauval_4 = tau;
                    Best_lamb1_4 = lamb1(k1);
                    Best_lamb2_4 = lamb2(k1);
                    Time_4 = time4;
                end
                if Index_Acc_5>Best_Acc_5
                    Best_Acc_5 = Index_Acc_5;
                    Best_C5 = C5;
                    Best_tauval_5 = tau;
                    Best_lamb1_5 = lamb1(k1);
                    Best_lamb2_5 = lamb2(k1);
                    Time_5 = time5;
                end
                if Index_Acc_6>Best_Acc_6
                    Best_Acc_6 = Index_Acc_6;
                    Best_C6 = C6;
                    Best_lamb1_6 = lamb1(k1);
                    Best_lamb2_6 = lamb2(k1);
                    Time_6 = time6;
                end
                if Index_Acc_7>Best_Acc_7
                    Best_Acc_7 = Index_Acc_7;
                    Best_C7 = C7;
                    Best_lamb1_7 = lamb1(k1);
                    Best_lamb2_7 = lamb2(k1);
                    Time_7 = time7;
                end
            end
        end
    end
    acc_1(jj) = max(acc_k1);
    acc_2(jj) = max(acc_k2);
    acc_5(jj) = max(max(max(acc_k5)));
end
fprintf('Best_Acc_0 = %f\n', Best_Acc_0);
fprintf('Best_Acc_1 = %f\n', Best_Acc_1);
fprintf('Best_Acc_2 = %f\n', Best_Acc_2);
fprintf('Best_Acc_3 = %f\n', Best_Acc_3);
fprintf('Best_Acc_4 = %f\n', Best_Acc_4);
fprintf('Best_Acc_5 = %f\n', Best_Acc_5);
fprintf('Best_Acc_6 = %f\n', Best_Acc_6);
%% 开始对预测集预测
s = Fuzzy_MemberShip_FCM(Ctrain, dtrain);
[acc0_predict,] = Unified_pin_svm(Ctrain, dtrain, Ctest,dtest, kernel, 0,Best_C0,p1);  %SVM
[acc1_predict,] = Unified_pin_svm(Ctrain, dtrain, Ctest,dtest, kernel, Best_tauval_1,Best_C1,p1);  %UPSVM
[acc2_predict,] = pin_svm(Ctrain, dtrain, Ctest,dtest, kernel, Best_tauval_2,Best_C2,p1); %PSVM
[acc3_predict,] = Unified_pin_fsvm(Ctrain, dtrain, Ctest,dtest, kernel, Best_tauval_3,Best_C3,p1,s); %FUPSVM
[acc4_predict,] = Unified_pin_ldm(Ctrain, dtrain, Ctest,dtest, kernel, Best_tauval_4,Best_C4,p1,Best_lamb1_4,Best_lamb2_4);  %UPLDM
[acc5_predict,] = Unified_pin_fldm(Ctrain, dtrain, Ctest,dtest, kernel, Best_tauval_5,Best_C5,p1,s,Best_lamb1_5,Best_lamb2_5);  %FUPLDM
[acc6_predict,] = Unified_pin_ldm(Ctrain, dtrain, Ctest,dtest, kernel, 0,Best_C6,p1,Best_lamb1_5,Best_lamb2_6);  %LDM
[acc7_predict,] = Unified_pin_fldm(Ctrain, dtrain, Ctest,dtest, kernel, 0,Best_C7,p1,s,Best_lamb1_5,Best_lamb2_7);  %FLDM
%% 结果写入
f = fopen(['cross_',data_name,'_result.txt'],'a+');
fprintf(f,'N_time=%f\n', N_times);
fprintf(f,'SVM:\n');
fprintf(f,'ACC_Predict:%f\nBest_C:%f\nTime:%fs\n', [acc0_predict,Best_C0,Time_0]);
fprintf(f,'UPSVM:\n');
fprintf(f,'ACC_Predict:%f\nBest_C:%f\nBest_tau:%f\nTime:%fs\n', [acc1_predict,Best_C1,Best_tauval_1,Time_1]);
fprintf(f,'PSVM:\n');
fprintf(f,'ACC_Predict:%f\nBest_C:%f\nBest_tau:%f\nTime:%fs\n', [acc2_predict,Best_C2,Best_tauval_2,Time_2]);
fprintf(f,'FUPSVM:\n');
fprintf(f,'ACC_Predict:%f\nBest_C:%f\nBest_tau:%f\nTime:%fs\n', [acc3_predict,Best_C3,Best_tauval_3,Time_3]);
fprintf(f,'UPLDM:\n');
fprintf(f,'ACC_Predict:%f\nBest_C:%f\nBest_tau:%f\nBest_lamb1:%f\nBest_lamb2:%f\nTime:%fs\n', [acc4_predict,Best_C4,Best_tauval_4,Best_lamb1_4,Best_lamb2_4,Time_4]);
fprintf(f,'FUPLDM:\n');
fprintf(f,'ACC_Predict:%f\nBest_C:%f\nBest_tau:%f\nBest_lamb1:%f\nBest_lamb2:%f\nTime:%fs\n', [acc5_predict,Best_C5,Best_tauval_5,Best_lamb1_5,Best_lamb2_5,Time_5]);
fprintf(f,'LDM:\n');
fprintf(f,'ACC_Predict:%f\nBest_C:%f\nBest_lamb1:%f\nBest_lamb2:%f\nTime:%fs\n', [acc6_predict,Best_C6,Best_lamb1_6,Best_lamb2_6,Time_6]);
fprintf(f,'FLDM:\n');
fprintf(f,'ACC_Predict:%f\nBest_C:%f\nBest_lamb1:%f\nBest_lamb2:%f\nTime:%fs\n', [acc7_predict,Best_C7,Best_lamb1_7,Best_lamb2_7,Time_7]);
%% 画图
Ymat=[acc_1',acc_2',acc_5'];
h=createfigure(tauval,Ymat);