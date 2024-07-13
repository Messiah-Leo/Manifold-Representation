function [Accuracy_BPNN2,AccuForNMF2,AccuracyDLDA2,AccuracyKPCA2,AccuracyPCA2,Accuracy,Accuracy2,t1,AUC_2layer]=twolayer(NumofAlltest,Database,Test_id,MaxAccuRank,NumOfClass,TrainingVector,NumOfTraining,NumOfTesting,TestingVector)
    Q=[];
    Priors=[];
    Means=[];
    Covs=[];
% for j=1:NumOfClass
    neuralalltrainid=[];
        dem=size(TrainingVector,1);
        X=TrainingVector';
        K=3;

    for i=1:NumOfClass
        tic
        X_class=X(1+NumOfTraining*(i-1):NumOfTraining*i,:);
        [covs,q,priors,means]=mix(X_class,NumOfTesting,NumOfTraining,NumOfClass,dem,K);
        t11=toc
        Q=[Q;q];
        Means=[Means;means];
        Priors=[Priors,priors];
%         Covs_2(:,1+(item-1)*K_2:item*K_2)=covs_2;
        Covs=[Covs,covs]
        parfor j=1:K
%             dis(:,j)=diag((TrainingVector(:,1+NumOfTraining*(i-1):NumOfTraining*i)-means(j,:)'*ones(1,NumOfTraining))'*pinv(covs(:,:,j))*(TrainingVector(:,1+NumOfTraining*(i-1):NumOfTraining*i)-means(j,:)'*ones(1,NumOfTraining)));
            dis(:,j)=exp(-diag((TrainingVector(:,1+NumOfTraining*(i-1):NumOfTraining*i)-means(j,:)'*ones(1,NumOfTraining))'*(TrainingVector(:,1+NumOfTraining*(i-1):NumOfTraining*i)-means(j,:)'*ones(1,NumOfTraining))/(covs(:,j))^2))./(covs(:,j));
%            dis(:,j)=diag((TrainingVector(:,1+NumOfTraining*(i-1):NumOfTraining*i)-means(j,:)'*ones(1,NumOfTraining))'*(TrainingVector(:,1+NumOfTraining*(i-1):NumOfTraining*i)-means(j,:)'*ones(1,NumOfTraining)));
        end 
        [~,class_neural] = sort(dis,2);
        neuralalltrainid=[neuralalltrainid;class_neural(:,end)+K*(i-1)];     
        
    end
 
    Q_2=[];
    Priors_2=[];
    Means_2=[];
    Covs_2=[];
    Priors_select=[];
    kernel_id=[];
    K_2=7;
          t_k=0;        
        for item=1:K*NumOfClass
      tic
      aa=sum(find(neuralalltrainid==item))
      if aa~=0
          t_k=t_k+1;
          Priors_select=[Priors_select,Priors(item)];
          kernel_id=[kernel_id,item]
            X_class_2=[];
            X_class_2=X(find(neuralalltrainid==item),:);
            nn=size(X_class_2,1);
            [covs_2,q_2,priors_2,means_2]=mix_2(X_class_2,NumOfTraining,nn,dem,K_2);
            Q_2=[Q_2;q_2];
            Means_2=[Means_2;means_2];
%             Means_2(1+(item-1)*K_2:item*K_2,:)=means_2;
            Priors_2=[Priors_2,priors_2];
%         Priors_2(:,1+(item-1)*K_2:item*K_2)=priors_2;
%         Covs_2(:,1+(item-1)*K_2:item*K_2)=covs_2;
             Covs_2=[Covs_2,covs_2];   
        t12=toc  
      end
        end

t1=t11+t12
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[~,KerTest]=KerGramMatrix(Means_2',TestingVector,4,NumOfClass,neuralalltrainid,Covs_2);
[~,KerTrain]=KerGramMatrix(Means_2',TrainingVector,4,NumOfClass,neuralalltrainid,Covs_2);
Meannon=[];
   for iiii=1:NumOfClass
    %create MeanVector
        MeanTemp=mean(KerTrain(:,((iiii-1)*NumOfTraining+1):(iiii*NumOfTraining)),2);  
        Meannon=[Meannon,MeanTemp];
   end
PPP=Priors_2;
AA_train=KerTrain.*(PPP'*ones(1,size(KerTrain,2)));
Meannon=Meannon.*(PPP'*ones(1,size(Meannon,2)));
AA_test=KerTest.*(PPP'*ones(1,size(KerTest,2)));
p_sum=[];
for item2=1:t_k
    p_sum= blkdiag(p_sum,ones(1,K_2));
end
AA_train=p_sum*AA_train;
Meannon=p_sum*Meannon;
AA_test=p_sum*AA_test;
size(Priors_select)

AA_train=AA_train.*(Priors_select'*ones(1,size(AA_train,2)));
Meannon=Meannon.*(Priors_select'*ones(1,size(Meannon,2)));
AA_test=AA_test.*(Priors_select'*ones(1,size(AA_test,2)));


% AA_train=AA_train.*(p_sum'*Priors'*ones(1,size(AA_train,2)));
% Meannon=Meannon.*(p_sum'*Priors'*ones(1,size(Meannon,2)));
% AA_test=AA_test.*(p_sum'*Priors'*ones(1,size(AA_test,2)));



[FAR_NMF2,GAR_NMF2,AccuForNMF2,~]=ComputNMF(NumOfClass,NumOfTesting,NumOfTraining,AA_train,AA_test,MaxAccuRank,Test_id,20) ;
[FAR_DLDA2,GAR_DLDA2,AccuracyDLDA2,~,AUC_DLDA2]=ComputeDLDA(Database,NumOfClass,NumOfTesting,NumOfTraining,AA_train,AA_test,MaxAccuRank,Test_id);
[FAR_PCA2,GAR_PCA2,AccuracyPCA2,~,AUC_PCA2]=ComputePCA(NumOfClass,NumOfTesting,NumOfTraining,AA_train,AA_test,MaxAccuRank);
[AccuracyKPCA2,AUC_KPCA2]=RBF_ELM(NumOfClass,NumOfTesting,NumOfTraining,TrainingVector,TestingVector,36,1000);
[Accuracy_BPNN2,~,~,AUCBP2]=ComputeBPNN2(NumOfClass,NumOfTesting,NumOfTraining,AA_train,AA_test,MaxAccuRank,21,NumOfClass);

AA_test_sum=[];
size(AA_test)
   for iiii=1:NumOfClass
    %create MeanVector
    kel_sel_id=find(kernel_id>=(iiii-1)*K+1&kernel_id<=iiii*K)
    MeanTemp=mean(AA_test(kel_sel_id,:),1);  
    
%         MeanTemp=mean(AA_test(((iiii-1)*K+1):(iiii*K),:),1);  
        AA_test_sum=[AA_test_sum;MeanTemp];
   end

 [AA_test_sum,AA_test_sum_id]=sort(AA_test_sum);  
truenum=sum(AA_test_sum_id(end,:)==Test_id);
Accuracy2=100*truenum/NumofAlltest;
%%%%%%%Accuracy2RecallPrecision

RandP_ca=reshape(AA_test_sum_id(end,:)',NumOfTesting,NumOfClass)';
for i_RandP=1:NumOfClass
    RandP_nu(i_RandP,:)=sum(RandP_ca==i_RandP*ones(size(RandP_ca)),2)';
end
Precision=diag(RandP_nu)'./sum(RandP_nu);
Recall=diag(RandP_nu)'./sum(RandP_nu,2)';

PR_id=-ones(NumOfClass,NumOfClass*NumOfTesting); 
for i=1:NumOfClass
    PR_id(i,(i-1)*NumOfTesting+1:i*NumOfTesting)=ones(1,NumOfTesting);
    s=exp(-abs(AA_test_sum_id(end,:)-i));
[TPR,TNR,info] = vl_roc(PR_id(i,:), s);
AUC2(i)=info.auc;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  [KerGram,KerTrainTest]=KerGramMatrix(TrainingVector,TestingVector,1,NumOfClass,neuralalltrainid,Covs);
% 
% K_id_tr_te=[neuralalltrainid,KerTrainTest];
% K_id_tr=[neuralalltrainid,KerGram];
% [~,IDkernel]=sort(neuralalltrainid);
% K_id_tr_te= K_id_tr_te(IDkernel,:);  
% K_id_tr= K_id_tr(IDkernel,:);  
% numofkernel=histc(neuralalltrainid,unique(neuralalltrainid));
%    
% b_tend = unique (neuralalltrainid) ;
% 
% un_kernel_id=setdiff(1:K*NumOfClass,b_tend);
% 
% 
%    m_tend=[];
%    for iii=1:numel(numofkernel)
%    m_tend=blkdiag(m_tend,ones(numofkernel(iii),1)/numofkernel(iii));
%    end
%    
%    AA=m_tend'*K_id_tr(:,2:end);% shenjingyuan  hebing hou de  yangben juzhen
%    AA_test=m_tend'*K_id_tr_te(:,2:end);% shenjing yuan  hebing hou de  teshi yangben juzhen 
%  
% PPP=Priors;
% PPP(un_kernel_id)=[];
% %    AA=AA.*(Priors'*ones(1,size(AA,2)));
% %    AA_test=AA_test.*(Priors'*ones(1,size(AA_test,2)));
%    AA=AA.*(PPP'*ones(1,size(AA,2)));
%    AA_test=AA_test.*(PPP'*ones(1,size(AA_test,2)));
%    
%  [FAR_NMF2,GAR_NMF2,AccuForNMF2,MatrixProjectionW]=ComputNMF(NumOfClass,NumOfTesting,NumOfTraining,AA,AA_test,MaxAccuRank,Test_id,20) ;
%  [FAR_DLDA2,GAR_DLDA2,AccuracyDLDA2,~]=ComputeDLDA(Database,NumOfClass,NumOfTesting,NumOfTraining,AA,AA_test,MaxAccuRank,Test_id);
%  [FAR_PCA2,GAR_PCA2,AccuracyPCA2,~]=ComputePCA(NumOfClass,NumOfTesting,NumOfTraining,AA,AA_test,MaxAccuRank);
%  [FAR_KPCA2,GAR_KPCA2,AccuracyKPCA2]=ComputeKPCA(NumOfClass,NumOfTesting,NumOfTraining,AA,AA_test,MaxAccuRank,Test_id);
% 
%    
%    
%    Meannon=[];
%    for iiii=1:NumOfClass
%     %create MeanVector
%         MeanTemp=mean(AA(:,((iiii-1)*NumOfTraining+1):(iiii*NumOfTraining)),2);  
%         Meannon=[Meannon,MeanTemp];
%    end
% % %    
%    
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%% create DistanceArray
   
 DistanceArray=zeros(NumofAlltest,NumOfClass);
   
 for j=1:NumofAlltest
   dist_temp=(repmat(AA_test(:,j),1,NumOfClass)-Meannon);
   DistanceArray(j,:)=sqrt(sum((dist_temp).^2));
 end
%  DistanceArray=AA_test_sum';
[DisSort,IndClass]=sort(DistanceArray,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CMC
Accuracy=zeros(1,MaxAccuRank);  ss=0;

for k=1:MaxAccuRank
    ID=IndClass(:,k)';
    ss=ss+sum(ID==Test_id);
    Accuracy(k)=100*ss/NumofAlltest;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ROC
% UB=max(max(DistanceArray));
% LB=min(min(DistanceArray));
UB=max(DisSort(:,NumOfClass));
LB=min(DisSort(:,1));

foot=(UB-LB)/97;
start=LB-2*foot;

for i=1:100
    threshold=start+i*foot;
    DistanceJudge=(DistanceArray<=threshold);
    Numgar=0;
    for j=1:NumOfClass
        Numgar=Numgar+sum(DistanceJudge(((j-1)*NumOfTesting+1):(j*NumOfTesting),j));    
        DistanceJudge(((j-1)*NumOfTesting+1):(j*NumOfTesting),j)=0;
    end
    GAR(i)=Numgar/NumofAlltest; %%% Genuine accept
    FAR(i)=sum(sum(DistanceJudge))/(NumofAlltest*(NumOfClass-1));  %%% False accept
end

AUC_2layer=max([mean(AUC2),mean(AUC_DLDA2),mean(AUC_PCA2),mean(AUC_KPCA2)]);




