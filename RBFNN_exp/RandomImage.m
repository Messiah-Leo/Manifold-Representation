function [TrainingVector,TestingVector,ID_Train,ID_Test]=RandomImage(InitialVector,NumPerClass,NumOfTraining,NumOfClass,NormType,Database);
% InitialVector is the OriginalImagevector matrix
[dim,NumofAllImage]=size(InitialVector);

ID=1:NumofAllImage;
ID_Train=[];
ID_Test=[];

for i=1:NumOfClass
    flag=randperm(NumPerClass); 
    ID(((i-1)*NumPerClass+1):(i*NumPerClass))=flag+(i-1)*NumPerClass;

    ID_Train=[ID_Train,ID(((i-1)*NumPerClass+1):((i-1)*NumPerClass+NumOfTraining))];
    ID_Test=[ID_Test,ID(((i-1)*NumPerClass+NumOfTraining+1):(i*NumPerClass))];
end 

% ID_Train
% ID_Test
% pause

TrainingVector=InitialVector(:,ID_Train);
TestingVector=InitialVector(:,ID_Test);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  data normalizations        
% NormType=0 means the data is not normalized

if NormType==1   %  mean-zero vector normalization      

    TrV=mean(TrainingVector); TeV=mean(TestingVector);
%        MTr=ones(size(TrainingVector,1),1)*TrV;  MTe=ones(size(TestingVector,1),1)*TeV;
       MTr=repmat(TrV,size(TrainingVector,1),1);    MTe=repmat(TeV,size(TrainingVector,1),1);
       
       TrainingVector=TrainingVector-MTr;
       TestingVector=TestingVector-MTe;

elseif NormType==2   % std normalization
    
        stdTrV=std(TrainingVector); stdTeV=std(TestingVector);
%         stdTr=ones(size(TrainingVector,1),1)*stdTrV;  stdTe=ones(size(TestingVector,1),1)*stdTeV;
        stdMTr=repmat(stdTrV,size(TrainingVector,1),1);    stdMTe=repmat(stdTeV,size(TrainingVector,1),1);
        %%%%%%%
       TrainingVector=TrainingVector./stdMTr;
       TestingVector=TestingVector./stdMTe; 

elseif NormType==3   %%%%%%%%%  R=max-min       
       RTrV=max(TrainingVector)-min(TrainingVector);  RTeV=max(TestingVector)-min(TestingVector);
%         RTr=ones(size(TrainingVector,1),1)*RTrV;  RTe=ones(size(TestingVector,1),1)*RTeV; 
        RTr=repmat(RTrV,size(TrainingVector,1),1);    RTe=repmat(RTeV,size(TrainingVector,1),1);
        %%%%%%%%
       TrainingVector=TrainingVector./RTr;
       TestingVector=TestingVector./RTe; 
       
       
elseif NormType==4    %L2 norm normalization
     N2TrV=sqrt(sum(TrainingVector.*TrainingVector)); N2TeV=sqrt(sum(TestingVector.*TestingVector));
     
%      N2Tr=ones(size(TrainingVector,1),1)*N2TrV;  N2Te=ones(size(TestingVector,1),1)*N2TeV; 
      N2Tr=repmat(N2TrV,size(TrainingVector,1),1);    N2Te=repmat(N2TeV,size(TrainingVector,1),1);
      %%%%%%%%
      TrainingVector=TrainingVector./N2Tr;
      TestingVector=TestingVector./N2Te;
end      