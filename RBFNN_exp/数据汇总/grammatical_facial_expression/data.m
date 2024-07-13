a=importdata('E:\Research\grammatical_facial_expression\a_affirmative_datapoints.txt');
a_data=a.data(:,2:end);
a_label=importdata('E:\Research\grammatical_facial_expression\a_affirmative_targets.txt');
X1=[a_data,a_label];
clear a
clear a_data
clear a_label
ID1=find(X1(:,end)==1);
ID11=find(X1(:,end)==0);
data1=X1(ID1,:); 
data11=X1(ID11,:); 



a=importdata('E:\Research\grammatical_facial_expression\a_conditional_datapoints.txt');
a_data=a.data(:,2:end);
a_label=importdata('E:\Research\grammatical_facial_expression\a_conditional_targets.txt');
X2=[a_data,a_label];
clear a
clear a_data
clear a_label
ID2=find(X2(:,end)==1);
data2=X2(ID2,:); 
data2(:,end)=data2(:,end)+1;

a=importdata('E:\Research\grammatical_facial_expression\a_doubt_question_datapoints.txt');
a_data=a.data(:,2:end);
a_label=importdata('E:\Research\grammatical_facial_expression\a_doubts_question_targets.txt');
X3=[a_data,a_label];
clear a
clear a_data
clear a_label
ID3=find(X3(:,end)==1);
data3=X3(ID3,:); 
data3(:,end)=data3(:,end)+2;

a=importdata('E:\Research\grammatical_facial_expression\a_emphasis_datapoints.txt');
a_data=a.data(:,2:end);
a_label=importdata('E:\Research\grammatical_facial_expression\a_emphasis_targets.txt');
X4=[a_data,a_label];
clear a
clear a_data
clear a_label
ID4=find(X4(:,end)==1);
data4=X4(ID4,:); 
data4(:,end)=data4(:,end)+3;

a=importdata('E:\Research\grammatical_facial_expression\a_negative_datapoints.txt');
a_data=a.data(:,2:end);
a_label=importdata('E:\Research\grammatical_facial_expression\a_negative_targets.txt');
X5=[a_data,a_label];
clear a
clear a_data
clear a_label
ID5=find(X5(:,end)==1);
data5=X5(ID5,:); 
data5(:,end)=data5(:,end)+4;

a=importdata('E:\Research\grammatical_facial_expression\a_relative_datapoints.txt');
a_data=a.data(:,2:end);
a_label=importdata('E:\Research\grammatical_facial_expression\a_relative_targets.txt');
X6=[a_data,a_label];
clear a
clear a_data
clear a_label
ID6=find(X6(:,end)==1);
data6=X6(ID6,:); 
data6(:,end)=data6(:,end)+5;


a=importdata('E:\Research\grammatical_facial_expression\a_topics_datapoints.txt');
a_data=a.data(:,2:end);
a_label=importdata('E:\Research\grammatical_facial_expression\a_topics_targets.txt');
X7=[a_data,a_label];
clear a
clear a_data
clear a_label
ID7=find(X7(:,end)==1);
data7=X7(ID7,:); 
data7(:,end)=data7(:,end)+6;



a=importdata('E:\Research\grammatical_facial_expression\a_wh_question_datapoints.txt');
a_data=a.data(:,2:end);
a_label=importdata('E:\Research\grammatical_facial_expression\a_wh_question_targets.txt');
X8=[a_data,a_label];
clear a
clear a_data
clear a_label
ID8=find(X8(:,end)==1);
data8=X8(ID8,:); 
data8(:,end)=data8(:,end)+7;



a=importdata('E:\Research\grammatical_facial_expression\a_yn_question_datapoints.txt');
a_data=a.data(:,2:end);
a_label=importdata('E:\Research\grammatical_facial_expression\a_yn_question_targets.txt');
X9=[a_data,a_label];
clear a
clear a_data
clear a_label
ID9=find(X9(:,end)==1);
data9=X9(ID9,:); 
data9(:,end)=data9(:,end)+8;


a=importdata('E:\Research\grammatical_facial_expression\b_affirmative_datapoints.txt');
a_data=a.data(:,2:end);
a_label=importdata('E:\Research\grammatical_facial_expression\b_affirmative_targets.txt');
X10=[a_data,a_label];
clear a
clear a_data
clear a_label
ID10=find(X10(:,end)==1);
data10=X10(ID10,:); 
data10(:,end)=data10(:,end)+9;


a=importdata('E:\Research\grammatical_facial_expression\b_conditional_datapoints.txt');
a_data=a.data(:,2:end);
a_label=importdata('E:\Research\grammatical_facial_expression\b_conditional_targets.txt');
X11=[a_data,a_label];
clear a
clear a_data
clear a_label
ID11=find(X11(:,end)==1);
data11=X11(ID11,:); 
data11(:,end)=data11(:,end)+10;

a=importdata('E:\Research\grammatical_facial_expression\b_doubt_question_datapoints.txt');
a_data=a.data(:,2:end);
a_label=importdata('E:\Research\grammatical_facial_expression\b_doubt_question_targets.txt');
X12=[a_data,a_label];
clear a
clear a_data
clear a_label
ID12=find(X12(:,end)==1);
data12=X12(ID12,:); 
data12(:,end)=data12(:,end)+11;

a=importdata('E:\Research\grammatical_facial_expression\b_emphasis_datapoints.txt');
a_data=a.data(:,2:end);
a_label=importdata('E:\Research\grammatical_facial_expression\b_emphasis_targets.txt');
X13=[a_data,a_label];
clear a
clear a_data
clear a_label
ID13=find(X13(:,end)==1);
data13=X13(ID13,:); 
data13(:,end)=data13(:,end)+12;


a=importdata('E:\Research\grammatical_facial_expression\b_negative_datapoints.txt');
a_data=a.data(:,2:end);
a_label=importdata('E:\Research\grammatical_facial_expression\b_negative_targets.txt');
X14=[a_data,a_label];
clear a
clear a_data
clear a_label
ID14=find(X14(:,end)==1);
data14=X14(ID14,:); 
data14(:,end)=data14(:,end)+13;

a=importdata('E:\Research\grammatical_facial_expression\b_relative_datapoints.txt');
a_data=a.data(:,2:end);
a_label=importdata('E:\Research\grammatical_facial_expression\b_relative_targets.txt');
X15=[a_data,a_label];
clear a
clear a_data
clear a_label
ID15=find(X15(:,end)==1);
data15=X15(ID15,:); 
data15(:,end)=data15(:,end)+14;


a=importdata('E:\Research\grammatical_facial_expression\b_topics_datapoints.txt');
a_data=a.data(:,2:end);
a_label=importdata('E:\Research\grammatical_facial_expression\b_topics_targets.txt');
X16=[a_data,a_label];
clear a
clear a_data
clear a_label
ID16=find(X16(:,end)==1);
data16=X16(ID16,:); 
data16(:,end)=data16(:,end)+15;

a=importdata('E:\Research\grammatical_facial_expression\b_wh_question_datapoints.txt');
a_data=a.data(:,2:end);
a_label=importdata('E:\Research\grammatical_facial_expression\b_wh_question_targets.txt');
X17=[a_data,a_label];
clear a
clear a_data
clear a_label
ID17=find(X17(:,end)==1);
data17=X17(ID17,:); 
data17(:,end)=data17(:,end)+16;


a=importdata('E:\Research\grammatical_facial_expression\b_yn_question_datapoints.txt');
a_data=a.data(:,2:end);
a_label=importdata('E:\Research\grammatical_facial_expression\b_yn_question_targets.txt');
X18=[a_data,a_label];
clear a
clear a_data
clear a_label
ID18=find(X18(:,end)==1);
data18=X18(ID18,:); 
data18(:,end)=data18(:,end)+17;


%将标签排序后的样本进行重新的排序
data=[data1;data2;data3;data4;data5;data6;data7;data8;data9;data10;data11;data12;data13;data14;data15;data16;data17;data18];


save('E:\Research\GFE.mat','data'); 








