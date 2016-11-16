clear all
close all


trainpath = '/research2/iccv2015/HCI/train';
valpath = '/research2/iccv2015/HCI/test';
traindata = dir(fullfile(trainpath,'*.h5'));
valdata = dir(fullfile(valpath,'*.h5'));

%%%%%%%%%%%%%% Vertical %%%%%%%%%%%%%%%%
LF_input=[];
LF_label=[];
for tt=1:length(traindata)
    LF = hdf5read(fullfile(trainpath,traindata(tt).name),'/LF');
    LF = permute(LF,[5,4,3,2,1]);
    [S,T,hei,wid,ch] = size(LF);
    [LF_in,LF_out] = angular_extract_patch(LF,'vertical'); 
    LF_input = [LF_input LF_in];
    LF_label = [LF_label LF_out];
    clear LF_in LF_out LF

end
save('HCI_train_vertical_input.mat','LF_input');
save('HCI_train_vertical_gt.mat','LF_label');  

clear LF_input LF_label
%%%%%%%%%%%%%% Horizontal %%%%%%%%%%%%%%%%
LF_input=[];
LF_label=[];
for tt=1:length(traindata)
    
    LF = hdf5read(fullfile(trainpath,traindata(tt).name),'/LF');
    LF = permute(LF,[5,4,3,2,1]);
    [S,T,hei,wid,ch] = size(LF);
    [LF_in,LF_out] = angular_extract_patch(LF,'horizontal'); 
    LF_input = [LF_input LF_in];
    LF_label = [LF_label LF_out];
    clear LF_in LF_out LF

end
save('HCI_train_horizontal_input.mat','LF_input');
save('HCI_train_horizontal_gt.mat','LF_label');   
clear LF_input LF_label

%%%%%%%%%%%%% 4 views %%%%%%%%%%%%%%%%%%%%

LF_input=[];
LF_label=[];
count =1;
for tt=1:length(traindata)
    LF = hdf5read(fullfile(trainpath,traindata(tt).name),'/LF');
    LF = permute(LF,[5,4,3,2,1]);
    [S,T,hei,wid,ch] = size(LF);
    [LF_in,LF_out] = angular_extract_patch(LF,'views'); 
    LF_input = [LF_input LF_in];
    LF_label = [LF_label LF_out];
    clear LF_in LF_out LF
    if tt ==6
        save(sprintf('HCI_train_4views_input_%d.mat',count),'LF_input');
        save(sprintf('HCI_train_4views_gt_%d.mat',count),'LF_label'); 
        count = count +1;
        LF_input=[];
        LF_label=[];
    elseif tt==length(traindata)
        save(sprintf('HCI_train_4views_input_%d.mat',count),'LF_input');
        save(sprintf('HCI_train_4views_gt_%d.mat',count),'LF_label'); 
        count = count +1;
        LF_input=[];
        LF_label=[];
    end
end

clear LF_input LF_label
