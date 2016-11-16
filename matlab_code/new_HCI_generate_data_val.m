clear all
close all


trainpath = '/research2/iccv2015/HCI/train';
valpath = '/research2/iccv2015/HCI/test';
traindata = dir(fullfile(trainpath,'*.h5'));
valdata = dir(fullfile(valpath,'*.h5'));

%%%%%%%%%%%%%% Vertical %%%%%%%%%%%%%%%%
LF_input=[];
LF_label=[];
for tt=1:length(valdata)
    LF = hdf5read(fullfile(valpath,valdata(tt).name),'/LF');
    LF = permute(LF,[5,4,3,2,1]);
    [S,T,hei,wid,ch] = size(LF);
    [LF_in,LF_out] = angular_extract_patch(LF,'vertical'); 
    LF_input = [LF_input LF_in];
    LF_label = [LF_label LF_out];
    clear LF_in LF_out LF

end
save('HCI_val_vertical_input.mat','LF_input');
save('HCI_val_vertical_gt.mat','LF_label');  

clear LF_input LF_label
%%%%%%%%%%%%%% Horizontal %%%%%%%%%%%%%%%%
LF_input=[];
LF_label=[];
for tt=1:length(valdata)
    
    LF = hdf5read(fullfile(valpath,valdata(tt).name),'/LF');
    LF = permute(LF,[5,4,3,2,1]);
    [S,T,hei,wid,ch] = size(LF);
    [LF_in,LF_out] = angular_extract_patch(LF,'horizontal'); 
    LF_input = [LF_input LF_in];
    LF_label = [LF_label LF_out];
    clear LF_in LF_out LF

end
save('HCI_val_horizontal_input.mat','LF_input');
save('HCI_val_horizontal_gt.mat','LF_label');   
clear LF_input LF_label

%%%%%%%%%%%%% 4 views %%%%%%%%%%%%%%%%%%%%

LF_input=[];
LF_label=[];
for tt=1:length(valdata)
    LF = hdf5read(fullfile(valpath,valdata(tt).name),'/LF');
    LF = permute(LF,[5,4,3,2,1]);
    [S,T,hei,wid,ch] = size(LF);
    [LF_in,LF_out] = angular_extract_patch(LF,'views'); 
    LF_input = [LF_input LF_in];
    LF_label = [LF_label LF_out];
    clear LF_in LF_out LF

end
save('HCI_val_4views_input.mat','LF_input');
save('HCI_val_4views_gt.mat','LF_label'); 
clear LF_input LF_label

