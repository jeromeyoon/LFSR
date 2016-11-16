clear all
close all


trainpath = '/research2/iccv2015/HCI/train';
valpath = '/research2/iccv2015/HCI/test';
traindata = dir(fullfile(trainpath,'*.h5'));
valdata = dir(fullfile(valpath,'*.h5'));
scale =2;
%%%%%%%%%%%%%% Vertical %%%%%%%%%%%%%%%%
%LF_input=[];
for tt=1:length(valdata)
    LF = hdf5read(fullfile(valpath,valdata(tt).name),'/LF');
    LF = permute(LF,[5,4,3,2,1]);
    [S,T,hei,wid,ch] = size(LF);
    LF_input={};
    for v=1:2:T
        for h =1:2:T
            tmp  =rgb2ycbcr(squeeze(LF(v,h,:,:,:)));
            tmp = modcrop(tmp,scale);
            LF_input{end+1} = imresize(imresize(tmp,1/scale,'bicubic'),[size(tmp,1),size(tmp,2)],'bicubic');
        end
    end
    save(sprintf('HCI_val_low_input_%d.mat',tt),'LF_input');
    LF_input={};
end
