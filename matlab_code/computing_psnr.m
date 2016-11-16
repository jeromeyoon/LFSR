% computing PSNR and ssim 

clear all
close all
border =10;
outputpath ='/research2/SPL/HCI/angSR/allviews3/buddha';
ver_outputfiles = dir(fullfile(outputpath,'ang_ver_*'));
hor_outputfiles = dir(fullfile(outputpath,'ang_hor_*'));
views_outputfiles = dir(fullfile(outputpath,'ang_views_*'));
valpath  ='/research2/iccv2015/HCI/test';
valdata = dir(fullfile(valpath,'*.h5'));
LF = hdf5read(fullfile(valpath,valdata(1).name),'/LF');
LF = permute(LF,[5,4,3,2,1]);
sum_psnr =0.0;
sum_ssim =0.0;
min_psnr =100;
max_psnr =-100;
min_ssim =100;
max_ssim =-100;
numfiles =56;
%%%%%%%% vertical PSNR and SSIM %%%%%%%

vv=[2,4,6,8];
hh=[1,3,5,7,9];
count =1;
for v=1:4
    for h=1:5
        fprintf('vertical %d/%d \n',count,20);

        gt= squeeze(LF(vv(v),hh(h),:,:,:));
        output = load(fullfile(outputpath,ver_outputfiles(count).name));
        output =output.Predict;
        output =ycbcr2rgb(uint8(output*255));
        output = output(border:end-border,border:end-border,:);
        gt= gt(border:end-border,border:end-border,:);
        val_psnr =psnr(output,gt);
        val_ssim =ssim(output,gt);
        [min_psnr,max_psnr]=findminmax(val_psnr,min_psnr,max_psnr);
        sum_psnr = sum_psnr+val_psnr;
        [min_ssim,max_ssim]=findminmax(val_ssim,min_ssim,max_ssim);
        sum_ssim = sum_ssim+val_ssim;
        count =count+1;
        fprintf('psnr:%.4f \n',val_psnr);
    end
end


%%%%%%%% horizontal PSNR and SSIM %%%%%%%

hh=[2,4,6,8];
vv=[1,3,5,7,9];
count = 1;


for v=1:5
    for h=1:4
        fprintf('horizontal %d/%d \n',count,20);

        gt= squeeze(LF(vv(v),hh(h),:,:,:));
        output = load(fullfile(outputpath,hor_outputfiles(count).name));
        output =output.Predict;
        output =ycbcr2rgb(uint8(output*255));
        output = output(border:end-border,border:end-border,:);
        gt= gt(border:end-border,border:end-border,:);
        val_psnr =psnr(output,gt);
        val_ssim =ssim(output,gt);
        [min_psnr,max_psnr]=findminmax(val_psnr,min_psnr,max_psnr);
        sum_psnr = sum_psnr+val_psnr;
        [min_ssim,max_ssim]=findminmax(val_ssim,min_ssim,max_ssim);
        sum_ssim = sum_ssim+val_ssim;
        count = count+1;
        fprintf('psnr:%.4f \n',val_psnr);
    end
end



%%%%%%%% views PSNR and SSIM %%%%%%%

hh=[2,4,6,8];
vv=[2,4,6,8];
count = 1;


for v=1:4
    for h=1:4
        fprintf('4views %d/%d \n',count,16);
        gt= squeeze(LF(vv(v),hh(h),:,:,:));
        output = load(fullfile(outputpath,views_outputfiles(count).name));
        output =output.Predict;
        output =ycbcr2rgb(uint8(output*255));
        output = output(border:end-border,border:end-border,:);
        gt= gt(border:end-border,border:end-border,:);
        val_psnr =psnr(output,gt);
        val_ssim =ssim(output,gt);
        [min_psnr,max_psnr]=findminmax(val_psnr,min_psnr,max_psnr);
        sum_psnr = sum_psnr+val_psnr;
        [min_ssim,max_ssim]=findminmax(val_ssim,min_ssim,max_ssim);
        sum_ssim = sum_ssim+val_ssim;
        count = count +1;
        fprintf('psnr:%.4f \n',val_psnr);
    end
end

mean_psnr = sum_psnr/numfiles;
mean_ssim = sum_ssim/numfiles;
fprintf('min_psr: %.6f max_psnr: %.6f mean_psnr: %.6f min_ssim: %.6f max_ssim:%.6f mean_ssim: %.6f \n', ...
        min_psnr,max_psnr,mean_psnr,min_ssim,max_ssim,mean_ssim);
