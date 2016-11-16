function [low_Input,SR_gt,Ang_gt] = angular_extract_patch(LFGT,option ) 

[t,s,y,x,ch] = size(LFGT);
scale=2;

low_Input={};
SR_gt={};
Ang_gt={};
if strcmp('vertical',option)
    
    for m =1:s-2
        for n=1:t
            high_res=[];
            low_res =[];
            out=[];
                    
            tmp  =rgb2ycbcr(uint8(squeeze(LFGT(m,n,:,:,:))*255));
            tmp = modcrop(tmp(:,:,1),scale);
            high_res = cat(3,high_res,tmp);
            tmp = imresize(imresize(tmp,1/scale,'bicubic'),[size(tmp,1),size(tmp,2)],'bicubic');
            low_res = cat(3,low_res,tmp);
            
            
            tmp  =rgb2ycbcr(uint8(squeeze(LFGT(m+2,n,:,:,:))*255));
            tmp = modcrop(tmp(:,:,1),scale);
            high_res = cat(3,high_res,tmp);
            tmp = imresize(imresize(tmp,1/scale,'bicubic'),[size(tmp,1),size(tmp,2)],'bicubic');
            low_res = cat(3,low_res,tmp);
            
            low_Input{end+1} = low_res;
            SR_gt{end+1} = high_res;
            
            tmp = rgb2ycbcr(uint8(squeeze(LFGT(m+1,n,:,:,:))*255));
            Ang_gt{end+1} = modcrop(tmp(:,:,1),scale);           
            
        end
    end
    
elseif strcmp('horizontal',option)
    for m =1:s
        for n=1:t-2
            high_res=[];
            low_res =[];
            out=[];
            
            tmp  =rgb2ycbcr(uint8(squeeze(LFGT(m,n,:,:,:))*255));
            tmp = modcrop(tmp(:,:,1),scale);
            high_res = cat(3,high_res,tmp);
            tmp = imresize(imresize(tmp,1/scale,'bicubic'),[size(tmp,1),size(tmp,2)],'bicubic');
            low_res = cat(3,low_res,tmp);
            
            tmp  =rgb2ycbcr(uint8(squeeze(LFGT(m,n+2,:,:,:))*255));
            tmp = modcrop(tmp(:,:,1),scale);
            high_res = cat(3,high_res,tmp);
            tmp = imresize(imresize(tmp,1/scale,'bicubic'),[size(tmp,1),size(tmp,2)],'bicubic');
            low_res = cat(3,low_res,tmp);
            low_Input{end+1} = low_res;
            SR_gt{end+1} = high_res;
            
            tmp = rgb2ycbcr(uint8(squeeze(LFGT(m,n+1,:,:,:))*255));
            Ang_gt{end+1} = modcrop(tmp(:,:,1),scale);

        end
    end
    
elseif strcmp('views',option)
    for m =1:s-2
        for n=1:t-2
            
            high_res=[];
            low_res =[];
            out=[];
            
            tmp  =rgb2ycbcr(uint8(squeeze(LFGT(m,n,:,:,:))*255));
            tmp = modcrop(tmp(:,:,1),scale);
            high_res = cat(3,high_res,tmp);
            tmp = imresize(imresize(tmp,1/scale,'bicubic'),[size(tmp,1),size(tmp,2)],'bicubic');
            low_res = cat(3,low_res,tmp);
            
            tmp  =rgb2ycbcr(uint8(squeeze(LFGT(m,n+2,:,:,:))*255));
            tmp = modcrop(tmp(:,:,1),scale);
            high_res = cat(3,high_res,tmp);
            tmp = imresize(imresize(tmp,1/scale,'bicubic'),[size(tmp,1),size(tmp,2)],'bicubic');
            low_res = cat(3,low_res,tmp);
            
            tmp  =rgb2ycbcr(uint8(squeeze(LFGT(m+2,n,:,:,:))*255));
            tmp = modcrop(tmp(:,:,1),scale);
            high_res = cat(3,high_res,tmp);
            tmp = imresize(imresize(tmp,1/scale,'bicubic'),[size(tmp,1),size(tmp,2)],'bicubic');
            low_res = cat(3,low_res,tmp);
            
            
            tmp  =rgb2ycbcr(uint8(squeeze(LFGT(m+2,n+2,:,:,:))*255));
            tmp = modcrop(tmp(:,:,1),scale);
            high_res = cat(3,high_res,tmp);
            tmp = imresize(imresize(tmp,1/scale,'bicubic'),[size(tmp,1),size(tmp,2)],'bicubic');
            low_res = cat(3,low_res,tmp);
            
            low_Input{end+1} = low_res;
            SR_gt{end+1} = high_res;
             
            tmp = rgb2ycbcr(uint8(squeeze(LFGT(m+1,n+1,:,:,:))*255));
            Ang_gt{end+1} = modcrop(tmp(:,:,1),scale);
            
            
        end
    end
end

clear LFGT
