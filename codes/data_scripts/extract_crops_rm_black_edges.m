clear
warning off
LR_Dir = '540p_train';
HR_Dir = '4K_train';
out_dir = '/home/ssddata/AI_4K/dataset/images';
img_fold = dir(LR_Dir);
img_fold = img_fold(3:end);
lr_patch_size = 96;
hr_patch_size = lr_patch_size*4;

rm_edge = 64;
lr_idx = rm_edge+1:540-rm_edge;
hr_idx = rm_edge*4+1:2160-rm_edge*4;

parfor i = 1:numel(img_fold)
    frames = dir(fullfile(LR_Dir,img_fold(i).name,'*.png'));
    hr_im = imread(fullfile(HR_Dir,img_fold(i).name,'frame_051.png'));
    hr_im_sum = sum(sum(hr_im,3),2);
    for j = 1:10:numel(frames)
        lr_im = imread(fullfile(LR_Dir,img_fold(i).name,frames(j).name));
        hr_im = imread(fullfile(HR_Dir,img_fold(i).name,frames(j).name));
        if sum(hr_im_sum<5000) > 200
            lr_im = lr_im(lr_idx,:,:);
            hr_im = hr_im(hr_idx,:,:);
        end
        [lr_h, lr_w, ~] = size(lr_im);
        [hr_h, hr_w, ~] = size(hr_im);
        
        crop_id = 0;
        for h_id = 1:floor(hr_h/hr_patch_size)
            for w_id = 1:floor(hr_w/hr_patch_size)
                crop_id = crop_id + 1;
                hr_patch = hr_im((h_id-1)*hr_patch_size+1:h_id*hr_patch_size,(w_id-1)*hr_patch_size+1:w_id*hr_patch_size,:);
                lr_patch = lr_im((h_id-1)*lr_patch_size+1:h_id*lr_patch_size,(w_id-1)*lr_patch_size+1:w_id*lr_patch_size,:);
                hr_out_path = fullfile(out_dir,'HR_train',[img_fold(i).name, '_', frames(j).name(1:end-4),'_crop_', num2str(crop_id), '.png']);
                lr_out_path = fullfile(out_dir,'LR_train',[img_fold(i).name, '_', frames(j).name(1:end-4),'_crop_', num2str(crop_id), '.png']);
%                 a = psnr(imresize(lr_patch,4),hr_patch);
                imwrite(lr_patch,lr_out_path);
                imwrite(hr_patch,hr_out_path);
            end
        end
    end
end
