%% Generating visualizations for SPIE abstract
clc
%close all

base_dir = 'C:\\Users\\Sam\\Desktop\\Collagen_Segmentation\\';
%b_img_dir = strcat(base_dir,'B\\');
b_img_dir = strcat(base_dir,'second_round\\B_2nd\\');
%gt_img_dir = strcat(base_dir,'Converted_masks\\');
%f_img_dir = strcat(base_dir,'F Normalized\\');
gt_img_dir = strcat(base_dir,'second_round\\C white excluded\\');
f_img_dir = strcat(base_dir,'second_round\\B_2nd\\');

% test_images = {'H 17938 X 13 Y 17 Cx 640Cy 1040','H 17938 X 13 Y 17 Cx 2744Cy 1856',...
%     'H 17938 X 13 Y 17 Cx 2608Cy 60','H 17998 X 9 Y 15 Cx 2204Cy 472',...
%     'H 17998 X 9 Y 15 Cx 2564Cy 2124'};
test_images = {'11519 X 17 Y 24 Cx 2844Cy 632','11518 X 10 Y 15 Cx 1280Cy 1560'...
    '11507 X 6 Y 19 Cx 784Cy 68','11519 X 17 Y 24 Cx 104Cy 1876'}

%test_models = {'Collagen_NF_RGB_FullSize_UN_CustPlus_CV_Segmentation_Output',...
%    'Collagen_Bright_UN_CustPlus_CV_Segmentation_Output',...
%    'Collagen_UN_MSE_0.0005_GrayScale_CV_Segmentation_Output',...
%    'Collagen_NF_CV_Bin_Segmentation_Output'};

%test_models = {'Collagen_Brightfield_Bin_CV_Segmentation_Output'};

test_models = {'Same_Training_Set\\B_Segmentation_Output','Same_Training_Set\\NF_0_Segmentation_Output'}


n_models = length(test_models);

for e =1:length(test_images)
    
    ex_img = test_images{e};
    b_img = imread(strcat(b_img_dir,ex_img,'.jpg'));
    %figure, imshow(b_img), axis image, title(test_images{e})
    f_img = imread(strcat(f_img_dir,ex_img,'.jpg'));
    %figure, imshow(f_img), axis image, title(test_images{e})
    gt_img = imread(strcat(gt_img_dir,ex_img,'.jpg'));
    %figure, imagesc(gt_img), axis image, title(test_images{e})

    figure, imshow(b_img), axis image, title(test_images{e})
    hold on
    h = imagesc(gt_img);
    h.AlphaData = 0.6;
    hold off
    
    figure
    for d = 1:n_models

        pred_img_dir = strcat(base_dir,'multiple_tries\\',test_models{d},'\\Testing_Output\\');
        thresh = 0.1;
        
        threshed_pred = imread(strcat(pred_img_dir,'Test_Example_',ex_img,'.tif'));
        threshed_pred(threshed_pred<255*thresh) = 0;
        threshed_pred = medfilt2(threshed_pred,[5,5]);
        if size(threshed_pred,1) ~= 512
            threshed_pred = imresize(threshed_pred,[512,512],'nearest');
        end
        %figure, imagesc(threshed_pred), axis image, title(num2str(d))
        
        %figure, imshow(b_img), axis image, title(num2str(d))
        subplot(1,n_models,d), imshow(b_img), axis image, title(strrep(test_models{d},'_',' '))
        
        hold on
        h = imagesc(threshed_pred);
        h.AlphaData = 0.6;
        hold off
        
        
    end
end






