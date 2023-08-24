%% Finding optimal threshold and generating ROC 
clc
%close all
%clear all
hold on

base_dir = 'C:\\Users\\Sam\\Desktop\\Collagen_Segmentation\\';
GT_masks_dir = strcat(base_dir,'second_round\\C white excluded\\');
%GT_masks_dir = strcat(base_dir,'Converted_masks\\');
GT_masks = dir(GT_masks_dir);
GT_masks = GT_masks(~ismember({GT_masks.name},{'.','..'}));
test_output_dir = strcat(base_dir,'Results\\');

test_models = {'Brightfield_Grayscale','Fluorescence_Grayscale','MultiChannel_Grayscale',...
    'Brightfield_RGB','Fluorescence_RGB','MultiChannel_RGB'};

opt_threshold = 0.1;
for t=1:length(test_models)
    test_model = test_models{t};

    test_output = strcat(test_output_dir,test_model,'\\Testing_Output\\');
    
    test_imgs = dir(test_output);
    test_imgs = test_imgs(~ismember({test_imgs.name},{'.','..'}));
    test_imgs = test_imgs(~contains({test_imgs.name},{'.csv','.fig'}));
    test_imgs = {test_imgs.name};
    
    auc_storage = zeros(9,length(test_imgs));
    mse_storage = zeros(9,length(test_imgs));
    
    sens_storage = zeros(1,length(test_imgs));
    spec_storage = zeros(1,length(test_imgs));
    
    roc_storage_x = zeros(length(test_imgs),512);
    roc_storage_y = zeros(length(test_imgs),512);
    for t = 1:length(test_imgs)
    
        test_pred = imread(strcat(test_output,test_imgs{t}));
        gt_mask = imread(strcat(GT_masks_dir,strrep(strrep(test_imgs{t},'Test_Example_',''),'.tif','.jpg')));
        %gt_mask = imresize(gt_mask,[256,256]);
    
        %figure
        %hold on
        thresh_store = zeros(9,1);
        mse_thresh_store = zeros(9,1);
        roc_store_x = zeros(9,512);
        roc_store_y = zeros(9,512);
        ct = 1;
        for thresh = 0.1:0.1:0.9
    
            threshed_gt = im2bw(gt_mask,thresh);
            if length(unique(threshed_gt))>1
                [x,y,~,auc] = perfcurve(reshape(threshed_gt,[],1),reshape(double(test_pred)./255,[],1),1);
                display(strcat('AUC for thresh:',num2str(thresh),'_is:',num2str(auc)))
                %plot(x,y,'DisplayName',num2str(thresh))
                thresh_store(ct,1) = auc;
                
                if ismembertol(thresh,opt_threshold,0.0000001)
                    % Threshed prediction for binary segmentation performance
                    threshed_prediction = im2bw(double(test_pred./255),thresh);
                    reshaped_gt = reshape(threshed_gt,1,[]);
                    reshaped_pred = reshape(threshed_prediction,1,[]);
                    one_hot_gt = [reshaped_gt;~reshaped_gt];
                    one_hot_pred = [reshaped_pred;~reshaped_pred];
                    [c,cm,ind,per] = confusion(double(one_hot_gt),double(one_hot_pred));
                    sensitivity = 1-per(1,1)
                    specificity = 1-per(1,2)
    
                    sens_storage(1,t) = sensitivity;
                    spec_storage(1,t) = specificity;
                end
    
                size_diff = 512-size(y,1);
                roc_store_y(ct,:) = [y;ones(size_diff,1)]';
                roc_store_x(ct,:) = [x;ones(size_diff,1)]';
    
                % Calculating MSE and Norm MSE from these thresholds
                threshed_continuous = double(test_pred)./255;
                threshed_continuous(threshed_continuous<thresh) = 0;
    
                square_diff = ((double(gt_mask)./255)-threshed_continuous).^2;
                mean_square_diff = mean(square_diff,'all','omitnan');
                mse_thresh_store(ct,1) = mean_square_diff;
    
            else
                display('Threshed out')
                thresh_store(ct,1) = nan;
                mse_thresh_store(ct,1)=nan;
                roc_store_x(ct,:) = nan;
                roc_store_y(ct,:) = nan;
            end
            ct = ct+1;
        end
        auc_storage(:,t) = thresh_store;
        mse_storage(:,t) = mse_thresh_store;
        roc_storage_x(t,:) = mean(roc_store_x,1,'omitnan');
        roc_storage_y(t,:) = mean(roc_store_y,1,'omitnan');
        %hold off
        %pause(2)
        %close all
    end
            
    roc_average_x = mean(roc_storage_x,1,'omitnan');
    roc_average_y = mean(roc_storage_y,1,'omitnan');
    plot(roc_average_x,roc_average_y,'DisplayName',strrep(test_model,'_',' ')), title('Average ROC Plot')
    legend
    
    writematrix(auc_storage,strcat(test_output,'AUC_Thresh.csv'))
    row_means = mean(auc_storage,2,'omitnan')
    row_stds = std(auc_storage,0,2,'omitnan')
    row_mean_mse = mean(mse_storage,2,'omitnan')
    row_mse_std = std(mse_storage,0,2,'omitnan')
    row_sens_mean = mean(sens_storage,'omitnan')
    row_sens_std = std(sens_storage,1,'omitnan')
    row_spec_mean = mean(spec_storage,'omitnan')
    row_spec_std = std(spec_storage,1,'omitnan')
    average_roc = trapz(roc_average_x,roc_average_y)

end



