%% Binarizing and calculating metrics for comparison between non-binary and binary models
clc
clear all
close all


base_dir = 'C:\\Users\\Sam\\Desktop\\Collagen_Segmentation\\';
gt_dir = 'second_round\\C white excluded\\';
nb_model = 'Results\\MultiChannel_Grayscale\\Testing_Output\\';

% Reading training and testing phase
testing_names = readtable(strcat(base_dir,'Last_Fold_Testing.csv'),'Delimiter',',');
test_files = testing_names.F_Testing_Image_Paths;
test_files = cellfun(@(x) strsplit(x,'/'),test_files,'UniformOutput',false);
test_files = cellfun(@(x) x{end},test_files,'UniformOutput',false);

images_dir = strcat(base_dir,nb_model,'\\');
image_list = dir(images_dir);
image_list = image_list(~ismember({image_list.name},{'.','..'}));
image_list = image_list(~contains({image_list.name},{'.csv'}));

image_list = {image_list.name};
%length(image_list)

% Checking contents of gt_dir
gt_list = dir(strcat(base_dir,gt_dir));
gt_list = gt_list(~ismember({gt_list.name},{'.','..'}));
gt_list = {gt_list.name};
%length(gt_list)

Accuracy = zeros(1,length(image_list));
Dice = zeros(1,length(image_list));
Recall = zeros(1,length(image_list));
Precision = zeros(1,length(image_list));
Specificity = zeros(1,length(image_list));
ImgLabel = cell(1,length(image_list));
Phase = cell(1,length(image_list));

thresh = 0.1;
for i = 1:length(image_list)

    img_name = image_list{i};
    gt_name = strrep(strrep(img_name,'Test_Example_',''),'tif','jpg');
    if ismember(gt_name,gt_list)
        img = imread(strcat(images_dir,img_name));
        gt = imread(strcat(base_dir,gt_dir,gt_name));
        
        % Binarizing ground truth
        bin_gt = imbinarize(rgb2gray(gt));
        % Binarizing prediction by set threshold
        bin_img = imbinarize(img,thresh);

        % Generating confusion matrix
        confusion_matrix = confusionmat(reshape(bin_gt,[],1),reshape(bin_img,[],1),"Order",[0,1]);
        tp = confusion_matrix(2,2);
        tn = confusion_matrix(1,1);
        fn = confusion_matrix(2,1);
        fp = confusion_matrix(1,2);


        acc = (tp+tn)/(sum(confusion_matrix,'all'));
        rec = (tp)/(tp+fn);
        f1 = (2*tp)/(2*tp+fp+fn);
        spec = (tn)/(tn+fp);
        prec = (tp)/(tp+fp);

        Accuracy(i) = acc;
        Recall(i) = rec;
        Dice(i) = f1;
        Specificity(i) = spec;
        Precision(i) = prec;
        ImgLabel{i} = gt_name;

        if ismember(gt_name,test_files)
            Phase{i} = 'Test';
        else
            Phase{i} = 'Training';
        end


    else
        display(strcat('No GT for',img_name,gt_name))
    end

end

metrics_table = table(Dice',Accuracy',Recall',Precision',Specificity',ImgLabel',Phase');
metrics_table.Properties.VariableNames = {'Dice','Accuracy','Recall','Precision','Specificity','ImgLabel','Phase'};
writetable(metrics_table,strcat(images_dir,'Binary_Metrics.csv'),'Delimiter',',')



