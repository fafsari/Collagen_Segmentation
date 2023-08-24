%% Qualitative image label performance analysis
% training & testing set composition comparison
clc
clear all
% Defining paths to necessary csv files
% Qualitative image labels:
qual_labels = 'C:\\Users\\Sam\\Desktop\\Collagen_Segmentation\\second_round\\B_2nd\\Image_Labels.csv';
% Training and testing set images
train_set = 'C:\\Users\\Sam\\Desktop\\Collagen_Segmentation\\Last_Fold_Training.csv';
test_set = 'C:\\Users\\Sam\\Desktop\\Collagen_Segmentation\\Last_Fold_Testing.csv';

qual_label_table = readtable(qual_labels,'Delimiter',',','ReadVariableNames',true);
train_set_table = readtable(train_set,'Delimiter',',','ReadVariableNames',true);
test_set_table = readtable(test_set,'Delimiter',',','ReadVariableNames',true);

% Getting the names from the train and test set tables (currently image
% paths in HiPerGator)
train_paths = train_set_table.Training_Image_Paths;
test_paths = test_set_table.B_Testing_Image_Paths;

train_names = cellfun(@(x) strsplit(x,'/'),train_paths,'UniformOutput',false);
test_names = cellfun(@(x) strsplit(x,'/'), test_paths, 'UniformOutput',false);

train_names = cellfun(@(x) x{end},train_names, 'UniformOutput',false);
test_names = cellfun(@(x) x{end}, test_names, 'UniformOutput',false);

% Getting the labels for images in the training and testing sets
train_labels = qual_label_table(find(ismember(qual_label_table.Image_Names,train_names)),:);
test_labels = qual_label_table(find(ismember(qual_label_table.Image_Names,test_names)),:);

% Creating comparison pie charts
t = tiledlayout(1,2,'TileSpacing','compact');

ax1 = nexttile;
pie(ax1,categorical(train_labels.Labels))
title('Training Set Composition')
ax2 = nexttile;
pie(ax2,categorical(test_labels.Labels))
title('Testing Set Composition')
lgd = legend(unique(qual_label_table.Labels));
lgd.Layout.Tile = 'east';

% Now looking at relative performance of different models for differently
% labeled images
model_base_dir = 'C:\\Users\\Sam\\Desktop\\Collagen_Segmentation\\Results\\';
model_list = {'Fluorescence_RGB\\Testing_Output\\',...
    'Fluorescence_Grayscale\\Testing_Output\\',...
    'Brightfield_RGB\\Testing_Output\\',...
    'Brightfield_Grayscale\\Testing_Output\\',...
    'MultiChannel_RGB\\Testing_Output\\',...
    'MultiChannel_Grayscale\\Testing_Output\\'};

model_names = {'Fluorescent RGB','Fluorescent Grayscale','Brightfield RGB',...
    'Brightfield Grayscale','Concatenated RGB','Concatenated Grayscale'};


for m = 1:length(model_list)
    
    current_model = model_list{m};
    current_model_name = model_names{m}
    test_results_path = strcat(model_base_dir,current_model);
    
    test_imgs = dir(test_results_path);
    test_imgs = test_imgs(~cell2mat({test_imgs.isdir}));
    test_imgs = test_imgs(~ismember({test_imgs.name},{'.','..'}));
    test_imgs = test_imgs(~contains({test_imgs.name},{'.csv','.fig','.xlsx'}));
    test_imgs = {test_imgs.name};

    test_imgs = cellfun(@(x) strrep(x,'Test_Example_',''),test_imgs,'UniformOutput',false);
    test_imgs = cellfun(@(x) strrep(x,'.tif','.jpg'),test_imgs,'UniformOutput',false);

    % AUC results
    auc_results = readtable(strcat(test_results_path,'AUC_Thresh.csv'),'Delimiter',',');
    auc_results = auc_results{1,:}';
    auc_struct.AUC = auc_results;
    auc_struct.Image_Names = test_imgs';

    auc_table = struct2table(auc_struct);

    % Binary metrics results (Dice, Accuracy, Recall, Precision,
    % Specifiicity, ImgLabel,Phase)
    bin_results = readtable(strcat(test_results_path,'Binary_Metrics.csv'),'Delimiter',',');
    bin_results.Properties.VariableNames = {'Dice','Accuracy','Recall','Precision','Specificity','Image_Names','Phase'};

    % Merging results tables with the qualitative labels table
    combined_results_table = innerjoin(bin_results,auc_table);
    merged_results_labels_table = innerjoin(combined_results_table,qual_label_table);

    writetable(merged_results_labels_table,strcat(test_results_path,'Merged_Results_Table.csv'))
    
    training_results_table = merged_results_labels_table(strcmp(merged_results_labels_table.Phase,'Training'),:);
    training_results_table.Phase = [];
    training_results_table.Image_Names = [];

    testing_results_table = merged_results_labels_table(strcmp(merged_results_labels_table.Phase,'Test'),:);
    testing_results_table.Phase = [];
    testing_results_table.Image_Names = [];

    all_results_table = merged_results_labels_table;
    all_results_table.Image_Names = [];

    metric_names = {'Dice','Accuracy','Recall','Precision','Specificity','AUC'};

    mkdir(strcat(test_results_path,'Qualitative_Metrics_Comparisons'))

    % Making side-by-side bar plots
    for n = 1:length(metric_names)
        current_metric = metric_names{n}
        
        mean(testing_results_table.(current_metric))
        std(testing_results_table.(current_metric))

        figure
        t = tiledlayout(1,3,"TileSpacing","compact");
    
        ax1 = nexttile;
        boxplot(training_results_table.(current_metric),training_results_table.Labels);
        title('Training Set Performance')
    
        ax2 = nexttile;
        test_fig = boxplot(testing_results_table.(current_metric),testing_results_table.Labels);
        title('Testing Set Performance')
        
        ax3 = nexttile;
        boxplot(all_results_table.(current_metric),all_results_table.Labels)
        title('Combined Performance')

        savefig(strcat(test_results_path,'Qualitative_Metrics_Comparisons',filesep,current_metric,'.fig'))
        delete(t)
        close all

    end
    
    if contains(current_model,'RGB')
        rgb_test_figs.(strrep(current_model_name,' ','_')) = testing_results_table;
    else
        g_test_figs.(strrep(current_model_name,' ','_')) = testing_results_table;
    end

        
end

% Going through all the test figures for each metric
rgb_models = fieldnames(rgb_test_figs);
g_models = fieldnames(g_test_figs);
for m = 1:length(metric_names)

    current_metric = metric_names{m};
    figure
    t = tiledlayout(2,length(model_list)/2);

    % Iterating through models
    for r =1:length(rgb_models)
        rgb_model = rgb_models{r};
        ax = nexttile;
        boxplot(rgb_test_figs.(rgb_model).(current_metric),rgb_test_figs.(rgb_model).Labels)
        title(strrep(rgb_model,'_',' '))
        ylim([0.5,1])
    end

    for g = 1:length(g_models)
        g_model = g_models{g};
        ax = nexttile;
        boxplot(g_test_figs.(g_model).(current_metric),g_test_figs.(g_model).Labels)
        title(strrep(g_model,'_',' '))
        ylim([0.5,1])
    end

    title(t,current_metric)

end




