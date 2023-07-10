"""

Script for organizing input parameters into training and testing parameters


"""

import os

def organize_parameters(parameters,nept_run):

    input_parameters = parameters['input_parameters']

    phase = input_parameters['phase']
    if 'image_dir' in input_parameters:
        image_dir = input_parameters['image_dir']+'/'
    elif 'f_image_dir' in input_parameters:
        f_image_dir = input_parameters['f_image_dir']+'/'
        b_image_dir = input_parameters['b_image_dir']+'/'

    print(f'Phase is: {phase}')

    if phase == 'train':
        train_parameters = parameters['training_parameters']
        test_parameters = parameters['testing_parameters']

        train_parameters['output_dir'] = input_parameters['output_dir']
        test_parameters['output_dir'] = input_parameters['output_dir']

        if 'color_transform' in input_parameters:
            train_parameters['color_transform'] = input_parameters['color_transform']
            test_parameters['color_transform'] = input_parameters['color_transform']

        else:
            test_parameters['color_transform'] = []
            train_parameters['color_transform'] = []

        if 'image_dir' in input_parameters:
            train_parameters['image_dir'] = input_parameters['image_dir']+'/'
            test_parameters['image_dir'] = input_parameters['image_dir']+'/'
        else:
            train_parameters['f_image_dir'] = input_parameters['f_image_dir']+'/'
            train_parameters['b_image_dir'] = input_parameters['b_image_dir']+'/'
            test_parameters['f_image_dir'] = input_parameters['f_image_dir']+'/'
            test_parameters['b_image_dir'] = input_parameters['b_image_dir']+'/'

        train_parameters['label_dir'] = input_parameters['label_dir']+'/'
        test_parameters['label_dir'] = input_parameters['label_dir']+'/'

        train_parameters['k_folds'] = input_parameters['k_folds']

        train_parameters['ann_classes'] = train_parameters['ann_classes']

    elif phase == 'test':
        train_parameters = {}
        test_parameters = parameters['testing_parameters']
        test_parameters['output_dir'] = input_parameters['output_dir']
        test_parameters['model_file'] = input_parameters['model_file']

        # Adding parameters for Reinhard color normalization
        if 'color_transform' in input_parameters:
            test_parameters['color_transform'] = input_parameters['color_transform']

        else:
            test_parameters['color_transform'] = []

        if 'label_dir' in input_parameters:
            test_parameters['label_dir'] = input_parameters['label_dir']+'/'


        if 'image_dir' in input_parameters:
            test_parameters['image_dir'] = input_parameters['image_dir']+'/'
        else:
            test_parameters['f_image_dir'] = input_parameters['f_image_dir']+'/'
            test_parameters['b_image_dir'] = input_parameters['b_image_dir']+'/'

        
        test_parameters['ann_classes'] = test_parameters['ann_classes']

    # Adding this option here for if using binary input masks or probabilistic (grayscale)
    test_parameters['target_type'] = input_parameters['target_type']

    test_parameters['output_dir'] = input_parameters['output_dir']

    if not os.path.isdir(test_parameters['output_dir']):
        os.makedirs(test_parameters['output_dir'])

    model_dir = test_parameters['output_dir']+'/models/'
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    try:
        nept_run['image_dir'] = image_dir
    except:
        nept_run['f_image_dir'] = f_image_dir
        nept_run['b_image_dir'] = b_image_dir
    try:
        if 'label_dir' in input_parameters:
            nept_run['label_dir'] = test_parameters['label_dir']
    except:
        nept_run['label_bin_dir'] = test_parameters['label_bin_dir']
        nept_run['label_reg_dir'] = test_parameters['label_reg_dir']
    nept_run['output_dir'] = test_parameters['output_dir']
    nept_run['model_dir'] = model_dir
    nept_run['Classes'] = test_parameters['ann_classes']
    nept_run['Target_Type'] = test_parameters['target_type']

    return train_parameters, test_parameters, phase









