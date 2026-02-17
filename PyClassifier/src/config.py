###########################
# Author: Capuzzo Daniele #
###########################

#### CONFIGURATION ####
# -- dataset directories
directory = "../../Malaria_Dataset_FromGT"
save_directory = "../saved_model/"
val_split = 0.2
SEED = 123          # for reproducibility

# -- image parameters
img_height = 256    # 128
img_width = 256     # 128
channels = 3

# -- hyperparameters
input_shape = (img_height, img_width, channels)
batch_size = 64

# -- tuner settings
tuner_epochs = 100
max_epochs = 8
factor = 3
early_stop_patience = 3
tuned_models_dir = '../tuned_models'
project_name = 'malaria_det'

# -- model settings
epochs_train = 50
patience_train = 3
evaluate = True