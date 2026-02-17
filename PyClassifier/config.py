###########################
# Author: Capuzzo Daniele #
###########################

#### CONFIGURATION ####
# -- dataset directories
directory = "../Malaria_Dataset_FromGT"
save_directory = "saved_models/"
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
max_epochs = 20
factor = 3
early_stop_patience = 5
tuned_models_dir = 'tuned_models'
project_name = 'malaria_det'
