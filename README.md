The scripts provided here for classifying nighttime cloud images 

The  scripts are included here as follows:

"channel_selection.py": split the raw image color channels into R G1 G2 B and crop the nighttime cloud images for further study

"illumination_conditions.py": calculate the altitude and azimuth of the Sun and the Moon

"get_features_test": the scripts include subregion division and feature extraction (sky background, star density, cloud gray values and cloud movement)

"PSO-XGBoost(1).py": load a feature file, train the PSO-XGBoost model and predict cloud types for individual subregions

"models.py": comparative models include SVM, KNN, RF and LightGBM

All these example scripts are intended to work with the example data provided and should be run in this order.
