# -*- coding: utf-8 -*-

# Importing libraries
import pandas as pd
import numpy as np
import susi
from sklearn import linear_model
from sklearn.metrics import precision_score, recall_score, auc, f1_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler, BorderlineSMOTE, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Dropout, Input
from keras.models import Model, Sequential
from keras import regularizers

##### FUNCTIONS
## Returns 8 variables: X and Y for ROS, RUS, SMOTE, and BorderlineSMOTE
# Inputs:   X - Training set for X in an array
#           y - Training set for y in an array
# Outputs:  X_ros, y_ros, X_rus, y_rus, X_smote, y_smote, X_bsmote, y_bsmote
def split_training_data_imbalance_methods(X, y):
    # Creating ROS training set with a 1:1 ratio
    ros = RandomOverSampler(random_state=0)
    X_ros, y_ros = ros.fit_resample(X, y)
    
    # Creating RUS training set with a 1:1 ratio
    rus = RandomUnderSampler( random_state=0)
    X_rus, y_rus = rus.fit_resample(X, y)
    
    # Creating the SMOTE training set with a 1:1 ratio
    smote = SMOTE(random_state=0)
    X_smote, y_smote = smote.fit_resample(X, y)
    
    # Creating the BorderlineSMOTE training set with a 1:1 ratio
    bsmote = BorderlineSMOTE(random_state=0)
    X_bsmote, y_bsmote = bsmote.fit_resample(X, y)
    
    # Creating the SMOTE + Tomek links training set with a 1:1 ratio
    smote_tomek = SMOTETomek(random_state=0)
    X_smote_tomek, y_smote_tomek = smote_tomek.fit_resample(X, y)
    
    # Creating the SMOTE + ENN training set with a 1:1 ratio
    smote_enn = SMOTEENN(random_state=0)
    X_smote_enn, y_smote_enn = smote_enn.fit_resample(X, y)
    
    return X_ros, y_ros, X_rus, y_rus, X_smote, y_smote, X_bsmote, y_bsmote, X_smote_tomek, y_smote_tomek, X_smote_enn, y_smote_enn

## Normalizes training array and testing array values to be between 0 and 1
# Inputs:   train - Training array to normalize
#           test - Testing array to normalize
# Outputs:  train, test
def normalize_features(train, test):
    sc = MinMaxScaler(feature_range = (0, 1))
    train = sc.fit_transform(train)
    test = sc.transform(test)
    return train, test

## Calculates and returns precision, recall, f1, cm, fpr, tpr, thresholds, auc_score
# Inputs:   ytest - Testing set for y
#           ypred - Predicted set for y
# Outputs:  precision, recall, f1, cm, fpr, tpr, thresholds, auc_score
def calculate_metrics(ytest, ypred):
    precision = precision_score(ytest, ypred)
    recall = recall_score(ytest, ypred)
    f1 = f1_score(ytest, ypred)
    fpr, tpr, thresholds = roc_curve(ytest, ypred)
    auc_score = auc(fpr, tpr)
    print('Precision: %s Recall: %s F1 Score: %s AUC: %s' %(precision, recall, f1, auc_score))
    
    # Creating tn, fp, fn, tp from confusion matrix
    cm = confusion_matrix(ytest, ypred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate inverse precision
    if tn + fn == 0:
        inverse_precision = 0
    else:
        inverse_precision = tn / (tn + fn)
    
    # Calculate inverse recall
    if tn + fp == 0:
        inverse_recall = 0
    else:
        inverse_recall = tn / (tn + fp)
    
    # Calculate markedness
    markedness = precision + inverse_precision - 1

    # Calculate informedness
    informedness = recall + inverse_recall - 1
    
    return precision, recall, f1, cm, fpr, tpr, thresholds, auc_score, inverse_precision, inverse_recall, markedness, informedness 

## Calculates and returns precision, recall, f1, cm, fpr, tpr, thresholds, auc_score for DAE
# Inputs:   cm - Confusion matrix of the DAE
#           err_df - Error dataframe of the DAE
# Outputs:  precision, recall, f1, cm, fpr, tpr, thresholds, auc_score
def calculate_metrics_dae(cm, err_df):
    # Extract values from cm
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate precision
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    
    # Calculate recall
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
        
    # Calculate inverse precision
    if tn + fn == 0:
        inverse_precision = 0
    else:
        inverse_precision = tn / (tn + fn)
    
    # Calculate inverse recall
    if tn + fp == 0:
        inverse_recall = 0
    else:
        inverse_recall = tn / (tn + fp)
    
    # Calculate f1, fpr, and tpr
    f1 = 2 * tp / (2 * tp + fp + fn)
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    
    # Calculate AUC score
    false_pos_rate, true_pos_rate, thresholds = roc_curve(err_df.True_class, err_df.Reconstruction_error)
    auc_score = auc(false_pos_rate, true_pos_rate,)
    
    # Calculate markedness
    markedness = precision + inverse_precision - 1

    # Calculate informedness
    informedness = recall + inverse_recall - 1
    
    return precision, recall, f1, cm, fpr, tpr, thresholds, auc_score, inverse_precision, inverse_recall, markedness, informedness

# Creates an MLP with 1 input layer, 2 hideen layers, and 1 output layer
# Fits the model based on the training data and makes predictions
# Precision, recall, f1, auc and a confuson matrix are output
# Inputs:   Xt - Training array for features to train model
#           yt - Training array for predicted variable to train model
#           Xtest - Test array for features to be predicted against
#           ytest - Test array for predicted variable to be predicted against
# Outputs:  precision, recall, f1, cm, fpr, tpr, thresholds, auc_score
def create_fit_mlp(Xt, yt, Xtest, ytest):
    # Initialize MLP
    mlp = Sequential()
    
    # Adding input layer and the first and second hidden layer
    mlp.add(Dense(15, activation = 'relu', kernel_initializer = 'uniform', input_dim=30))
    mlp.add(Dense(15, activation = 'relu', kernel_initializer = 'uniform'))
    mlp.add(Dense(15, activation = 'relu', kernel_initializer = 'uniform'))
    
    # Adding output layer
    mlp.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    # Compile the MLP
    mlp.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Fit the MLP to the training set
    mlp.fit(Xt, yt, batch_size = 128, epochs = 5)
    
    # Predicting on the test set
    ypred = mlp.predict(Xtest)
    ypred = (ypred > 0.5)
    
    return calculate_metrics(ytest, ypred)

# Creates a CNN with 1 convolution layer, 1 max pooling layer, 1 full connection layer and 1 output layer
# Fits the model based on the training data and makes predictions
# Precision, recall, f1, auc and a confuson matrix are output
# Inputs:   Xt - Training array for features to train model
#           yt - Training array for predicted variable to train model
#           Xtest - Test array for features to be predicted against
#           ytest - Test array for predicted variable to be predicted against
# Outputs:  precision, recall, f1, cm, fpr, tpr, thresholds, auc_score
def create_fit_cnn(Xt, yt, Xtest, ytest):
    # Reshaping Xt, yt, Xtest, and ytest to be useable with the CNN
    Xt, Xtest = np.reshape(Xt, (Xt.shape[0], Xt.shape[1], 1)), np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], 1))
    Xt, yt, Xtest, ytest = np.array(Xt), np.array(yt), np.array(Xtest), np.array(ytest)
    
    # Initializing CNN
    cnn = Sequential()
    
    # Creating input layer
    cnn.add(Conv1D(filters = 4, kernel_size = 2, input_shape = (30, 1)))
    
    # Creating pooling layer
    cnn.add(MaxPooling1D(pool_size = 1))
    
    # Flatten
    cnn.add(Flatten())
    
    # Adding full connection layer
    cnn.add(Dense(units = 64, activation = 'relu'))
    
    # Adding output layer
    cnn.add(Dense(units = 1, activation = 'sigmoid'))
    
    # Compile the CNN
    cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Fit the CNN to the training set
    cnn.fit(Xt, yt, batch_size = 128, epochs = 5)
    
    # Predicting on the test set
    ypred = cnn.predict(Xtest)
    ypred = (ypred > 0.5)

    return calculate_metrics(ytest, ypred)

# Creates a RNN with 4 LSTM layers, and 1 output layer
# Fits the model based on the training data and makes predictions
# Precision, recall, f1, auc and a confuson matrix are output
# Inputs:   Xt - Training array for features to train model
#           yt - Training array for predicted variable to train model
#           Xtest - Test array for features to be predicted against
#           ytest - Test array for predicted variable to be predicted against
def create_fit_rnn(Xt, yt, Xtest, ytest):
    # Reshaping Xt, yt, Xtest, and ytest to be useable with the RNN
    Xt, Xtest = np.reshape(Xt, (Xt.shape[0], Xt.shape[1], 1)), np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], 1))
    Xt, yt, Xtest, ytest = np.array(Xt), np.array(yt), np.array(Xtest), np.array(ytest)
    
    # Initializing RNN
    regressor = Sequential()
    
    # Creating first LSTM layer and adding dropout
    regressor.add(LSTM(units = 15, return_sequences = True, input_shape = (Xt.shape[1], 1)))
    regressor.add((Dropout(0.2)))
    
    # Adding 2nd LSTM layer and adding dropout
    regressor.add(LSTM(units = 15, return_sequences = True))
    regressor.add((Dropout(0.2)))
    
    # Adding 3rd LSTM layer and adding dropout
    regressor.add(LSTM(units = 15, return_sequences = True))
    regressor.add((Dropout(0.2)))
    
    # Adding 4th LSTM layer and adding dropout
    regressor.add(LSTM(units = 15))
    regressor.add((Dropout(0.2)))
    
    # Adding output layer
    regressor.add(Dense(1, activation = 'sigmoid'))
    
    # Compile the CNN
    regressor.compile(optimizer = 'adam', loss = 'binary_crossentropy',  metrics = ['accuracy'])
    
    # Fit the RNN to the training set
    regressor.fit(Xt, yt, validation_data = (Xtest, ytest), epochs = 5, batch_size = 128)
    
    # Predicting on the test set
    ypred = regressor.predict(Xtest)
    ypred = (ypred > 0.5)

    return calculate_metrics(ytest, ypred)

# Creates a supervised SOM 
# Fits the model based on the training data and makes predictions
# Precision, recall, f1, auc and a confuson matrix are output
# Inputs:   Xt - Training array for features to train model
#           yt - Training array for predicted variable to train model
#           Xtest - Test array for features to be predicted against
#           ytest - Test array for predicted variable to be predicted against
# Outputs:  precision, recall, f1, cm, fpr, tpr, thresholds, auc_score
def create_fit_som(Xt, yt, Xtest, ytest):
    # Initializing SOM
    som = susi.SOMClassifier(n_rows = 15, n_columns= 15, verbose = 1)
    
    # Fit the SOM to the training data
    som.fit(Xt, yt)
    ypred = som.predict(Xtest)
    ypred = (ypred > 0.5)
    
    return calculate_metrics(ytest, ypred)

# Creates an RBM to extract features and passes them to a Logistic Regression model to evaluate 
# Fits the model based on the training data and makes predictions
# Precision, recall, f1, auc and a confuson matrix are output
# Inputs:   Xt - Training array for features to train model
#           yt - Training array for predicted variable to train model
#           Xtest - Test array for features to be predicted against
#           ytest - Test array for predicted variable to be predicted against
# Outputs:  precision, recall, f1, cm, fpr, tpr, thresholds, auc_score
def create_fit_rbm(Xt, yt, Xtest, ytest):
    # Creating Logistic Regression model to evaluate features
    logistic = linear_model.LogisticRegression(solver='newton-cg', tol=1)
    
    # Creating RBM
    rbm = BernoulliRBM(random_state = 0, verbose = True)
    rbm_features = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
    
    # Setting training parameters
    rbm.learning_rate = .05
    rbm.n_iter = 5
    rbm.n_components = 60
    
    # Setting inverse of regularization strength
    logistic.C = 3000
    
    # Fit RBM to the training data
    rbm_features.fit(Xt, yt)
    
    # Predicting on the test set
    ypred = rbm_features.predict(Xtest)
    ypred = (ypred > 0.5)
    
    return calculate_metrics(ytest, ypred)

# Creates a Deep Autoencoder with 3 layers for the encoder and 3 layers for the decoder
# Fits the model based on the training data and makes predictions
# Precision, recall, f1, auc and a confuson matrix are output
# Inputs:   Xt - Training array for features to train model
#           yt - Training array for predicted variable to train model
#           Xtest - Test array for features to be predicted against
#           ytest - Test array for predicted variable to be predicted against
# Outputs:  precision, recall, f1, cm, fpr, tpr, thresholds, auc_score
def create_fit_dae(Xt, Xtest, ytest):
    # Setting learning rate
    learning_rate = 1e-7
    
    # Creating DAE with 3 layers for the encoder and 3 layers for the decoder
    input_layer = Input(shape=(30, ))
    encoder = Dense(16, activation="relu", activity_regularizer=regularizers.l1(learning_rate))(input_layer)
    encoder = Dense(8, activation="relu")(encoder)
    encoder = Dense(4, activation="relu")(encoder)
    decoder = Dense(4, activation='relu')(encoder)
    decoder = Dense(8, activation='relu')(encoder)
    decoder = Dense(30, activation='sigmoid')(decoder)
    
    # Initializing and compiling DAE
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(metrics=['accuracy'],
                        loss='mean_squared_error',
                        optimizer='adam')

    # Fitting the model
    autoencoder.fit(Xt, Xt,
                        epochs= 5,
                        batch_size= 64,
                        shuffle=True,
                        validation_data=(Xtest, Xtest))
    
    # Generating predictions
    Xtest_predictions = autoencoder.predict(Xtest)
    
    # Generating confusion matrix to calculate metrics
    mse = np.mean(np.power(Xtest - Xtest_predictions, 2), axis=1)
    error_df = pd.DataFrame({'Reconstruction_error': mse,
                            'True_class': ytest})
    threshold_fixed = 0.005
    y_pred = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
    conf_matrix = confusion_matrix(error_df.True_class, y_pred)
    
    return calculate_metrics_dae(conf_matrix, error_df)

################################################################### Script Start
##### Import dataset and Split into training set and testing set
dataset = pd.read_csv('European Credit Card Transactions.csv')
x = dataset.iloc[:, 0:30].values
y = dataset.iloc[:, 30]

# Split dataset into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

# Normalize features
X_train, X_test = normalize_features(X_train, X_test)

# Create copies of the training set using ROS, RUS, SMOTE, SMOTETomek, and SMOTEENN
X_train_ros, y_train_ros, X_train_rus, y_train_rus, X_train_smote, y_train_smote, X_train_bsmote, y_train_bsmote, X_train_smote_tomek, y_train_smote_tomek, X_train_smote_enn, y_train_smote_enn = split_training_data_imbalance_methods(X_train, y_train)

#### Create MLP Classifier and Calculate Metrics
# Predict from original data
mlp_precision, mlp_recall, mlp_f1, mlp_cm, mlp_tpr, mlp_fpr, mlp_thresholds, mlp_auc_score, mlp_inverse_precision, mlp_inverse_recall, mlp_markedness, mlp_informedness  = create_fit_mlp(X_train, y_train, X_test, y_test)

# Predict from sampled datasets
mlp_precision_ros, mlp_recall_ros, mlp_f1_ros, mlp_cm_ros, mlp_fpr_ros, mlp_tpr_ros, mlp_thresholds_ros, mlp_auc_score_ros, mlp_inverse_precision_ros, mlp_inverse_recall_ros, mlp_markedness_ros, mlp_informedness_ros = create_fit_mlp(X_train_ros, y_train_ros, X_test, y_test)
mlp_precision_rus, mlp_recall_rus, mlp_f1_rus, mlp_cm_rus, mlp_fpr_rus, mlp_tpr_rus, mlp_thresholds_rus, mlp_auc_score_rus, mlp_inverse_precision_rus, mlp_inverse_recall_rus, mlp_markedness_rus, mlp_informedness_rus = create_fit_mlp(X_train_rus, y_train_rus, X_test, y_test)
mlp_precision_smote, mlp_recall_smote, mlp_f1_smote, mlp_cm_smote, mlp_fpr_smote, mlp_tpr_smote, mlp_thresholds_smote, mlp_auc_score_smote, mlp_inverse_precision_smote, mlp_inverse_recall_smote, mlp_markedness_smote, mlp_informedness_smote = create_fit_mlp(X_train_smote, y_train_smote, X_test, y_test)
mlp_precision_bsmote, mlp_recall_bsmote, mlp_f1_bsmote, mlp_cm_bsmote, mlp_fpr_bsmote, mlp_tpr_bsmote, mlp_thresholds_bsmote, mlp_auc_score_bsmote, mlp_inverse_precision_bsmote, mlp_inverse_recall_bsmote, mlp_markedness_bsmote, mlp_informedness_bsmote = create_fit_mlp(X_train_bsmote, y_train_bsmote, X_test, y_test)
mlp_precision_smote_tomek, mlp_recall_smote_tomek, mlp_f1_smote_tomek, mlp_cm_smote_tomek, mlp_fpr_smote_tomek, mlp_tpr_smote_tomek, mlp_thresholds_smote_tomek, mlp_auc_score_smote_tomek, mlp_inverse_precision_smote_tomek, mlp_inverse_recall_smote_tomek, mlp_markedness_smote_tomek, mlp_informedness_smote_tomek = create_fit_mlp(X_train_smote_tomek, y_train_smote_tomek, X_test, y_test)
mlp_precision_smote_enn, mlp_recall_smote_enn, mlp_f1_smote_enn, mlp_cm_smote_enn, mlp_fpr_smote_enn, mlp_tpr_smote_enn, mlp_thresholds_smote_enn, mlp_auc_score_smote_enn, mlp_inverse_precision_smote_enn, mlp_inverse_recall_smote_enn, mlp_markedness_smote_enn, mlp_informedness_smote_enn = create_fit_mlp(X_train_smote_enn, y_train_smote_enn, X_test, y_test)

#### Create CNN Classifier and Calculate Metrics
# Predict from original data
cnn_precision, cnn_recall, cnn_f1, cnn_cm, cnn_tpr, cnn_fpr, cnn_thresholds, cnn_auc_score, cnn_inverse_precision, cnn_inverse_recall, cnn_markedness, cnn_informedness  = create_fit_cnn(X_train, y_train, X_test, y_test)

# Predict from sampled datasets
cnn_precision_ros, cnn_recall_ros, cnn_f1_ros, cnn_cm_ros, cnn_fpr_ros, cnn_tpr_ros, cnn_thresholds_ros, cnn_auc_score_ros, cnn_inverse_precision_ros, cnn_inverse_recall_ros, cnn_markedness_ros, cnn_informedness_ros = create_fit_cnn(X_train_ros, y_train_ros, X_test, y_test)
cnn_precision_rus, cnn_recall_rus, cnn_f1_rus, cnn_cm_rus, cnn_fpr_rus, cnn_tpr_rus, cnn_thresholds_rus, cnn_auc_score_rus, cnn_inverse_precision_rus, cnn_inverse_recall_rus, cnn_markedness_rus, cnn_informedness_rus = create_fit_cnn(X_train_rus, y_train_rus, X_test, y_test)
cnn_precision_smote, cnn_recall_smote, cnn_f1_smote, cnn_cm_smote, cnn_fpr_smote, cnn_tpr_smote, cnn_thresholds_smote, cnn_auc_score_smote, cnn_inverse_precision_smote, cnn_inverse_recall_smote, cnn_markedness_smote, cnn_informedness_smote = create_fit_cnn(X_train_smote, y_train_smote, X_test, y_test)
cnn_precision_bsmote, cnn_recall_bsmote, cnn_f1_bsmote, cnn_cm_bsmote, cnn_fpr_bsmote, cnn_tpr_bsmote, cnn_thresholds_bsmote, cnn_auc_score_bsmote, cnn_inverse_precision_bsmote, cnn_inverse_recall_bsmote, cnn_markedness_bsmote, cnn_informedness_bsmote = create_fit_cnn(X_train_bsmote, y_train_bsmote, X_test, y_test)
cnn_precision_smote_tomek, cnn_recall_smote_tomek, cnn_f1_smote_tomek, cnn_cm_smote_tomek, cnn_fpr_smote_tomek, cnn_tpr_smote_tomek, cnn_thresholds_smote_tomek, cnn_auc_score_smote_tomek, cnn_inverse_precision_smote_tomek, cnn_inverse_recall_smote_tomek, cnn_markedness_smote_tomek, cnn_informedness_smote_tomek = create_fit_cnn(X_train_smote_tomek, y_train_smote_tomek, X_test, y_test)
cnn_precision_smote_enn, cnn_recall_smote_enn, cnn_f1_smote_enn, cnn_cm_smote_enn, cnn_fpr_smote_enn, cnn_tpr_smote_enn, cnn_thresholds_smote_enn, cnn_auc_score_smote_enn, cnn_inverse_precision_smote_enn, cnn_inverse_recall_smote_enn, cnn_markedness_smote_enn, cnn_informedness_smote_enn = create_fit_cnn(X_train_smote_enn, y_train_smote_enn, X_test, y_test)

##### Create RNN Classifier and Calculate Metrics
# Predict from original data
rnn_precision, rnn_recall, rnn_f1, rnn_cm, rnn_tpr, rnn_fpr, rnn_thresholds, rnn_auc_score, rnn_inverse_precision, rnn_inverse_recall, rnn_markedness, rnn_informedness  = create_fit_rnn(X_train, y_train, X_test, y_test)

# Predict from sampled datasets
rnn_precision_ros, rnn_recall_ros, rnn_f1_ros, rnn_cm_ros, rnn_fpr_ros, rnn_tpr_ros, rnn_thresholds_ros, rnn_auc_score_ros, rnn_inverse_precision_ros, rnn_inverse_recall_ros, rnn_markedness_ros, rnn_informedness_ros = create_fit_rnn(X_train_ros, y_train_ros, X_test, y_test)
rnn_precision_rus, rnn_recall_rus, rnn_f1_rus, rnn_cm_rus, rnn_fpr_rus, rnn_tpr_rus, rnn_thresholds_rus, rnn_auc_score_rus, rnn_inverse_precision_rus, rnn_inverse_recall_rus, rnn_markedness_rus, rnn_informedness_rus = create_fit_rnn(X_train_rus, y_train_rus, X_test, y_test)
rnn_precision_smote, rnn_recall_smote, rnn_f1_smote, rnn_cm_smote, rnn_fpr_smote, rnn_tpr_smote, rnn_thresholds_smote, rnn_auc_score_smote, rnn_inverse_precision_smote, rnn_inverse_recall_smote, rnn_markedness_smote, rnn_informedness_smote = create_fit_rnn(X_train_smote, y_train_smote, X_test, y_test)
rnn_precision_bsmote, rnn_recall_bsmote, rnn_f1_bsmote, rnn_cm_bsmote, rnn_fpr_bsmote, rnn_tpr_bsmote, rnn_thresholds_bsmote, rnn_auc_score_bsmote, rnn_inverse_precision_bsmote, rnn_inverse_recall_bsmote, rnn_markedness_bsmote, rnn_informedness_bsmote = create_fit_rnn(X_train_bsmote, y_train_bsmote, X_test, y_test)
rnn_precision_smote_tomek, rnn_recall_smote_tomek, rnn_f1_smote_tomek, rnn_cm_smote_tomek, rnn_fpr_smote_tomek, rnn_tpr_smote_tomek, rnn_thresholds_smote_tomek, rnn_auc_score_smote_tomek, rnn_inverse_precision_smote_tomek, rnn_inverse_recall_smote_tomek, rnn_markedness_smote_tomek, rnn_informedness_smote_tomek = create_fit_rnn(X_train_smote_tomek, y_train_smote_tomek, X_test, y_test)
rnn_precision_smote_enn, rnn_recall_smote_enn, rnn_f1_smote_enn, rnn_cm_smote_enn, rnn_fpr_smote_enn, rnn_tpr_smote_enn, rnn_thresholds_smote_enn, rnn_auc_score_smote_enn, rnn_inverse_precision_smote_enn, rnn_inverse_recall_smote_enn, rnn_markedness_smote_enn, rnn_informedness_smote_enn = create_fit_rnn(X_train_smote_enn, y_train_smote_enn, X_test, y_test)

##### Create SOM Classifier and Calculate Metrics
# Predict from original data
som_precision, som_recall, som_f1, som_cm, som_tpr, som_fpr, som_thresholds, som_auc_score, som_inverse_precision, som_inverse_recall, som_markedness, som_informedness  = create_fit_som(X_train, y_train, X_test, y_test)

# Predict from sampled datasets
som_precision_ros, som_recall_ros, som_f1_ros, som_cm_ros, som_fpr_ros, som_tpr_ros, som_thresholds_ros, som_auc_score_ros, som_inverse_precision_ros, som_inverse_recall_ros, som_markedness_ros, som_informedness_ros = create_fit_som(X_train_ros, y_train_ros, X_test, y_test)
som_precision_rus, som_recall_rus, som_f1_rus, som_cm_rus, som_fpr_rus, som_tpr_rus, som_thresholds_rus, som_auc_score_rus, som_inverse_precision_rus, som_inverse_recall_rus, som_markedness_rus, som_informedness_rus = create_fit_som(X_train_rus, y_train_rus, X_test, y_test)
som_precision_smote, som_recall_smote, som_f1_smote, som_cm_smote, som_fpr_smote, som_tpr_smote, som_thresholds_smote, som_auc_score_smote, som_inverse_precision_smote, som_inverse_recall_smote, som_markedness_smote, som_informedness_smote = create_fit_som(X_train_smote, y_train_smote, X_test, y_test)
som_precision_bsmote, som_recall_bsmote, som_f1_bsmote, som_cm_bsmote, som_fpr_bsmote, som_tpr_bsmote, som_thresholds_bsmote, som_auc_score_bsmote, som_inverse_precision_bsmote, som_inverse_recall_bsmote, som_markedness_bsmote, som_informedness_bsmote = create_fit_som(X_train_bsmote, y_train_bsmote, X_test, y_test)
som_precision_smote_tomek, som_recall_smote_tomek, som_f1_smote_tomek, som_cm_smote_tomek, som_fpr_smote_tomek, som_tpr_smote_tomek, som_thresholds_smote_tomek, som_auc_score_smote_tomek, som_inverse_precision_smote_tomek, som_inverse_recall_smote_tomek, som_markedness_smote_tomek, som_informedness_smote_tomek = create_fit_som(X_train_smote_tomek, y_train_smote_tomek, X_test, y_test)
som_precision_smote_enn, som_recall_smote_enn, som_f1_smote_enn, som_cm_smote_enn, som_fpr_smote_enn, som_tpr_smote_enn, som_thresholds_smote_enn, som_auc_score_smote_enn, som_inverse_precision_smote_enn, som_inverse_recall_smote_enn, som_markedness_smote_enn, som_informedness_smote_enn = create_fit_som(X_train_smote_enn, y_train_smote_enn, X_test, y_test)

##### Create RBM Classifier and Calculate Metrics
# Predict from original data
rbm_precision, rbm_recall, rbm_f1, rbm_cm, rbm_tpr, rbm_fpr, rbm_thresholds, rbm_auc_score, rbm_inverse_precision, rbm_inverse_recall, rbm_markedness, rbm_informedness  = create_fit_rbm(X_train, y_train, X_test, y_test)

# Predict from sampled datasets
rbm_precision_ros, rbm_recall_ros, rbm_f1_ros, rbm_cm_ros, rbm_fpr_ros, rbm_tpr_ros, rbm_thresholds_ros, rbm_auc_score_ros, rbm_inverse_precision_ros, rbm_inverse_recall_ros, rbm_markedness_ros, rbm_informedness_ros = create_fit_rbm(X_train_ros, y_train_ros, X_test, y_test)
rbm_precision_rus, rbm_recall_rus, rbm_f1_rus, rbm_cm_rus, rbm_fpr_rus, rbm_tpr_rus, rbm_thresholds_rus, rbm_auc_score_rus, rbm_inverse_precision_rus, rbm_inverse_recall_rus, rbm_markedness_rus, rbm_informedness_rus = create_fit_rbm(X_train_rus, y_train_rus, X_test, y_test)
rbm_precision_smote, rbm_recall_smote, rbm_f1_smote, rbm_cm_smote, rbm_fpr_smote, rbm_tpr_smote, rbm_thresholds_smote, rbm_auc_score_smote, rbm_inverse_precision_smote, rbm_inverse_recall_smote, rbm_markedness_smote, rbm_informedness_smote = create_fit_rbm(X_train_smote, y_train_smote, X_test, y_test)
rbm_precision_bsmote, rbm_recall_bsmote, rbm_f1_bsmote, rbm_cm_bsmote, rbm_fpr_bsmote, rbm_tpr_bsmote, rbm_thresholds_bsmote, rbm_auc_score_bsmote, rbm_inverse_precision_bsmote, rbm_inverse_recall_bsmote, rbm_markedness_bsmote, rbm_informedness_bsmote = create_fit_rbm(X_train_bsmote, y_train_bsmote, X_test, y_test)
rbm_precision_smote_tomek, rbm_recall_smote_tomek, rbm_f1_smote_tomek, rbm_cm_smote_tomek, rbm_fpr_smote_tomek, rbm_tpr_smote_tomek, rbm_thresholds_smote_tomek, rbm_auc_score_smote_tomek, rbm_inverse_precision_smote_tomek, rbm_inverse_recall_smote_tomek, rbm_markedness_smote_tomek, rbm_informedness_smote_tomek = create_fit_rbm(X_train_smote_tomek, y_train_smote_tomek, X_test, y_test)
rbm_precision_smote_enn, rbm_recall_smote_enn, rbm_f1_smote_enn, rbm_cm_smote_enn, rbm_fpr_smote_enn, rbm_tpr_smote_enn, rbm_thresholds_smote_enn, rbm_auc_score_smote_enn, rbm_inverse_precision_smote_enn, rbm_inverse_recall_smote_enn, rbm_markedness_smote_enn, rbm_informedness_smote_enn = create_fit_rbm(X_train_smote_enn, y_train_smote_enn, X_test, y_test)

##### Create Deep Autoencoder and Calculate Metrics
# Predict from original data
dae_precision, dae_recall, dae_f1, dae_cm, dae_tpr, dae_fpr, dae_thresholds, dae_auc_score, dae_inverse_precision, dae_inverse_recall, dae_markedness, dae_informedness  = create_fit_dae(X_train, X_test, y_test)

# Predict from sampled datasets
dae_precision_ros, dae_recall_ros, dae_f1_ros, dae_cm_ros, dae_fpr_ros, dae_tpr_ros, dae_thresholds_ros, dae_auc_score_ros, dae_inverse_precision_ros, dae_inverse_recall_ros, dae_markedness_ros, dae_informedness_ros = create_fit_dae(X_train_ros, X_test, y_test)
dae_precision_rus, dae_recall_rus, dae_f1_rus, dae_cm_rus, dae_fpr_rus, dae_tpr_rus, dae_thresholds_rus, dae_auc_score_rus, dae_inverse_precision_rus, dae_inverse_recall_rus, dae_markedness_rus, dae_informedness_rus = create_fit_dae(X_train_rus, X_test, y_test)
dae_precision_smote, dae_recall_smote, dae_f1_smote, dae_cm_smote, dae_fpr_smote, dae_tpr_smote, dae_thresholds_smote, dae_auc_score_smote, dae_inverse_precision_smote, dae_inverse_recall_smote, dae_markedness_smote, dae_informedness_smote = create_fit_dae(X_train_smote, X_test, y_test)
dae_precision_bsmote, dae_recall_bsmote, dae_f1_bsmote, dae_cm_bsmote, dae_fpr_bsmote, dae_tpr_bsmote, dae_thresholds_bsmote, dae_auc_score_bsmote, dae_inverse_precision_bsmote, dae_inverse_recall_bsmote, dae_markedness_bsmote, dae_informedness_bsmote = create_fit_dae(X_train_bsmote, X_test, y_test)
dae_precision_smote_tomek, dae_recall_smote_tomek, dae_f1_smote_tomek, dae_cm_smote_tomek, dae_fpr_smote_tomek, dae_tpr_smote_tomek, dae_thresholds_smote_tomek, dae_auc_score_smote_tomek, dae_inverse_precision_smote_tomek, dae_inverse_recall_smote_tomek, dae_markedness_smote_tomek, dae_informedness_smote_tomek = create_fit_dae(X_train_smote_tomek, X_test, y_test)
dae_precision_smote_enn, dae_recall_smote_enn, dae_f1_smote_enn, dae_cm_smote_enn, dae_fpr_smote_enn, dae_tpr_smote_enn, dae_thresholds_smote_enn, dae_auc_score_smote_enn, dae_inverse_precision_smote_enn, dae_inverse_recall_smote_enn, dae_markedness_smote_enn, dae_informedness_smote_enn = create_fit_dae(X_train_smote_enn, X_test, y_test)

### Print all metrics
# Write Results to File
f= open("results.txt","w+")

# Print Metrics for MLP
f.write('MLP\n')
f.write('Base - Precision: %s Recall: %s Inverse Precision:%s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(mlp_precision, mlp_recall, mlp_inverse_precision, mlp_inverse_recall, mlp_f1, mlp_auc_score, mlp_markedness, mlp_informedness))
f.write('========================= Sampling Methods\n')
f.write('ROS - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall:%s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(mlp_precision_ros, mlp_recall_ros, mlp_inverse_precision_ros, mlp_inverse_recall_ros, mlp_f1_ros, mlp_auc_score_ros, mlp_markedness_ros, mlp_informedness_ros))
f.write('RUS - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall:%s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(mlp_precision_rus, mlp_recall_rus, mlp_inverse_precision_rus, mlp_inverse_recall_rus, mlp_f1_rus, mlp_auc_score_rus, mlp_markedness_rus, mlp_informedness_rus))
f.write('SMOTE - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall:%s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(mlp_precision_smote, mlp_recall_smote, mlp_inverse_precision_smote, mlp_inverse_recall_smote, mlp_f1_smote, mlp_auc_score_smote, mlp_markedness_smote, mlp_informedness_smote))
f.write('BorderlineSMOTE - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(mlp_precision_bsmote, mlp_recall_bsmote, mlp_inverse_precision_bsmote, mlp_inverse_recall_bsmote, mlp_f1_bsmote, mlp_auc_score_bsmote, mlp_markedness_bsmote, mlp_informedness_bsmote))
f.write('SMOTE + Tomek Links - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(mlp_precision_smote_tomek, mlp_recall_smote_tomek, mlp_inverse_precision_smote_tomek, mlp_inverse_recall_smote_tomek, mlp_f1_smote_tomek, mlp_auc_score_smote_tomek, mlp_markedness_smote_tomek, mlp_informedness_smote_tomek))
f.write('SMOTE + ENN - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(mlp_precision_smote_enn, mlp_recall_smote_enn, mlp_inverse_precision_smote_enn, mlp_inverse_recall_smote_enn, mlp_f1_smote_enn, mlp_auc_score_smote_enn, mlp_markedness_smote_enn, mlp_informedness_smote_enn))

# f.write Metrics for CNN
f.write('\nCNN\n')
f.write('Base - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(cnn_precision, cnn_recall, cnn_inverse_precision, cnn_inverse_recall, cnn_f1, cnn_auc_score, cnn_markedness, cnn_informedness))
f.write('========================= Sampling Methods\n')
f.write('ROS - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(cnn_precision_ros, cnn_recall_ros, cnn_inverse_precision_ros, cnn_inverse_recall_ros, cnn_f1_ros, cnn_auc_score_ros, cnn_markedness_ros, cnn_informedness_ros))
f.write('RUS - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(cnn_precision_rus, cnn_recall_rus, cnn_inverse_precision_rus, cnn_inverse_recall_rus, cnn_f1_rus, cnn_auc_score_rus, cnn_markedness_rus, cnn_informedness_rus))
f.write('SMOTE - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(cnn_precision_smote, cnn_recall_smote, cnn_inverse_precision_smote, cnn_inverse_recall_smote, cnn_f1_smote, cnn_auc_score_smote, cnn_markedness_smote, cnn_informedness_smote))
f.write('BorderlineSMOTE - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(cnn_precision_bsmote, cnn_recall_bsmote, cnn_inverse_precision_bsmote, cnn_inverse_recall_bsmote, cnn_f1_bsmote, cnn_auc_score_bsmote, cnn_markedness_bsmote, cnn_informedness_bsmote))
f.write('SMOTE + Tomek Links - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(cnn_precision_smote_tomek, cnn_recall_smote_tomek, cnn_inverse_precision_smote_tomek, cnn_inverse_recall_smote_tomek, cnn_f1_smote_tomek, cnn_auc_score_smote_tomek, cnn_markedness_smote_tomek, cnn_informedness_smote_tomek))
f.write('SMOTE + ENN - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(cnn_precision_smote_enn, cnn_recall_smote_enn, cnn_inverse_precision_smote_enn, cnn_inverse_recall_smote_enn, cnn_f1_smote_enn, cnn_auc_score_smote_enn, cnn_markedness_smote_enn, cnn_informedness_smote_enn))

# f.write Metrics for RNN
f.write('\nRNN\n')
f.write('Base - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(rnn_precision, rnn_recall, rnn_inverse_precision, rnn_inverse_recall, rnn_f1, rnn_auc_score, rnn_markedness, rnn_informedness))
f.write('========================= Sampling Methods\n')
f.write('ROS - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(rnn_precision_ros, rnn_recall_ros, rnn_inverse_precision_ros, rnn_inverse_recall_ros, rnn_f1_ros, rnn_auc_score_ros, rnn_markedness_ros, rnn_informedness_ros))
f.write('RUS - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(rnn_precision_rus, rnn_recall_rus, rnn_inverse_precision_rus, rnn_inverse_recall_rus, rnn_f1_rus, rnn_auc_score_rus, rnn_markedness_rus, rnn_informedness_rus))
f.write('SMOTE - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(rnn_precision_smote, rnn_recall_smote, rnn_inverse_precision_smote, rnn_inverse_recall_smote, rnn_f1_smote, rnn_auc_score_smote, rnn_markedness_smote, rnn_informedness_smote))
f.write('BorderlineSMOTE - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(rnn_precision_bsmote, rnn_recall_bsmote, rnn_inverse_precision_bsmote, rnn_inverse_recall_bsmote, rnn_f1_bsmote, rnn_auc_score_bsmote, rnn_markedness_bsmote, rnn_informedness_bsmote))
f.write('SMOTE + Tomek Links - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(rnn_precision_smote_tomek, rnn_recall_smote_tomek, rnn_inverse_precision_smote_tomek, rnn_inverse_recall_smote_tomek, rnn_f1_smote_tomek, rnn_auc_score_smote_tomek, rnn_markedness_smote_tomek, rnn_informedness_smote_tomek))
f.write('SMOTE + ENN - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(rnn_precision_smote_enn, rnn_recall_smote_enn, rnn_inverse_precision_smote_enn, rnn_inverse_recall_smote_enn, rnn_f1_smote_enn, rnn_auc_score_smote_enn, rnn_markedness_smote_enn, rnn_informedness_smote_enn))

# f.write Metrics for SOM
f.write('\nSOM\n')
f.write('Base - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(som_precision, som_recall, som_inverse_precision, som_inverse_recall, som_f1, som_auc_score, som_markedness, som_informedness))
f.write('========================= Sampling Methods\n')
f.write('ROS - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(som_precision_ros, som_recall_ros, som_inverse_precision_ros, som_inverse_recall_ros, som_f1_ros, som_auc_score_ros, som_markedness_ros, som_informedness_ros))
f.write('RUS - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(som_precision_rus, som_recall_rus, som_inverse_precision_rus, som_inverse_recall_rus, som_f1_rus, som_auc_score_rus, som_markedness_rus, som_informedness_rus))
f.write('SMOTE - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(som_precision_smote, som_recall_smote, som_inverse_precision_smote, som_inverse_recall_smote, som_f1_smote, som_auc_score_smote, som_markedness_smote, som_informedness_smote))
f.write('BorderlineSMOTE - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(som_precision_bsmote, som_recall_bsmote, som_inverse_precision_bsmote, som_inverse_recall_bsmote, som_f1_bsmote, som_auc_score_bsmote, som_markedness_bsmote, som_informedness_bsmote))
f.write('SMOTE + Tomek Links - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(som_precision_smote_tomek, som_recall_smote_tomek, som_inverse_precision_smote_tomek, som_inverse_recall_smote_tomek, som_f1_smote_tomek, som_auc_score_smote_tomek, som_markedness_smote_tomek, som_informedness_smote_tomek))
f.write('SMOTE + ENN - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(som_precision_smote_enn, som_recall_smote_enn, som_inverse_precision_smote_enn, som_inverse_recall_smote_enn, som_f1_smote_enn, som_auc_score_smote_enn, som_markedness_smote_enn, som_informedness_smote_enn))

# f.write Metrics for RBM
f.write('\nRBM\n')
f.write('Base - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(rbm_precision, rbm_recall, rbm_inverse_precision, rbm_inverse_recall, rbm_f1, rbm_auc_score, rbm_markedness, rbm_informedness))
f.write('========================= Sampling Methods\n')
f.write('ROS - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(rbm_precision_ros, rbm_recall_ros, rbm_inverse_precision_ros, rbm_inverse_recall_ros, rbm_f1_ros, rbm_auc_score_ros, rbm_markedness_ros, rbm_informedness_ros))
f.write('RUS - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(rbm_precision_rus, rbm_recall_rus, rbm_inverse_precision_rus, rbm_inverse_recall_rus, rbm_f1_rus, rbm_auc_score_rus, rbm_markedness_rus, rbm_informedness_rus))
f.write('SMOTE - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(rbm_precision_smote, rbm_recall_smote, rbm_inverse_precision_smote, rbm_inverse_recall_smote, rbm_f1_smote, rbm_auc_score_smote, rbm_markedness_smote, rbm_informedness_smote))
f.write('BorderlineSMOTE - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(rbm_precision_bsmote, rbm_recall_bsmote, rbm_inverse_precision_bsmote, rbm_inverse_recall_bsmote, rbm_f1_bsmote, rbm_auc_score_bsmote, rbm_markedness_bsmote, rbm_informedness_bsmote))
f.write('SMOTE + Tomek Links - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(rbm_precision_smote_tomek, rbm_recall_smote_tomek, rbm_inverse_precision_smote_tomek, rbm_inverse_recall_smote_tomek, rbm_f1_smote_tomek, rbm_auc_score_smote_tomek, rbm_markedness_smote_tomek, rbm_informedness_smote_tomek))
f.write('SMOTE + ENN - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(rbm_precision_smote_enn, rbm_recall_smote_enn, rbm_inverse_precision_smote_enn, rbm_inverse_recall_smote_enn, rbm_f1_smote_enn, rbm_auc_score_smote_enn, rbm_markedness_smote_enn, rbm_informedness_smote_enn))

# f.write Metrics for Deep Autoencoder
f.write('\nDAE\n')
f.write('Base - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(dae_precision, dae_recall, dae_inverse_precision, dae_inverse_recall, dae_f1, dae_auc_score, dae_markedness, dae_informedness))
f.write('========================= Sampling Methods\n')
f.write('ROS - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(dae_precision_ros, dae_recall_ros, dae_inverse_precision_ros, dae_inverse_recall_ros, dae_f1_ros, dae_auc_score_ros, dae_markedness_ros, dae_informedness_ros))
f.write('RUS - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(dae_precision_rus, dae_recall_rus, dae_inverse_precision_rus, dae_inverse_recall_rus, dae_f1_rus, dae_auc_score_rus, dae_markedness_rus, dae_informedness_rus))
f.write('SMOTE - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(dae_precision_smote, dae_recall_smote, dae_inverse_precision_smote, dae_inverse_recall_smote, dae_f1_smote, dae_auc_score_smote, dae_markedness_smote, dae_informedness_smote))
f.write('BorderlineSMOTE - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(dae_precision_bsmote, dae_recall_bsmote, dae_inverse_precision_bsmote, dae_inverse_recall_bsmote, dae_f1_bsmote, dae_auc_score_bsmote, dae_markedness_bsmote, dae_informedness_bsmote))
f.write('SMOTE + Tomek Links - Precision: %s Recall: %s Inverse Precision:%s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(dae_precision_smote_tomek, dae_recall_smote_tomek, dae_inverse_precision_smote_tomek, dae_inverse_recall_smote_tomek, dae_f1_smote_tomek, dae_auc_score_smote_tomek, dae_markedness_smote_tomek, dae_informedness_smote_tomek))
f.write('SMOTE + ENN - Precision: %s Recall: %s Inverse Precision: %s Inverse Recall: %s F1-Measure: %s AUC: %s Markedness:%s Informedness:%s\n' %(dae_precision_smote_enn, dae_recall_smote_enn, dae_inverse_precision_smote_enn, dae_inverse_recall_smote_enn, dae_f1_smote_enn, dae_auc_score_smote_enn, dae_markedness_smote_enn, dae_informedness_smote_enn))

f.close()