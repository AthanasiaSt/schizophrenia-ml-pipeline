--------------------------------------------------------------
Gene Expression Data Classification of post-mortem tissue 
from the anterior prefrontal cortex of schizophrenic patients
--------------------------------------------------------------

The aim of this project was to develop a fully automated, algorithmic approach (ML pipeline) in order to classify with as high unbiased performance as possible the chosen dataset (GDS4523). It contains 54675 gene expression profiles from 51 post-mortem tissues of the anterior prefrontal cortex (Brodmann Area 10, BA10) of schizophrenic and control patients. In the pipeline were inserted preprocessing techniques (mean imputator and standard scaling), unsupervised techniques (pca, kmeans) for visualization, Lasso regression for feature selection and finally two classifiers (SVM and Random forest) only for binary classification based on the disease. The tuning of the hyperparameters (lasso --> alpha, SVM rbf --> C, gamma, Random forest --> max_depth, n_estimators), the model evaluation and selection of the best configuration were accomplished by utilizing Nrepeated stratified cross validation and three metrics (Accuracy, F1, AUC-ROC). 
--------------------------------------------------------------

----------
functions
----------

'pipeline':

description: It is the main pipeline that 'calls' the rest of the functions

-inputs: 2 --> (GDS_name of the dataset, 'yes' to download the dataset or anything else as a 'no')
-outputs: 4 --> (SVM_configurations (nested list), RF_configurations (nested_list), selected model trained on all data, names of selected genes)

example: SVM, RF, Final_model, genes_selec = pipeline('GDS4523', download = 'no')
result: All the procedures are automatically conducted (with the inner defined hyperparameters) but the dataset will not be downloaded. 

----------------------------------------------------------------------------

'download_data':

description: It downloads and unzips the dataset

-inputs: 2 --> (GDS_name of the dataset, 'yes' to download the dataset or anything else as a 'no')
-outputs: 1 --> (The filename of the unziped dataset --> file.soft)

example: file_soft =  download_data('GDS4523', download = 'yes')
result: The dataset GDS4523 is downloaded and then unziped.

----------------------------------------------------------------------------

'clean_data':

description: It clears the initial information of the dataset

-inputs: 2 --> (file.soft name, GDS ID)
-outputs: 1 --> (The filename of the cleaned dataset --> file.clean)

example: file_clean = clean_data('GDS4523.soft', 'GDS4523')
result: The dataset is cleaned and the file.clean is returned 

----------------------------------------------------------------------------

'target_extraction':

description: It extracts all the potential targets (disease, sex, age), the corresponding sample names and loads the dataset into a dataframe (samples x features) while keeping the gene names

-inputs: 2 --> (file.soft name, file.clean name)
-outputs: 5 --> (target_disease(classes and the corresponding samples), target_sex(classes and the corresponding samples), target_age(classes and the corresponding samples), Dataframe of the dataset (features x samples), gene-feature names)

example: target_disease, target_sex, target_age, df_raw, gene_names = target_extraction('GDS4523.soft',"GDS4523.clean")
result: all targets, a dataframe and the gene names are returned

----------------------------------------------------------------------------

'matrix':

description: It makes the dataframe into a matrix with samples as rows and features as columns. Also it makes the categorical targets into numerical(Only binary classes) 

-inputs: 2 --> (dataframe, targets)
-outputs: 2 --> (matrix of the data(samples x features), targets(as zero or one))

example: X,y = matrix(df_raw, target_disease)
result: the numerical targets based on the disease and a matrix are returned

----------------------------------------------------------------------------

'Prepro':

description: It performs two preprocessing techniques on the matrix given. It utilizes a mean imputator for the missing data and a standard scaler. It is trained on the dataset only if the argument fit = 'yes', else it utilizes the imputator and the standard scaler given to it as arguments.  

-inputs: 4 --> (Matrix X (samples x features), imputator model, standard scaler model, fit = yes for the model to be trained on the given dataset or fit = anything else for the models given as arguments to be used on the transformation of the dataset given)
-outputs: 3 --> (The resulting matrix X, the model used for imputation, the model used for standard scaling)

example: X, imp, std_scl = Prepro(X,imputator,std_scaler,fit='yes')
result: X will be used for training of the imputator and standard scaler and then it will be transformed. If fit = anything else then the imputator and std_scaler models would be used only to transform X. Finally the model used and the transformed X are returned. 

----------------------------------------------------------------------------

'pca_kmeans':

description: It performs conventional PCA (90% of the total variance) and then Kmeans (k is decided based on silhouette scores) on the resulting principal components. Also four plots are produced. 

-inputs: 2 --> (matrix X (samples x features), targets y (numerical))
-outputs: 1 --> (four plots, a matrix with the resulting principal components)

example: X_pca = pca_kmeans(X,y)
result: four plots are produced and a matrix with the resulting principal components is returned

----------------------------------------------------------------------------

'lasso_':

description: It performs lasso regression on the given dataset based on the given alpha hyperparameter, with tol = 0.001 and max_iterations = 5000 and all the other parameters set to default case. If zero features are returned then one of the features is picked randomly.

-inputs: 3 --> (matrix X (samples x features), targets y (numerical), alpha)
-outputs: 1--> (a boolean array indicating which features to be kept)

example: feat_selec = (X_FS, y_FS, alpha = 0.01)
result: a boolean array is returned from the current trial

----------------------------------------------------------------------------

'config':

description: It is the main algorithm that trains all the configurations and does the evaluation

-inputs: 7 --> (matrix X (samples x features), targets y (numerical), C (a list with the hyperparameters), gamma (a list with the hyperparameters), max_depth (a list with the hyperparameters), estimators (a list with the hyperparameters), alpha (a list with the hyperparameters))
-outputs: 2 --> (SVM hyperparameters, metrics of evaluation, trained models of each configuration in a nested list, the same for Random forest)

example: SVM, RF = config(X,y,C,gamma,max_depth,estimators,alpha)
result: two nested list for each classifier containing the evaluation (Accuracy, F1, ROC-AUC) of each configuration

----------------------------------------------------------------------------

'model_sel':

description: It checks which classifier accomplished the highest score based on the three metrics. Then returns the model and the hyperparameters that accomplished the highest AUC-ROC and trains it on all the dataset.

-inputs: 4 --> (matrix X (samples x features), targets y (numerical), SVM nested list with the results of all the configurations, RF nested list with the results of all the configurations)
-outputs: 2 --> (Final model trained on all data, selected features)

example: Final_model, features = model_sel(X, y, SVM, RF)
result: it returnes the final model and a boolean list indicating the selected features



--------
Results
--------
The best model is determined by the highest scores for each one of the three metrics. So the returned classifier may accηιωε two of the three highest metrics and not all of them. Finally the selected configuration is returned based on the ROC-AUC score (in this case the same configuration achieves the highest of all scores). 

---------------------------------------------------------

for the default run: 

Elapsed time: 3271.070493221283 seconds (for 200 configurations to be trained and tested)

The best model that accomplished the highest score based on the three metrics, is SVM with c: 100.0 and gamma: 0.01 . Also, alpha is equal to: 0.00031622776601683794
For the selected configuration, F1 score is equal to: 0.5414978354978356 and Accuracy is equal to: 0.5819047619047619.
The highest returned accuracy is equal to: 0.5819047619047619 , the highest F1 score: 0.5414978354978356 and the highest roc-auc score: 0.5683333333333334.  


SVM best model based on AUC-ROC, Accuracy and F1

(100.0, #C
 0.01, #gamma
 SVC(C=100.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
     decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
     max_iter=-1, probability=False, random_state=None, shrinking=True,
     tol=0.001, verbose=False),
 0.5819047619047619, #Accuracy
 0.5414978354978356, #F1
 0.5683333333333334, #AUC-ROC
 0.00031622776601683794 #Alpha )


RF best model based on AUC-ROC, Accuracy and F1

(20, #max depth
 200, #number of estimators
 RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                        criterion='gini', max_depth=20, max_features='auto',
                        max_leaf_nodes=None, max_samples=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=1, min_samples_split=2,
                        min_weight_fraction_leaf=0.0, n_estimators=200,
                        n_jobs=None, oob_score=False, random_state=0, verbose=0,
                        warm_start=False),
 0.5761904761904761, # Accuracy
 0.533079365079365, #F1
 0.5716666666666667, #AUC-ROC
 0.0031622776601683794 #alpha)


-------------------------------------------------------------------

To run the same 'experiment' you should only execute the below command:

SVM, RF, Final_model, genes_selec = pipeline('GDS4523', download = 'no') if you already have the dataset 
and SVM, RF, Final_model, genes_selec = pipeline('GDS4523', download = 'yes') to download the dataset. 

The run will take approximately 45 minutes and the hyperparameters are defined in the function 'pipeline':

    C = [1e-3,1e-2, 1, 1e2,1e3] #SVM
    gamma = [1e-2,1e-1, 1, 1e1,1e2]
    max_depth = [20, 40, 60, 80, None] #Random forest 
    estimators = [100, 200, 300, 400, 500]
    alpha = np.logspace(-4, -0.5, 8) # Lasso

At total 200 configurations are tested. 







