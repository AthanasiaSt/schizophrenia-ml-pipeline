import subprocess
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns 
import operator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.feature_selection import  SelectFromModel
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import time
import random


def download_data(filename, download): #insert the name of the dataset according to ID GDS 
    #-------------dowload and unzip the dataset----------
    if download == 'yes':
       print('Dowloading the dataset')
       os.system(('wget ftp://ftp.ncbi.nlm.nih.gov/geo/datasets/GDS4nnn/{i}/soft/{i}.soft.gz').format(i = filename))
    subprocess.call(["gunzip", "{i}.soft.gz".format(i = filename)])
    print('File unzipped')

    return "{i}.soft".format(i = filename)


def clean_data(file_name, ID):
    os.system(('grep \'^[^!#^]\' {file_name} > {ID}.clean').format(file_name=file_name, ID = ID)) #cleaning the dataset 
    print('File {ID}.clean ready'.format(ID=ID))

    file_clean = '{ID}.clean'.format(ID=ID)
    return file_clean


def target_extraction(file_soft, file_clean):
    #-----initiating some lists-----
    
    samples_info = {}
    target_disease=[]
    target_sex=[]
    target_age=[]
    targets = ['Disease', 'Age', 'Sex']
    
    #--------Exctracting the targets based on the disease, the sex and the age--------
    
    with open("GDS4523.soft") as f:
        for line in f:
            if line.startswith("!dataset_table_begin"):
                break
            elif line.startswith("!subset_description"):
                subset_description = line.split("=")[1].strip()
    
            elif line.startswith("!subset_sample_id"):
                subset_ids = line.split("=")[1].split(",")
                subset_ids = [x.strip() for x in subset_ids]
    
                for k in subset_ids:
                    samples_info[k] = subset_description
                    if samples_info[k]  == 'schizophrenia' or samples_info[k]  == 'control' or samples_info[k]  == 'n/a': 
                        target_disease.append((k,samples_info[k]))
    
                    elif samples_info[k]  == 'male' or samples_info[k]  == 'female': 
                        target_sex.append((k,samples_info[k]))
                    else:
                        target_age.append((k,samples_info[k]))
    
    target_disease = sorted(target_disease, key=lambda tup: tup[0])
    target_disease = list(zip(*target_disease)) #targets of disease
    target_sex = sorted(target_sex, key=lambda tup: tup[0])
    target_sex = list(zip(*target_sex)) #targets of sex
    target_age = sorted(target_age, key=lambda tup: tup[0])
    target_age = list(zip(*target_age)) #targets of age
    
    #-------loading data as a Dataframe---------
    df_raw = pd.read_csv("GDS4523.clean", delimiter='\t')
    gene_names = df_raw['IDENTIFIER'] #store the gene-feature names 

    return target_disease, target_sex, target_age, df_raw, gene_names


#######-----making of matrix --------------

def matrix(df, target): #making the dataframe into matrix and the targets numerical
            
    df = df.drop(['ID_REF', 'IDENTIFIER'], axis=1) 
    
    df = df.T #samples X features or genes
    df = df.reindex(target[0]) #reindex the samples based on the order of the targets
    
    N = df.shape[0] #dimensions of dataset
    D=df.shape[1]
    
    print('\n The dataset contains',N,'samples and',D,'gene measurements\n')
    
    X = df.values #matrix
    
    y_disease=[] #labels in numerical form
    for i in target[1]:
           if i == 'schizophrenia':
               y_disease.append(1)
           else:
               y_disease.append(0)
    
    y = np.array(y_disease)
    
    return X, y 


##------Preprocessing-------------

def Prepro(X, imputator, std_scaler, fit): #Dealing with missing data and Standard scaling
          
    #------dealing with the missing data---------
    
    if fit == 'yes':
        # print('--------imputing missing values----------\n')
        imp = SimpleImputer(missing_values=np.nan, strategy='mean') #fill the missing data with 
        imp.fit(X)                                                 #the mean of each column (feature)
        X = imp.transform(X)
        
        #-------Standardize------
        # print('-----------Standard scaling--------------\n')
        std_scl = StandardScaler()
        std_scl.fit(X)
        X = std_scl.transform(X)
    else:
        X = imputator.transform(X)
        X = std_scaler.transform(X)

        imp = imputator
        std_scl = std_scaler
        
    return X, imp, std_scl
    

##----unsupervised techniques---------
def pca_kmeans(X,y): #pca and kmeans 
    pcs=[]
    
    N = X.shape[0] #dimensions of dataset
    D=X.shape[1]
    
    #----------making the labels-------------
    Targets=[] #labels
    for i in y:
       if i == 0:
           Targets.append('Control')
       else:
           Targets.append('schizophrenia')
       
    # #--------plotting------------
    sns.set()
    plt.figure()
      
    sns.set_palette('colorblind')
    sns_plot=sns.scatterplot(x=X[:,0], y=X[:,1], hue=Targets).set_title('The first two dimensions of the standardized data')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    
    # fig = sns_plot.get_figure()
    # fig.savefig('1_age_STDData.png', dpi=300)
    #-------------------------------------
    
    #-----------PCA------------
    
    #-----find the number of PCs that return approximately 0.9% of the total variance -----------
    print('-----------Conducting Conventional PCA----------------')
    
    #-----deciding the number of components based on the variance explained
    i=N
    while i != 0 :
        
        pca = PCA(n_components=i)
        pca.fit(X)
        principalComponents = pca.transform(X)
    
        explained_var = pca.explained_variance_ratio_
        variance = np.sum(explained_var)
        
        if variance <= 0.9:
            components=i
            break 
        
        i = i-1
        
    print('\n',components,'PCs were kept, explaining',variance,'% of the total variance\n')
    
    ##--------PCA plot----------
    sns.set()
    plt.figure()
        
    sns.set_palette('colorblind')
    sns_plot=sns.scatterplot(x=principalComponents[:,0], y=principalComponents[:,1], hue=Targets).set_title('The first two principal components')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    
    # fig = sns_plot.get_figure()
    # fig.savefig('3_disease_PCA.png', dpi=300)
    
    #-------making a dataframe with the principal components---------------
    for i in range (1,components+1):
        pcs.append('PC'+str(i))
        
    principalDF = pd.DataFrame(data = principalComponents, columns = pcs)    
        
    #----------------Kmeans---------------------------
    #-------find the optimal number of clusters-------
    print('------------Conducting Kmeans------------')
    silhouette_scores=[]
    n_clust = []
    for n_clusters in list(range(2,int(N/2))):
        
        clusterer = KMeans(n_clusters=n_clusters)
        preds = clusterer.fit_predict(principalComponents)
        # centers = clusterer.cluster_centers_
    
        score = silhouette_score(X, preds)
        #print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))
        silhouette_scores.append(score)
        n_clust += [n_clusters]
    
    max_index, max_value = max(enumerate(silhouette_scores), key=operator.itemgetter(1))
    
    # print(silhouette_scores) #Optimal --> 2 clusters
    print('\n The max silhouette score is accomplished for', n_clust[max_index],'clusters and is equal to',max_value)
    
    #-------plotting the Sihlouette values--------------------
    sns.set()
    fig_sil = plt.figure(figsize = (7,6))
    ax_sil = fig_sil.add_subplot(1,1,1)
    
    ax_sil.plot(range(2,int(N/2)), silhouette_scores, marker = 'o', c = "black")
    ax_sil.set_title('Silhouette values vs number of clusters')
    ax_sil.set_xlabel('Number of Clusters')
    ax_sil.set_ylabel('Silhouette score')
    plt.show()
    
    # fig = ax_sil.get_figure()
    # fig.savefig('4_silhouette.png', dpi=300)
    
    #-------------Κmeans with the best number of clusters---------------
    
    y_pred=KMeans(n_clusters=n_clust[max_index], random_state=0).fit_predict(principalComponents)
    
    #------------plotting----------------------------------
    sns.set()
    fig1 = plt.figure(figsize = (6,5))   
    ax1 = fig1.add_subplot(1,1,1)
    ax1.set_xlabel('Component 1', fontsize = 15)
    ax1.set_ylabel('Component 2', fontsize = 15)
    ax1.set_title('Kmeans Clustering', fontsize = 20)
    
    target_names = ["Cluster " + str(s+1) for s in list(range(n_clust[max_index]))]
    colors = ["r", "g", "b", "c", "y", "m", "k", "o"]
    for c, i, target_name in zip(colors, list(range(n_clust[max_index])), target_names):
        ax1.scatter(principalComponents[y_pred==i,0], principalComponents[y_pred==i,1], c=c, label=target_name, alpha = 0.8, s=15)
    plt.legend()
    plt.show()
    
    # fig = ax1.get_figure()
    # fig.savefig('5_K-means.png', dpi=300)
        
    X = principalDF.values #return a matrix with the principal components

    return X


#----------Feature selection, Lasso-------------
def lasso_(X,y,alpha):
  
    lasso = Lasso()
        
    #conducting Lasso using the alpha of the current configuration
    lass = Lasso(alpha=alpha, max_iter = 5000, tol = 0.001) #increased max_iter and tol because of 
    lass.fit(X,y)                                      # a difficulty in convergence
    
    sel=SelectFromModel(lass)
    sel.fit(X,y)
    bool_select = sel.get_support() #boolean for the selected features
    
    #in case no features are returned, then pick one randomly 
    if np.sum(bool_select) == 0:
        print('\nLasso returned zero features\n')
        num = random.randint(0, len(bool_select))
        bool_select[num] = True
        
    # print('The number of selected features (Lasso) is:',np.sum(bool_select),'\n')
    return bool_select


#-------Testing all the configurations-----------
#testing only two hyperparameters for each classifier (Random forest and SVM)
#and testing the aplha hyperparameter of lasso

def config(X, y, C, gamma, max_depth, estimators, alpha):
            
    hyper_1 = [] #nested lists of the hyperparameters of the classifiers
    hyper_2 = []
    hyper_1.extend([list(i) for i in zip(C,max_depth)]) 
    hyper_2.extend([list(i) for i in zip(gamma,estimators)])
    
    classifiers_svm = [] 
    classifiers_RF = []

    start = time.time() #measuring time

    # For all the configurations manually------
    for c, depth in hyper_1:
        for Gamma, estim in hyper_2:
            for Alpha in alpha:
    
                print('\n---------New combination of parameters------')
                
                #initialization of the imputator and standard scaler
                imputator=SimpleImputer(missing_values=np.nan, strategy='mean')
                std_scaler=StandardScaler()
                
                #defining the models of the current configuration
                clf = SVC(C=c, gamma=Gamma) #svm 'Rbf'
                clf_RF = RandomForestClassifier(max_depth=depth, n_estimators=estim, random_state=0) #random forest
    
                #divide the dataset for feature selection and classification
                X_CL, X_FS, y_CL, y_FS = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=0, stratify=y) 
                         
                #preprocessing before feature selection
                X_FS, imp, std_scl = Prepro(X_FS, imputator, std_scaler, fit='yes')
                                                                                                 
                #feature selection, Lasso
                feat_selec = lasso_(X_FS,y_FS, Alpha) #returns a boolean array
                  
                #Conducting Repeated stratified Kfold Cross validation for each configuration
                # Number of trials
                NUM_TRIALS = 5
                
                #initiating lists to store the models and the evalulation scores
                mean_accu_cv = []
                mean_F1_cv = []
                mean_roc_auc_cv=[]
                accuracy_RF_mean_cv = []
                F1_RF_mean_cv = []
                roc_auc_RF_mean_cv = []
            
                for i in range(NUM_TRIALS):
            
                    #stratified kfold splits 
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
                    
                    mean_accu=[]
                    F1_mean=[]
                    roc_auc_mean=[]
                    accuracy_RF_mean=[]
                    F1_RF_mean=[]
                    roc_auc_RF_mean=[]
                    score = []
                    
                    for train_index, test_index in skf.split(X_CL, y_CL):
                        
                        X_train, X_test = X_CL[train_index], X_CL[test_index]
                        y_train, y_test = y_CL[train_index], y_CL[test_index]
                    
                        # #--------Preprocessing, separately for train and test set
                        X_train, imp, std_scl = Prepro(X_train, imputator, std_scaler, fit='yes') #fit and transform
                        X_test, imp, std_scl = Prepro(X_test,imp, std_scl,fit='no') #only transform
                    
                        #---------Training of the classifiers using only the selected features
                        #svm
                        clf.fit(X_train[:,feat_selec], y_train)
                        y_pred = clf.predict(X_test[:,feat_selec])
                        
                        #store the metrics, accuracy, F1 score, roc_auc score for SVM
                        mean_accu += [accuracy_score(y_test, y_pred)]
                        F1_mean += [f1_score(y_test, y_pred, average='macro')]
                        roc_auc_mean += [roc_auc_score(y_test, y_pred)]
                        
                        #Random forest
                        clf_RF.fit(X_train[:,feat_selec], y_train)  
                        y_pred = clf_RF.predict(X_test[:,feat_selec])
                        #store the metrics, accuracy, F1 score, roc_auc score for RF
                        accuracy_RF_mean += [accuracy_score(y_test, y_pred)]
                        F1_RF_mean += [f1_score(y_test, y_pred, average='macro')]
                        roc_auc_RF_mean += [roc_auc_score(y_test, y_pred)]
                        
                    
                    #compute the mean_metrics from cross validation
                    #svm
                    mean_accu_cv +=[sum(mean_accu)/len(mean_accu)]
                    mean_F1_cv +=[sum(F1_mean)/len(F1_mean)]
                    mean_roc_auc_cv +=[sum(roc_auc_mean)/len(roc_auc_mean)]
                    
                    #random forest
                    accuracy_RF_mean_cv += [sum(accuracy_RF_mean)/len(accuracy_RF_mean)]
                    F1_RF_mean_cv += [sum(F1_RF_mean)/len(F1_RF_mean)]
                    roc_auc_RF_mean_cv +=  [sum(roc_auc_RF_mean)/len(roc_auc_RF_mean)]
                
                    print('------------',i ,'th round of Ntrials-------------')
                 
                #store the metrics, the hyperparameters and the models in nested lists
                classifiers_svm.append((c, Gamma, clf, sum(mean_accu_cv)/NUM_TRIALS,sum(mean_F1_cv)/NUM_TRIALS,sum(mean_roc_auc_cv)/NUM_TRIALS, Alpha, np.sum(feat_selec)))
                classifiers_RF.append((depth, estim, clf_RF, sum(accuracy_RF_mean_cv)/NUM_TRIALS,sum(F1_RF_mean_cv)/NUM_TRIALS,sum(roc_auc_RF_mean_cv)/NUM_TRIALS, Alpha))
       
    end = time.time()
    print('Elapsed time:',end - start,'seconds')
       
    return  classifiers_svm, classifiers_RF #return the nested lists with the results


#---------Model selection------------
def model_sel(X, y, classifiers_svm, classifiers_RF):
    #initialization of the imputator and standard scaler
    imputator=SimpleImputer(missing_values=np.nan, strategy='mean')
    std_scaler=StandardScaler()
    
    #find the best configuration of each classifier according to each metric
    max_Accu_SVM = max(classifiers_svm, key=lambda x: x[3])
    max_F1_SVM = max(classifiers_svm, key=lambda x: x[4])
    max_AUCROC_SVM = max(classifiers_svm, key=lambda x: x[5])
    
    max_Accu_RF = max(classifiers_RF, key=lambda x: x[3])
    max_F1_RF = max(classifiers_RF, key=lambda x: x[4])
    max_AUCROC_RF = max(classifiers_RF, key=lambda x: x[5])
    
    #return the classifier that had the highest score and highest scores 
    #in the three metrics
    score_SVM = 0
    score_RF = 0
    
    if max_Accu_SVM[3] > max_Accu_RF[3]:
        score_SVM +=1
        score_RF = score_RF -1
    else:
        score_SVM = score_SVM -1
        score_RF +=1
        
        
    if max_F1_SVM[4] > max_F1_RF[4] :
        score_SVM +=1
        score_RF = score_RF -1
    else:
        score_SVM = score_SVM -1
        score_RF +=1
        
    if max_AUCROC_SVM[5] > max_AUCROC_RF[5]:
        score_SVM +=1
        score_RF = score_RF -1
    else:
        score_SVM = score_SVM -1
        score_RF +=1
        
    if score_SVM > score_RF :
        
        #return the best model based on roc-auc score
        model = max_AUCROC_SVM[2]
        C = max_AUCROC_SVM[0]
        gamma = max_AUCROC_SVM[1]
        Alpha = max_AUCROC_SVM[6]
        
        print('\nThe best model, based on the highest ROC-AUC score, is SVM with c:',C,'and gamma:',gamma,'. Also, alpha is equal to:',Alpha)
        print('For the selected configuration, F1 score is equal to:',max_AUCROC_SVM[4],'and accuracy is equal to:',max_AUCROC_SVM[3])
        print('The highest returned accuracy is equal to:',max_Accu_SVM[3],', the highest F1 score:',max_F1_SVM[4],', and the highest roc-auc score:',max_AUCROC_SVM[5],'\n')
    
    else:
        
        model = max_AUCROC_RF[2]
        max_depth = max_AUCROC_RF[0]
        n_estim = max_AUCROC_RF[1]
        Alpha = max_AUCROC_RF[6]
        
        print('\nThe best model, based on the highest ROC-AUC score, is Random forest with max_depth:',max_depth,'and number of estimators:',n_estim,'. Also, alpha is equal to:',Alpha)
        print('For the selected configurations F1 score is equal to:',max_F1_RF[4],'and accuracy is equal to:',max_Accu_RF[3])
        print('The highest returned accuracy is equal to:',max_Accu_RF[3],', the highest F1 score:',max_F1_RF[4],', the highest roc-auc score:',max_AUCROC_RF[5],'\n')
    
    
    #Train the final model on all data
    
    #-----preprocessing
    X,imp,std_scl = Prepro(X, imputator,std_scaler, fit='yes')
    feat_selec= lasso_(X,y, Alpha) #lasso with the selected alpha
    model.fit(X[:,feat_selec], y) #train the selected model
    
    return model, feat_selec
    
    
def pipeline(GDS_ID, download):
    #initialization of the imputator and standard scaler
    imputator=SimpleImputer(missing_values=np.nan, strategy='mean')
    std_scaler=StandardScaler()
    
    #hyperparameters
    C = [1e-3,1e-2, 1, 1e2,1e3] #SVM
    gamma = [1e-2,1e-1, 1, 1e1,1e2]
    max_depth = [20, 40, 60, 80, None] #Random forest 
    estimators = [100, 200, 300, 400, 500]
    alpha = np.logspace(-4, -0.5, 8) # Lasso
    
    #downloading and unzipping the dataset
    file_soft =  download_data(GDS_ID, download)
    file_clean = clean_data(file_soft, GDS_ID)
    target_disease, target_sex, target_age, df_raw, gene_names = target_extraction(file_soft,file_clean)

    #visualization
    X,y = matrix(df_raw, target_disease)
    X, imp, std_scl = Prepro(X,imputator,std_scaler,fit='yes')
    X_pca = pca_kmeans(X,y)
    
    #model selection
    X,y = matrix(df_raw, target_disease)
    SVM, RF = config(X,y, C,gamma,max_depth,estimators,alpha)
    Final_model, features = model_sel(X, y, SVM, RF)
    genes = gene_names[features] 
    
    return SVM, RF, Final_model, genes
    
SVM, RF, Final_model, genes_selec = pipeline('GDS4523', download = 'no')


###########################################
#RESULTS


# Elapsed time: 3271.070493221283 seconds (for 200 configurations to be trained and tested)

# The best model that accomplished the highest score based on the three metrics, is SVM with c: 100.0 and gamma: 0.01 . Also, alpha is equal to: 0.00031622776601683794
# For the selected configuration, F1 score is equal to: 0.5414978354978356 and Accuracy is equal to: 0.5819047619047619.
# The highest returned accuracy is equal to: 0.5819047619047619 , the highest F1 score: 0.5414978354978356 and the highest roc-auc score: 0.5683333333333334.  


# SVM best model based on AUC-ROC, Accuracy and F1

# (100.0, #C
#  0.01, #gamma
#  SVC(C=100.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
#      decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
#      max_iter=-1, probability=False, random_state=None, shrinking=True,
#      tol=0.001, verbose=False),
#  0.5819047619047619, #Accuracy
#  0.5414978354978356, #F1
#  0.5683333333333334, #AUC-ROC
#  0.00031622776601683794 #Alpha )


# RF best model based on AUC-ROC, Accuracy and F1

# (20, #max depth
#  200, #number of estimators
#  RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
#                         criterion='gini', max_depth=20, max_features='auto',
#                         max_leaf_nodes=None, max_samples=None,
#                         min_impurity_decrease=0.0, min_impurity_split=None,
#                         min_samples_leaf=1, min_samples_split=2,
#                         min_weight_fraction_leaf=0.0, n_estimators=200,
#                         n_jobs=None, oob_score=False, random_state=0, verbose=0,
#                         warm_start=False),
#  0.5761904761904761, # Accuracy
#  0.533079365079365, #F1
#  0.5716666666666667, #AUC-ROC
#  0.0031622776601683794 #alpha)
