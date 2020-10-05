################################################################################
"""
This is a Programming Project for Affective, Social Neuroscience (SoSe 2020).
This module provides the predictive modeling for level of immersion based on
 valence and arousal using the dataset:
<https://box.fu-berlin.de/s/2bP2cdDaeefBy2n >

Those are requirement to run this module:
    - python
    - keras
    - matplotlib
    - numpy
    - pandas
    - seaborn
    - statsmodels
    - sklearn
    - google
    - xgboost

This modules consists of four main part:
    - PART I: Group and Data Selection
    - PART II: Data Preprocessing
    - PART III: Predictive Modeling
    - PART IV: Statistical Analysis with BIG FIVE

We are highly recommend to use the google colab using GPU.

Authors: Andreea Al-Afuni, JiHoon Kim, Angela Sofia Royo Romero, and Bati Yilmaz
Last-modified : 4th October 2020
"""
################################################################################# 0. Import the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from google.colab import drive                                                  
from keras.models import Model, Sequential
from keras import models
from keras import layers
from keras import Input
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn import svm, linear_model, metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
sns.set(style="darkgrid", color_codes=True)
np.random.seed(20201005)                                                        # To ensure the reproducibility of the programm
################################################################################ PART I: Group and Data Selection
                                                                                # 1. Load the dataset: SENT_GROUP_INFO and SENT_RATING_DATA
drive.mount('/gdrive', force_remount=True)                                      # Mount the files on the google drive to be used with google colab

sg_index = "B:E,AE:AI,CW:DJ,DM,FN"                                              # Select the necessary dataset based on the column indices

s_grp_dir = '/gdrive/My Drive/SCAN_seminar_data/SENT_GROUP_INFO.xlsx'           # Load the dataset SENT_GROUP_INFO 
s_group = pd.read_excel(s_grp_dir, usecols = sg_index)                          # Only loading columns B to E (Case, Text, Condition, Language)
                                                                                #                      AE to AI (BFI scores)
                                                                                #                      CW to DJ (Reading experience ratings)
                                                                                #                      DM, FN (Attention check, Minus points for fast completion = DEG Time)
s_rat_dir = '/gdrive/My Drive/SCAN_seminar_data/SENT_RATING_DATA.xlsx'          # Load the dataset: SENT_RATING_DATA (all columns)
s_rating = pd.read_excel(s_rat_dir)
################################################################################# 2. Group the Data: Coherent ENG+GER, HARRY AND PIPPI

                                                                                # SENT_GROUP_INFO.xlsx
sg_harry = s_group['TEXT'] == 'HARRY'                                           # return true only for HARRY
sg_pippi = s_group['TEXT'] == 'PIPPI'                                           # return true only for PIPPI
sg_coherent = s_group['CONDITION'] == 'COHERENT'                                # return true only for COHERENT
sg_scrambled = s_group['CONDITION'] == 'SCRAMBLED'                              # return true only for SCRAMBLED
sg_eng = s_group['QESTN_LANGUAGE'] == 'ENG'                                     # return true only for ENG
sg_ger = s_group['QESTN_LANGUAGE'] == 'GER'                                     # return true only for GER

                                                                                # SENT_RATING_DATA.xlsx
sr_harry = s_rating['TEXT'] == 'HARRY'                                          # retrun true only for HARRY
sr_pippi = s_rating['TEXT'] == 'PIPPI'                                          # retrun true only for PIPPI
sr_coherent = s_rating['CONDITION'] == 'COHERENT'                               # retrun true only for COHERENT
sr_scrambled = s_rating['CONDITION'] == 'SCRAMBLED'                             # retrun true only for SCRAMBLED
sr_eng = s_rating['LANGUAGE'] == 'ENG'                                          # retrun true only for ENG
sr_ger = s_rating['LANGUAGE'] == 'GER'                                          # retrun true only for GER

                                                                                # Data filtering in the certain conditions
sg_co_harry = s_group[sg_harry & sg_coherent]                                   # Text: HARRY; Condition: COHERENT
sg_sc_harry = s_group[sg_harry & sg_scrambled]                                  # Text: HARRY; Condition: SCRAMBLED
sg_co_pippi = s_group[sg_pippi & sg_coherent]                                   # Text: PIPPI; Condition: COHERENT
sg_sc_pippi = s_group[sg_pippi & sg_scrambled]                                  # Text: PIPPI; Condition: SCRAMBLED

categories = ['(29) SC_HARRY','(26) CO_HARRY','CO_PIPPI (21)','SC_PIPPI (21)']  # Plot a pie chart of the distribution of the whole dataset (N=97)
sizes = [len(sg_sc_harry),len(sg_co_harry),len(sg_sc_pippi),len(sg_sc_pippi)]
colors = ['#ff9999', '#ffc000', '#8fd9b6', '#d395d0']
wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 5}

fig, ax = plt.subplots()
ax.pie(sizes,labels=categories,autopct='%1.1f%%',startangle=255,
       colors=colors,wedgeprops=wedgeprops)
plt.title('Dataset (N=97)')
plt.show()
################################################################################# 3. Data Selection (Exclude the "bad data") and visualisation
sg_co = s_group[sg_coherent]
sr_co = s_rating[sr_coherent]
                                                                                ### Selecting the cases with "bad data"

                                                                                #Criteria that renders a case as "bad data" are:
b_DEG = sg_co.loc[sg_co['DEG_TIME']>100]['CASE']                                #       1. DEG_TIME > 100 (completion of the survey was too fast)

b_ATT = sg_co.loc[sg_co['ATTENTION_CHECKS_COUNT_WRONG_ANSWERS']>0]['CASE']      #       2. ATTENTION_CHECKS_COUNT_WRONG_ANSWERS > 0 (clicking random answeres)

b_PAG = sr_co.loc[sr_co['PAGE_TIME']>999]['CASE']                               #       3. PAGE_TIME > 999 (pausing while completing the survey, can interfere with immersion)

b_list = pd.concat([b_DEG, b_ATT, b_PAG], axis = 0)                             # Concatenate the "bad data" described above into a single list

b_list = b_list.drop_duplicates()                                               # Delete the duplicates
b_case = list(b_list.values)                                                    # Make of list of the cases considered to be "bad data" (list of case number)
print('bad data case list:',b_case)

                                                                                
                                                                                ### Excluding the "bad data" cases from the dataset

a_group = sg_co.copy()                                                          # Copy the SENT_GROUP_INFO.coherent
a_rating = sr_co.copy()                                                         # Copy the # SENT_RATING_DATA.coherent

for i in b_case:                                                                # Exclude the bad case from the coherent dataset
    a_rating = a_rating.drop(a_rating[a_rating['CASE'] == i].index)
    a_group = a_group.drop(a_group[a_group['CASE'] == i].index)

sg_co_harry = a_group[sg_harry & sg_coherent]                                   # Text: HARRY; Condition: COHERENT, bad data excluded
sg_sc_harry = a_group[sg_harry & sg_scrambled]                                  # Text: HARRY; Condition: SCRAMBLED, bad data excluded
sg_co_pippi = a_group[sg_pippi & sg_coherent]                                   # Text: PIPPI; Condition: COHERENT, bad data excluded
sg_sc_pippi = a_group[sg_pippi & sg_scrambled]                                  # Text: PIPPI; Condition: SCRAMBLED, bad data excluded
sg_co_en_pippi = a_group[sg_pippi & sg_coherent & sg_eng]                       # Text: PIPPI; Condition: COHERENT; Language: ENG
sg_co_ge_pippi = a_group[sg_pippi & sg_coherent & sg_ger]                       # Text: PIPPI; Condition: COHERENT; Language: GER

categories = ['CO_PIPPI (17)','CO_HARRY (24)']                                  # Plot the pie chart of the distribution of the COHERENT dataset (N=41)
                                                                                # Two pie chart slices:                                                                               
sizes = [len(sg_co_pippi),len(sg_co_harry)]                                     #    Text: PIPPI; Condition: COHERENT
colors = ['#ff9999', '#ffc000']                                                 #    Text: HARRY; Condition: COHERENT
wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 5}

fig, ax = plt.subplots()
ax.pie(sizes,labels=categories,autopct='%1.1f%%',startangle=105,
       colors=colors,wedgeprops=wedgeprops)
plt.title('Coherent Dataset (N=41)')                                            
plt.show()                                                                     


categories = ['CO_ENG_PIPPI (12)','CO_GER_PIPPI (5)','(24) CO_HARRY']           # Plot the pie chart of the distribution of the COHERENT dataset (N=41)
sizes = [len(sg_co_en_pippi),len(sg_co_ge_pippi),len(sg_co_harry)]              # Three pie chart slices: 
colors = ['#ff9999', '#ffc000','#d395d0']                                       #    Text: HARRY; Condition: COHERENT 
wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 5}                     #    Text: PIPPI; Condition: COHERENT; Language: ENG
                                                                                #    Text: PIPPI; Condition: COHERENT; Language: GER
fig, ax = plt.subplots()
ax.pie(sizes,labels=categories,autopct='%1.1f%%',startangle=105,                # 
       colors=colors,wedgeprops=wedgeprops)                                     # 
plt.title('Selected Dataset (N=41)')                                            #
plt.show()                                                                      #
################################################################################# 4. Set the data with Condition: COHERENT only
                                                                                # Loading data from SENT_RATING file for:
ar_co_harry = a_rating[sr_harry & sr_coherent]                                  #   Text: HARRY; Condition: COHERENT
ar_co_pippi = a_rating[sr_pippi & sr_coherent & sr_eng]                         #   Text: PIPPI; Condition: COHERENT; Language: ENG

                                                                                #  Loading data from SENT_GROUP_INFO file for:
ag_co_harry = a_group[sg_harry & sg_coherent]                                   #   Text: HARRY; Condition: COHERENT
ag_co_pippi = a_group[sg_pippi & sg_coherent & sg_eng]                          #   Text: PIPPI; Condition: COHERENT; Language: ENG

################################################################################# 5. drop out the NaN column from the data from SENT_GROUP_INFO file
gh = ag_co_harry                                                                # Text: HARRY; Condition: COHERENT
gp = ag_co_pippi                                                                # Text: PIPPI; Condition: COHERENT; Language: ENG
                                                                                # Charachter names: legend
rh = ar_co_harry.loc[:,['CASE','AROUSAL_RATING','VALENCE_RATING']]              # Text: HARRY; Condition: COHERENT in SENT_RATING_DATA
                                                                                # rh = ratings harry from file SENT_RATING
                                                                                #      contains: CASE, AROUSAL ratings, VALENCE ratings for Text: HARRY; Condition: COHERENT
rp = ar_co_pippi.loc[:,['CASE','AROUSAL_RATING','VALENCE_RATING']]              # Text: PIPPI; Condition: COHERENT; Language: ENG in SENT_RAITING_DATA
rp = rp.dropna(axis=0)                                                          # drop out NaN row
                                                                                # rp = ratings pippi from file SENT_RATING
                                                                                #      contains: CASE, AROUSAL ratings, valence ratings for Text: PIPPI; Condition: COHERENT; Language: ENG
print('HARRY')
print(rh.isnull().sum(), 'NaN')                                                 # check the NaN
print('PIPPI')
print(rp.isnull().sum(), 'NaN')                                                 # check the NaN

print("Coherent demographic informations: ")                                    # Demographic info on the countable columns
print(ag_co_harry.describe())
print(ar_co_harry.describe())

plt.title('DEG_TIME and CASE (N=41)')                                           # Plot the DEG_TIME
plt.scatter(ag_co_harry['CASE'], ag_co_harry['DEG_TIME'], c='orange')
plt.scatter(ag_co_pippi['CASE'], ag_co_pippi['DEG_TIME'], c='purple')
plt.plot(sg_co['CASE'], sg_co['DEG_TIME'])
plt.show()

print('b_DEG', list(b_DEG))                                                     # Print the case of DEG_TIME

plt.title('ATTENTION_COUNT_WRONG_ANSWERS and CASE (N=41)')                      # Plot the ATTENTION_COUNT_WRONG_ANSWERS
plt.scatter(ag_co_harry['CASE'],ag_co_harry['ATTENTION_CHECKS_COUNT_WRONG_ANSWERS'],c='orange')
plt.scatter(ag_co_pippi['CASE'],ag_co_pippi['ATTENTION_CHECKS_COUNT_WRONG_ANSWERS'],c='purple')
plt.plot(sg_co['CASE'],sg_co['ATTENTION_CHECKS_COUNT_WRONG_ANSWERS'])
plt.show()
print('b_ATT', list(b_ATT))                                                     # Print the case of ATTENTION_CHECKS_COUNT_WRONG_ANSWERS

plt.title('PAGE_TIME and CASE (N=41)')                                          # Plot the PAGE_TIME
plt.scatter(ar_co_harry['CASE'], ar_co_harry['PAGE_TIME'], c='orange')
plt.scatter(ar_co_pippi['CASE'], ar_co_pippi['PAGE_TIME'], c='purple')
plt.plot(sr_co['CASE'], sr_co['PAGE_TIME'])
plt.show()
print('b_PAG', list(b_PAG))                                                     # Print the case of PAGE_TIME

s_rating_filtered = rh.filter(["VALENCE_RATING", "AROUSAL_RATING"])             # Plot the AROUSAL_RATING and VALENCE_RATING HARRY distribution
sns.distplot(s_rating_filtered["AROUSAL_RATING"])                               # AROUSAL_RATING in Text: HARRY; Condition: COHERENT

sns.distplot(s_rating_filtered['VALENCE_RATING'], color='orange')               # Plot the VALENCE_RATING in Text: HARRY; Condition: COHERENT

sns.jointplot(x='AROUSAL_RATING',y='VALENCE_RATING',                            # Joint plot of the both VALENCE_RATING and AROUSAL_RATING in HARRY
              data = s_rating_filtered,kind='reg')                              # Text: HARRY; Condition: COHERENT

s_rating_filtered = rp.filter(["VALENCE_RATING", "AROUSAL_RATING"])             # Plot the AROUSAL_RATING and VALENCE_RATING PIPPI distribution
sns.distplot(s_rating_filtered["AROUSAL_RATING"])                               # AROUSAL_RATING in Text: PIPPI; Condition: COHERENT; Language: ENG

sns.distplot(s_rating_filtered['VALENCE_RATING'], color='orange')               # Plot the VALENCEL_RATING in Text: PIPPI; Condition: COHERENT; Language: ENG

sns.jointplot(x='AROUSAL_RATING',y='VALENCE_RATING',                            # Joint plot of the both VALENCE_RATING and AROUSAL_RATING in 
              data = s_rating_filtered,kind='reg',color ='purple')              # Text: PIPPI; Condition: COHERENT; Language: ENG

################################################################################ PART II: Data Preprocessing
                                                                                # 6. Compound the reader_response for immersion
rg_set = pd.concat([gh,gp], axis = 0)                                           # Merge the HARRY and PIPPI dataset
a_group = rg_set.copy()                                                         # Copy the dataset
h_group = gh.copy()
p_group = gp.copy()

## Characters names: legend:
# gh = data from SENT_GROUP_INFO Text: HARRY; Condition: COHERENT
#
# gp = data from SENT_GROUP_INFO Text: PIPPI; Condition: COHERENT; Language: ENG
#
# a_group: reading experience ratings for Text: HARRY; Condition: COHERENT and Text: PIPPI; Condition: COHERENT; Language: ENG
# h_group: reading experience ratings for Text: HARRY; Condition: COHERENT
# p_group: reading experience ratings for Text: PIPPI; Condition: COHERENT; Language: ENG

a_group['IMMERSION'] = 0                                                        # Create the new column of 'IMMERSION'

reader_response = [16,17]                                                       # Select the reader's responses
                                                                                # If you need it then check the columns order, "print(a_group.columns)"
for i in reader_response:                                                       # Sum the reader's response
    a_group['IMMERSION'] += a_group.iloc[:,i]

a_group['IMMERSION']=a_group['IMMERSION']/len(reader_response)                  # Using the arithmetic mean of the reader's response
print(a_group.head(10))
sns.distplot(a_group["IMMERSION"])

h_group['IMMERSION'] = 0                                                        # Create the new column of 'IMMERSION' for 
                                                                                # Text: HARRY, Condition: COHERENT
reader_response = [8,9,10,11,15,16,21]                                          # Select the reader's responses
                                                                                # If you need it then check the columns order, "print(a_group.columns)"
for i in reader_response:                                                       # Sum the reader's response
    h_group['IMMERSION'] += h_group.iloc[:,i]

h_group['IMMERSION']=h_group['IMMERSION']/len(reader_response)                  # Using the arithmetic mean of the reader's response
print(h_group.head(10))
sns.distplot(h_group["IMMERSION"])                                              

p_group['IMMERSION'] = 0                                                        # Create the new column of 'IMMERSION'
                                                                                # For Text: PIppi, Condition: COHERENT, Language: ENG
reader_response = [8,9,10,11,15,16,21]                                          # Select the reader's responses
                                                                                # If you need it then check the columns order, "print(a_group.columns)"
for i in reader_response:                                                       # Sum the reader's response
    p_group['IMMERSION'] += p_group.iloc[:,i]

p_group['IMMERSION']=p_group['IMMERSION']/len(reader_response)                  # Using the arithmetic mean of the reader's response
print(p_group.head(10))
sns.distplot(p_group["IMMERSION"])

################################################################################# 7. Preprocess the dataset
gh_case = list(a_group.loc[a_group['TEXT']=='HARRY'].loc[:,['CASE']]['CASE'])   # HARRY CASE LIST "print(len(gh_case))"
gp_case = list(a_group.loc[a_group['TEXT']=='PIPPI'].loc[:,['CASE']]['CASE'])   # PIPPI CASE LIST "print(len(gp_case))"

tr_hr_case = gh_case[2:22]                                                      # training set for Text: HARRY, Condition: COHERENT
te_hr_case = gh_case[0:2] + gh_case[22:25]                                      # testing set for Text: HARRY, Condition: COHERENT
hr_case = [tr_hr_case, te_hr_case]                                              # training and testing cases for Text: HARRY, Condition: COHERENT

tr_pi_case = gp_case[0:10]                                                      # training set for Text: PIPPI, Condition: COHERENT, Language = ENG
te_pi_case = gp_case[10:14]                                                     # testing set for Text: PIPPI, Condition: COHERENT, Language = ENG
pi_case = [tr_pi_case, te_pi_case]                                              # PIPPI case

rh_set = rh.copy()                                                              # Load the independent variable related dataset
rp_set = rp.copy()

hr_train_arousal = []
hr_train_valence = []
hr_test_arousal = []
hr_test_valence = []

for i in hr_case:                                                               # Text: HARRY
    for j in i:                                                                 # Extract Arousal and Valence
        if i == tr_hr_case:
            set = rh_set[rh_set['CASE']==j]
            #new = set.sort_values(by='SENTENCE_NUMBER')
            hr_train_arousal.append(list(set.loc[:,['AROUSAL_RATING']]['AROUSAL_RATING']))
            hr_train_valence.append(list(set.loc[:,['VALENCE_RATING']]['VALENCE_RATING']))
        else:
            set = rh_set[rh_set['CASE']==j]
            #new = set.sort_values(by='SENTENCE_NUMBER')
            hr_test_arousal.append(list(set.loc[:,['AROUSAL_RATING']]['AROUSAL_RATING']))
            hr_test_valence.append(list(set.loc[:,['VALENCE_RATING']]['VALENCE_RATING']))

pi_train_arousal = []
pi_train_valence = []
pi_test_arousal = []
pi_test_valence = []

for i in pi_case:                                                               # Text: PIPPI
    for j in i:                                                                 # Extract Arousal and Valence
        if i == tr_pi_case:
            set = rp_set[rp_set['CASE']==j]
            #new = set.sort_values(by='SENTENCE_NUMBER')
            pi_train_arousal.append(list(set.loc[:,['AROUSAL_RATING']]['AROUSAL_RATING']))
            pi_train_valence.append(list(set.loc[:,['VALENCE_RATING']]['VALENCE_RATING']))
        else:
            set = rp_set[rp_set['CASE']==j]
            #new = set.sort_values(by='SENTENCE_NUMBER')
            pi_test_arousal.append(list(set.loc[:,['AROUSAL_RATING']]['AROUSAL_RATING']))
            pi_test_valence.append(list(set.loc[:,['VALENCE_RATING']]['VALENCE_RATING']))

print('Coherent HARRY Dataset')                                                 # Set the input data for training set (as x_Train) and testing set (as x_Test)
print('train_arousal:',len(hr_train_arousal),hr_train_arousal)                  # The input data consists of the arounsal and valence ratings
print('train_valence:',len(hr_train_valence),hr_train_valence)
print('test_arousal:',len(hr_test_arousal),hr_test_arousal)
print('test_valence:',len(hr_test_valence),hr_test_valence)

print('Coherent PIPPI Dataset')
print('train_arousal:',len(pi_train_arousal),pi_train_arousal)
print('train_valence:',len(pi_train_valence),pi_train_valence)
print('test_arousal:',len(pi_test_arousal),pi_test_arousal)
print('test_valence:',len(pi_test_valence),pi_test_valence)

################################################################################ Set the output data for the training set (as Y_Train) and test set (as Y_Test)
hr_train_immersion = []                                                         # The output data consists of the Immersion level as calculated above
hr_test_immersion = []

for i in hr_case:                                                               # HARRY IMMERSION DATASET
    for j in i:
        if i == tr_hr_case:
            hr_train_immersion.append(list(a_group[a_group['CASE']==j]['IMMERSION']))
        else:
            hr_test_immersion.append(list(a_group[a_group['CASE']==j]['IMMERSION']))

print('hr_train_immersion',len(hr_train_immersion),hr_train_immersion)
print('hr_test_immersion',len(hr_test_immersion),hr_test_immersion)

# PIPPI IMMERSION DATASET
pi_train_immersion = []
pi_test_immersion = []

for i in pi_case:                                                               # PIPPI IMMERSION DATASET
    for j in i:
        if i == tr_pi_case:
            pi_train_immersion.append(list(a_group[a_group['CASE']==j]['IMMERSION']))
        else:
            pi_test_immersion.append(list(a_group[a_group['CASE']==j]['IMMERSION']))

print('pi_train_immersion',len(pi_train_immersion),pi_train_immersion)
print('pi_test_immersion',len(pi_test_immersion),pi_test_immersion)

################################################################################ Dataset as np.array
print('HARRY Dataset')                                                          # HARRY DATASET
hr_tr_arousal = np.array(hr_train_arousal)
hr_tr_valence = np.array(hr_train_valence)
hr_tr_immersion = np.array(hr_train_immersion)

hr_te_arousal = np.array(hr_test_arousal)
hr_te_valence = np.array(hr_test_valence)
hr_te_immersion = np.array(hr_test_immersion)

print('HARRY Training set')                                                     
print('hr_tr_arousal',hr_tr_arousal.shape,hr_tr_arousal)
print('hr_tr_valence',hr_tr_valence.shape,hr_tr_valence)
print('hr_tr_immersion',hr_tr_immersion.shape,hr_tr_immersion)

print('HARRY Test set')
print(hr_te_arousal)
print(hr_te_valence)
print(hr_te_immersion)

print('PIPPI Dataset')                                                          # PIPPI DATASET
pi_tr_arousal = np.array(pi_train_arousal)
pi_tr_valence = np.array(pi_train_valence)
pi_tr_immersion = np.array(pi_train_immersion)

pi_te_arousal = np.array(pi_test_arousal)
pi_te_valence = np.array(pi_test_valence)
pi_te_immersion = np.array(pi_test_immersion)

print('PIPPI Training set')
print('pi_tr_arousal',pi_tr_arousal.shape,pi_tr_arousal)
print('pi_tr_valence',pi_tr_valence.shape,pi_tr_valence)
print('pi_tr_immersion',pi_tr_immersion.shape,pi_tr_immersion)

print('PIPPI Test set')
print(pi_te_arousal)
print(pi_te_valence)
print(pi_te_immersion)

################################################################################# Concatenate to design flatten (1,) dataset
hr_tr_dataset = np.concatenate((hr_tr_arousal, hr_tr_valence), axis = 1)        #######   HARRY dataset
hr_te_dataset = np.concatenate((hr_te_arousal, hr_te_valence), axis = 1)        # In order to pass the input data in an optimal manner into the 
print(hr_tr_dataset.shape,'tr_dataset')                                         # machine learning algorithm, we will create an input dataset in the shape of 
print(hr_tr_dataset)                                                            # a two dimensional array (a, b) where
print(hr_te_dataset.shape,'te_dataset')                                         #   a=number of cases (2- fot the training set, 4 for the test set)
print(hr_te_dataset)                                                            #   b=250, in which the first 125 values represent the arousal scores 
                                                                                #     and the next 125 values represent the the valence scores

pi_tr_dataset = np.concatenate((pi_tr_arousal, pi_tr_valence), axis = 1)        #######    PIPPI dataset
pi_te_dataset = np.concatenate((pi_te_arousal, pi_te_valence), axis = 1)        # Same (a, b) structure as described above

################################################################################ PART III: Predictive Modeling
def MAE(y_train, y_pred):                                                       # 8. Define Evaluation Score, Mean Absolute Error (MAE)
  return np.mean(np.abs((y_train - y_pred)))

################################################################################# 9. Multiple Linear Regression for HARRY dataset (TEXT: Harry, CONDITION: COHERENT)
x_train = hr_tr_dataset.copy()
x_test = hr_te_dataset.copy()
y_train = hr_tr_immersion.copy()
y_test = hr_te_immersion.copy()

mlr = LinearRegression()
mlr.fit(x_train, y_train) 

print("train score")
y_pred = mlr.predict(x_train)

for i, e in enumerate(y_pred):
    print("expected_value",y_pred[i],'\t', "real_value",y_train[i])

print('MAE:',MAE(y_train, y_pred))

print("test score")
y_pred0 = mlr.predict(x_test)

for i, e in enumerate(y_pred0):
    print("expected_value",y_pred0[i],'\t', "real_value",y_train[i])

print('MAE:',MAE(y_test, y_pred0))

################################################################################# 10. k-Neighbors-Regression for HARRY (TEXT: HARRY, CONDITION: COHERENT)
x_train = hr_tr_dataset.copy()
x_test = hr_te_dataset.copy()
y_train = hr_tr_immersion.copy()
y_test = hr_te_immersion.copy()

neigh = KNeighborsRegressor(n_neighbors = 5, weights = "distance")
neigh.fit(x_train, y_train) 

print("train score")
y_pred1 = neigh.predict(x_train)

for i, e in enumerate(y_pred1):
    print("expected_value",y_pred1[i],'\t', "real_value",y_train[i])

print('MAE:',MAE(y_train, y_pred1))

print("test score")
y_pred2 = neigh.predict(x_test)

for i, e in enumerate(y_pred2):
    print("expected_value",y_pred2[i],'\t', "real_value",y_train[i])

print('MAE:',MAE(y_test, y_pred2))

################################################################################# 11. SVR, Support Vector Regressoion for HARRY (TEXT: HARRY, CONDITION: COHERENT)
x_train = hr_tr_dataset.copy()
x_test = hr_te_dataset.copy()
y_train = hr_tr_immersion.copy()
y_test = hr_te_immersion.copy()

SupportVectorRegModel = SVR()
SupportVectorRegModel.fit(x_train,y_train)

print("train score")
y_pred3 = SupportVectorRegModel.predict(x_train)

for i, e in enumerate(y_pred3):
    print("expected_value",y_pred3[i],'\t', "real_value",y_train[i])

print('MAE:',MAE(y_train, y_pred3))

print("test score")
y_pred4 = SupportVectorRegModel.predict(x_test)

for i, e in enumerate(y_pred4):
    print("expected_value",y_pred4[i],'\t', "real_value",y_train[i])

print('MAE:',MAE(y_test, y_pred4))

################################################################################# 12. XGB Regression for HARRY (TEXT: HARRY, CONDITION: COHERENT)
x_train = hr_tr_dataset.copy()
x_test = hr_te_dataset.copy()
y_train = hr_tr_immersion.copy()
y_test = hr_te_immersion.copy()

xgb1 = XGBRegressor()
parameters = {'nthread':[4],
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07],
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500]}

xgb_grid = GridSearchCV(xgb1,parameters,cv = 2,n_jobs = 5,verbose=True)
xgb_grid.fit(x_train,y_train)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)

print("train score")
y_pred5= xgb_grid.predict(x_train)
for i, e in enumerate(y_pred5):
    print("expected_value",y_pred5[i], '\t', "real_value", y_train[i])

print('MAE:',MAE(y_train, y_pred5))

print("test score")
y_pred6 = xgb_grid.predict(x_test)
for i, e in enumerate(y_pred6):
    print("expected_value",y_pred6[i], '\t', "real_value", y_test[i])

print('MAE',MAE(y_test, y_pred6))

################################################################################# 13. NN Regression for HARRY (TEXT: HARRY, CONDITION: COHERENT)
x_train = hr_tr_dataset.copy()                                                  # Characteristics: 5 layers
x_test = hr_te_dataset.copy()                                                   # Layer 1: Input layer, size = x_train.shape[1] which is 250
y_train = hr_tr_immersion.copy()                                                # Layer 2, 3, 4: size = 64
y_test = hr_te_immersion.copy()                                                 # Layer 5: Output layer, size = 1

model = models.Sequential()
model.add(layers.Dense(64, activation='relu',
                       input_shape=(x_train.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
model.summary()

history = model.fit(x_train, y_train,epochs = 100,batch_size=1, verbose=0)

print("train score")
test_mse_score, test_mae_score = model.evaluate(x_train, y_train)
result1 = model.predict(x_train, verbose=0)
for i, e in enumerate(result1):
    print("expected_value",sum(e,0.0)/len(e), '\t', "real_value", y_train[i])

print("test score")
test_mse_score, test_mae_score = model.evaluate(x_test, y_test)

result = model.predict(x_test, verbose=0)
for i, e in enumerate(result):
    print("expected_value",sum(e,0.0)/len(e), '\t', "real_value", y_test[i])

################################################################################# 14. Multiple Linear Regression for PIPPI (TEXT: PIPPI, CONDITION: COHERENT)
x_train = pi_tr_dataset.copy()
x_test = pi_te_dataset.copy()
y_train = pi_tr_immersion.copy()
y_test = pi_te_immersion.copy()

mlr = LinearRegression()
mlr.fit(x_train, y_train) 

print("train score")
y_pred7 = mlr.predict(x_train)

for i, e in enumerate(y_pred7):
    print("expected_value",y_pred7[i],'\t', "real_value",y_train[i])

print('MAE:',MAE(y_train, y_pred7))

print("test score")
y_pred8 = mlr.predict(x_test)

for i, e in enumerate(y_pred8):
    print("expected_value",y_pred8[i],'\t', "real_value",y_train[i])

print('MAE:',MAE(y_test, y_pred8))

################################################################################# 15. k-Neighbors-Regression for PIPPI (TEXT: PIPPI, CONDITION: COHERENT)

x_train = pi_tr_dataset.copy()
x_test = pi_te_dataset.copy()
y_train = pi_tr_immersion.copy()
y_test = pi_te_immersion.copy()

neigh = KNeighborsRegressor(n_neighbors = 5, weights = "distance")
neigh.fit(x_train, y_train) 

print("train score")
y_pred9 = neigh.predict(x_train)

for i, e in enumerate(y_pred9):
    print("expected_value",y_pred9[i],'\t', "real_value",y_train[i])

print('MAE:',MAE(y_train, y_pred9))

print("test score")
y_pred10 = neigh.predict(x_test)

for i, e in enumerate(y_pred10):
    print("expected_value",y_pred10[i],'\t', "real_value",y_train[i])

print('MAE:',MAE(y_test, y_pred10))

################################################################################# 16. SVR, Support Vector Regression for PIPPI (TEXT: PIPPI, CONDITION: COHERENT)
x_train = pi_tr_dataset.copy()
x_test = pi_te_dataset.copy()
y_train = pi_tr_immersion.copy()
y_test = pi_te_immersion.copy()

SupportVectorRegModel = SVR()
SupportVectorRegModel.fit(x_train,y_train)

print("train score")
y_pred11 = SupportVectorRegModel.predict(x_train)

for i, e in enumerate(y_pred11):
    print("expected_value",y_pred11[i],'\t', "real_value",y_train[i])

print('MAE:',MAE(y_train, y_pred11))

print("test score")
y_pred12 = SupportVectorRegModel.predict(x_test)

for i, e in enumerate(y_pred12):
    print("expected_value",y_pred12[i],'\t', "real_value",y_train[i])

print('MAE:',MAE(y_test, y_pred12))

################################################################################# 17. XGB Regression for PIPPI (TEXT: PIPPI, CONDITION: COHERENT)
x_train = pi_tr_dataset.copy()
x_test = pi_te_dataset.copy()
y_train = pi_tr_immersion.copy()
y_test = pi_te_immersion.copy()

xgb1 = XGBRegressor()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07], #so called `eta` value
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500]}

xgb_grid = GridSearchCV(xgb1,parameters,cv = 2,n_jobs = 5,verbose=True)
xgb_grid.fit(x_train,y_train)

print("train score")
y_pred13= xgb_grid.predict(x_train)
for i, e in enumerate(y_pred13):
    print("expected_value",y_pred13[i], '\t', "real_value", y_train[i])
print('MAE',MAE(y_test, y_pred13))

print("test score")
y_pred14 = xgb_grid.predict(x_test)
for i, e in enumerate(y_pred14):
    print("expected_value",y_pred14[i], '\t', "real_value", y_test[i])

print('MAE',MAE(y_test, y_pred14))

################################################################################# 18. NN Regression for PIPPI (TEXT: PIPPI, CONDITION: COHERENT)
x_train = pi_tr_dataset.copy()                                                  # Characteristics of the neural network: see above (For Text: HARRY)
x_test = pi_te_dataset.copy()
y_train = pi_tr_immersion.copy()
y_test = pi_te_immersion.copy()

model = models.Sequential()
model.add(layers.Dense(64, activation='relu',
                       input_shape=(x_train.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

history = model.fit(x_train, y_train,epochs = 100,batch_size=1, verbose=0)

print("train score")
test_mse_score, test_mae_score = model.evaluate(x_train, y_train)
result1 = model.predict(x_train, verbose=0)
for i, e in enumerate(result1):
    print("expected_value",sum(e,0.0)/len(e), '\t', "real_value", y_train[i])

print("test score")
test_mse_score, test_mae_score = model.evaluate(x_test, y_test)

result = model.predict(x_test, verbose=0)
for i, e in enumerate(result):
    print("expected_value",sum(e,0.0)/len(e), '\t', "real_value", y_test[i])

################################################################################# 19. Neural network, Regression: Hyperparameter Optimization
x_train = hr_tr_dataset.copy()
x_test = hr_te_dataset.copy()
y_train = hr_tr_immersion.copy()
y_test = hr_te_immersion.copy()

def gridsearch_model(neurons1, neurons2, neurons3):                             ######### Grid Search function
                                                                                ############  Input:
    model = models.Sequential()
    model.add(layers.Dense(neurons1, activation='relu',                         #   neurons1 : integer, first dropout layer parameter      
                       input_shape=(x_train.shape[1],)))
    model.add(Dense(neurons2,activation = 'relu'))                              #   neurons2 : integer, first dropout layer parameter
    model.add(Dense(neurons3,activation = 'relu'))                              #   neurons3 : integer, second dropout layer parameter
    model.add(Dense(1))
    
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])             ###### Output:
    return model                                                                #    This function returns a model

model = KerasRegressor(build_fn=gridsearch_model, nb_epoch=100,                 # Grid Search, Hyper-parameter tuning for Text: HARRY, Condition: COHERENT
                       batch_size=4, verbose=0)

neurons1 = [32,64,128]                                                               
neurons2 = [32,64,128]
neurons3 = [32,64,128]

param_grid = dict(neurons1 = neurons1,neurons2 = neurons2,neurons3=neurons3)    # Grid Search parameter on hyper parameters
grid = GridSearchCV(estimator = model, param_grid = param_grid,n_jobs=-1)       # GridSearchCV process constructs and evaluates a model 
grid_result = grid.fit(x_train,y_train)                                         # for each combination of parameters.

means = grid_result.cv_results_['mean_test_score']                              # Summarize results
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean,stdev,param in zip(means,stds,params):
    print("%f (%f) with: %r" % (mean,stdev,param))
print("Best: %f using %s" % (grid_result.best_score_,grid_result.best_params_))

################################################################################# 20. Nueral network, Regression: Cross-Validation
para = grid_result.best_params_                                                 # Cross Validation for Text: HARRY, Condition: COHERENT
append = []
a_list = ['neurons1','neurons2','neurons3']
for i in a_list:
    append.append(para[i])
print(append)

k = 5                                                                           # K-fold cross-validation
num_val_samples = len(x_train) // k
num_epochs = 200
all_mae_scores = []

def build_model():                                                              ####### Define the model
    model = models.Sequential()                                                 # No Input
    model.add(layers.Dense(append[0], activation='relu',                        # Output: this function returns a model
                           input_shape=(x_train.shape[1],)))
    model.add(layers.Dense(append[1], activation='relu'))
    model.add(layers.Dense(append[2], activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

for i in range(k):
    print('processing fold #:', i+1)
    val_data = x_train[i*num_val_samples: (i + 1)*num_val_samples]
    val_targets = y_train[i*num_val_samples: (i+1)*num_val_samples]
                                                                                # Build training and test sets
    partial_x_train = np.concatenate(                                           
        [x_train[:i * num_val_samples],
         x_train[(i+1) * num_val_samples:]],
         axis = 0)                                                              # partial_x_train: contains k-1 subgroups of the total train input dataset
    partial_train_targets = np.concatenate(
        [y_train[:i * num_val_samples],
         y_train[(i + 1) * num_val_samples:]],
         axis=0)                                                                # partial_train_targets: contains k-1 subgroups of the total train output data

    model = build_model()
    history = model.fit(partial_x_train, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs = num_epochs, batch_size=4, verbose=0)
    mae_history = history.history['val_mae']
    all_mae_scores.append(mae_history)

average_mae_history = [np.mean([x[i] for x in all_mae_scores]) for i in range(num_epochs)]

plt.plot(range(1, len(average_mae_history)+1), average_mae_history)             # Validation MAE Visualization
plt.xlabel('Epochs')
plt.ylabel('Average Validation MAE')
plt.show()

################################################################################# 21. Neural network, Regression: Evaluation
para = grid_result.best_params_                                                 # Final train and test of Text: HARRY, Condition: COHERENT
append = []
alist = ['neurons1','neurons2','neurons3']
for i in alist:
    print(para[i])
    append.append(para[i])
print(append)

model = models.Sequential()
model.add(layers.Dense(append[0], activation='relu',
                       input_shape=(x_train.shape[1],)))
model.add(layers.Dense(append[1], activation='relu'))
model.add(layers.Dense(append[2], activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
model.summary()

history = model.fit(x_train,y_train,epochs=115,batch_size=20,verbose=0)

print("train score")
test_mse_score, test_mae_score = model.evaluate(x_train, y_train)
result1 = model.predict(x_train, verbose=0)
for i, e in enumerate(result1):
    print("expected_value",sum(e,0.0)/len(e), '\t', "real_value", y_train[i])

print("test score")
test_mse_score, test_mae_score = model.evaluate(x_test, y_test)

result = model.predict(x_test, verbose=0)
for i, e in enumerate(result):
    print("expected_value",sum(e,0.0)/len(e), '\t', "real_value", y_test[i])

################################################################################ PART IV: Statistical Analysis with BIG FIVE
################################################################################# 22. Statistical Analysises between Immersion and 5 BFI (TEXT: HARRY; PIPPI)
                                                                                # Using linear regression, ordinary least square
e_group = a_group.copy()                                                        # Text: HARRY; PIPPI, Condition: COHERENT, Immersion: 2 Reader_Response
e_group.head()

bfi_list = []
for i in range(5):
    bfi_list.append(e_group.columns[i+4])

def immersion_bfi_sa(bfi_list):                                                 # Function for the Statistical Analysis
    for i, bfi in enumerate(bfi_list):
        bfi_list = 'IMMERSION ~ '+ bfi
        res = smf.ols(formula =bfi_list, data = e_group).fit()
        print(bfi)
        print('R2: ', res.rsquared)
        print(res.summary())
        print('')

def immersion_bfi_pl(bfi_list):                                                 # Function for the Plot the IMMERSION and BFI in linear regression
    color = ['red','yellow','green','blue','indigo','purple']
    for i, bfi in enumerate(bfi_list):    
        sns.regplot(x = bfi, y="IMMERSION", data = e_group, color = color[i])

def immersion_bfi_pl2(bfi_list):                                                # Function for the Plot the IMMERSION and BFI in nonlinear regression
    color = ['red','orange','yellow','green','blue','indigo','purple',
             'red','orange','yellow','green','blue','indigo','purple']
    for i, bfi in enumerate(bfi_list):    
        sns.regplot(x = bfi, y = "IMMERSION", data = e_group,
                    color = color[i], order = 2)

immersion_bfi_sa(bfi_list)                                                      # Statistical Analysis
immersion_bfi_pl(bfi_list)                                                      # Plot the IMMERSION and BFI in linear regression
immersion_bfi_pl2(bfi_list)                                                     # Plot the IMMERSION and BFI in nonlinear regression

################################################################################# BFI_OPENNESS and IMMERSION distribution
x = e_group['BFI_OPENNESS']                                                     # BFI_OPENNESS and IMMERSION
y = e_group['IMMERSION']
plt.scatter(x,y)

################################################################################ IMMERSION and BFI OPENNESS
res = smf.ols(formula='IMMERSION ~ BFI_OPENNESS', data = e_group).fit()         # Using linear regression, oridnary least square
sns.regplot(x="BFI_OPENNESS", y="IMMERSION", data=e_group, color='purple')
print('BFI_OPENESS')
print(res.summary())

plt.figure(figsize=(10, 2))                                                     # Check the outlier lower -2, over 2 considered as outlier
plt.stem(res.resid_pearson)
plt.axhline(2, c="g", ls="--")
plt.axhline(-2, c="g", ls="--")
plt.title("")
plt.show()

################################################################################# BFI_EXTRAVERSION and IMMERSION
x = e_group['BFI_EXTRAVERSION']                                                 # BFI_EXTRAVERSION and IMMERSION distribution
y = e_group['IMMERSION']
plt.scatter(x,y)

################################################################################# IMMERSION AND BFI_EXTRAVERSION
                                                                                # Using linear regression, oridnary least square
res = smf.ols(formula='IMMERSION ~ BFI_EXTRAVERSION', data = e_group).fit()
sns.regplot(x="BFI_EXTRAVERSION", y="IMMERSION", data=e_group, color='purple')
print('BFI_EXTRAVERSION')
print(res.summary())

plt.figure(figsize=(10, 2))                                                     # Check the outlier lower -2, over 2 considered as outlier
plt.stem(res.resid_pearson)
plt.axhline(2, c="g", ls="--")
plt.axhline(-2, c="g", ls="--")
plt.title("")
plt.show()

################################################################################# 23. Statistical Analysises between Immersion and 5 BFI (TEXT: HARRY)
                                                                                # Using linear regression, ordinary least square
e_group = h_group.copy()                                                        # Text: HARRY, Condition: COHERENT, Immersion: 2 Reader_Response
e_group.head()

################################################################################# Statistical Analysis and the Plot
def immersion_bfi_sa(bfi_list):                                                 # Function for the Statistical Analysis
    for i, bfi in enumerate(bfi_list):
        bfi_list = 'IMMERSION ~ '+ bfi
        res = smf.ols(formula =bfi_list, data = e_group).fit()
        print(bfi)
        print('R2: ', res.rsquared)
        print(res.summary())
        print('')

def immersion_bfi_pl(bfi_list):                                                 # Function for the Plot the IMMERSION and BFI in linear regression
    color = ['red','yellow','green','blue','indigo','purple']
    for i, bfi in enumerate(bfi_list):    
        sns.regplot(x = bfi, y="IMMERSION", data = e_group, color = color[i])

def immersion_bfi_pl2(bfi_list):                                                # Function for the Plot the IMMERSION and BFI in nonlinear regression
    color = ['red','orange','yellow','green','blue','indigo','purple',
             'red','orange','yellow','green','blue','indigo','purple']
    for i, bfi in enumerate(bfi_list):    
        sns.regplot(x = bfi, y = "IMMERSION", data = e_group,
                    color = color[i], order = 2)

bfi_list = []
for i in range(5):
    bfi_list.append(e_group.columns[i+4])

immersion_bfi_sa(bfi_list)                                                      # Statistical Analysis
immersion_bfi_pl(bfi_list)                                                      # Plot the IMMERSION and BFI in linear regression
immersion_bfi_pl2(bfi_list)                                                     # Plot the IMMERSION and BFI in nonlinear regression

################################################################################# 24. Statistical Analysises between Immersion and 5 BFI (TEXT: PIPPI)
                                                                                # Using linear regression, ordinary least square
e_group = p_group.copy()                                                        # Text: PIPPI, Condition: COHERENT, Immersion: 3 Reader_Response
e_group.head()

bfi_list = []
for i in range(5):
    bfi_list.append(e_group.columns[i+4])

immersion_bfi_sa(bfi_list)                                                      # Statistical Analysis
immersion_bfi_pl(bfi_list)                                                      # Plot the IMMERSION and BFI in linear regression
immersion_bfi_pl2(bfi_list)                                                     # Plot the IMMERSION and BFI in nonlinear regression

################################################################################# Statistical Analysises between Immersion and 5 BFI: Artihmetic Mean
                                                                                # 25. Immersion and BIG_FIVE: All Reader_response (CONDITION: COHERENT; SCRAMBLED)
e_group = p_group.copy()                                                        # Text: HARRY; PIPPI, Condition: COHERENT; SCRAMBLED
e_group.head()

b_DEG = s_group.loc[s_group['DEG_TIME']>100]['CASE']
b_ATT = s_group.loc[s_group['ATTENTION_CHECKS_COUNT_WRONG_ANSWERS']>0]['CASE']
b_PAG = s_rating.loc[s_rating['PAGE_TIME']>999]['CASE']
b_list = pd.concat([b_DEG, b_ATT, b_PAG], axis = 0)
b_list = b_list.drop_duplicates()
b_case = list(b_list.values)
print(b_case)

e_group = []
e_rating = []
a_group = s_group.copy()
a_rating = s_rating.copy()

for i in b_case:
    a_rating = a_rating.drop(a_rating[a_rating['CASE'] == i].index)
    a_group = a_group.drop(a_group[a_group['CASE'] == i].index)
    if i == 46:
        e_group = a_group
        e_rating = a_rating

e_group['IMMERSION'] = 0

for i in range (14):
    e_group['IMMERSION'] += e_group.iloc[:,i+4]

e_group['IMMERSION']=e_group['IMMERSION']/14
print(e_group.head(10))
sns.distplot(e_group["IMMERSION"])

################################################################################# Correlation between BIG FIVE and IMMERSION 
bfi_list = []
for i in range(5):
  bfi_list.append(e_group.columns[i+4])

immersion_bfi_sa(bfi_list)
immersion_bfi_pl(bfi_list)
immersion_bfi_pl2(bfi_list)

################################################################################ IMMERSION and BFI OPENNESS
res = smf.ols(formula='IMMERSION ~ BFI_CONSCIENTIOUSNESS', data = e_group).fit()         # Using linear regression, oridnary least square
sns.regplot(x="BFI_CONSCIENTIOUSNESS", y="IMMERSION", data=e_group, color='purple')
print('BFI_OPENESS')
print(res.summary())

plt.figure(figsize=(10, 2))                                                     # Check the outlier lower -2, over 2 considered as outlier
plt.stem(res.resid_pearson)
plt.axhline(2, c="g", ls="--")
plt.axhline(-2, c="g", ls="--")
plt.title("")
plt.show()

################################################################################ IMMERSION and BFI OPENNESS
res = smf.ols(formula='IMMERSION ~ BFI_OPENNESS', data = e_group).fit()         # Using linear regression, oridnary least square
sns.regplot(x="BFI_OPENNESS", y="IMMERSION", data=e_group, color='purple')
print('BFI_OPENESS')
print(res.summary())

plt.figure(figsize=(10, 2))                                                     # Check the outlier lower -2, over 2 considered as outlier
plt.stem(res.resid_pearson)
plt.axhline(2, c="g", ls="--")
plt.axhline(-2, c="g", ls="--")
plt.title("")
plt.show()

################################################################################# Statistical Analysises between Immersion and 5 BFI: Artihmetic Mean
                                                                                # 26. Immersion and BIG_FIVE: All Reader_response (CONDITION: COHERENT)
sg_co = s_group[sg_coherent]                                                    # Text: HARRY; PIPPI, Condition: COHERENT
sr_co = s_rating[sr_coherent]
                                                                                ### Selecting the cases with "bad data"

                                                                                # Criteria that renders a case as "bad data" are:
b_DEG = sg_co.loc[sg_co['DEG_TIME']>100]['CASE']                                #       1. DEG_TIME > 100 (completion of the survey was too fast)

b_ATT = sg_co.loc[sg_co['ATTENTION_CHECKS_COUNT_WRONG_ANSWERS']>0]['CASE']      #       2. ATTENTION_CHECKS_COUNT_WRONG_ANSWERS > 0 (clicking random answeres)

b_PAG = sr_co.loc[sr_co['PAGE_TIME']>999]['CASE']                               #       3. PAGE_TIME > 999 (pausing while completing the survey, can interfere with immersion)

b_list = pd.concat([b_DEG, b_ATT, b_PAG], axis = 0)                             # Concatenate the "bad data" described above into a single list

b_list = b_list.drop_duplicates()                                               # Delete the duplicates
b_case = list(b_list.values)                                                    # Make of list of the cases considered to be "bad data" (list of case number)
print('bad data case list:',b_case)
                                                                                
                                                                                ### Excluding the "bad data" cases from the dataset

a_group = sg_co.copy()                                                          # Copy the SENT_GROUP_INFO.coherent
a_rating = sr_co.copy()                                                         # Copy the # SENT_RATING_DATA.coherent

for i in b_case:                                                                # Exclude the bad case from the coherent dataset
    a_rating = a_rating.drop(a_rating[a_rating['CASE'] == i].index)
    a_group = a_group.drop(a_group[a_group['CASE'] == i].index)

sg_co_harry = a_group[sg_harry & sg_coherent]                                   # Text: HARRY; Condition: COHERENT, bad data excluded
sg_sc_harry = a_group[sg_harry & sg_scrambled]                                  # Text: HARRY; Condition: SCRAMBLED, bad data excluded
sg_co_pippi = a_group[sg_pippi & sg_coherent]                                   # Text: PIPPI; Condition: COHERENT, bad data excluded
sg_sc_pippi = a_group[sg_pippi & sg_scrambled]                                  # Text: PIPPI; Condition: SCRAMBLED, bad data excluded
sg_co_en_pippi = a_group[sg_pippi & sg_coherent & sg_eng]                       # Text: PIPPI; Condition: COHERENT; Language: ENG
sg_co_ge_pippi = a_group[sg_pippi & sg_coherent & sg_ger]                       # Text: PIPPI; Condition: COHERENT; Language: GER

                                                                                #  Loading data from SENT_GROUP_INFO file for:
ag_co_harry = a_group[sg_harry & sg_coherent]                                   #   Text: HARRY; Condition: COHERENT
ag_co_pippi = a_group[sg_pippi & sg_coherent & sg_eng]                          #   Text: PIPPI; Condition: COHERENT; Language: ENG
                                                                                # drop out the NaN column from the data from SENT_GROUP_INFO file:
gh = ag_co_harry                                                                # Text: HARRY; Condition: COHERENT
gp = ag_co_pippi                                                                # Text: PIPPI; Condition: COHERENT; Language: ENG

rg_set = pd.concat([gh,gp], axis = 0)                                           # Merge the HARRY and PIPPI dataset
a_group = rg_set.copy()                                                         # Copy the dataset
h_group = gh.copy()
p_group = gp.copy()

a_group['IMMERSION'] = 0                                                        # Create the new column of 'IMMERSION'

reader_response = [9,10,11,12,13,14,15,16,17,18,19,20,21,22]                    # Use all reader's response
                                                                                # If you need it then check the columns order, "print(a_group.columns)"
for i in reader_response:                                                       # Sum the reader's response
    a_group['IMMERSION'] += a_group.iloc[:,i]

a_group['IMMERSION']=a_group['IMMERSION']/len(reader_response)                  # Using the arithmetic mean of the reader's response
print(a_group.head(10))
sns.distplot(a_group["IMMERSION"])

################################################################################# Statistical Analysises between Immersion and 5 BFI: Mean case
                                                                                # Using linear regression, ordinary least square
e_group = a_group.copy()
e_group.head()

bfi_list = []
for i in range(5):
    bfi_list.append(e_group.columns[i+4])

immersion_bfi_sa(bfi_list)                                                      # Statistical Analysis
immersion_bfi_pl(bfi_list)                                                      # Plot the IMMERSION and BFI in linear regression
immersion_bfi_pl2(bfi_list)                                                     # Plot the IMMERSION and BFI in nonlinear regression

################################################################################ IMMERSION and BFI OPENNESS
res = smf.ols(formula='IMMERSION ~ BFI_EXTRAVERSION', data = e_group).fit()         # Using linear regression, oridnary least square
sns.regplot(x="BFI_EXTRAVERSION", y="IMMERSION", data=e_group, color='purple')
print('BFI_OPENESS')
print(res.summary())

plt.figure(figsize=(10, 2))                                                     # Check the outlier lower -2, over 2 considered as outlier
plt.stem(res.resid_pearson)
plt.axhline(2, c="g", ls="--")
plt.axhline(-2, c="g", ls="--")
plt.title("")
plt.show()

################################################################################# 27. Reader Response and BIG FIVE (TEXT: HARRY; PIPPI, CONDITION:SCRAMBLED)
for reader in ['FOCUSING_OF_ATTENTION','TEXT_ABSORPTION','IMAGINABILITY',
               'SPATIAL_INVOLVEMENT','GENERAL_READING_ENJOYMENT',
               'IDENTIFICATION','EASE_OF_COGNITIVE_ACCESS']:
    e_group = a_group.copy()
    e_group.head()

    bfi_list = []

    for i in range(5):
        bfi_list.append(e_group.columns[i+4])

    def reader_bfi_sa(reader, bfi_list):                                        # Function for the Statistical Analysis
        for i, bfi in enumerate(bfi_list):
            bfi_list = reader +'~ '+ bfi
            res = smf.ols(formula =bfi_list, data = e_group).fit()
            print(bfi)
            print('R2: ', res.rsquared)
            print(res.summary())
            print('')

    reader_bfi_sa(reader, bfi_list)