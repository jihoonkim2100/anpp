# anpp

This is a group programming project for Affective and Social Neuroscience (SoSe 2020)

This is the guideline documentation to run the code.

Library Package
Those are requirement to run this module with .py file:

- keras pip install keras
- matplotlib pip install matplotlib
- numpy pip install numpy
- pandas pip install pandas
- seaborn pip install seaborn
- statsmodels pip install statsmodels
- sklearn pip install sklearn
- tensorflow pip install tensorflow
- xgboost pip install xgboost

Attachment: SCAN_seminar_data.zip
Name Content

- Group1_Presentation.pptx Presentation slides
- Group1_Presentation.pdf PDF file, but there is some animation part is excluded
- Group1_Ver73.ipynb It contains the result which was presented in the seminar.
  We uploaded the dataset files using Google Drive.
- Group1_Ver73.py .py version: it must install libraries. (so, we recommend
  using google colaboratory. Because it does not need
  to install packages)
- Group1_GDrive.ipynb Connect the dataset from the google drive (You can set
  the runtime type, Hardware accelerator: GPU
- SENT_GROUP_INFO.xlsx Dataset 1
- SENT_RATING_DATA.xlsx Dataset 2

The Code Workflow
There are four main task and to complete the task it divided into 28 sub tasks:

- PART I: Group and Data Selection
- PART II: Data Preprocessing
- PART III: Predictive Modeling
- PART IV: Statistical Analysis with BIG FIVE

0. Import the library
   PART I: Group and Data Selection
1. Loading the dataset: SENT_GROUP_INFO and SENT_RATING_DATA
2. Grouping the data: Coherent ENG+GER, HARRY AND PIPPI
3. Data Selection (Exclude the "bad data") and visualization
4. Set the data with Condition: COHERENT only
5. drop out the NaN column from the data from SENT_GROUP_INFO file
   PART II: Data Preprocessing
6. Compound the reader_response for immersion
7. Preprocess the dataset
   PART III: Predictive Modeling
8. Define Evaluation Score, Mean Absolute Error (MAE)
9. Multiple Linear Regression for HARRY dataset (TEXT: Harry, CONDITION: COHERENT)
10. k-Neighbors-Regression for HARRY (TEXT: HARRY, CONDITION: COHERENT)
11. SVR, Support Vector Regression for HARRY (TEXT: HARRY, CONDITION: COHERENT)
12. XGB Regression for HARRY (TEXT: HARRY, CONDITION: COHERENT)
13. NN Regression for HARRY (TEXT: HARRY, CONDITION: COHERENT)
14. Multiple Linear Regression for PIPPI (TEXT: PIPPI, CONDITION: COHERENT)
15. k-Neighbors-Regression for PIPPI (TEXT: PIPPI, CONDITION: COHERENT)
16. SVR, Support Vector Regression for PIPPI (TEXT: PIPPI, CONDITION: COHERENT)
17. XGB Regression for PIPPI (TEXT: PIPPI, CONDITION: COHERENT)
18. NN Regression for PIPPI (TEXT: PIPPI, CONDITION: COHERENT)
19. Neural network, Regression: Hyperparameter Optimization
20. Neural network, Regression: Cross-Validation
21. Neural network, Regression: Evaluation
    PART IV: Statistical Analysis with BIG FIVE
22. Statistical Analyses between Immersion and 5 BFI (TEXT: HARRY; PIPPI)
23. Statistical Analyses between Immersion and 5 BFI (TEXT: HARRY)
24. Statistical Analyses between Immersion and 5 BFI (TEXT: PIPPI)
25. Immersion and BIG_FIVE: All Reader_response (CONDITION: COHERENT; SCRAMBLED)
26. Immersion and BIG_FIVE: All Reader_response (CONDITION: COHERENT)
27. Reader Response and BIG FIVE (TEXT: HARRY; PIPPI, CONDITION:COHERENT)
