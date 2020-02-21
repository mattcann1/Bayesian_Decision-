# Bayesian_Decision
Applies Bayesian decision  theory to a 2D problem with 3 classes decribed by priors, mean vectors and covariance matrices. 

The program completes two main goals:
  1) Plots the decision boundaries for a maximum likelihood (ML) and Maximum A Posteriori (MAP) decision rule alongside the means and first standard devaition contours for the classes.
  2) Generates a sample dataset from the class information and classifies the data using both ML and MAP classifers. The confusion matices are generated for both method and the experimental error rate is presented. 
  
The three classes used for the program have the following properties.
![](Figures/Capture.PNG)

1) The decision boundaries for both the Maximum Likelihood (ML) and Maximum A Posteriori (MAP) classifier were determined and plotted alongside the means and first standard deviation contours.
![](Figures/Q2parta_NEW1.png)
(a) (b)  
Decision boundaries for each class of dataset for (a) ML and (b) MAP classifer

Using the prior probabilities of each class, a dataset of a total of 3000 data points was generated with N_(Class 1)=600,N_(Class 2)=2100,N_(Class 3)=300. The dataset was classified using the ML and MAP methods of part a and the resulting confusion matrices are shown in Figure 4.
![](Figures/heatmap_1.png)    ![](Figures/heatmap_2.png)

The experimental error of the two classifiers is 8.7% and 6.4% for ML and MAP classifiers, respectively. The MAP experimental error is improved because of the consideration of the priors in the classification. Although some classes were misclassified more, such as class 3 being labelled as class 1, the experimental error weighs misclassifications alongside the prior value. Since MAP classifier takes into consideration the priors, the misclassification of the largest priors is decreased which is demonstrated by the decrease of misclassifications in class 2 values. 
