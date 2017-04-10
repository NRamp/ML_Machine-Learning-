# ML_Project.Rmd


## Practical Machine Learning 
## Date : 4/9/2017 


### Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

### Data:
The training data for this project are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. 

### Goal:
The goal of the project is to predict the manner in which the exercise was done. This is the "classe" variable in the training set. Other variables may be used to predict with. A report describing how the model was built including how cross validation was used, the expected out of sample error and why the choices made were done. Also utilize the prediction model to predict 20 different test cases. 

### Loading data:

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.3.3
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.3.2
```

```r
library(rattle)
```

```
## Warning: package 'rattle' was built under R version 3.3.3
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 4.1.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(rpart)
```

```
## Warning: package 'rpart' was built under R version 3.3.3
```

```r
library(rpart.plot)
```

```
## Warning: package 'rpart.plot' was built under R version 3.3.3
```

```r
trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training <- read.csv(url(trainUrl), na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(url(testUrl), na.strings=c("NA","#DIV/0!",""))
dim(training) ; dim(testing)
```

```
## [1] 19622   160
```

```
## [1]  20 160
```
## Data Cleaning: 
Let's do some cleaning before we split the data.
Let's delete rows with missing values and columns that may be contextual and not provide any prediction

```r
training <- training[, colSums(is.na(training)) == 0]
testing <- testing[, colSums(is.na(testing)) == 0]
training <- training[, -c(1:7)]
testing <- testing[, -c(1:7)]
dim(training); dim(testing)
```

```
## [1] 19622    53
```

```
## [1] 20 53
```

Now partition the dataset into 2 pieces for training and validating

```r
inTrain <- createDataPartition(training$classe, p=0.7, list=FALSE)
myTraining <- training[inTrain, ]
myvalidating <- training[-inTrain, ]
dim(myTraining); dim(myvalidating)
```

```
## [1] 13737    53
```

```
## [1] 5885   53
```
### Prediction Algorithms:
We will use K-fold cross validation and Random forests to predict the outcome.

K-fold cross validation: 
Let's use the default 10 fold cross validation here. 


```r
foldcontrol <- trainControl(method = "cv", number = 10)
fitrpart <- train(classe ~ ., data = myTraining, method = "rpart", 
                   trControl = foldcontrol)
print(fitrpart, digits = 4)
```

```
## CART 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 12362, 12363, 12364, 12364, 12364, 12363, ... 
## Resampling results across tuning parameters:
## 
##   cp       Accuracy  Kappa  
##   0.02441  0.5724    0.45072
##   0.04335  0.4694    0.30190
##   0.11535  0.3233    0.05931
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.02441.
```


```r
fancyRpartPlot(fitrpart$finalModel)
```

![](ML-_Project_files/figure-html/unnamed-chunk-5-1.png)<!-- -->
## Predict outcomes using the "myvalidating " set

```r
predict_rpart <- predict(fitrpart, myvalidating)

## Show output
confusionMatrix(myvalidating$classe, predict_rpart)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1030  178  317  142    7
##          B  198  644  237   60    0
##          C   27   48  798  153    0
##          D   38  149  504  273    0
##          E   16  255  246   75  490
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5497          
##                  95% CI : (0.5369, 0.5625)
##     No Information Rate : 0.3572          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4353          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7869   0.5055   0.3796  0.38834  0.98592
## Specificity            0.8593   0.8926   0.9397  0.86665  0.89013
## Pos Pred Value         0.6153   0.5654   0.7778  0.28320  0.45287
## Neg Pred Value         0.9337   0.8673   0.7316  0.91262  0.99854
## Prevalence             0.2224   0.2165   0.3572  0.11946  0.08445
## Detection Rate         0.1750   0.1094   0.1356  0.04639  0.08326
## Detection Prevalence   0.2845   0.1935   0.1743  0.16381  0.18386
## Balanced Accuracy      0.8231   0.6991   0.6597  0.62749  0.93802
```

```r
confusionMatrix(myvalidating$classe, predict_rpart)$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   5.497026e-01   4.353388e-01   5.368844e-01   5.624716e-01   3.571793e-01 
## AccuracyPValue  McnemarPValue 
##  1.662908e-198  2.584594e-259
```

```r
confusionMatrix(myvalidating$classe, predict_rpart)$overall[1]
```

```
##  Accuracy 
## 0.5497026
```
The confustion Matric shows an accuracy rate with K-fold cross validation (10 fold) is only 0.49.

Now lets see what Random forest can do.

```r
fit_rf <- train(classe ~ ., data = myTraining, method = "rf", 
                   trControl = foldcontrol)
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.3.3
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
print(fit_rf, digits = 4)
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 12363, 12365, 12363, 12363, 12365, 12364, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa 
##    2    0.9923    0.9902
##   27    0.9911    0.9888
##   52    0.9877    0.9844
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

Now let's look at the prediction on the validation set..


```r
predict_rf_validating <- predict(fit_rf, myvalidating)
conf_rf_validating <- confusionMatrix(myvalidating$classe, predict_rf_validating)
conf_rf_validating
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    8 1126    5    0    0
##          C    0    8 1018    0    0
##          D    0    0   10  954    0
##          E    0    0    0    3 1079
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9942         
##                  95% CI : (0.9919, 0.996)
##     No Information Rate : 0.2858         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9927         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9952   0.9929   0.9855   0.9969   1.0000
## Specificity            1.0000   0.9973   0.9984   0.9980   0.9994
## Pos Pred Value         1.0000   0.9886   0.9922   0.9896   0.9972
## Neg Pred Value         0.9981   0.9983   0.9969   0.9994   1.0000
## Prevalence             0.2858   0.1927   0.1755   0.1626   0.1833
## Detection Rate         0.2845   0.1913   0.1730   0.1621   0.1833
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9976   0.9951   0.9919   0.9974   0.9997
```
For this dataset, random forest method is way better than classification tree method. The accuracy rate is 0.9927. 

##Observation:
The Random forest computationally was very expensive and was not efficient. 

## Prediction on the Testing Data:

Having identified Random forest is better of the two - we can now run it on the test data. 

Let now predict the outcome on the testing set..


```r
predict_rf_testing <-predict(fit_rf, testing)
predict_rf_testing
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


