Introduction
------------

The goal of this project is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict using machine learning which activity they were performance.

Load Data and Libraries
-----------------------

``` r
library(caret)
library(ggplot2)
```

Load the training and testing csvs, training is used for fitting the model and testing is used in the Coursera quiz. To estimate the out of sample error the training set is split 80/20 into a new training and testing set.

``` r
training_orig <- read.csv("pml-training.csv")
testing_orig <- read.csv("pml-testing.csv")

set.seed(1)
trainingSplit <- createDataPartition(y=training_orig$classe, p=0.8, list=FALSE)
training <- training_orig[trainingSplit, ]
testing <- training_orig[-trainingSplit, ]
```

Remove variables that have low variance, are mostly NA, or not appropreiate for use in the model (such as user\_name).

``` r
lowVar <- nearZeroVar(training)
training <- training[, -lowVar]
testing <- testing[, -lowVar]
testing_orig <- testing_orig[, -lowVar]

mostlyNA <- sapply(training, function(x) mean(is.na(x))) > 0.95
training <- training[, mostlyNA == FALSE]
testing <- testing[, mostlyNA == FALSE]
testing_orig <- testing_orig[, mostlyNA == FALSE]

training <- training[, -(1:5)]
testing <- testing[, -(1:5)]
testing_orig <- testing_orig[, -c(1:5)]
```

Modelling
---------

The following code fits a random forest to the activity data using all of the factors left in the dataset after preprocessing to predict classe. 5-fold cross validation is used to avoid overfitting hence improving likely performance on real world or out of sample test data.

``` r
cv5Fold <- trainControl(method="cv", number=5, verboseIter=FALSE)

fit <- train(classe ~ ., data=training, method="rf", trControl=cv5Fold)
fit$finalModel
```

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = param$mtry) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 27
    ## 
    ##         OOB estimate of  error rate: 0.19%
    ## Confusion matrix:
    ##      A    B    C    D    E  class.error
    ## A 4462    1    0    0    1 0.0004480287
    ## B    8 3025    4    1    0 0.0042791310
    ## C    0    4 2734    0    0 0.0014609204
    ## D    0    0    8 2565    0 0.0031092110
    ## E    0    1    0    2 2883 0.0010395010

The summary above shows that the model has a very low error of 0.2%.

Out of Sample Error
-------------------

To calculate the out of sample error we will use the 20% testing data set aside earlier.

``` r
preds <- predict(fit, newdata=testing)

confusionMatrix(testing$classe, preds)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1116    0    0    0    0
    ##          B    2  756    1    0    0
    ##          C    0    1  683    0    0
    ##          D    0    0    4  639    0
    ##          E    0    1    0    0  720
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.9977         
    ##                  95% CI : (0.9956, 0.999)
    ##     No Information Rate : 0.285          
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.9971         
    ##  Mcnemar's Test P-Value : NA             
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9982   0.9974   0.9927   1.0000   1.0000
    ## Specificity            1.0000   0.9991   0.9997   0.9988   0.9997
    ## Pos Pred Value         1.0000   0.9960   0.9985   0.9938   0.9986
    ## Neg Pred Value         0.9993   0.9994   0.9985   1.0000   1.0000
    ## Prevalence             0.2850   0.1932   0.1754   0.1629   0.1835
    ## Detection Rate         0.2845   0.1927   0.1741   0.1629   0.1835
    ## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
    ## Balanced Accuracy      0.9991   0.9982   0.9962   0.9994   0.9998

The model predicts classe with a 99.8% accuracy in the testing set hence the **estimated out of sample error is 0.2%**.

Predict Classe of 20 Test Case for the Quiz
-------------------------------------------

The model has a high accuracy level hence should be sufficient to pass the quiz.

``` r
testPreds <- predict(fit, newdata=testing_orig)
data.frame(classe=testPreds)
```

    ##    classe
    ## 1       B
    ## 2       A
    ## 3       B
    ## 4       A
    ## 5       A
    ## 6       E
    ## 7       D
    ## 8       B
    ## 9       A
    ## 10      A
    ## 11      B
    ## 12      C
    ## 13      B
    ## 14      A
    ## 15      E
    ## 16      E
    ## 17      A
    ## 18      B
    ## 19      B
    ## 20      B
