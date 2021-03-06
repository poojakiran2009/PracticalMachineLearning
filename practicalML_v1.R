library (data.table)
library (caret)
library (parallel)
library (doParallel)
# Read the files

training <- fread ('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv', header=T)
testing <- fread ('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv', header=T)

# See what columns are missing (NA or "")and ignore them
naCols <- which (training [,lapply(.SD, function(x) length(which(is.na(x))))] > 0)
emptyCols <- which (training [,lapply(.SD, function(x) length(which(x=="")))] > 0)
missingCols <- c(naCols, emptyCols)
missingCols <- unique (missingCols)
predCols <- names (training)[-missingCols]
# Columns to predict are the ones with belt, arm, dumbbell, forearm in them
isPredictor <- predCols [which (grepl("belt|[^(fore)]arm|dumbbell|forearm", predCols))]

# We want only the predictor variables and the target classe
training <- training [,c(isPredictor, "classe"),with=F]
testing <- testing [,c(isPredictor),with=F]

# remove variables with zero variance
mynzv <- nearZeroVar (as.data.frame (training)[,isPredictor], saveMetrics = TRUE)
zeroNms <- isPredictor [mynzv$zeroVar]
	# none to remove

isPredictor <- isPredictor [!(isPredictor %in% zeroNms)]

# remove highly correlated variables
mycor <- findCorrelation (cor (as.data.frame (training)[,isPredictor]))
isPredictor <- isPredictor[!(isPredictor %in% isPredictor [mycor])]

# Split dataset into 80% train and 20% validation
seed <- 11
set.seed(seed)
inTrain <- createDataPartition(training$classe, p=0.8)
mytrain <- training [inTrain[[1]]]
myvalidation <- training[-inTrain[[1]]]

# center and scale the variables
mypreproc <- preProcess (mytrain[,isPredictor, with=F], method = c("center", "scale"))
# apply the preprocess to training, validation and testing
mytrain <- predict (mypreproc, mytrain)
myvalidation <- predict (mypreproc, myvalidation)
testing <- predict (mypreproc, testing)


cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)
ctrl <- trainControl(classProbs=TRUE,
                     savePredictions=TRUE,
                     allowParallel=TRUE)
# Next we will try different methods. Let us try random forest first
myrf <- train (x=mytrain[,isPredictor,with=F], y = as.factor(mytrain$classe), method="rf")
# let us also try gbm
mygbm <- train (x=mytrain[,isPredictor,with=F], y = as.factor(mytrain$classe), method="gbm")

# Predict on the validation set
predrfvalid <- predict (myrf, myvalidation)
predgbmvalid <- predict (mygbm, myvalidation)

# Predict on the test set
predrftest <- predict (myrf, testing)
predgbmtest <- predict (mygbm, testing)

# Evaluate model on the validation set - RF results
confusionMatrix (myvalidation$classe, predrfvalid)
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1115    1    0    0    0
         B    5  753    1    0    0
         C    0    2  682    0    0
         D    0    0    7  636    0
         E    0    1    1    3  716

Overall Statistics
                                          
               Accuracy : 0.9946          
                 95% CI : (0.9918, 0.9967)
    No Information Rate : 0.2855          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9932          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9955   0.9947   0.9870   0.9953   1.0000
Specificity            0.9996   0.9981   0.9994   0.9979   0.9984
Pos Pred Value         0.9991   0.9921   0.9971   0.9891   0.9931
Neg Pred Value         0.9982   0.9987   0.9972   0.9991   1.0000
Prevalence             0.2855   0.1930   0.1761   0.1629   0.1825
Detection Rate         0.2842   0.1919   0.1738   0.1621   0.1825
Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
Balanced Accuracy      0.9976   0.9964   0.9932   0.9966   0.9992

# Evaluate model on the validation set - gbm results
confusionMatrix (myvalidation$classe, predgbmvalid)

Finally, as the Random Forest model gave us the best result, we will apply that to our validation set and create the documents to submit.
Confusion Matrix and Statistics

          Reference
Prediction   A   B   C   D   E
         A  49   0   0 121 946
         B  30   9   1  96 623
         C  26   0   0 132 526
         D  48   1   0 161 433
         E  57  13   0  88 563

Overall Statistics
                                          
               Accuracy : 0.1993          
                 95% CI : (0.1869, 0.2122)
    No Information Rate : 0.7879          
    P-Value [Acc > NIR] : 1               
                                          
                  Kappa : 0.0161          
 Mcnemar's Test P-Value : <2e-16          

Statistics by Class:

                     Class: A Class: B  Class: C Class: D Class: E
Sensitivity           0.23333 0.391304 0.0000000  0.26923   0.1821
Specificity           0.71263 0.807692 0.8255992  0.85504   0.8101
Pos Pred Value        0.04391 0.011858 0.0000000  0.25039   0.7809
Neg Pred Value        0.94264 0.995575 0.9996913  0.86677   0.2105
Prevalence            0.05353 0.005863 0.0002549  0.15243   0.7879
Detection Rate        0.01249 0.002294 0.0000000  0.04104   0.1435
Detection Prevalence  0.28448 0.193474 0.1743564  0.16391   0.1838
Balanced Accuracy     0.47298 0.599498 0.4127996  0.56213   0.4961


# code as suggested by Coursera
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

pml_write_files(predrftest)



