
rm(list = ls())

setwd("F:/ML/8 Practical Machine Learning/wk4")

# wk 4 - Project
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
              "training.csv")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
              "testing.csv")

library(caret)
library(dplyr)

training <- read.csv("training.csv", na.strings=c("",".","NA"))
testing <- read.csv("testing.csv", na.strings=c("",".","NA"))

testing$cvtd_timestamp <- as.Date(as.character(testing$cvtd_timestamp),
                                       "%d/%m/%Y %H:%M")


qplot( cvtd_timestamp, roll_belt, data = training, colour = user_name)

# exactly according to the specification (Class A), throwing the elbows 
# to the front (Class B), lifting the dumbbell only halfway (Class C), 
# lowering the dumbbell only halfway (Class D) and throwing the hips to
# the front (Class E).

## data preprocessing

# removing columns with more than 5% NA values
selNames <- apply(training, 2, function(x){ (1-sum(is.na(x))/length(x)) > 0.95} )
names(training)[!selNames]
selNames <- names(training)[selNames]
training <- training[, selNames]
testing <- testing[, c(selNames[-60], "problem_id")]


# grouping the predictor variables
# removing dates and time

trainPredVar <- training[,-c(1,3,4)]
trainPredVar$cvtd_timestamp <- as.Date(as.character(trainPredVar$cvtd_timestamp),
                                       "%d/%m/%Y %H:%M")
## training testing and validation data
set.seed(100)
inTV <- createDataPartition(trainPredVar$roll_belt, p=0.8, list = FALSE)
trainingPV <- trainPredVar[inTV,]
testingPV <- trainPredVar[-inTV,]
    

## find correlations/required nummber of variables - PCA?
notNume <- c(1:4, 57) # removing non-numeric variables
trainPc <- prcomp(trainingPV[, -notNume], scale = TRUE)
testPc <- prcomp(testingPV[, -notNume], scale = TRUE) 

biplot(trainPc, scale = 0)

# PC variance plot
plot(trainPc$sdev^2/sum(trainPc$sdev^2))
plot(cumsum(trainPc$sdev^2/sum(trainPc$sdev^2))*100, ylab = "Cumulative Variance %",
     xlab = "Number of predictors")
abline(90, 0, col = "red")
abline( v = 20, col = "red")

# re-including excluded rows
# by including only the first 20 variables of the PCA we are able to account for 
# more than 90% variance

trainNew <- data.frame(trainPc$x[, 1:20])
trainNew <- cbind(trainNew, trainingPV[, notNume])
testNew <- data.frame(testPc$x[, 1:20])
testNew <- cbind(testNew, testingPV[, notNume])

##--------------------------------------------------------------##
## train obtained data using random forest

set.seed(1000)

# fit model

# rf
modFit3 <- train(classe ~ . , data = trainNew, method = "rf", 
                trControl = trainControl(method="cv", number = 3))
modFit7 <- train(classe ~ . , data = trainNew, method = "rf", 
                trControl = trainControl(method="cv", number = 7))
modFit10 <- train(classe ~ . , data = trainNew, method = "rf", 
                trControl = trainControl(method="cv", number = 10))

predVal3 <- predict(modFit3, trainNew)
predVal7 <- predict(modFit7, trainNew)
predVal10 <- predict(modFit10, trainNew)
# accuracy
sum(predVal3 == trainNew$classe)/length(predVal3)
sum(predVal7 == trainNew$classe)/length(predVal7)
sum(predVal10 == trainNew$classe)/length(predVal10)
# 1

plot(varImp(modFit3))
plot(varImp(modFit7))
plot(varImp(modFit10))

## accuracy on test set 
predValTest3 <- predict(modFit3, testNew)
predValTest7 <- predict(modFit7, testNew)
predValTest10 <- predict(modFit10, testNew)

sum(predValTest3 == testNew$classe)/length(predValTest3)
sum(predValTest7 == testNew$classe)/length(predValTest7)
sum(predValTest10 == testNew$classe)/length(predValTest10)
# 0.9587051






#####

control <- trainControl(method="repeatedcv", number=10, repeats=3)
tunegrid <- expand.grid(.mtry= sqrt(ncol(trainingPV)))
modFitnew <- train(classe ~., data = trainingPV, method = "rf", ntree = 10,
                    metric = "Accuracy", 
                    tuneGrid=tunegrid, trControl=control)
print(modFitnew)
# prediction on split test case
prednew <- predict(modFitnew, testingPV)
print(sum(prednew == testingPV$classe)/length(prednew))
# importance of variables
varImp(modFitnew)

# prediction on provided test case
prednewTest <- predict(modFitnew, testing)
print(prednewTest)
#  B A B A A E D B A A B C B A E E A B B B