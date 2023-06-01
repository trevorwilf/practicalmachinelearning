#---
#This is just the R code - to see it run, or review documentation please review the index.Rmd file
#title: "Practical Machine Learning"
#author: "Trevor"
#date: "5/07/2016"
#output: html_document
#---


setwd("~/ShinyApps/machinelearning")

library(caret); library(randomForest); library(gbm); library(xgboost); library(plyr)

set.seed(3234)

library(doMC)
numcores <- floor(detectCores()) - 4
registerDoMC(cores = numcores)

pmldata_train <- read.csv(file="pml-training.csv",head=TRUE,sep=",")
pmldata_test <- read.csv(file="pml-testing.csv",head=TRUE,sep=",")

pmldata_train.omit <- pmldata_train[ , apply(pmldata_train, 2, 
                                             function(x) !any(is.na(x) | x == ""))]
pmldata_test.omit <- pmldata_test[ , apply(pmldata_test, 2, 
                                           function(x) !any(is.na(x) | x == ""))]

goodstuff <- pmldata_train.omit[,grep("arm|belt|dumb|class", 
                                      names(pmldata_train.omit))]
goodstuff.zero <- nearZeroVar(goodstuff, saveMetrics = TRUE)
goodstuff <- goodstuff[, goodstuff.zero[, "nzv"] == FALSE]

resamplenum <- 10
repeatsnum <- 1
#length is = (resamplenum)+1
seeds <- vector(mode = "list", length = ((resamplenum) + 1)) 
#3 is the number of tuning parameter for rf and gbm
for(i in 1:resamplenum) seeds[[i]]<- sample.int(n=1000, 3) 
#for the last model
seeds[[resamplenum+1]]<-sample.int(1000, 1)

registerDoMC(cores = numcores)
myControl <- trainControl(method = "cv", number = resamplenum, 
                          repeats = repeatsnum, seeds=seeds)
test_gbm_10 <- train(classe ~ ., method = "gbm", data = goodstuff, 
                     verbose = TRUE, trControl = myControl)
test_rf_10 <- train(classe ~ ., method = "rf", data = goodstuff, 
                    verbose = TRUE, trControl = myControl)

# xgbTree has its own parallel functionality so we 
# need to turn the registered cores back to one
registerDoMC(cores = 1) 
seeds <- vector(mode = "list", length = (resamplenum + 1)) 
#12 is the number of tuning parameter for xgm
for(i in 1:resamplenum) seeds[[i]]<- sample.int(n=1000, 12) 
seeds[[resamplenum+1]]<-sample.int(1000, 1)

myControl <- trainControl(method = "cv", number = resamplenum, 
                          repeats = repeatsnum, seeds=seeds)
test_xgb_10 <- train(classe ~ ., method = "xgbTree", data = goodstuff, 
                     verbose = TRUE, trControl = myControl)

resamplenum <- 60
repeatsnum <- 1
seeds <- vector(mode = "list", length = ((resamplenum) + 1)) 
for(i in 1:resamplenum) seeds[[i]]<- sample.int(n=1000, 3) 
seeds[[resamplenum+1]]<-sample.int(1000, 1)

registerDoMC(cores = numcores)
myControl <- trainControl(method = "cv", number = resamplenum, 
                          repeats = repeatsnum, seeds=seeds)
test_gbm_60 <- train(classe ~ ., method = "gbm", data = goodstuff, 
                     verbose = TRUE, trControl = myControl)
test_rf_60 <- train(classe ~ ., method = "rf", data = goodstuff, 
                    verbose = TRUE, trControl = myControl)

registerDoMC(cores = 1) 
seeds <- vector(mode = "list", length = (resamplenum + 1)) 
for(i in 1:resamplenum) seeds[[i]]<- sample.int(n=1000, 12) 
seeds[[resamplenum+1]]<-sample.int(1000, 1)

myControl <- trainControl(method = "cv", number = resamplenum, 
                          repeats = repeatsnum, seeds=seeds)
test_xgb_60 <- train(classe ~ ., method = "xgbTree", data = goodstuff, 
                     verbose = TRUE, trControl = myControl)

resamplenum <- 20
repeatsnum <- 3
seeds <- vector(mode = "list", length = ((resamplenum) + 1)) 
for(i in 1:resamplenum) seeds[[i]]<- sample.int(n=1000, 3) 
seeds[[resamplenum+1]]<-sample.int(1000, 1)

registerDoMC(cores = numcores)
myControl <- trainControl(method = "cv", number = resamplenum, 
                          repeats = repeatsnum, seeds=seeds)
test_gbm_20x3 <- train(classe ~ ., method = "gbm", data = goodstuff, 
                     verbose = TRUE, trControl = myControl)
test_rf_20x3 <- train(classe ~ ., method = "rf", data = goodstuff, 
                    verbose = TRUE, trControl = myControl)

registerDoMC(cores = 1) 
seeds <- vector(mode = "list", length = (resamplenum + 1)) 
for(i in 1:resamplenum) seeds[[i]]<- sample.int(n=1000, 12) 
seeds[[resamplenum+1]]<-sample.int(1000, 1)

myControl <- trainControl(method = "cv", number = resamplenum, 
                          repeats = repeatsnum, seeds=seeds)
test_xgb_20x3 <- train(classe ~ ., method = "xgbTree", data = goodstuff, 
                     verbose = TRUE, trControl = myControl)

resamplenum <- 200
repeatsnum <- 1
seeds <- vector(mode = "list", length = ((resamplenum) + 1)) 
for(i in 1:resamplenum) seeds[[i]]<- sample.int(n=1000, 3) 
seeds[[resamplenum+1]]<-sample.int(1000, 1)

registerDoMC(cores = numcores)
myControl <- trainControl(method = "cv", number = resamplenum, 
                          repeats = repeatsnum, seeds=seeds)
test_gbm_200 <- train(classe ~ ., method = "gbm", data = goodstuff, 
                     verbose = TRUE, trControl = myControl)
test_rf_200 <- train(classe ~ ., method = "rf", data = goodstuff, 
                    verbose = TRUE, trControl = myControl)

registerDoMC(cores = 1) 
seeds <- vector(mode = "list", length = (resamplenum + 1)) 
for(i in 1:resamplenum) seeds[[i]]<- sample.int(n=1000, 12) 
seeds[[resamplenum+1]]<-sample.int(1000, 1)

myControl <- trainControl(method = "cv", number = resamplenum, 
                          repeats = repeatsnum, seeds=seeds)
test_xgb_200 <- train(classe ~ ., method = "xgbTree", data = goodstuff, 
                     verbose = TRUE, trControl = myControl)

registerDoMC(cores = numcores)

registerDoMC(cores = numcores)

#Generalized Boosted Regression Modeling
test <- confusionMatrix(test_gbm_10)
accuracy <- sum(diag(test$table))/100
test <- confusionMatrix(test_gbm_20x3)
accuracy <- rbind(accuracy, (sum(diag(test$table))/100))
test <- confusionMatrix(test_gbm_60)
accuracy <- rbind(accuracy, (sum(diag(test$table))/100))
test <- confusionMatrix(test_gbm_200)
accuracy <- rbind(accuracy, (sum(diag(test$table))/100))

#Random Forest
test <- confusionMatrix(test_rf_10)
accuracy <- rbind(accuracy, (sum(diag(test$table))/100))
test <- confusionMatrix(test_rf_20x3)
accuracy <- rbind(accuracy, (sum(diag(test$table))/100))
test <- confusionMatrix(test_rf_60)
accuracy <- rbind(accuracy, (sum(diag(test$table))/100))
test <- confusionMatrix(test_rf_200)
accuracy <- rbind(accuracy, (sum(diag(test$table))/100))

#Extreme Gradient Boosting
test <- confusionMatrix(test_xgb_10)
accuracy <- rbind(accuracy, (sum(diag(test$table))/100))
test <- confusionMatrix(test_xgb_20x3)
accuracy <- rbind(accuracy, (sum(diag(test$table))/100))
test <- confusionMatrix(test_xgb_60)
accuracy <- rbind(accuracy, (sum(diag(test$table))/100))
test <- confusionMatrix(test_xgb_200)
accuracy <- rbind(accuracy, (sum(diag(test$table))/100))

colnames(accuracy)[1] <- "accuracy"
accuracy <- cbind(accuracy, apply(accuracy, 1, function(x) {1 - x}))


#Generalized Boosted Regression Modeling
prediction_gbm_10 <- as.character(predict(test_gbm_10, pmldata_test.omit))
prediction_gbm_20x3 <- as.character(predict(test_gbm_20x3, pmldata_test.omit))
prediction_gbm_60 <- as.character(predict(test_gbm_60, pmldata_test.omit))
prediction_gbm_200 <- as.character(predict(test_gbm_200, pmldata_test.omit))

#Random Forest
prediction_rf_10 <- as.character(predict(test_rf_10, pmldata_test.omit))
prediction_rf_20x3 <- as.character(predict(test_rf_20x3, pmldata_test.omit))
prediction_rf_60 <- as.character(predict(test_rf_60, pmldata_test.omit))
prediction_rf_200 <- as.character(predict(test_rf_200, pmldata_test.omit))

#Extreme Gradient Boosting
prediction_xgb_10 <- as.character(predict(test_xgb_10, pmldata_test.omit))
prediction_xgb_20x3 <- as.character(predict(test_xgb_20x3, pmldata_test.omit))
prediction_xgb_60 <- as.character(predict(test_xgb_60, pmldata_test.omit))
prediction_xgb_200 <- as.character(predict(test_xgb_200, pmldata_test.omit))

result_test <- rbind(prediction_gbm_10, prediction_gbm_20x3, prediction_gbm_60, prediction_gbm_200,
            prediction_rf_10, prediction_rf_20x3, prediction_rf_60, prediction_rf_200,
            prediction_xgb_10, prediction_xgb_20x3, prediction_xgb_60, prediction_xgb_200)

colnames(result_test) <- 1:20
      
finalresult <- cbind(accuracy, result_test)
rownames(finalresult) <- rownames(result_test)
colnames(finalresult)[2] <- "outofsampleerror"

#finalresult
subset(finalresult, select = c("accuracy", "outofsampleerror"))

write.csv(finalresult, file = "MyData.csv")

registerDoMC(cores = 1)

