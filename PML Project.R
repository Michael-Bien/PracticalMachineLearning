#PML Sandbox 
#fread from website
library(data.table)
training <- fread("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
head(training[1:10,1:100])

#examine first column grouping -- useful?
library(ggplot2)
ggplot(training, aes(x=raw_timestamp_part_2, group=classe))+
  geom_density()

training <- training[,-(1:5)]
table(training$new_window)


#which variables have no variance?
library(caret)
nearZeroVar_cols <- nearZeroVar(training)
training2 <- training[,-..nearZeroVar_cols]
#which variables are all NA's?
colMeans(is.na(training2))
colMeans(is.na(preObj))
mostly_na <-colMeans(is.na(training2)) > .75
training3 <- training2[,-..mostly_na]
dim(training3)
str(training3)
View(training3[1:10,])

#what does this do?
preObj <- preProcess(training[,-"classe"], method="knnImpute")

#Correlation plot
featurePlot(x=training3[,], y=training$classe, plot="pairs")

#Exploratory plots


#try some kmeans clustering?
kMeans1 <- kmeans(training3[,-"classe"], centers=5)
training3$clusters <- kMeans1$cluster
confusionMatrix(training3$clusters, training3$classe)

#lasso for variable selection
lasso <- train(classe~., data=training3, method="lasso")

# basic rpart plot
rpart_fit <- train(classe ~ ., method="rpart", data=training3)
library(rattle)
fancyRpartPlot(rpart_fit$finalModel)
# GBM
gbm_fit <- train(classe ~ ., method="gbm", data=training3, verbose=FALSE)

# RANDOM FOREST
rf_fit <- train(classe ~ ., method="rf", data=training3, prox=TRUE)

# ENSEMBLED MODELS
predDf <- data.frame(pred1, pred2, target = training3$classe)
combModFit <- train(classe ~ ., method="gam", data=predDF)
combPred <- predict(combModFit, predDF)

# ACCURACY METRICS
