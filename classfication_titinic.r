setwd("C:/choidontouch/dataset/titanic")
#setwd("C:/choi/bigdata/data/auction")
train <- read.csv("train.csv", na.strings=c("", NA))
test <- read.csv("test.csv" , na.strings=c("", NA))
str(iris)
str(train)
idx <- createDataPartition(train$Survived, p=0.7, list=F)
sam = sample(1:150, size=30, replace=F)

table(train$Parch)


train$Survived <- as.factor(train$Survived)
##sample과 createDataPatition의 차이
tit_train <- train[idx,]
tit_test <- train[-idx,]
table(tit_test$Survived)
str(train)
##나이브베이즈 기법
library(e1071)
naive.result <- naiveBayes(tit_train, tit_train$Survived, laplace = 1)
navie.pred <- predict(naive.result, tit_test, type='class')
confusionMatrix(navie.pred, tit_test$Survived)
table(navie.pred, tit_test$Survived)

##다항로지스틱 회귀
library(nnet)
multi.result <- multinom(Survived~ Pclass+ Sex+Age+Fare, tit_train)
multi.pred<-predict(multi.result, tit_test)
table(multi.pred, tit_test$Survived)


##의사결정 트리 기법
library(rpart)
rpart.result<-rpart(Survived~., tit_train)
rpart.result
rpart.pred<-predict(rpart.result, tit_test, type="class")
table(rpart.pred, tit_test$Survived)
confusionMatrix(rpart.pred, tit_test$Survived)

#SVM함수 사용
install.packages("kernlab")
library(kernlab)
svm.result<-ksvm(Survived ~Pclass+Sex, data= tit_train, kernel='rbfdot')
svm.pred<-predict(svm.result, tit_test, type="response")
table(svm.pred, tit_test$Survived)
confusionMatrix(svm.pred, tit_test$Survived)

##앙상블 기법
install.packages("randomForest")
library(randomForest)
rf.result<-randomForest(Survived~Pclass+Sex+Fare, tit_train, ntree=500)
rf.pred<-predict(svm.result, tit_test, type="response")
table(rf.pred, tit_test$Survived)
confusionMatrix(svm.pred, tit_test$Survived)


install.packages("Amelia")
library(Amelia)
missmap(train)
