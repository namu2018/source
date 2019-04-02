install.packages("caret")
library(caret)
idx <- createDataPartition(iris$Species, p=0.8, list=F)
sam <- samlpe(1:150, size=50, replace=F)
##sample과 createDataPatition의 차이
iris_train <- iris[idx,]
iris_test <- iris[-idx,]
table(iris_test$species)

##나이브베이즈 기법
install.packages("e1071")
library(e1071)
naive.result <- naiveBayes(iris_train, iris_train$Species, laplace = 1)
navie.pred <- predict(naive.result, iris_test, type='class')
confusionMatrix(navie.pred, iris_test$Species)
table(navie.pred, iris_test$Species)

##다항로지스틱 회귀
library(nnet)
multi.result <- multinom(Species~., iris_train)
multi.pred<-predict(multi.result, iris_test)
table(multi.pred, iris_test$Species)


##의사결정 트리 기법
library(rpart)
rpart.result<-rpart(Species~., iris_train)
rpart.result
rpart.pred<-predict(rpart.result, iris_test, type="class")
table(rpart.pred, iris_test$Species)
confusionMatrix(rpart.pred, iris_test$Species)

#SVM함수 사용
install.packages("kernlab")
library(kernlab)
svm.result<-ksvm(Species ~ . , data= iris_train, kernel='rbfdot')
svm.pred<-predict(svm.result, iris_test, type="response")
table(svm.pred, iris_test$Species)
confusionMatrix(svm.pred, iris_test$Species)

##앙상블 기법
install.packages("randomForest")
library(randomForest)
rf.result<-randomForest(Species~., iris_train, ntree=500)
rf.pred<-predict(svm.result, iris_test, type="response")
table(rf.pred, iris_test$Species)
confusionMatrix(svm.pred, iris_test$Species)
