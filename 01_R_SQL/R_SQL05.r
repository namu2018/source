library(RODBC)
my <- odbcConnect("mysql", uid="root", pwd="toto1234");
sqlQuery(my, "show databases");
sqlQuery(my, "use db01;");
sqlQuery(my, "show tables;")
         
exeQuery <- function(query) {
  res <- c()
  for (i in 1:length(query)) {
    # print(query[i])
    tmp <- sqlQuery(my,query[i])
    res <- list(res, tmp)
  }
  return (res)
}

v <- c('show databases', 'use db01;', "show tables;")
exeQuery(v)

## 01. insert
v <- c("desc iris_tbl;")
exeQuery(v)

v <- c("INSERT INTO iris_tbl VALUES (1.1,2.2,3.3,4.4,'setosa');", "select * from iris_tbl")
exeQuery(v)

## 01. insert(2) - 변수로 값을 넣어보기
iris[1,1]
val <- paste(iris[3,1],iris[3,2],iris[3,3],iris[3,4],  sep=",")
val
query1 <- paste("INSERT INTO iris_tbl VALUES (", val,",'setosa');")
query1
v <- c(query1, "select * from iris_tbl")
exeQuery(v)

## 02. 여려행 insert하기
v <- c(
  "INSERT INTO iris_tbl VALUES ( 4.7,3.2,1.3,0.2 ,'setosa');",
  "INSERT INTO iris_tbl VALUES ( 4.7,3.2,1.3,0.2 ,'setosa');",
  "INSERT INTO iris_tbl VALUES ( 4.5,3.2,1.3,0.2 ,'setosa');",
  "INSERT INTO iris_tbl VALUES ( 4.2,3.2,1.3,0.2 ,'versicolor');",
  "INSERT INTO iris_tbl VALUES ( 4.1,3.2,1.3,0.2 ,'versicolor');",
  "INSERT INTO iris_tbl VALUES ( 3.7,3.2,1.3,0.2 ,'virginica');",
  "INSERT INTO iris_tbl VALUES ( 3.7,3.2,1.3,0.2 ,'virginica');",
  "INSERT INTO iris_tbl VALUES ( 4.7,3.2,1.3,0.2 ,'virginica');",
  "select * from iris_tbl;")
exeQuery(v)

## 03. update하기 
## 지정 또는 조건에 해당되는 값을 변경한다.
'''
UPDATE `student` SET address="서울";
UPDATE `student` SET name="이진경" WHERE id=1;
UPDATE `iris_tbl` SET Species="versicolor" WHERE Sepal_length=5;
'''

## update
## 종류가 versicolor인 것을 Sepal_length=4.2로 변경한다.
v <- c(
  "select * from iris_tbl;",
  "UPDATE `iris_tbl` SET Sepal_length=4.2  WHERE  Species='versicolor';",
  "select * from iris_tbl;")
exeQuery(v)

## sqlQuery(my, "UPDATE `iris_tbl` SET Sepal_length=5  WHERE  Species='versicolor';")
## sqlQuery(my, "select * from iris_tbl;")


## 04. 삭제하기 
## 지정 또는 조건에 해당되는 값을 삭제한다.
## DELETE FROM 테이블명 [WHERE 삭제하려는 칼럼 명=값]


v <- c(
  "select * from iris_tbl;",
  "delete FROM `iris_tbl` WHERE  Species='versicolor';",
  "select * from iris_tbl;")
exeQuery(v)

v <- c(
  "INSERT INTO iris_tbl VALUES ( 4.7,3.2,1.3,0.2 ,'setosa');",
  "INSERT INTO iris_tbl VALUES ( 4.7,3.2,1.3,0.2 ,'setosa');",
  "INSERT INTO iris_tbl VALUES ( 4.5,3.2,1.3,0.2 ,'setosa');",
  "INSERT INTO iris_tbl VALUES ( 4.2,3.2,1.3,0.2 ,'versicolor');",
  "INSERT INTO iris_tbl VALUES ( 4.1,3.2,1.3,0.2 ,'versicolor');",
  "select * from iris_tbl;")
exeQuery(v)

## 05. 테이블의 전체값 삭제하기(TRUNCATE)
## TRUNCATE 테이블명
## TRUNCATE iris_tbl
v <- c(
  "TRUNCATE iris_tbl;",
  "select * from iris_tbl;")
exeQuery(v)


