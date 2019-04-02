### titanic data set 분석해 보기
# Use the titanic data set
library(sqldf)
install.packages("PASWR")
data(titanic3, package="PASWR")
colnames(titanic3)

'''
[1] "pclass"    "survived"  "name"      "sex"       "age"       "sibsp"     "parch"     "ticket"   
[9] "fare"      "cabin"     "embarked"  "boat"      "body"      "home.dest"
'''
summary(titanic3)
head(titanic3)

## (QA1) 나이별 비어 있지 않은 친구들의 명수는 얼마나 되는가?
sqldf('select age, count(*) from titanic3 where age is not null group by age')
table(titanic3$age)

## (QA2) 나이가 결측치가 아닌 친구들을 ggplot2를 이용해서 histgram을 그려보자.
library(ggplot2)
DF=sqldf('select age from titanic3 where age != "NA"')
qplot(DF$age,data=DF, geom="histogram")


## (QA3) 29살 사람들의 산 사람과 죽은 사람은 얼마나 될까?
DF=sqldf('select survived,count(*) from titanic3 where age=29 group by survived')
DF

DF=sqldf('select count(*) from titanic3 where age=29 group by survived')
DF
DF2=t(DF)  # 행과 열을 바꾸기
DF2
colnames(DF2) = c('Died', 'Survived')
DF2


#### iris dataset
irisTable <- iris
names(irisTable) <- c("Sepal_length", "Sepal_Width", "Petal_length", "Petal_width", "Species")
sqldf("select Species, Sepal_length from irisTable")
sqldf("select Species, count(*) from irisTable group by Species")       
sqldf("select Species, avg(Sepal_length) from irisTable group by Species")
sqldf("select Species, Sepal_length from irisTable group by Species having Sepal_length between 5 and 6")
sqldf("select Species, Sepal_length from irisTable group by Species having Species='virginica'")

## (QA4) 
colnames(titanic3)
DF=sqldf('select pclass,count(*), min(fare), max(fare) from titanic3 group by pclass')
DF

# 항구별 살고 죽은 사람들,
DF=sqldf('select embarked,survived, count(*)  from titanic3 group by embarked, survived')
DF

# 그렇다면 남녀 비율은?
DF=sqldf('select embarked,survived,sex, count(*)  from titanic3 group by embarked, survived, sex')
DF

# 그렇다면 남녀 비율은? 이중에 Southampton만 보자.
DF=sqldf('select embarked,survived,sex, count(*)  from titanic3 group by embarked, survived, sex having embarked="Southampton"')
DF
