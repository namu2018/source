library(xgboost)
library(xggrid)
library(sqldf)
library(dplyr)
rm(list=ls())
options("scipen"=100)
setwd("C:/choidontouch/Auction_master_kr")
#setwd("C:/choi/bigdata/data/auction")
rent <- read.csv("Auction_rent.csv", na.strings=c("", NA))
train <- read.csv("Auction_master_train.csv", na.strings=c("", NA))
test <- read.csv("Auction_master_test.csv" , na.strings=c("", NA))
regist <- read.csv("Auction_regist.csv", na.strings=c("", NA))
sub <- read.csv("Auction_submission.csv", na.strings=c("", NA))


str(rent)

auc <- rbind(train, test)
names(rent) <- c("Auction_key", "Rent_class","Purpose_use" ,"Occupied_part","Rent_date",
                 "Rent_deposit","Rent_monthly_price","Specific"  )
rent$junip <- ifelse(rent$Rent_class=="전입    ",1,0)
rent$jumyu <- ifelse(rent$Rent_class=="점유    ",1,0)
head(rent)
junip <- sqldf("SELECT Auction_key, count(*) junipsu, sum(Rent_deposit) junsega, sum(Rent_monthly_price) yulsega FROM rent where junip=1 group by Auction_key") 

jumyu <- sqldf("SELECT Auction_key, count(*) jumyusu,sum(Rent_deposit) junsega, sum(Rent_monthly_price) yulsega FROM rent where jumyu=1 group by Auction_key")
junip

train_rent<- sqldf("SELECT * FROM auc LEFT JOIN
                   junip
                   USING(Auction_key)")
train_rent<- sqldf("SELECT * FROM train_rent LEFT JOIN
                   (SELECT Auction_key, count(*) jumyusu,sum(Rent_deposit) junsega1, sum(Rent_monthly_price) yulsega1 FROM rent where jumyu=1 group by Auction_key)
                   USING(Auction_key)")
head(jumyu,10)
head(junip,10)
head(train_rent)

str(regist)
table(regist$Regist_type)
togi_deungi <- sqldf("SELECT Auction_key,Regist_type togi FROM regist WHERE Regist_type='토지별도등기' ")
togi_deungi
deungibi <- sqldf("SELECT Auction_key, sum(Regist_price) deungibi FROM regist
                  WHERE Regist_price is not null
                  GROUP BY Auction_key")

train_rent_reg <-sqldf("SELECT * FROM train_rent LEFT JOIN
                       togi_deungi 
                       USING(Auction_key)")
train_rent_reg <-sqldf("SELECT * FROM train_rent_reg LEFT JOIN
                       deungibi
                       USING(Auction_key)")
names(train_rent_reg)
train_rent_reg$junipsu <- ifelse(is.na(train_rent_reg$junipsu)==TRUE, 0, train_rent_reg$junipsu)
train_rent_reg$junsega <- ifelse(is.na(train_rent_reg$junsega)==TRUE, 0, train_rent_reg$junsega)
train_rent_reg$yulsega <- ifelse(is.na(train_rent_reg$yulsega)==TRUE, 0, train_rent_reg$yulsega)
train_rent_reg$jumyusu <- ifelse(is.na(train_rent_reg$jumyusu)==TRUE, 0, train_rent_reg$jumyusu)
train_rent_reg$junsega1 <- ifelse(is.na(train_rent_reg$junsega1)==TRUE, 0, train_rent_reg$junsega1)
train_rent_reg$yulsega1 <- ifelse(is.na(train_rent_reg$yulsega1)==TRUE, 0, train_rent_reg$yulsega1)
train_rent_reg$togi <- ifelse(is.na(train_rent_reg$togi)==TRUE, 0, train_rent_reg$togi)
train_rent_reg$deungibi <- ifelse(is.na(train_rent_reg$deungibi)==TRUE, 0, train_rent_reg$deungibi)
train_rent_reg$togi<- ifelse(train_rent_reg$togi=='토지별도등기',1,0)


train_rent_reg$H_p <- train_rent_reg$Hammer_price/train_rent_reg$Total_building_auction_area
train_rent_reg$M_p <- train_rent_reg$Minimum_sales_price/train_rent_reg$Total_building_auction_area
train_rent_reg$T_p <- train_rent_reg$Total_appraisal_price/train_rent_reg$Total_building_auction_area
train_rent_reg$TPDMP <- round(train_rent_reg$T_p/train_rent_reg$M_p,digits = 2)
table(train_rent_reg$TPDMP)
#train_rent_reg$diff_p <- (train_rent_reg$Hammer_price-train_rent_reg$Total_appraisal_price)/train_rent_reg$Total_building_auction_area

train_rent_reg$First_auction_date<-as.integer(substring(train_rent_reg$First_auction_date,1,4))
train_rent_reg$Final_auction_date<-as.integer(substring(train_rent_reg$Final_auction_date,1,4))
train_rent_reg$Appraisal_date<-as.integer(substring(train_rent_reg$Appraisal_date,1,4))
train_rent_reg$Close_date<-as.integer(substring(train_rent_reg$Close_date,1,4))
train_rent_reg$Preserve_regist_date <-as.integer(substring(train_rent_reg$Preserve_regist_date,1,4))
train_rent_reg$Share_auction_YorN <- as.integer(train_rent_reg$Share_auction_YorN)
train_rent_reg$Apartment_usage <- as.integer(train_rent_reg$Apartment_usage) 
train_rent_reg$Appraisal_company <- as.integer(train_rent_reg$Appraisal_company)
train_rent_reg$Close_result <- as.integer(train_rent_reg$Close_result)
train_rent_reg$Auction_class <- as.integer(train_rent_reg$Auction_class)
train_rent_reg$Bid_class <- as.integer(train_rent_reg$Bid_class)
train_rent_reg$addr_do <- ifelse(train_rent_reg$addr_do=='서울',1,0)
train_rent_reg$addr_si <- as.integer(train_rent_reg$addr_si)
train_rent_reg$addr_dong <- as.integer(train_rent_reg$addr_dong)
train_rent_reg$Final_result <- as.integer(train_rent_reg$Final_result)
train_rent_reg$Creditor <- as.integer(train_rent_reg$Creditor)
names(train_rent_reg)

train_rent_reg <- train_rent_reg[,-c(34:36)]
train_rent_reg <- train_rent_reg[,-c(27,30)]
train_rent_reg <- train_rent_reg[,-27]
#train_rent_reg <- train_rent_reg[,-17]
str(train_rent_reg)


TEST_T <- train_rent_reg %>% filter(train_rent_reg$Hammer_price==0)
TRAIN_T <- train_rent_reg %>% filter(train_rent_reg$Hammer_price!=0)
subset(TRAIN_T,TPDMP==7.45) 
TRAIN_T<-TRAIN_T[-1767,]
str(train_rent_reg)
plot(TRAIN_T$Auction_count, TRAIN_T$TPDMP)

#######################################################################################

train_1 <- subset(TRAIN_T, TPDMP==1)
test_1 <- subset(TEST_T, TPDMP==1)
train_2 <- subset(TRAIN_T, TPDMP==1.25)
test_2 <- subset(TEST_T, TPDMP==1.25)
train_3 <- subset(TRAIN_T, TPDMP==1.56)
test_3 <- subset(TEST_T, TPDMP==1.56)
train_4 <- subset(TRAIN_T, TPDMP >=2.44)
test_4 <- subset(TEST_T, TPDMP >=2.44)
train_5 <- subset(TRAIN_T, TPDMP == 1.95)
test_5 <- subset(TEST_T, TPDMP == 1.95)
str(test_5)
dim(test_1);dim(test_2);dim(test_3);dim(test_4);dim(test_5)


lm <- lm(H_p~ M_p+junipsu, data=train_1)
pre <- predict(lm, newdata=test_1)
test_1$Hammer_price <- pre

lm <- lm(H_p~ M_p+junipsu, data=train_2)
pre <- predict(lm, newdata=test_2)
test_2$Hammer_price <- pre

lm <- lm(H_p~ M_p+Total_building_area, data=train_3)
pre <- predict(lm, newdata=test_3)
test_3$Hammer_price <- pre

lm <- lm(H_p~ M_p+junipsu+Auction_count+Total_land_real_area , data=train_4)
pre <- predict(lm, newdata=test_4)
test_4$Hammer_price <- pre

lm <- lm(H_p~ M_p+point.x+junipsu+Auction_count , data=train_5)
pre <- predict(lm, newdata=test_5)
test_5$Hammer_price <- pre



test
test <- rbind(test_1, test_2, test_3, test_4, test_5)
test$Hammer_price <- test$Hammer_price*test$Total_building_auction_area
test$Hammer_price

sub1 <- sqldf("SELECT s.Auction_key,t.Hammer_price   FROM sub s
              LEFT JOIN test t USING(Auction_key) ")
sub1
write.csv(sub1, file='try_1217lm1_mp.csv', row.names=FALSE)
#71691336
sqldf("SELECT Hammer_price from test where Auction_key=2384")
