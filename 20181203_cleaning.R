library(sqldf)
library(dplyr)
rm(list=ls())
options("scipen"=100)
getwd()
setwd("C:/choidontouch/Auction_master_kr")
setwd("C:/choi/bigdata/data/auction")
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
plot(train_rent_reg$Minimum_sales_price, train_rent_reg$Total_building_auction_area)
#train_rent_reg$diff_p <- (train_rent_reg$Hammer_price-train_rent_reg$Total_appraisal_price)/train_rent_reg$Total_building_auction_area

train_rent_reg$Final_yy<-as.integer(substring(train_rent_reg$Final_auction_date,1,4))
train_rent_reg$Final_mm <- as.integer(substring(train_rent_reg$Final_auction_date,6,7))
#train_rent_reg <- train_rent_reg[,-17]
names(train_rent_reg)
'''
train_rent_reg$T<- ifelse(train_rent_reg$Hammer_price==0,0,1)
train_rent_reg$T <- factor(train_rent_reg$T)
library(dplyr)
library(ggplot2)
View(train_rent_reg)
g1<-ggplot(train_rent_reg,aes(train_rent_reg$Total_appraisal_price,
                              train_rent_reg$Bid_class))+
  geom_point(aes(color=train_rent_reg$T))+xlim(0,5000000000)
g1

'''
table(TEST_T$TPDMP)

TEST_T <- train_rent_reg %>% filter(train_rent_reg$Hammer_price==0)
TRAIN_T <- train_rent_reg %>% filter(train_rent_reg$Hammer_price!=0)
subset(TRAIN_T,TPDMP==7.45) 
TRAIN_T<-TRAIN_T[-1767,]
str(train_rent_reg)
plot(TRAIN_T$Auction_count, TRAIN_T$TPDMP)
train_s <- sqldf("SELECT * FROM TRAIN_T WHERE addr_do='서울'")
train_b <- sqldf("SELECT * FROM TRAIN_T WHERE addr_do='부산'")
test_s <- sqldf("SELECT * FROM TEST_T WHERE addr_do='서울'")
test_b <- sqldf("SELECT * FROM TEST_T WHERE addr_do='부산'")
dim(train_s)
dim(train_b)
str(TRAIN_T)
table(TRAIN_T$TPDMP); table(TEST_T$TPDMP)
subset(T,TPDMP==7.45) 
TRAIN_T<-TRAIN_T[-1767,]
sqldf("SELECT * FROM TRAIN_T where TPDMP=7.45")
plot(train_s$diff_p, train_s$TPDMP)
plot(train_b$TPDMP, train_b$diff_p)
table(test_b$TPDMP)
table(train_b$TPDMP)
table
str(train_b)
TEST_T
TRAIN_T
names(TRAIN_T)

str(train_1)
str(test_1)
lm_1 <- lm(Hammer_price~., data=train_1)
pre_1 <- predict(lm_1, newdata=test_1)
sub$Hammer_price <- lm_pre
sub
write.csv(sub, file='try_1120_1.csv', row.names=FALSE)
