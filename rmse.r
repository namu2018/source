install.packages("tidyverse")
library(tidyverse)
library(stringr)
install.packages("xggrid")
library(lubridate)
library(DT)
library(caret)
library(ggplot2)
library(leaflet)
library(corrplot)
library(boot) #for diagnostic plots
library(dplyr)
library(xgboost)
library(xggrid)

fillColor = "#FFA07A"
fillColor2 = "#F1C40F"

datatable(head(train_s), style="bootstrap", class="table-condensed", options = list(dom = 'tp',scrollX = TRUE))
str(train_s)
TRAIN_T$Auction_class <- as.integer(TRAIN_T$Auction_class)
TEST_T$Auction_class <- as.integer(TEST_T$Auction_class)
TRAIN_T$Bid_class <- as.integer(TRAIN_T$Bid_class)
TEST_T$Bid_class <- as.integer(TEST_T$Bid_class)
TRAIN_T$addr_do <- ifelse(TRAIN_T$addr_do=='서울',1,0)
TEST_T$addr_do <- ifelse(TEST_T$addr_do=='서울',1,0)
TRAIN_T$addr_si <- as.integer(TRAIN_T$addr_si)
TEST_T$addr_si <- as.integer(TEST_T$addr_si)
TRAIN_T$addr_dong <- as.integer(TRAIN_T$addr_dong)
TEST_T$addr_dong <- as.integer(TEST_T$addr_dong)
TRAIN_T$Fiyear = as.integer(substr(TRAIN_T$Final_auction_date,1,4))
TRAIN_T$Fiyear
names(TRAIN_T)

TRAIN_T_N = TRAIN_T %>%
  select(-Auction_key,-Appraisal_company,-Appraisal_date, -c(16:19), -c(23:29),-c(32:38))

TEST_T_N = TEST_T %>%
  select(-Auction_key,-Appraisal_company,-Appraisal_date, -c(16:19), -c(23:29),-c(32:38))

str(TRAIN_T_N)


train_s <- sqldf("SELECT * FROM TRAIN_T_N WHERE addr_do=1")
test_s <- sqldf("SELECT * FROM TEST_T_N WHERE addr_do=1")
train_b <- sqldf("SELECT * FROM TRAIN_T_N WHERE addr_do=0")
test_s <- sqldf("SELECT * FROM TEST_T_N WHERE addr_do=1")

formula = Hammer_price~ .

fitControl <- trainControl(method="cv",number = 5)
model_s_s = train(formula, data = train_s_s,method = "lm",trControl = fitControl,metric="RMSE")
names(train_s_s)
train_s_s <- train_s %>% select(20,c(30:31))
model_g_b = train(formula, data = train_b,
                         method = "lm",trControl = fitControl,metric="RMSE")


model_xg_s = train(formula, data = train_s,
                            method = "xgbTree",trControl = fitControl,
                           na.action = na.pass,metric="RMSE")

importance = varImp(model_xg_s)

PlotImportance(importance)




PlotImportance = function(importance)
{
  varImportance <- data.frame(Variables = row.names(importance[[1]]), 
                              Importance = round(importance[[1]]$Overall,2))
  
  # Create a rank variable based on importance
  rankImportance <- varImportance %>%
    mutate(Rank = paste0('#',dense_rank(desc(Importance))))
  
  rankImportancefull = rankImportance
  
  ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                             y = Importance)) +
    geom_bar(stat='identity',colour="white", fill = fillColor) +
    geom_text(aes(x = Variables, y = 1, label = Rank),
              hjust=0, vjust=.5, size = 4, colour = 'black',
              fontface = 'bold') +
    labs(x = 'Variables', title = 'Relative Variable Importance') +
    coord_flip() + 
    theme_bw()
  
  
}

PlotImportance(importance)

model_xg_s


summary(train_s)
train_s$DIFF_p <- (train_s$Hammer_price - train_s$Total_appraisal_price)/train_s$Total_building_auction_area
plot(train_s$DIFF_p, train_s$TPDMP)
train_s$TPDMP <- train_s$T_p/train_s$M_p
plot(train_s$Total_appraisal_price, train_s$TPDMP)
table(train_s$TPDMP)
train_b$TPDMP <- train_b$T_p/train_b$M_p
table(train_b$TPDMP)

View(train_b)
table(train_s$deungibi)