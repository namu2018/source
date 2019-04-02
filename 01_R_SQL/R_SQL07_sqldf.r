install.packages("sqldf")
library(sqldf)


library(MASS)
str(Cars93)

# sqldf를 이용하여 다음을 풀어보자.
# (1) Type으로 정렬한다. 
# (2) MPG.city, MPG.highway의 평균값을 보여준다.
R_sqldf_1 <- sqldf('
  select "Type" as "Car_Type", 
                   avg("MPG.city") as "mean_MPG.city", 
                   avg("MPG.highway") as "mean_MPG.highway"  
                   from Cars93 
                   group by Type
                   order by Type
                   ')
R_sqldf_1

### 결과값.
'''
Car_Type mean_MPG.city mean_MPG.highway
1  Compact      22.68750         29.87500
2    Large      18.36364         26.72727
3  Midsize      19.54545         26.72727
4    Small      29.85714         35.47619
5   Sporty      21.78571         28.78571
6      Van      17.00000         21.88889
'''


