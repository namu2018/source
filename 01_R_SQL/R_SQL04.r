### 01. DB 관련

'''
[1] DB 생성
CREATE DATABASE `데이터베이스명` CHARACTER SET utf8 COLLATE utf8_general_ci;

[2] DB 삭제 
DROP DATABASE `데이터베이스명`;

[3] DB 열람
SHOW DATABASES;

[4] DB 선택
USE `데이터베이스명`
'''

library(RODBC)
my <- odbcConnect("mysql", uid="root", pwd="toto1234");
sqlQuery(my, "show databases");

DBShowSql = "show databases;"
### DB 생성
### CREATE DATABASE `데이터베이스명` CHARACTER SET utf8 COLLATE utf8_general_ci;
sql = "CREATE DATABASE `DB01` CHARACTER SET utf8 COLLATE utf8_general_ci;"
sqlQuery(my, sql);
sqlQuery(my, DBShowSql);

### DB 삭제 
### DROP DATABASE `데이터베이스명`;
sql = "DROP DATABASE `db01`"
sqlQuery(my, sql);
sqlQuery(my, DBShowSql);


### DB 열람
### SHOW DATABASES;

### DB 선택
### USE `데이터베이스명`
sqlQuery(my, "use mysql");
sqlQuery(my, "show tables;");

### 02. 테이블 관련
'''
[1] Table 생성

CREATE DATABASE `데이터베이스명` CHARACTER SET utf8 COLLATE utf8_general_ci;

[2] DB 삭제 
DROP DATABASE `데이터베이스명`;

[3] DB 열람
SHOW DATABASES;

[4] DB 선택
USE `데이터베이스명`
'''

## [1] Table 생성
sql = paste("CREATE DATABASE `DB01` CHARACTER SET utf8 COLLATE utf8_general_ci;")
sqlQuery(my, sql);
sqlQuery(my, "show databases");
sqlQuery(my, "use DB01");

R_sql <- sqlQuery(my,"
        CREATE TABLE  `iris_tbl`  (
            Sepal_length FLOAT(3),
            Sepal_Width FLOAT(3),
            Petal_length FLOAT(3),
            Petal_Width FLOAT(3),
            Species VARCHAR(20)
        );
")

R_sql <- sqlQuery(my,"
        CREATE TABLE  `student`  (
                  Sepal_length FLOAT(3),
                  Sepal_Width FLOAT(3),
                  Petal_length FLOAT(3),
                  Petal_Width FLOAT(3),
                  Species VARCHAR(20)
        );
")

sqlQuery(my, "show tables;");

## [2] 스키마 열람(schema list)
## desc table명 
sqlQuery(my, "desc iris_tbl;");

## [3] 테이블 삭제(table delete)
## DROP TABLE `테이블명`
sqlQuery(my, "drop table `student`;");
sqlQuery(my, "show tables;");



irisdf <- iris
irisdf
