# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 12:11:42 2018

@author: ktm
"""

# -*- coding: utf-8 -*-
## 여성의류 검색 단어
## url = "http://browse.auction.co.kr/search?keyword=%EC%97%AC%EC%84%B1%EC%9D%98%EB%A5%98&itemno=&nickname=&encKeyword=%25EC%2597%25AC%25EC%2584%25B1%25EC%259D%2598%25EB%25A5%2598&arraycategory=&frm=&dom=auction&isSuggestion=No&retry="
## url = "http://browse.auction.co.kr/search?keyword=%ED%99%94%EC%9E%A5%ED%92%88&itemno=&nickname=&encKeyword=%25ED%2599%2594%25EC%259E%25A5%25ED%2592%2588&arraycategory=&frm=&dom=auction&isSuggestion=No&retry="

## computer
## http://browse.auction.co.kr/search?keyword=computer&itemno=&nickname=&encKeyword=computer&arraycategory=&frm=&dom=auction&isSuggestion=No&retry=

## 스타일 난다.


#%% utf-8로 인코딩
## url = "http://browse.auction.co.kr/search?keyword=%EC%97%AC%EC%84%B1%EC%9D%98%EB%A5%98&itemno=&nickname=&encKeyword=%25EC%2597%25AC%25EC%2584%25B1%25EC%259D%2598%25EB%25A5%2598&arraycategory=&frm=&dom=auction&isSuggestion=No&retry="
import requests as rq
from bs4 import BeautifulSoup
import lxml
url = "http://browse.auction.co.kr/search?keyword={}&itemno=&nickname=&encKeyword={}&arraycategory=&frm=&dom=auction&isSuggestion=No&retry=".format("여성의류", "여성의류")
url

res = rq.get(url)
res.url
html = res.content
soup = BeautifulSoup(html, 'lxml')

#%% 타이틀 가져오기 
soup_item = soup.find_all("div", class_="section--itemcard")
soup_item[1]
num = len(soup_item)
num

title = []
price_sales = []
price_ori = []
company = []
for i in range(0, num):
    # 상품명
    soup_title = soup_item[i].find("span", class_="text--title")
    print(soup_title)
    if soup_title is not None:
        title_txt = soup_title.text
        print(title_txt)
        title.append(title_txt)
    else:
        title.append("")
    
    # 상품가격(할인적용금액)
    soup_price = soup_item[i].find("strong", class_="text--price_seller")
    print(soup_price)
    if soup_price is not None:
        price_txt = soup_price.text
        price_sales.append(price_txt)
    else:
        price_sales.append("")
    
    # 상품가격(원래 금액)
    soup_price_ori = soup_item[i].find("strong", class_="text--price_original")
    print(soup_price_ori)
    if soup_price_ori is not None:
        price_ori_txt = soup_price_ori.text
        price_ori.append(price_ori_txt)   
    else:
        price_ori.append("")

    # 회사
    soup_company = soup_item[i].find("span", class_="text")
    print(soup_company)
    if soup_company is not None:
        soup_company_txt = soup_company.text
        company.append(soup_company_txt)   
    else:
        company.append("스마일배송")
        
        
    # 판매자
    #soup_company = soup_item[i].find("span", class_="text")
    #print(soup_company)
#%%
import pandas as pd

print(title, len(title))
print(price_sales, len(price_sales))
print(price_ori, len(price_ori))
print(company, len(company))

title1 = pd.Series(title)
price_sales1 = pd.Series(price_sales)
price_ori1 = pd.Series(price_ori)
company_name = pd.Series(company)
#%% 
dat = pd.DataFrame({ "title" : title , 
                   "price_sales" : price_sales, 
                   "price_origin" : price_ori,
                   "company_name" : company_name }, columns=['title','price_sales','price_origin', "company_name"] )
dat

#dat.to_csv("company_info.csv", index=False, encoding="utf-8")  # MAC 유저
dat.to_csv("company_info.csv", index=False, encoding="EUCKR")  # Window EXCEL 한글 볼 때
#%%
dat