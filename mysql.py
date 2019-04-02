# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 09:06:19 2018

@author: ktm
"""

import mysql.connector
mydb = mysql.connector.connect(
        host ="localhost",
        user ="root",
        passwd ="qwer1234"
)

print(mydb)

mycursor = mydb.cursor()
