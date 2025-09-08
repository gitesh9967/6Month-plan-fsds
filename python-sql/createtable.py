import mysql.connector

conn = mysql.connector.connect(host="localhost", user="root", password = "123456", database = "pythondb")

mycursor = conn.cursor()
mycursor.execute("create table students (name varchar(50), branch varchar(10), id int)")
mycursor.execute("show tables")

for X in mycursor:
    print(X)
    
    