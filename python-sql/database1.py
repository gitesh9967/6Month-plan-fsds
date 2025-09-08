import mysql.connector

conn = mysql.connector.connect(host="localhost", user="root", password = "123456")

mycursor = conn.cursor()

mycursor.execute("show databases")
for X in mycursor:
    print(X)
    
    