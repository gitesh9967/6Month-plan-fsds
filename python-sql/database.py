import mysql.connector

conn = mysql.connector.connect(host="localhost", user="root", password = "123456")

if conn.is_connected():
    print("Connection established")
    
    
mycursor = conn.cursor()
mycursor.execute("create database pythondb")
print(mycursor)
    