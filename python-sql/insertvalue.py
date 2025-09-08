import mysql.connector

conn = mysql.connector.connect(host="localhost", user="root", password = "123456", database = "pythondb")

mycursor = conn.cursor()

sql = "insert into students (name, branch, id) values (%s, %s, %s)"
#val = ("Rohan", "CSE", 1)

#If user want to create multiple values then you can creat a list

val = [("Rohan", "CSE", 1), ("Sohan", "ECE", 2), ("Mohan", "ME", 3)]
mycursor.executemany(sql, val)
conn.commit()
print(mycursor.rowcount, "record inserted")

