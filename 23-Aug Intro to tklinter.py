import tkinter as tk

# Create the main appliction window
root = tk.Tk()
root.title("Simple Tkinter App") 
root.geometry("200x100")    #Set window size

#Function to print "Hello, World!"  in the console
def print_hello():
    print("Hello, World!")
    print('good bye')
    
#Creat a button that triggers the say_hello function
hello_button = tk.Button(root, text="Click Me", command=print_hello)  
hello_button.pack(pady=20)  #pack the button into the window

# Start the Tkinter event loop
root.mainloop()

