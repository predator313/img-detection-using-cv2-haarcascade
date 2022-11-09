import tkinter as tk 
from face_data_collect import register
from face_recognition import recognize

window = tk.Tk()  
window.title("Face_Recogniser") 
window.configure(background ='white') 
window.grid_rowconfigure(0, weight = 1) 
window.grid_columnconfigure(0, weight = 1) 
message = tk.Label( 
    window, text ="Face-Recognition-System",  
    bg ="green", fg = "white", width = 50,  
    height = 3, font = ('times', 30, 'bold'))  
      
message.place(x = 200, y = 20) 

lbl2 = tk.Label(window, text ="Name",  
width = 20, fg ="green", bg ="white",  
height = 2, font =('times', 15, ' bold '))  
lbl2.place(x = 400, y = 300) 
  
txt = tk.Entry(window, width = 20,  
bg ="white", fg ="green",  
font = ('times', 15, ' bold ')  ) 
txt.place(x = 700, y = 315) 
  
trainImg = tk.Button(window, text ="Data Collection",  
command=lambda :register(txt),
fg ="white", bg ="green",  
width = 20, height = 3, activebackground = "Red",  
font =('times', 15, ' bold '),
) 
trainImg.place(x = 500, y = 500) 

trackImg = tk.Button(window, text ="Testing",  
command = recognize, 
 fg ="white", bg ="green",  
width = 20, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
trackImg.place(x = 800, y = 500) 
window.mainloop() 