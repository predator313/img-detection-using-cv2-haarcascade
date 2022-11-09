# Write a Python Script that captures images from your webcam video stream
# Extracts all Faces from the image frame (using haarcascades)
# Stores the Face information into numpy arrays

# 1. Read and show video stream, capture images
# 2. Detect Faces and show bounding box (haarcascade)
# 3. Flatten the largest face image(gray scale) and save in a numpy array
# 4. Repeat the above for multiple people to generate training data


import cv2
import pandas as pd
import numpy as np
from tkinter import messagebox
import tkinter as tk
import os


def register(txt):
	t = tk.Tk()
	t.geometry('+1050+120')
	t.configure(background='#122c57')
	l1 = tk.Label(t,text="taking 10 photos\n",fg='white',bg='#122c57')
	l1.pack()
	#Init Camera

	cap = cv2.VideoCapture(0)

	# Face Detection
	face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

	skip = 0
	face_data = []
	dataset_path = './data/'
	name = txt.get().upper()
	# counter = 10
	while True:
		ret,frame = cap.read()

		if ret==False:
			continue

		frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		

		faces = face_cascade.detectMultiScale(frame,1.3,5)
		if len(faces)==0:
			# print('your face is not visible \n please get into the frame')
			continue
			
		faces = sorted(faces,key=lambda f:f[2]*f[3])

		# Pick the last face (because it is the largest face acc to area(f[2]*f[3]))
		for face in faces[-1:]:
			x,y,w,h = face
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

			#Extract (Crop out the required face) : Region of Interest
			offset = 10
			face_section = frame[y:y+h,x:x+w]
			face_section = cv2.resize(face_section,(100,100))

			skip += 1
			if skip%8==0:
				face_data.append(face_section)
				l2 = tk.Label(t,text=str(len(face_data))+"\n",fg='white',bg='#122c57')
				l2.pack()				
				print(len(face_data))


		cv2.imshow("Frame",frame)
		cv2.imshow("Face Section",face_section)
		t.update()
		key_pressed = cv2.waitKey(1) & 0xFF
		if key_pressed == ord('q') or len(face_data) >= 10:
			t.destroy()
			break



	
	cap.release()
	cv2.destroyAllWindows()

	# Convert our face list array into a numpy array
	face_data = np.asarray(face_data)
	face_data = face_data.reshape((face_data.shape[0],-1))
	print(face_data.shape)

	# Save this data into file system
	np.save(dataset_path+name+'.npy',face_data)
	print("Data Successfully save at "+dataset_path+name+'.npy')

	# Registering student in csv file
	
	# if file does not exist write header
	row = np.array([name]).reshape((1,2))
	df = pd.DataFrame(row) 
	if not os.path.isfile('name_data.csv'):
	   df.to_csv('name_data.csv', header=['name'],index=False)
	else: # else it exists so append without writing the header
	   df.to_csv('name_data.csv', mode='a', header=False,index=False)
		
	
	tk.messagebox.showinfo("Notification",) 
