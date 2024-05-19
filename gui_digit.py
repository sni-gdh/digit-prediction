#!/usr/bin/env python
# coding: utf-8

# In[13]:


from tkinter import *
import numpy as np
from PIL import ImageGrab
from DATA_RECOGNITION import make_predictions
import pickle

#from matplotlib import pyplot as plt

import time


window = Tk()
window.title("Digit recognition")

l1 = Label()


def MyProject():
    global l1

    widget = cv
    
    # Setting co-ordinates of canvas
    #have to add positions for different scales to get correct croped image for prediction.
    x = window.winfo_rootx() + widget.winfo_x()
    y = window.winfo_rooty() + widget.winfo_y()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()


    # Add delay to ensure window is fully rendered
    time.sleep(0.5)

    # Image is captured from canvas and is resized to (28 X 28) px
    img = ImageGrab.grab().crop((x, y, x1, y1)).resize((28,28))
    
    # Converting rgb to grayscale image
    img = img.convert('L')
    # Extracting pixel matrix of image and converting it to a vector of (784,1)
    x = np.asarray(img)
    vec = x.flatten().reshape(784,1)

#     plt.imshow(x, cmap='gray')
#     plt.title("Captured Image")
#     plt.show()
    # Loading weights and baises
    with open("trained_params.pkl", "rb") as dump_file:
        W1, b1, W2, b2 = pickle.load(dump_file)

    # Calling function for prediction
    pred = make_predictions(vec/255,W1,b1,W2,b2)

    # Displaying the result
    l1.destroy()
    l1 = Label(window, text="Digit = " + str(pred[0]), font=('Vrinda', 20))
    l1.place(x=230, y=420)


lastx, lasty = None, None


# Clears the canvas
def clear_widget():
    global cv, l1
    cv.delete("all")
    l1.destroy()


# Activate canvas
def event_activation(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y


# To draw on canvas
def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=20, fill='white', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y


# Label
L1 = Label(window, text="Digit Recoginition", font=('Vrinda', 25), fg="blue")
L1.place(x=35, y=10)

# Button to clear canvas
b1 = Button(window, text="1. Clear Canvas", font=('Vrinda', 15), bg="orange", fg="black", command=clear_widget)
b1.place(x=120, y=370)

# Button to predict digit drawn on canvas
b2 = Button(window, text="2.  Prediction", font=('Vrinda', 15), bg="white", fg="red", command=MyProject)
b2.place(x=320, y=372)

# Setting properties of canvas
cv = Canvas(window, width=330, height=300, bg='black')
cv.place(x=120, y=70)

cv.bind('<Button-1>', event_activation)

window.configure(background = 'whitesmoke')
window.geometry("600x600")
window.mainloop()


# In[ ]:





# In[ ]:




