"""
Open Anaconda prompt/Cmd and change directory to the local repo and run the following command:
streamlit run Rps_src.py
"""

import numpy as np
import cv2 as cv
import streamlit as st
import random
import time

st.title('Rock Paper Scissors')
st.write('Play a move:')
pos_mov = ['Rock','Paper','Scissors']
prev = time.time()
your_score = 0
comp_score = 0

def skinmask(img):
    hsvim = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")
    skinRegionHSV = cv.inRange(hsvim, lower, upper)

    blurred = cv.blur(skinRegionHSV, (2,2))
    ret, thresh = cv.threshold(blurred,0,255,cv.THRESH_BINARY)
    return thresh

def getcnthull(mask_img):
    contours, hierarchy = cv.findContours(mask_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv.contourArea(x))
    hull = cv.convexHull(contours)
    return contours, hull

def getdefects(contours):
    hull = cv.convexHull(contours, returnPoints=False)
    defects = cv.convexityDefects(contours, hull)
    return defects

if st.button('Start Playing'):
    st.text('Prepare your move, screenshot will be taken every 5 seconds\nPress q while the img window is selected to terminate the program')
    cap = cv.VideoCapture(0) # '0' for webcam
    while cap.isOpened():
        _, img = cap.read()
        
        roi = img[50:300, 50:300]
        cv.rectangle(img,(50,50),(300,300),(0,0,255),1)
        
        try:
            mask_img = skinmask(roi)
            contours, hull = getcnthull(mask_img)
            #cv.drawContours(roi, [contours], -1, (255,255,0), 2)
            cv.drawContours(roi, [hull], -1, (0, 255, 255), 2)  
            
            defects = getdefects(contours)#contours)
            
            index = -1
            
            if defects is not None:
                cnt = 0
                for i in range(defects.shape[0]):  # calculate the angle
                    s, e, f, d = defects[i][0]
                    start = tuple(contours[s][0])
                    end = tuple(contours[e][0])
                    far = tuple(contours[f][0])
                    a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    
                    #Extra: Distance between point and convex hull
                    s = (a+b+c)/2
                    ar = np.sqrt(s*(s-a)*(s-b)*(s-c))
                    d = (2*ar)/a
                    
                    angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  #      cosine theorem
                          
                    if angle <= np.pi / 2 and d>30:  # angle less than 90 degree, treat as fingers
                        cnt += 1
                        cv.circle(roi, far, 4, [0, 0, 255], -1)
                
                if cnt > 0:
                    cnt = cnt+1
                if cnt == 2:
                    cv.putText(img,"Scissors", (0, 50), cv.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv.LINE_AA)
                    index = 2
                elif cnt == 5:
                    cv.putText(img,"Paper", (0, 50), cv.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv.LINE_AA)
                    index = 1
                elif cnt == 0:
                    cv.putText(img,"Rock", (0, 50), cv.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv.LINE_AA)
                    index = 0
                #else:
                #    cv.putText(img,"Invalid",(0,50), cv.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv.LINE_AA)
                #cv.putText(img, str(cnt), (0, 50), cv.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv.LINE_AA)
            cv.imshow("img", img)
            chek = round(time.time()-prev,2)
            if chek > 5.00: 
            #if st.button('Start playing'):
                st.image(img, caption='Your move')
                rand_num = random.randint(0,2)
                move = pos_mov[rand_num]
                st.text('Computer plays:'+move)
                if index == rand_num:
                    st.text('It\'s a draw!!')
                elif index == 0 and rand_num == 1:
                    st.text('Computer wins!!')
                    comp_score += 1
                elif index == 0 and rand_num == 2:
                    st.text('You win!!')
                    your_score +=1
                elif index == 1 and rand_num == 0:
                    st.text('You win!!')
                    your_score += 1
                elif index == 1 and rand_num == 2:
                    st.text('Computer wins!')
                    comp_score+=1
                elif index == 2 and rand_num == 0:
                    st.text('Computer wins!!')
                    comp_score += 1
                elif index == 2 and rand_num == 1:
                    st.text('You win!!')
                    your_score += 1                
                
                st.subheader('Score tally')
                st.text('You: '+str(your_score)+'  Computer: '+str(comp_score))
                prev = time.time()
        except:
            pass
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()