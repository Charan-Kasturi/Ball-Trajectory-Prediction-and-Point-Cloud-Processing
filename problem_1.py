import cv2 as cv,os
import numpy as np
import matplotlib.pyplot as plt,math
# # Creating path
CURRENT_DIR = os.path.dirname(__file__)
path=os.path.join(CURRENT_DIR,'ball.mov')
# Loading the video
vid = cv.VideoCapture(path)

def ball_detection(xy,mask):
    count=0
    minx=0
    miny=0
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if mask[x][y]==255:
                if x<minx or minx==0:
                    minx=x
                if y<miny or miny==0:
                    miny=y
                count+=1
    #radius
    # print(count)
    # print(xy)
    if count!=0 and count>50:
        r=float(math.sqrt(count/3.14))
        centre_x= minx+r
        centre_y=miny+r
        xy=np.append(xy,[[centre_x,centre_y]],0)
    return xy

xy=np.empty((0,2))
while(vid.isOpened()):
    ret,frame=vid.read()
    if ret==True:
        hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
        
        red_1=np.array([7,5,10])
        red_2=np.array([30,255,255])
        mask=cv.inRange(frame,red_1,red_2)
        mask=cv.rectangle(mask,(650,0),(mask.shape[1],300),(0,0,0),-1)
        mask=cv.rectangle(mask,(0,465),(400,1000),(0,0,0),-1)
        mask=cv.rectangle(mask,(500,500),(1000,800),(0,0,0),-1)

        cv.imshow('frame',frame)
        
        key =cv.waitKey(1)
        if key ==ord('q'):
            break
        xy=ball_detection(xy,mask)
    else:
        break


#plot
x=np.linspace(0,1400,10000)
y=np.linspace(0,700,10000)
vid.release()
cv.destroyAllWindows()
plt.plot(xy[:,1],xy[:,0],'ro')
plt.show()

# ******************************************************************************************************************************

n=np.shape([x])
x_1=xy[:,0]**2
x_2=xy[:,0]*1
a1=np.array([x_1])
a2=np.array([x_2])
a3=np.ones_like(x_1)
#print(a1)
#print(a2)
#print(a3)
A=np.vstack((a1,a2,a3))
# A=np.array([[1,2,3],[2,3,4]])
A_T=A.T
y=xy[:,1]
# Y=np.array([y1 for y1 in y]).T
Y=np.vstack(y)
Y_T=Y.T
# inverse
A_T_inv=np.linalg.inv(np.dot(A,A_T))
P=np.dot(A_T,A_T_inv)
B=np.dot(Y_T,P)
# graph
print(B)
a=B[0:,0]
b=B[0:,1]
c=B[0:,2]

y = a*x**2 + b*x + c
plt.plot(x, y)

# Add labels and title
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title("Graph")

# Display the graph
plt.show()
# ************************************************************************************************************
y_new=xy[0][1]+300
# new equation
c_new=c-y_new
discriminant = b**2 - 4*a*c_new

# check if the roots are real or complex
if discriminant < 0:
    print("The roots are complex.")
else:
    # calculate the roots using the quadratic formula
    root1 = (-b + math.sqrt(discriminant)) / (2*a)
    root2 = (-b - math.sqrt(discriminant)) / (2*a)
    #nprint("The roots are:", root1, "and", root2)
    x_new=np.maximum(root1,root2)
    print("The x-coordinate of the ball’s landing spot in pixels is:",x_new)