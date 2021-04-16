# import the necessary packages
import imutils
import cv2
import math
 
# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread("img3.jpg")
image = cv2.resize(image,(800, 600))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
 
# threshold the image, then perform a series of erosions and dilations to remove any small regions of noise
thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)
 
# find contours in thresholded image, then grab the largest one
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)

# compute the center of the contour
M = cv2.moments(c)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
print("Center : ({}, {} ) ".format( cX, cY))


# draw the contour and center of the shape on the image
cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
cv2.putText(image, "(c_X, c_Y) ", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#find the extremeleft finger tip
extLeft = tuple(c[c[:, :, 0].argmin()][0])
(ex_x, ex_y) = extLeft
print ("Leftmost point : ({}, {}) ".format(ex_x, ex_y))

cv2.circle(image, extLeft, 7, (0, 0, 255), -1)

#calculating the distace between center and topmost point
leftdist = math.sqrt((ex_x - cX)**2 + (ex_y - cY)**2)
print("distance from center to leftmost finger : {} ".format(leftdist))


#find the rightmost finger tip
extRight = tuple(c[c[:, :, 0].argmax()][0])
(ex_x, ex_y) = extRight
print ("Rightmost point : ({}, {}) ".format(ex_x, ex_y))

cv2.circle(image, extRight, 7, (0, 0, 255), -1)

#calculating the distace between center and topmost point
rightdist = math.sqrt((ex_x - cX)**2 + (ex_y - cY)**2)
print("distance from center to rigthmost finger : {} ".format(rightdist))

#findig the upper most  finger point
extTop = tuple(c[c[:, :, 1].argmin()][0])
(top_x, top_y) = extTop
print ("Topmost point : ({}, {}) ".format(top_x, top_y))

cv2.circle(image, extTop, 8, (255, 0, 0), -1)
cv2.putText(image, "(top_x, top_y)", (top_x - 20, top_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#calculating the distace between center and topmost point
dist = math.sqrt((top_x - cX)**2 + (top_y - cY)**2)
print("distance from center to middle finger: {} ".format(dist))


#Drawing the Y axis point where line from center meets
cv2.circle(image, (cX, top_y), 7, (255, 0, 255), -1)
cv2.putText(image, "(X_center,top_y)", (cX - 20, top_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#Draw the line joining the point center to tip and center to y_axis
cv2.line(image, (cX,cY), extTop, (0,0,0), 2)
cv2.line(image, (cX,cY), (cX ,top_y), (255,255,0), 2)

#find angle 
theta = math.degrees(math.atan2(top_y - cY, top_x - cX) - math.atan2(top_y - cY, cX - cX))
print("angle : {} ".format(theta))



#rotate the image about the axis with angle calculated
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, theta, 1.0)
rotated = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

#Print the original image and the rotated image.
cv2.imshow("Image", image)
cv2.imshow("Rotated", rotated)
cv2.waitKey(30000)

