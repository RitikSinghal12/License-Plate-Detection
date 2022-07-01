import cv2 as cv 
import numpy as np
import imutils 
import easyocr

img = cv.imread('car2.jpeg')
img = cv.resize(img,(500,300))
cv.imshow('Image',img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Filtering
bfilter = cv.bilateralFilter(gray, 11, 17, 17)

# Edge
canny = cv.Canny(bfilter,170, 200)
#cv.imshow('Canny',canny)

# Contours
keypoints = cv.findContours(canny.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv.contourArea, reverse=True)[:50]

blank = np.zeros(img.shape, dtype='uint8') 
cv.drawContours(blank, contours, -1, (0,0,255), 1)
#cv.imshow('Contours', blank)

location = None
for contour in contours:
    perimeter = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.01 * perimeter, True)
    if len(approx) == 4:
        location = approx
        break

print(location)

mask = np.zeros(gray.shape, dtype='uint8')
new_image = cv.drawContours(mask,[location], 0, 255, -1)
new_image = cv.bitwise_and(img,img,mask = mask)

#cv.imshow('Masked Image', new_image)

(x,y) = np.where(mask==255)
(x1,y1) = (np.min(x), np.min(y))
(x2,y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]

cv.imshow('Number Plate', cropped_image)

reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)
print(result)
text = result[0][-2]
print(text)
font = cv.FONT_HERSHEY_SIMPLEX
#res = cv.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv.LINE_AA)
res = cv.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)

cv.imshow('Detected Number Plate', res)

cv.waitKey(0)
cv.destroyAllWindows()