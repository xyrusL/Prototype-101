import urllib.request
import cv2
import numpy as np

url = 'http://192.168.254.137/640x480.jpg'

while True:
    # Open the URL and read the stream
    imgResp = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)

    # all the opencv processing is done here
    cv2.imshow('test', img)

    # if 'q' key is pressed, exit the loop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
