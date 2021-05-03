from flask import Flask, request, Response
import numpy as np
import cv2
import easyocr
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import requests
import unicodedata
import json
import jsonify 
# Initialize the Flask application
app = Flask(__name__)

def url_to_image(url):
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.COLOR_BGR2RGB)
    
	return image

def crop(img):
    if img.shape[0]>600:
        img=img[150:-30,:-50,:]
    return img    

def erosion(image_binary):
    kernel = np.ones((5,5), np.uint8)
    erode=cv2.erode(image_binary,kernel)
    return erode
    
def translate(text,tl):
    # Translates some text into Hindi
    URL = 'https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl='+tl+'&dt=t&q=' + text
    r = requests.get(url=URL)
    data = r.json()
    return data[0][0][0]
# route http posts to this method
@app.route('/translate', methods=['GET'])
def index():

    if request.method == 'GET':

        url = request.args.get('imageUri')
        tl = request.args.get('tl')
        image = url_to_image(url)

        RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Crop Image
        image = crop(RGB_img)
        img_width = image.shape[0]
        img_height = image.shape[1]
        # Convert Image into Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # apply connected compogrnent analysis to the thresholded image
        output = cv2.connectedComponentsWithStats(
            thresh, 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output

        mask = np.zeros(gray.shape, dtype="uint8")
        # loop over the number of unique connected component labels, skipping
        # over the first label (as label zero is the background)
        for i in range(1, numLabels):
            # extract the connected component statistics for the current
            # label
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            
            # ensure the width, height, and area are all neither too small
            # nor too big
            keepWidth =  w < 0.8 * img_width
            keepHeight = h > 3 and h < 0.5 * img_height
            # ensure the connected component we are examining passes all
            # three tests
            if all((keepWidth, keepHeight)):
                # construct a mask for the current connected component and
                # then take the bitwise OR with the mask 
                print("[INFO] keeping connected component '{}'".format(i))
                componentMask = (labels == i).astype("uint8") * 255
                mask = cv2.bitwise_or(mask, componentMask)
        invert = cv2.bitwise_not(mask)
        reader = easyocr.Reader(['hi','en'])
        
        original_text = reader.readtext(invert,detail=0)
        text = ""
        for i in original_text:
            text +=i+" "
        translated_text = translate(text,tl)

    return json.dumps({'orig':text,'trans':translated_text},indent=4)


# start flask app
if __name__ == "__main__":
    app.run()
