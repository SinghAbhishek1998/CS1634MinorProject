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

        # Convert Image into Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (thresh, image_binary) = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

        # Eroded Image
        erode=erosion(image_binary)

        # Making Contours
        contours, hierarchy = cv2.findContours(erode, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        mask = list(hierarchy[0][:,-1]<=-1)
        ids = np.arange(len(hierarchy[0]))
        mid = ids[mask]
        new_cnts = [contours[i] for i in mid]
        cnts_temp = [i for i in new_cnts if cv2.contourArea(i)>1000.0]

        images =[]

        for sr, i in enumerate(cnts_temp):
            x, y, w, h = cv2.boundingRect(i)
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
            images.append(image[y:y+h,x:x+w])
        

        reader = easyocr.Reader(['hi','en'])
        
        for i in images:
            bounds = reader.readtext(i,detail=0)
        original_text = ' '.join(map(str,bounds))
        
        translated_text = translate(original_text,tl)

        result = {'orig':original_text,'trans':translated_text}
    return json.dumps(result, indent = 4)


# start flask app
if __name__ == "__main__":
    app.run()
