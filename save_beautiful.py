import requests
import re
import os
from icrawler.builtin import GoogleImageCrawler
import cv2
root_dir = "/Users/uemuraminato/Desktop/web_server/favorite/"
search = GoogleImageCrawler(storage={'root_dir':"photo/cool"})

from PIL import Image as resizer

def getImageFromIcra(num, words:str):
      for word in words:
        search.crawl(max_num=num,keyword=word)     
                   

def resizeImages(key):

    os.makedirs("./Face/" + str(key)  ,exist_ok=True)
    save_path = "./Face/" + str(key) + "/"

    input_dir = root_dir + "photo" + "/" + str(key) + "/"

    for image in os.listdir(input_dir):        
        img = resizer.open(input_dir + image,'r')
        img = img.resize((100,100))

        img.save(save_path + image,'png',quality=100,optimize=True )


    

def main():
    # 「人名,正面」にすると、より多くの正面画像を拾える
    #words = ['橋本環奈','大原櫻子','藤田ニコル']
    #words  =["上戸彩","本田翼","石田ゆりこ"]
    #words = ["石原さとみ","深田恭子","石原さとみ"]
    words = ["菜々緒","黒木メイサ","天海祐希"]
    getImageFromIcra(400,words)
    resizeImages("cool")
if __name__ == '__main__':
    main()