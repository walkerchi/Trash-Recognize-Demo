## Trash Recognition Demo

this project is mainly composed of a MaskRCNN([by matterport](https://github.com/matterport/Mask_RCNN)) trained on TACO dataset([by padropro](https://github.com/pedropro/TACO)) and a server written in python flask.

### 0) install required libs and download weight

install requirements

```powershell
pip3 install -r requirements.txt
```
[download](https://github.com/pedropro/TACO/pedropro/TACO/releases/download/1.0/taco_10_3.zip) weights and put the mask_rcnn_taco_0100.h5 file in the server folder
or you can run
```
python download_weight.py
```
to do the same thing.


### 1)to run the server

```powershell
python server.py
```



### 2) to visit the Interface

```
http://localhost:10000
```



### 3) the function of the  interface is quite simple at this moment

![example img](images\example_1.png)

First select the image file ,then click submit to recognize the image. 

And you will get the feedback like the one showed above.



### 4) you could also customize the configuration
in server.py
```
PORT = 10000                #you could modify the port to any one you like
CAPTION_SIZE = 8            #you could also change the size of the caption
CAPTION_COLOR = 'w'         #and the color of the caption
BOX_STYLE = 'dashed'        #the style of the border of the box
BOX_THICK = 2               #the border width of the box
```

