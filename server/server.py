PORT = 10000
MAP_PATH = 'map_10.csv'
WEIGHT_PATH = 'mask_rcnn_taco_0100.h5'
CAPTION_SIZE = 8
CAPTION_COLOR = 'w'



import flask
from flask import request
import csv
import random
import colorsys
import base64
import pickle as pkl
import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
from skimage.measure import find_contours
import numpy as np
import tensorflow as tf
from keras.models import model_from_json

from model import MaskRCNN
from config import Config
class TacoTestConfig(Config):
    NAME = "taco"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0
    NUM_CLASSES = 11
    USE_OBJECT_ZOOM = False
config = TacoTestConfig()


app = flask.Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16*1024*1024
@app.after_request
def cors(env):
    env.headers['Access-Control-Allow-Origin']='*'
    env.headers['Access-Control-Allow-Method']='*'
    env.headers['Access-Control-Allow-Headers']='x-requested-with,content-type'
    return env

model = MaskRCNN('inference',config,'models/logs')
model.load_weights(WEIGHT_PATH,None,by_name=True)
model.keras_model._make_predict_function()
class_names = []
with open(MAP_PATH,'r') as f:
    reader = csv.reader(f)
    class_names+=[row[1] for row in reader]
class_names_set = list(set(class_names))
class_names_set.sort(key = class_names.index)

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None,
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height, 0)
    ax.set_xlim(0, width)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            x = random.randint(x1, (x1 + x2) // 2)
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color=CAPTION_COLOR, size=CAPTION_SIZE, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    plt.gcf().set_size_inches(width / 100, height / 100)
    ax.imshow(masked_image.astype(np.uint8))
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)



@app.route('/',methods=['GET'])
def index():
    return app.send_static_file('index.html')

@app.route('/detect/file',methods=['POST'])
def detect_file():
    data = dict(request.form)
    head = data['img'].split(',')[0]
    img = base64.b64decode(data['img'].split(',')[-1])
    img = np.fromstring(img,np.uint8)
    img = cv.imdecode(img,cv.IMREAD_COLOR)
    # cv.imwrite('tmp.jpg',img)
    # img = cv.imread('tmp.jpg')

    results = model.detect([img])
    r = results[0]
    ax = get_ax(1)
    display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                            class_names_set, r['scores'], ax=ax)

    sio = BytesIO()
    plt.savefig(sio,format='jpeg')
    data = base64.encodebytes(sio.getvalue()).decode()
    plt.close()

    # img = base64.b64decode(data)
    # img = np.fromstring(img,np.uint8)
    # img = cv.imdecode(img,cv.IMREAD_COLOR)
    # cv.imwrite('tmp.png',img)

    data = head+','+data
    response = {
        'img': data,
        'message':'Here You Are'
    }
    return flask.jsonify(response),200
    # except Exception as e:
    #     print(e)
    #     response = {
    #         'img':None,
    #         'message':'Request error'
    #     }
    #     return flask.jsonify(response),400


if __name__ == '__main__':
    app.run(host='127.0.0.1',port=PORT)
    
    