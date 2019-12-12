import numpy as np
import cv2 as cv
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imutils
from itertools import chain 
from collections import namedtuple
import glob
import os
import json
#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label.
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


labels = [
    #           name     id trainId      category  catId hasInstances ignoreInEval            color
    Label(     'void' ,   0 ,     0,        'void' ,   0 ,      False ,      False , (  0,   0,   0) ),
    Label(    's_w_d' , 200 ,     1 ,   'dividing' ,   1 ,      False ,      False , ( 70, 130, 180) ),
    Label(    's_y_d' , 204 ,     2 ,   'dividing' ,   1 ,      False ,      False , (220,  20,  60) ),
    Label(  'ds_w_dn' , 213 ,     3 ,   'dividing' ,   1 ,      False ,       True , (128,   0, 128) ),
    Label(  'ds_y_dn' , 209 ,     4 ,   'dividing' ,   1 ,      False ,      False , (255, 0,   0) ),
    Label(  'sb_w_do' , 206 ,     5 ,   'dividing' ,   1 ,      False ,       True , (  0,   0,  60) ),
    Label(  'sb_y_do' , 207 ,     6 ,   'dividing' ,   1 ,      False ,       True , (  0,  60, 100) ),
    Label(    'b_w_g' , 201 ,     7 ,    'guiding' ,   2 ,      False ,      False , (  0,   0, 142) ),
    Label(    'b_y_g' , 203 ,     8 ,    'guiding' ,   2 ,      False ,      False , (119,  11,  32) ),
    Label(   'db_w_g' , 211 ,     9 ,    'guiding' ,   2 ,      False ,       True , (244,  35, 232) ),
    Label(   'db_y_g' , 208 ,    10 ,    'guiding' ,   2 ,      False ,       True , (  0,   0, 160) ),
    Label(   'db_w_s' , 216 ,    11 ,   'stopping' ,   3 ,      False ,       True , (153, 153, 153) ),
    Label(    's_w_s' , 217 ,    12 ,   'stopping' ,   3 ,      False ,      False , (220, 220,   0) ),
    Label(   'ds_w_s' , 215 ,    13 ,   'stopping' ,   3 ,      False ,       True , (250, 170,  30) ),
    Label(    's_w_c' , 218 ,    14 ,    'chevron' ,   4 ,      False ,       True , (102, 102, 156) ),
    Label(    's_y_c' , 219 ,    15 ,    'chevron' ,   4 ,      False ,       True , (128,   0,   0) ),
    Label(    's_w_p' , 210 ,    16 ,    'parking' ,   5 ,      False ,      False , (128,  64, 128) ),
    Label(    's_n_p' , 232 ,    17 ,    'parking' ,   5 ,      False ,       True , (238, 232, 170) ),
    Label(   'c_wy_z' , 214 ,    18 ,      'zebra' ,   6 ,      False ,      False , (190, 153, 153) ),
    Label(    'a_w_u' , 202 ,    19 ,  'thru/turn' ,   7 ,      False ,       True , (  0,   0, 230) ),
    Label(    'a_w_t' , 220 ,    20 ,  'thru/turn' ,   7 ,      False ,      False , (128, 128,   0) ),
    Label(   'a_w_tl' , 221 ,    21 ,  'thru/turn' ,   7 ,      False ,      False , (128,  78, 160) ),
    Label(   'a_w_tr' , 222 ,    22 ,  'thru/turn' ,   7 ,      False ,      False , (150, 100, 100) ),
    Label(  'a_w_tlr' , 231 ,    23 ,  'thru/turn' ,   7 ,      False ,       True , (255, 165,   0) ),
    Label(    'a_w_l' , 224 ,    24 ,  'thru/turn' ,   7 ,      False ,      False , (180, 165, 180) ),
    Label(    'a_w_r' , 225 ,    25 ,  'thru/turn' ,   7 ,      False ,      False , (107, 142,  35) ),
    Label(   'a_w_lr' , 226 ,    26 ,  'thru/turn' ,   7 ,      False ,      False , (201, 255, 229) ),
    Label(   'a_n_lu' , 230 ,    27 ,  'thru/turn' ,   7 ,      False ,       True , (0,   191, 255) ),
    Label(   'a_w_tu' , 228 ,    28 ,  'thru/turn' ,   7 ,      False ,       True , ( 51, 255,  51) ),
    Label(    'a_w_m' , 229 ,    29 ,  'thru/turn' ,   7 ,      False ,       True , (250, 128, 114) ),
    Label(    'a_y_t' , 233 ,    30 ,  'thru/turn' ,   7 ,      False ,       True , (127, 255,   0) ),
    Label(   'b_n_sr' , 205 ,    31 ,  'reduction' ,   8 ,      False ,      False , (255, 128,   0) ),
    Label(  'd_wy_za' , 212 ,    32 ,  'attention' ,   9 ,      False ,       True , (  0, 255, 255) ),
    Label(  'r_wy_np' , 227 ,    33 , 'no parking' ,  10 ,      False ,      False , (178, 132, 190) ),
    Label( 'vom_wy_n' , 223 ,    34 ,     'others' ,  11 ,      False ,       True , (128, 128,  64) ),
    Label(   'om_n_n' , 250 ,    35 ,     'others' ,  11 ,      False ,      False , (102,   0, 204) ),
    Label(    'noise' , 249 ,   255 ,    'ignored' , 255 ,      False ,       True , (  0, 153, 153) ),
    Label(  'ignored' , 255 ,   255 ,    'ignored' , 255 ,      False ,       True , (255, 255, 255) ),
]

name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]
        
color2label = { str(label.color[0])+' '+str(label.color[1])+' '+str(label.color[2]) : label for label in labels }


categories = []
for i in range(1,36):
    categories.append(dict(supercategory = "road_objects",id = i,name = trainId2label[i].name))

info = dict(url='https://github.com/devyash17', contributor='devyash', year=2019, description='alpha',
            date_created='2019', version=1.0)

licenses = ['private']

directory_path = '/home/intern/devyash/Apolloscape-Lane-Segmentation/Labels_road04/Label/Record001/Camera 5/'
imageList = os.listdir(directory_path)

images = []
image_id = 0
for image in imageList:
    an_image = dict(date_captured='secret_hehexd', id=image_id, coco_url='no-coco', height=2710, width=3384, license=0,
                    file_name=image[:-8]+'.jpg', flickr_url='who_uses_flicker?')
    images.append(an_image)
    image_id = image_id + 1

annotations = []

image_id = 0
annot_id = 0
for image in imageList:
#    if image_id >= 10:
#        break
    im = cv.imread(directory_path+image)
    im2 = cv.imread('/home/intern/devyash/Apolloscape-Lane-Segmentation/ColorImage_road04/ColorImage/Record001/Camera 5/'+image[:-8]+'.jpg')
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    edged = cv.Canny(gray, 30, 200) 
#    ret, thresh = cv.threshold(gray, 20, 255, 0)
#    _, contours, _ = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#    _, contours, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    _,contours,_ = cv.findContours(edged,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 
    temp_contours = []
    for c in contours:
        grain = np.int0(cv.boxPoints(cv.minAreaRect(c)))
        centroid = (grain[2][1]-(grain[2][1]-grain[0][1])//2, grain[2][0]-(grain[2][0]-grain[0][0])//2)
        color = im[centroid]
        key = str(color[2])+' '+str(color[1])+' '+str(color[0])
#        print(key)
        if key in color2label.keys():
#            cv.drawContours(im2, c, -1, (0,255,0), 3)
            category_id = color2label[key].trainId
            if category_id != 0 and category_id != 255:
                shape = c.shape
                coords = c.reshape(shape[0],shape[2])
                x0 = float(coords[np.argmin(coords[:,0])][0])
                y0 = float(coords[np.argmin(coords[:,1])][1])
                x1 = float(coords[np.argmax(coords[:,0])][0])
                y1 = float(coords[np.argmax(coords[:,1])][1])
                bbox = []
                bbox.append(x0)
                bbox.append(y0)
                bbox.append(x1-x0)
                bbox.append(y1-y0)
                area = bbox[2]*bbox[3]
                coords = coords.tolist()
                coords = list(chain.from_iterable(coords))
    #            if len(coords) >=40:            
                segmentation = [coords]
                annotation = dict(image_id=image_id,category_id=category_id,iscrowd=0,
                        id=annot_id,segmentation=segmentation,area=area,bbox=bbox)
                annotations.append(annotation)
                annot_id = annot_id + 1
                temp_contours.append(c)
        
#    cv.drawContours(im2, temp_contours, -1, (0,255,0), 3)
#    cv.imwrite('/home/intern/devyash/PANet/lane-segmentation/contours/'+image,im2)
#    plt.imshow(im2)
#    plt.savefig('/home/intern/devyash/PANet/lane-segmentation/contours/'+image)
#    plt.show()
    image_id = image_id + 1
    
data = dict(categories=categories, annotations=annotations, info=info, images=images, licenses=licenses)

with open('/home/intern/devyash/PANet/lane-segmentation/apollo_laneseg.json', 'w') as outfile:
    json.dump(data, outfile)