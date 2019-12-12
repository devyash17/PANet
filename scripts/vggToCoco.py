# convert VIA-VGG annotations to MSCOCO JSON format
import json
import numpy as np
import os

via_json = '/home/intern/devyash/PANet/sample5.json'
output_path_train = '/home/intern/devyash/PANet/train.json'
output_path_test = '/home/intern/devyash/PANet/test.json'
directory_path = '/home/intern/devyash/PANet/annotations/'
gt_path = '/home/intern/devyash/PANet/ground_truth/gt_boxes_'

json_files = [gt_path+'0.json',gt_path+'1.json',gt_path+'2.json']

def load_annotations(json_path):

    with open(json_path) as f:
        annotations = json.load(f)

    return annotations

coco_parts = dict(signage=0, traffic_sign=1, traffic_light=2)

categories = []
for i,name in enumerate(coco_parts.keys()):
    categories.append(dict(supercategory = "road objects",id=i,name = name))

# second: 'annotations' data
test_annos = []
train_annos= []
test_files = []
train_files = []
test_images = []
train_images = []
count = {'signage':0,'traffic_sign':0,'traffic_light':0}
gt_boxes_test = [{},{},{}]
area = 0
cnt = 0
iscrowd = 0
image_id = 0
empty = 0
no_of_images = 0
image_names = []

# load original via data
jsonList = os.listdir(directory_path)
for via_json in jsonList:    
    loaded_json = load_annotations(directory_path+via_json)
    for a_key in loaded_json.keys():
        an_item = loaded_json[a_key]
        filename = an_item['filename']
        if filename not in image_names:
            image_names.append(filename)
            no_of_images += 1
        regions = an_item['regions']
        if (len(regions)) == 0:
            empty += 1
        flag = 0     
        for a_region in regions:
            shape = a_region['shape_attributes']
            region = a_region['region_attributes']
            if region['Label'] in coco_parts.keys():  # recognize mscoco part
                flag = 1
                segmentation = [[]]
                if shape['name'] == 'rect':
                    x0 = shape['x']
                    w = shape['width']
                    y0 = shape['y']
                    h = shape['height']
                    segmentation[0].append(x0)
                    segmentation[0].append(y0)
                    segmentation[0].append(x0)
                    segmentation[0].append(y0+h)
                    segmentation[0].append(x0+w)
                    segmentation[0].append(y0+h)
                    segmentation[0].append(x0+w)
                    segmentation[0].append(y0)
                else:
                    x0 = min(shape['all_points_x'])
                    y0 = min(shape['all_points_y'])
                    w = max(shape['all_points_x'])-x0
                    h = max(shape['all_points_y'])-y0
                    
                    for (x,y) in zip(shape['all_points_x'],shape['all_points_y']):
                        segmentation[0].append(x)
                        segmentation[0].append(y)

                area = w*h
                bbox = []
                bbox.append(int(x0))
                bbox.append(int(y0))
                bbox.append(int(w))
                bbox.append(int(h))
                
                bbox_test = bbox
                bbox_test[2] = bbox[0]+w
                bbox_test[3] = bbox[1]+h
                
                category_id=coco_parts[region['Label']]
                
                annotation = dict(image_id=image_id, category_id=category_id, iscrowd=iscrowd,
                          id=cnt, segmentation=segmentation, area=area, bbox=bbox)
                # take 3 out of every 10 images for testing
                if image_id % 10 >= 7:
                    test_annos.append(annotation)
#                    if filename not in gt_boxes_test[category_id]:
#                        gt_boxes_test[category_id][filename] = []
#                    gt_boxes_test[category_id][filename].append(bbox_test)
                else:
                    count[region['Label']] += 1
                    train_annos.append(annotation)
                    
                cnt = cnt+1
                    
        if flag == 1:
            an_image = dict(date_captured='secret_hehexd', id=image_id, coco_url='no-coco', height=1080, width=1920, license=0,
                        file_name=filename, flickr_url='who_uses_flicker?')
            if image_id % 10 >= 7:
                test_files.append(filename)
                test_images.append(an_image)
            else:
                train_files.append(filename)
                train_images.append(an_image)
            image_id = image_id + 1
            
# third: 'info' data
info = dict(url='https://github.com/harrisonford', contributor='harrisonford', year=2019, description='alpha',
            date_created='2019', version=1.0)

# fifth: 'licenses' data
licenses = ['private']

# put data in final dictionary
test_data = dict(categories=categories, annotations=test_annos, info=info, images=test_images, licenses=licenses)
train_data = dict(categories=categories, annotations=train_annos, info=info, images=train_images, licenses=licenses)

with open(output_path_train, 'w') as outfile:
    json.dump(train_data, outfile)

with open(output_path_test, 'w') as outfile:
    json.dump(test_data, outfile)

#m = 0
#for gt_path in json_files:
#    with open(gt_path, 'w') as outfile:
##        gt_boxes_test = gt_boxes_test.replace("\'", "\"")
#        json.dump(gt_boxes_test[m],outfile)
#    m += 1


#
#
## create text file for downloading images
#link = 'http://10.4.71.100/stage/maze/vs/trackSticker.php?action=getImage&image='
#textfile_test = 'test.txt'
#textfile_train = 'train.txt'
#    
#file1 = open(textfile_test,'w')
#for a_file in test_files:
#    file1.write(a_file)
#    file1.write('\n')
#    file1.write(link+a_file)
#    file1.write('\n')
#
#file1.close()
#
#file1 = open(textfile_train,'w')
#for a_file in train_files:
#    file1.write(a_file)
#    file1.write('\n')
#    file1.write(link+a_file)
#    file1.write('\n')
#
#file1.close()