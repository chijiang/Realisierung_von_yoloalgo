from config import *
import os
from xml.dom.minidom import parse

def convert_annotations(annotations_folder, target_folder_name="test"):
    for file in os.listdir(annotations_folder):
        out_file_name = file.replace(".xml", ".txt")
        image_file_name = file.replace(".xml", ".jpg")
        file_name = annotations_folder + '/' + file
        tree = parse(file_name)
        collection = tree.documentElement
        im_width = float(collection.getElementsByTagName("size")[0].getElementsByTagName('width')[0].childNodes[0].data)
        im_height = float(collection.getElementsByTagName("size")[0].getElementsByTagName('height')[0].childNodes[0].data)
        grid_width, grid_height = im_width / GRID_COUNT, im_height / GRID_COUNT
        obj_list = collection.getElementsByTagName("object")
        content = ""
        for obj in obj_list:
            obj_name = obj.getElementsByTagName("name")[0].childNodes[0].data
            if obj_name not in CLASS_LIST:
                continue
            box = [float(obj.getElementsByTagName("bndbox")[0].getElementsByTagName("xmin")[0].childNodes[0].data),
                   float(obj.getElementsByTagName("bndbox")[0].getElementsByTagName("ymin")[0].childNodes[0].data),
                   float(obj.getElementsByTagName("bndbox")[0].getElementsByTagName("xmax")[0].childNodes[0].data),
                   float(obj.getElementsByTagName("bndbox")[0].getElementsByTagName("ymax")[0].childNodes[0].data)]
            xi, yi = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            grid_pos = (int(xi / grid_width), int(yi / grid_height))
            x, y = xi % grid_width / grid_width, yi % grid_height / grid_height
            w, h = (box[2] - box[0]) / im_width, (box[3] - box[1]) / im_height
            class_idx = CLASS_LIST.index(obj_name)
            content += "{} {} {} {} {} {} {} \n".format(grid_pos[0], grid_pos[1], x, y, w, h, class_idx)
        with open("./data/" + target_folder_name + "/" + out_file_name, "w") as f:
            temp = f.write(content)
            
### RD ds ###
