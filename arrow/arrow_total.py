
import ast
import copy
import cv2
import logging
import os
import torch
from . import arrow_box, find_head_tail
from PIL import Image

cpu_device = torch.device("cpu")
parent_dir= os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

def arrow_head_tail(args):
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    image_directory= parent_dir+'/'+ args.input
    for image_name in os.listdir(args.input):
        path= image_directory+'/'+image_name
        bboxes= arrow_box.arrow_distinguish_test(args.threshold, path, device)
        if bboxes == None:
            continue
        image= cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        width= image.shape[1]
        height= image.shape[0]
        x_scale= width/800
        y_scale= height/800
        image= cv2.resize(image,(800,800))
        sample= cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        image_crop_list=[]
        dst_list=[]
        arrow_bbox_list=[]
        number=0
        for box in bboxes:
            arrow_coor= box

            image_crop= image[max(arrow_coor[1]-5,0):min(arrow_coor[3]+5,800),max(arrow_coor[0]-5,0):min(arrow_coor[2]+5,800)]
            dst= find_head_tail.find_head_tail(image_crop)

            for point in dst:
                if dst == ('None', 'None'):
                    arrow_bbox_list.append([number,[box[0]*x_scale, box[1]*y_scale, box[2]*x_scale, box[3]*y_scale]])
                    if arrow_coor[2]-arrow_coor[0]> arrow_coor[3]-arrow_coor[1]:
                        dst_list.append([(int(arrow_coor[0]*x_scale),int((arrow_coor[1]+arrow_coor[3])*y_scale/2)),'None',number])
                        dst_list.append([(int(arrow_coor[2]*x_scale),int((arrow_coor[1]+arrow_coor[3])*y_scale/2)),'None',number])
                        number+=1
                    else:
                        dst_list.append([(int((arrow_coor[0]+arrow_coor[2])*x_scale/2), int(arrow_coor[1]*y_scale)),'None',number])
                        dst_list.append([(int((arrow_coor[0]+arrow_coor[2])*x_scale/2), int(arrow_coor[3]*y_scale)),'None',number])
                        number+=1
                        
                    break
                    
                dst_coor= point[0]
                dst_head_tail= point[1]
                dst_coor_x= dst_coor[0]+ max(arrow_coor[0]-5,0)
                dst_coor_y= dst_coor[1]+ max(arrow_coor[1]-5,0)
                dst_list.append([(int(dst_coor_x*x_scale),int(dst_coor_y*y_scale)),dst_head_tail,number])
            arrow_bbox_list.append([number,[box[0]*x_scale, box[1]*y_scale, box[2]*x_scale, box[3]*y_scale]])
            number+=1

        if not 'arrow_detection_result' in os.listdir(args.output):
            os.mkdir(parent_dir+'/'+args.output+'/arrow_detection_result')
        if not 'arrow_head_tail_result' in os.listdir(args.output):
            os.mkdir(parent_dir+'/'+args.output+'/arrow_head_tail_result')

        f= open(parent_dir+'/'+args.output+'/arrow_head_tail_result/result_'+image_name+'.txt','w')
        for dst in dst_list:
            head_tail='head'
            if dst[1]=='no':
                head_tail= 'tail'
            elif dst[1]=='None':
                head_tail= 'None'
            f.write("arrow_number:{} ; coor:{} ; head_tail:{}".format(dst[2],dst[0],head_tail))
            f.write('\n')
        f.close()

        f= open(parent_dir+'/'+ args.output+'/arrow_detection_result/arrow_bbox_'+image_name+'.txt','w')
        for bbox in arrow_bbox_list:
            f.write("arrow_number:{} ; coor:{}".format(bbox[0],bbox[1]))
            f.write("\n")
        f.close()
        logging.info(image_name+' done')
