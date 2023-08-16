import ast
import cv2
import itertools
import numpy as np
import os 
import subprocess
from cv2 import groupRectangles
from PIL import Image

def intersection(rectA, rectB): 
    a, b = rectA, rectB
    
    if a[4]=='1' or b[4]=='1':
        return False
    
    dx= abs((a[0]+a[2])/2 - (b[0]+b[2])/2)
    dy= abs((a[1]+a[3])/2 - (b[1]+b[3])/2)
    max_length= max(abs(a[2]-a[0]), abs(b[2]-b[0])) 
    min_height= (abs(a[3]-a[1])+abs(b[3]-b[1]))/2
    
    if dx< max_length/2 and dy <= 1.2* min_height:
        return dx**2+ dy**2
    else:
        return False    
    
    
def combineRect(rectA, rectB):
    a, b = rectA, rectB
    startX = min( a[0], b[0] )
    startY = min( a[1], b[1] )
    endX = max( a[2], b[2] )
    endY = max( a[3], b[3] )

    return (startX, startY, endX, endY,'1')


def checkIntersectAndCombine(rects):
    if rects is None:
        return None
    mainRects = rects
    revise_mainRects=[]
    for rect in mainRects:
        revise_mainRects.append((rect[0],rect[1],rect[2],rect[3],'0'))
    noIntersect = False
    
    while not noIntersect:
        revise_mainRects= list(set(revise_mainRects))
        newRectsArray = dict()
        for rectA, rectB in itertools.combinations(revise_mainRects, 2):
            if intersection(rectA, rectB):
                if not rectA in newRectsArray and not rectB in newRectsArray:
                    newRectsArray[rectA]= [rectB,intersection(rectA,rectB)]
                    newRectsArray[rectB]= [rectA,intersection(rectA,rectB)]
                elif rectA in newRectsArray:
                    before_distance= newRectsArray[rectA][1]
                    before_correspond= newRectsArray[rectA][0]
                    if intersection(rectA, rectB) < before_distance:
                        newRectsArray[rectA]= [rectB,intersection(rectA,rectB)]
                        newRectsArray[rectB]= [rectA,intersection(rectA,rectB)]
                        del newRectsArray[before_correspond]
                        
                elif rectB in newRectsArray:
                    before_distance= newRectsArray[rectB][1]
                    before_correspond= newRectsArray[rectB][0]
                    if intersection(rectA, rectB) < before_distance:
                        newRectsArray[rectB]= [rectA,intersection(rectA,rectB)]
                        newRectsArray[rectA]= [rectB,intersection(rectA,rectB)]
                        del newRectsArray[before_correspond]
                        
        if len(newRectsArray) == 0:
            noIntersect = True
        else:
            for rect1, related_inform in newRectsArray.items():
                rect2= related_inform[0]
                if rect1 in revise_mainRects:
                    revise_mainRects.remove(rect1)
                if rect2 in revise_mainRects:
                    revise_mainRects.remove(rect2)
                newRect = combineRect(rect1, rect2)
                if not newRect in revise_mainRects:
                    revise_mainRects = revise_mainRects + [newRect]
    revise_mainRects= [tuple(rect) for rect in revise_mainRects]
    return revise_mainRects

def find_and_combine_ocr_bbox(args):
    special_unit= '@#$%^&*+/↑→↓←><~!?:;'
    basic_formula=['OH','HO','NH','HN','SH','HS','H','O','S','N','COOH']
    
    paddleocr_binary_output = subprocess.check_output(['paddleocr', '--image_dir', '%s'%args.input])
    paddleocr_output_list = paddleocr_binary_output.decode("utf-8").split("\n")
    
    paddleocr_pathway_id = None
    paddleocr_pathway_info = {}
    for idx, elem in enumerate(paddleocr_output_list):
        if "**********" in elem:
            if paddleocr_pathway_id != elem.split("**********")[1]:
                paddleocr_pathway_id = elem.split("**********")[1].split('/')[-1]
                paddleocr_pathway_info[paddleocr_pathway_id] = []
        if "[[[" in elem:
            info = ast.literal_eval(elem.split("root INFO: ")[1])
            info_dict = {}
            info_dict["transcription"] = info[1][0]
            info_dict["points"] = info[0]
            paddleocr_pathway_info[paddleocr_pathway_id].append(info_dict)

    print('OCR process ended')
    print('OCR revise process start....')
    
    f = open(args.output+'/'+'system_revise_results.txt', 'w')
    
    for pathway_id in paddleocr_pathway_info:
        ocr_list = []
        ocr_dict = {}
        ocr_result= paddleocr_pathway_info[pathway_id]
        num=0
        for word in ocr_result:
            word_name= word['transcription']
            if word_name in special_unit or word_name in basic_formula or len(word_name)==1:
                continue
            for letter in word_name:
                if letter in special_unit:
                    word_name= word_name.replace(letter,'')
            word_coor= word['points']
            min_x= word_coor[0][0]
            min_y= word_coor[0][1]
            max_x= word_coor[2][0]
            max_y= word_coor[2][1]
            ocr_list.append((min_x,min_y,max_x,max_y))
            ocr_dict[num]=(word_name,min_x,min_y,max_x,max_y)
            num+=1
        output= checkIntersectAndCombine(ocr_list)

        box_ocr_match=dict()
        for group in output:
            box_ocr_match[group]=dict()
            for key,value in ocr_dict.items():
                coord= value[1:]
                if group[0] <=coord[0] and group[1] <= coord[1] and coord[2]<=group[2] and coord[3]<= group[3]:
                    box_ocr_match[group][key]= value[0]

        final_box_ocr= list()
        for box, ocr in box_ocr_match.items():
            each_box_dict=dict()
            sentence=''
            ocr_sort= sorted(ocr.items())
            for each_word in ocr_sort:
                if len(sentence)>0 and sentence[-1]=='-':
                    sentence+= each_word[1]
                else:
                    sentence+= each_word[1]+' '
            sentence= sentence.strip()
            each_box_dict['transcription']= sentence
            each_box_dict['points']= [[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]]
            final_box_ocr.append(each_box_dict)
        
        f.write(pathway_id+'\t')
        f.write(str(final_box_ocr)+'\n')
    f.close()

