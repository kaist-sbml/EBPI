import itertools
from PIL import Image
from cv2 import groupRectangles
import cv2
import os    
import numpy as np
import ast


def intersection(rectA, rectB): 
    '''
    a, b = rectA, rectB
    
    min_dist_x= min(a[2]-a[0],b[2]-b[0])
    min_dist_y= min(a[3]-a[1],b[3]-b[1])
    
    
    startX = max( min(a[0], a[2]), min(b[0], b[2]) )
    startY = max( min(a[1], a[3]), min(b[1], b[3]) )
    endX = min( max(a[0], a[2]), max(b[0], b[2]) )
    endY = min( max(a[1], a[3]), max(b[1], b[3]) )
    
    
    if startX <= endX+1 and startY <= endY+1:
        if (endX-startX)/min_dist_x>0.7 or (endY-startY)/min_dist_y>0.7:
            return True
        else:
            return False
    else:
        return False
    '''
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
    
    '''
    while noIntersect == False:
        revise_mainRects= list(set(revise_mainRects))
        newRectsArray = list()
        for rectA, rectB in itertools.combinations(revise_mainRects, 2):
            newRect = []
            if intersection(rectA, rectB):
                newRect = combineRect(rectA, rectB)
                
                newRectsArrayappend(newRect)
                noIntersect = False
            
                if rectA in revise_mainRects:
                    revise_mainRects.remove(rectA)
                if rectB in revise_mainRects:
                    revise_mainRects.remove(rectB)
        if len(newRectsArray) == 0:
            noIntersect = True
        else:
            revise_mainRects = revise_mainRects + newRectsArray
    revise_mainRects= [tuple(rect) for rect in revise_mainRects]
    return revise_mainRects'''

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

def revise_ocr_bbox(args):
    ocr_result_path= args.draw_img_save_dir
    ocr_total= open(ocr_result_path+'/'+'system_results.txt','r').read()
    ocr_inform= ocr_total.split('\n')
    special_unit= '@#$%^&*+/↑→↓←><~!?:;'
    basic_formula=['OH','HO','NH','HN','SH','HS','H','O','S','N']
    f = open(ocr_result_path+'/'+'system_revise_results.txt', 'w')
    for pathway in ocr_inform:
        try:
            image_name,ocr_result= pathway.split('\t')
        except:
            continue
        ocr_list=list()
        ocr_dict=dict()
        
        ocr_result= ast.literal_eval(ocr_result)
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
        f.write(image_name+'\t')
        f.write(str(final_box_ocr)+'\n')
    f.close()

