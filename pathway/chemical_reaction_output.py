#!/usr/bin/env python
# coding: utf-8

# In[174]:


import os
import ast
import numpy as np
import time
import math
import pandas as pd

def middle_criteria(arrow_point1, arrow_point2, ocr_middle_point):
    a = np.array([arrow_point1[0], arrow_point1[1]]) 
    b = np.array([arrow_point2[0], arrow_point2[1]])
    c = np.array([ocr_middle_point[0], ocr_middle_point[1]])

    va1 = b - a
    vb1 = c - a
    norm_a1 = np.linalg.norm(va1)
    norm_b1 = np.linalg.norm(vb1)
    dot_ab1 = np.dot(va1, vb1)
    cos_th1 = dot_ab1 / (norm_a1 * norm_b1)
    if -1>cos_th1:
        cos_th1= -1
    elif 1<cos_th1:
        cos_th1= 1
    rad1 = math.acos(cos_th1)
    deg1 = math.degrees(rad1)
    
    va2 = a - b
    vb2 = c - b
    norm_a2 = np.linalg.norm(va2)
    norm_b2 = np.linalg.norm(vb2)
    dot_ab2 = np.dot(va2, vb2)
    cos_th2 = dot_ab2 / (norm_a2 * norm_b2)
   
    if -1>cos_th2:
        cos_th2= -1
    elif 1<cos_th2:
        cos_th2= 1
    rad2 = math.acos(cos_th2)
    deg2 = math.degrees(rad2)
    
    if deg1<110 and deg2<110:
        return True
    else:
        return False
    
def sort_arrow_head_tail(arrow_inform, image_ocr, bbox_inform, basic_formula, special_unit):
    arrow_new_inform= list()
    arrow_min_distance= dict()
    for point_head_tail in arrow_inform:
        if point_head_tail=='':
            continue
        arrow_name= point_head_tail.split(' ; ')[0]
        head_tail= point_head_tail.split(' ; ')[2].replace('head_tail:','')
        min_set=0
        min_word=0
        min_distance=10**9
        point_coord= ast.literal_eval(point_head_tail.split(' ; ')[1].replace('coor:',''))
        if point_coord=='':
            continue

        arrow_i= [ast.literal_eval(i.split(' ; ')[1].replace('coor:','')) for i in bbox_inform if arrow_name == i.split(' ; ')[0]][0]
        arrow_i_list= [ast.literal_eval(i.split(' ; ')[1].replace('coor:','')) for i in arrow_inform if arrow_name == i.split(' ; ')[0]]
        middle_point= ((int(arrow_i[0])+int(arrow_i[2]))/2,(int(arrow_i[1])+int(arrow_i[3]))/2)
        
        for word in image_ocr:
            if len(word['transcription'])==1 or word['transcription'].upper in basic_formula or word['transcription'] in special_unit:
                continue
            name= word['transcription']
            points= word['points']
            '''
            if arrow_name in arrow_min_distance:
                if points == arrow_min_distance[arrow_name][0][2]:
                    continue'''
            for letter in name:
                if letter in special_unit:
                    name= name.replace(letter,'')

            # whether ocr is metabolite or not
            point_x_sum=0
            point_y_sum=0
            for point in points:
                point_x_sum+=point[0]
                point_y_sum+=point[1]
            point_x_avg= point_x_sum/len(points)
            point_y_avg= point_y_sum/len(points)
            middle_distance= (middle_point[0]-point_x_avg)**2+ (middle_point[1]-point_y_avg)**2
            
            if middle_criteria(arrow_i_list[0], arrow_i_list[1], (point_x_avg, point_y_avg)):
                continue
                
            distance= (point_coord[0]-point_x_avg)**2+ (point_coord[1]-point_y_avg)**2 
            if distance < min_distance:
                min_distance= distance

        if not arrow_name in arrow_min_distance:
            arrow_min_distance[arrow_name]= [(min_distance, point_head_tail)]

        else:
            before= arrow_min_distance[arrow_name]
            after= before+ [(min_distance, point_head_tail)]
            arrow_min_distance[arrow_name]= after
    for key, value in arrow_min_distance.items():    
        arrow1= value[0]
        arrow2= value[1]
        if arrow1[0]<arrow2[0]:
            arrow_new_inform.append(arrow1[1])
            arrow_new_inform.append(arrow2[1])
        else:
            arrow_new_inform.append(arrow2[1])
            arrow_new_inform.append(arrow1[1])            
            
    return arrow_new_inform
                   
def arrow_ocr_match_function(arrow_inform, image_ocr, bbox_inform, basic_formula, special_unit, word_class_total):
    arrow_ocr_match= dict()
    ocr_contained= list()
    arrow_name_set= set()
    for point in arrow_inform:
        if point=='':
            continue
        arrow_name= point.split(' ; ')[0]
        arrow_name_set.add(arrow_name)
        head_tail= point.split(' ; ')[2].replace('head_tail:','')
        min_set=0
        min_word=0
        min_distance=10**9
        point_coord= ast.literal_eval(point.split(' ; ')[1].replace('coor:',''))
        if point_coord=='':
            continue

        arrow_i= [ast.literal_eval(i.split(' ; ')[1].replace('coor:','')) for i in bbox_inform if arrow_name == i.split(' ; ')[0]][0]
        arrow_i_list= [ast.literal_eval(i.split(' ; ')[1].replace('coor:','')) for i in arrow_inform if arrow_name == i.split(' ; ')[0]] 
        middle_point= ((int(arrow_i[0])+int(arrow_i[2]))/2,(int(arrow_i[1])+int(arrow_i[3]))/2)
        arrow_i_max_len= max(abs(arrow_i[2]-arrow_i[0]), abs(arrow_i[3]-arrow_i[1]))
        
        for i in range(len(image_ocr)):
            word= image_ocr[i]
            word_class= word_class_total[i]
            if len(word['transcription'])==1 or word['transcription'].upper in basic_formula or word['transcription'] in special_unit:
                #\ or np.argmax(np.array(word_class)) != 3:
                continue
            name= word['transcription']
            points= word['points']
            if arrow_name in arrow_ocr_match:
                if not arrow_ocr_match[arrow_name][0][1]=='':
                    if points == arrow_ocr_match[arrow_name][0][1]:
                        continue
            for letter in name:
                if letter in special_unit:
                    name= name.replace(letter,'')

            # whether ocr is metabolite or not
            point_x_sum=0
            point_y_sum=0
            for point in points:
                point_x_sum+=point[0]
                point_y_sum+=point[1]
            point_x_avg= point_x_sum/len(points)
            point_y_avg= point_y_sum/len(points)
            
            if middle_criteria(arrow_i_list[0], arrow_i_list[1], (point_x_avg, point_y_avg)):
                continue

            middle_distance= (point_coord[0]-point_x_avg)**2+ (point_coord[1]-point_y_avg)**2 
            if middle_distance< min_distance:
                min_distance= middle_distance
                min_set=(name,points,head_tail)
                min_word= word
                
            for point in points:
                distance= (point_coord[0]-point[0])**2+ (point_coord[1]-point[1])**2
                if distance< min_distance:
                    min_distance= distance
                    min_set=(name,points,head_tail)
                    min_word= word

        ocr_contained.append(min_word)
        if not arrow_name in arrow_ocr_match:
            if min_set==0:
                arrow_ocr_match[arrow_name]=[('None_ocr_matched', '', 'None')]
            else:
                arrow_ocr_match[arrow_name]= [min_set]

        else:
            before= arrow_ocr_match[arrow_name]
            after= before+ [min_set]
            arrow_ocr_match[arrow_name]= after
            
    return arrow_ocr_match, ocr_contained, arrow_name_set
    
def make_reaction_and_text_classifier(args,text_classifier):
    each_image_ocr= dict()
    ocr_result= open(args.draw_img_save_dir+'/'+'system_revise_results.txt','r').read()
    ocr_inform= ocr_result.split('\n')
    dataframe= pd.DataFrame(columns=['image_name','reaction','gene','gene_protein','protein_complex','others'])

    
    if not 'final_output' in os.listdir(args.output_dir):
        os.mkdir(args.output_dir+'/final_output')
    for each_image in ocr_inform:
        if each_image=='':
            continue
        name, inform= each_image.split('\t')
        each_image_ocr[name]= ast.literal_eval(inform)

    special_unit= '@#$%^&*+/↑→↓←><~!?:;'
    basic_formula=['OH','HO','NH','HN','SH','HS','H','O','S','N','C']
    for file in os.listdir(args.output_dir+'/arrow_head_tail_result'):
        if 'result_' in file:
            arrow_file_name= file.replace('result_','')
            print(arrow_file_name.replace('.txt','')+' start')
            f= open(args.output_dir+'/arrow_head_tail_result/'+file,'r')
            data= f.read()
            arrow_inform= data.split('\n')
            image_ocr= each_image_ocr[arrow_file_name.replace('.txt','')]
            bbox= open(args.output_dir+'/arrow_detection_result/arrow_bbox_'+arrow_file_name,'r')
            bbox_data= bbox.read()
            bbox_inform= bbox_data.split('\n')
            ocr_not_contained=list()
            ocr_not_contained_name= list()
            total_output=list()
            arrow_inform= sort_arrow_head_tail(arrow_inform,image_ocr,bbox_inform, basic_formula, special_unit)
            ocr_list= [word['transcription'] for word in image_ocr if not len(word['transcription'])==1 or word['transcription'].upper in basic_formula or word['transcription'] in special_unit]
            if ocr_list==[]:
                ocr_not_contained_arrow_match=dict()
            else:
                word_class_total= text_classifier(args,ocr_list).detach().numpy()
                image_ocr=[ocr for ocr in image_ocr if ocr['transcription'] in ocr_list]
                arrow_ocr_match, ocr_contained, arrow_name_set= arrow_ocr_match_function(arrow_inform, image_ocr, bbox_inform, basic_formula, special_unit,word_class_total)
                
                #ocr not contained list        
                for i in range(len(image_ocr)):
                    word_ocr= image_ocr[i]
                    if not word_ocr in ocr_contained and not word_ocr['transcription'] in basic_formula and not word_ocr['transcription'] in special_unit:
                        ocr_not_contained.append([word_ocr,i])
                        ocr_not_contained_name.append(word_ocr['transcription'])
                ocr_not_contained_arrow_match= dict()
                for arrow_name in arrow_name_set:
                    ocr_not_contained_arrow_match[arrow_name]=[]  
            
            #ocr that are not metabolites
            try:
                if len(ocr_not_contained_name)!=0:
                    #word_class_total= text_classifier(args,ocr_not_contained_name).detach().numpy()
                    for word_ocr_inform in ocr_not_contained:
                        word_ocr= word_ocr_inform[0]
                        index_number= word_ocr_inform[1]
                        word_class= word_class_total[index_number]
                        x_sum=0
                        y_sum=0
                        word_ocr_coor= word_ocr['points']
                        arrow_set=True
                        for point in word_ocr_coor:
                            x_sum+= point[0]
                            y_sum+= point[1]
                        x_avg= x_sum/len(word_ocr_coor)
                        y_avg= y_sum/len(word_ocr_coor)

                        min_dist= 10**9

                        for arrow_name in arrow_name_set:
                            near_criteria=False
                            middle_criterian=False
                            if arrow_name=='':
                                continue
                            each_arrow_min_dist= 10**9
                            arrow_i= [ast.literal_eval(i.split(' ; ')[1].replace('coor:','')) for i in bbox_inform if arrow_name == i.split(' ; ')[0]][0]
                            arrow_i_list= [ast.literal_eval(i.split(' ; ')[1].replace('coor:','')) for i in arrow_inform if arrow_name == i.split(' ; ')[0]]
                            middle_point= ((int(arrow_i[0])+int(arrow_i[2]))/2,(int(arrow_i[1])+int(arrow_i[3]))/2)
                            arrow_i_max_len= max(abs(arrow_i[2]-arrow_i[0]), abs(arrow_i[3]-arrow_i[1]))

                            #whether ocr is near the arrow

                            for point in word_ocr_coor:
                                near_dist=(point[0]-middle_point[0])**2+(point[1]-middle_point[1])**2
                                near_dist2= (x_avg-middle_point[0])**2+(y_avg-middle_point[1])**2
                                if min(near_dist**(1/2),near_dist2**(1/2)) < arrow_i_max_len:
                                    near_criteria=True
                            if middle_criteria(arrow_i_list[0], arrow_i_list[1], (x_avg, y_avg)):
                                middle_criterian= True
                            if near_criteria and middle_criterian:
                                for point in word_ocr_coor:
                                    dist= (point[0]-middle_point[0])**2+(point[1]-middle_point[1])**2
                                    if each_arrow_min_dist > dist:
                                        each_arrow_min_dist=dist
                                    for arrow in arrow_i_list:
                                        dist_corner= (arrow[0]-point[0])**2+(arrow[1]-point[1])**2
                                        if each_arrow_min_dist > dist_corner:
                                            each_arrow_min_dist=dist_corner
                                middle_point_dist= (x_avg-middle_point[0])**2+(y_avg-middle_point[1])**2
                                min_each_arrow= min(middle_point_dist, each_arrow_min_dist)

                                if min_dist> min_each_arrow:
                                    min_dist= min_each_arrow
                                    arrow_set= arrow_name

                        if arrow_set==True:
                            continue

                        if np.argmax(np.array(word_class))==0:
                            before_inform= ocr_not_contained_arrow_match[arrow_set]
                            before_inform.append((word_ocr['transcription'],'gene'))
                            ocr_not_contained_arrow_match[arrow_set]=before_inform
                        elif np.argmax(np.array(word_class))==1:
                            before_inform= ocr_not_contained_arrow_match[arrow_set]
                            before_inform.append((word_ocr['transcription'],'gene_protein'))
                            ocr_not_contained_arrow_match[arrow_set]=before_inform
                        elif np.argmax(np.array(word_class))==2:
                            before_inform= ocr_not_contained_arrow_match[arrow_set]
                            before_inform.append((word_ocr['transcription'],'protein_complex'))
                            ocr_not_contained_arrow_match[arrow_set]=before_inform

                        elif np.argmax(np.array(word_class))==3:
                            before_inform= ocr_not_contained_arrow_match[arrow_set]
                            before_inform.append((word_ocr['transcription'],'other'))
                            ocr_not_contained_arrow_match[arrow_set]=before_inform

                        else:
                            continue

            except:
                print(arrow_file_name.replace('.txt','')+' error')
            
            #make reaction
            try:
                for arrow_ocr, inform in arrow_ocr_match.items():
                    continue_criteria=False
                    for result_inform in inform:
                        if result_inform==0:
                            continue_criteria=True
                    if continue_criteria==True:
                        continue
                    metabolite1_name= inform[0][0]
                    metabolite1_head_tail= inform[0][2]
                    metabolite2_name= inform[1][0]
                    metabolite2_head_tail= inform[1][2]
                    if metabolite1_head_tail=='head' and metabolite2_head_tail=='tail':
                        reaction= metabolite2_name+' -> '+ metabolite1_name
                    elif metabolite1_head_tail=='head' and metabolite2_head_tail=='head':
                        reaction= metabolite1_name+' <-> '+ metabolite2_name
                    elif metabolite1_head_tail=='tail' and metabolite2_head_tail=='head':
                        reaction= metabolite1_name+' -> '+ metabolite2_name
                    elif metabolite1_head_tail=='None' and metabolite2_head_tail=='None':
                        reaction= metabolite1_name+' |-| '+ metabolite2_name
                    elif metabolite1_head_tail=='tail' and metabolite2_head_tail=='tail':
                        reaction= metabolite1_name+' --- '+ metabolite2_name

                    gene_enzyme_other_inform= ocr_not_contained_arrow_match[arrow_ocr]
                    total_output.append([reaction,gene_enzyme_other_inform])
                image_inform=[]
                
                for each_inform in total_output:
                    key= each_inform[0]
                    inform= each_inform[1]
                    gene_list=[]
                    gene_protein_list=[]
                    protein_complex_list=[]
                    other_list=[]
                    for other_ocr in inform:
                        if other_ocr[1]=='gene':
                            gene_list.append(other_ocr[0])
                        elif other_ocr[1]=='gene_protein':
                            gene_protein_list.append(other_ocr[0])
                        elif other_ocr[1]=='protein_complex':
                            protein_complex_list.append(other_ocr[0])
                        else:
                            other_list.append(other_ocr[0])
                    image_inform.append([arrow_file_name.replace('.txt',''), key,gene_list, gene_protein_list, protein_complex_list,other_list])
                result= pd.DataFrame(image_inform, columns=['image_name','reaction','gene','gene_protein','protein_complex','others'])
                dataframe= pd.concat([dataframe, result])
                dataframe.to_excel(args.output_dir+'/output.xlsx')
                
                '''
                f = open(args.output_dir+'/final_output/final_'+arrow_file_name, 'w')
                for each_inform in total_output:
                    key= each_inform[0]
                    inform= each_inform[1]
                    f.write('REACTION: '+key+'\n')
                    gene_list=[]
                    gene_protein_list=[]
                    protein_complex_list=[]
                    other_list=[]
                    for other_ocr in inform:
                        if other_ocr[1]=='gene':
                            gene_list.append(other_ocr[0])
                        elif other_ocr[1]=='gene_protein':
                            gene_protein_list.append(other_ocr[0])
                        elif other_ocr[1]=='protein_complex':
                            protein_complex_list.append(other_ocr[0])
                        else:
                            other_list.append(other_ocr[0])
                    f.write('GENE: ')
                    for gene in gene_list:
                        f.write(gene+' , ')
                    f.write('\n')
                    f.write('GENE_PROTEIN: ')
                    for gene_protein in gene_protein_list:
                        f.write(gene_protein+' , ')
                    f.write('\n')
                    f.write('PROTEIN_COMPLEX: ')    
                    for protein_complex in protein_complex_list:
                        f.write(protein_complex+' , ')
                    f.write('\n')
                    f.write('OTHERS: ')    
                    for other in other_list:
                        f.write(other+' , ')

                    f.write('\n')
                    f.write('---------------------------------------------------')
                    f.write('\n')
                f.close()
                
                '''
                print(arrow_file_name.replace('.txt','')+' finished')
            
            except:
                print(arrow_file_name.replace('.txt','')+' error')