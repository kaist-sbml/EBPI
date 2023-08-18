#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import subprocess
import time
from arrow.arrow_total import arrow_head_tail
from paddleocr.tools.infer.predict_system import main
from pathway.chemical_reaction_output import make_reaction_and_text_classifier
from text.ocr_bbox import find_and_combine_ocr_bbox
from text.text_classifier_model import text_classifier
from bulkdownload_and_imageclassification.bulk_download import bulkdownload
from bulkdownload_and_imageclassification.classification import classification

t1 = time.time()

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
group = parser.add_argument_group('EBPI')
group.add_argument('-i', '--input', dest='input', default='./input',
                  help='Use this argument to specify an input folder\n\n')

#bulk download 
group.add_argument('-m', '--metabolite', dest='metabolite', default=False,
                  help='Use this argument to bulk download of specific target product\n\n')
group.add_argument('-he', '--header', dest='header', type= str,
                  help='Use this argument to headers of chrome\n\n')
#header example= 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'

group.add_argument('-e', '--email', dest='email', type= str,
                  help='Use this argument to email of pmc\n\n')
group.add_argument('-l', '--len', dest='len', default=10000,
                  help='Use this argument to how many paper to bring from pmc about specific target product\n\n')


group.add_argument('-o', '--output', dest='output', default='./output',
                  help='Use this argument to specify an output folder\n\n')
group.add_argument('-t', '--threshold', dest='threshold', type=float, default=0.9,
                  help='Use this argument to set confidence score of object detection\n\n')
group.add_argument('-g', '--gpu', dest='gpu', type=str, default='cuda',
                  help='Use this argument for gpu usage\n\n')


args = parser.parse_args()

#bulk download of specific metabolite
if args.metabolite:
    pmc_name_dict=dict()
    abs_path= os.path.dirname(__file__)
    pmc_list1= pd.read_csv(abs_path + '/bulkdownload_and_imageclassification/oa_non_comm_use_pdf.csv')
    pmc_list2= pd.read_csv(abs_path + '/bulkdownload_and_imageclassification/oa_comm_use_file_list.csv')

    for name in list(pmc_list1['File']):
        pmc_name_dict[name.split('.')[1]]= 'https://ftp.ncbi.nlm.nih.gov/pub/pmc/'+ name

    for name in list(pmc_list2['File']):
        pmc_name_dict[name.split('/')[-1].strip('.tar.gz')] = 'https://ftp.ncbi.nlm.nih.gov/pub/pmc/'+ name
    
    metabolites= args.metabolite.split(' ')
    original_input_dir = args.input
    original_output_dir = args.output
    
    for metabolite in metabolites:

        args.metabolite= metabolite
        args.input= os.path.join(original_input_dir,metabolite)
        args.output= os.path.join(original_output_dir,metabolite)
        
        if os.path.isdir(args.input) == False:
            os.makedirs(args.input)
        if os.path.isdir(args.output) == False:
            os.makedirs(args.output)

        print(metabolite+" start")
        print("Bulk downloading....")
        bulkdownload_result = bulkdownload(args, pmc_name_dict)
        print("Bulk downloading ended")
        
        if bulkdownload_result == []:
            print("There is no result to analyze")
            
        else:
            print("Image classification....")
            classification(args, bulkdownload_result)
            print("Image classification ended")
            #OCR
            print('OCR finding....')
            find_and_combine_ocr_bbox(args)
            print('OCR revise process ended')

            #Head tail
            print('arrow_head_tail finding....')
            arrow_head_tail(args)
            print('arrow_head_tail detection ended')

            #final processing
            print('start final processing....')
            output= make_reaction_and_text_classifier(args, text_classifier)
            print('final output save to '+ args.output)

else:    
    if os.path.isdir(args.input) == False:
        os.mkdir(args.input)

    if os.path.isdir(args.output) == False:
        os.mkdir(args.output)
    #OCR
    print('OCR finding....')
    find_and_combine_ocr_bbox(args)
    print('OCR revise process ended')

    #Head tail
    print('arrow_head_tail finding....')
    arrow_head_tail(args)
    print('arrow_head_tail detection ended')

    #final processing
    print('start final processing....')
    output= make_reaction_and_text_classifier(args, text_classifier)
    print('final output save to '+ args.output)


t2 = time.time()
print('Execution time: %.3f minutes'%((t2-t1)/60))
