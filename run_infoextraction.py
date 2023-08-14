#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import subprocess
from arrow.arrow_total import arrow_head_tail
from paddleocr.tools.infer.predict_system import main
from pathway.chemical_reaction_output import make_reaction_and_text_classifier
from text.ocr_bbox import find_and_combine_ocr_bbox
from text.text_classifier_model import text_classifier
from bulkdownload_and_imageclassification.bulk_download import bulkdownload
from bulkdownload_and_imageclassification.classification import classification


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
group = parser.add_argument_group('EBPI')
group.add_argument('-i', '--input', dest='input', default='./input',
                  help='Use this argument to specify an input folder\n\n')

#bulk download 
group.add_argument('-m', '--metabolite', dest='metabolite', default=False,
                  help='Use this argument to bulk download of specific target product\n\n')
group.add_argument('-he', '--header', dest='header', type= str,
                  help='Use this argument to headers of chrome\n\n')
group.add_argument('-e', '--email', dest='email', type= str,
                  help='Use this argument to email of pmc\n\n')
group.add_argument('-l', '--len', dest='len', default=10000,
                  help='Use this argument to how many paper to bring from pmc about specific target product\n\n')


group.add_argument('-o', '--output', dest='output', default='output',
                  help='Use this argument to specify an output folder\n\n')
group.add_argument('-t', '--threshold', dest='threshold', type=float, default=0.9,
                  help='Use this argument to set confidence score of object detection\n\n')
group.add_argument('-g', '--gpu', dest='gpu', type=str, default='cuda',
                  help='Use this argument for gpu usage\n\n')

#header example
header = {'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'}

args = parser.parse_args()

if os.path.isdir(args.output) == False:
    os.mkdir(args.output)

#bulk download of specific metabolite
if args.metabolite:
    print("Bulk downloading....")
    bulkdownload_result = bulkdownload(args.header, args.metabolite, args.email, args.len)
    print("Bulk downloading ended")
    print("Image classification....")
    classification(bulkdownload_result, args.metabolite, args.gpu)
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

