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

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
group = parser.add_argument_group('EBPI')
group.add_argument('-i', '--input', dest='input', default='./input',
                  help='Use this argument to specify an input folder\n\n')
group.add_argument('-o', '--output', dest='output', default='./output',
                  help='Use this argument to specify an output folder\n\n')
group.add_argument('-t', '--threshold', dest='threshold', type=float, default=0.9,
                  help='Use this argument to set confidence score of object detection\n\n')
group.add_argument('-g', '--gpu', dest='gpu', type=str, default='cuda',
                  help='Use this argument for gpu usage\n\n')
args = parser.parse_args()

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

