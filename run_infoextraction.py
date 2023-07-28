#!/usr/bin/env python
# coding: utf-8

import argparse
import subprocess
from arrow.arrow_total import arrow_head_tail
from paddleocr.tools.infer.predict_system import main
from pathway.chemical_reaction_output import make_reaction_and_text_classifier
from text.ocr_bbox import find_and_combine_ocr_bbox
from text.text_classifier_model import text_classifier

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
group = parser.add_argument_group('EBPI')
group.add_argument('-i', '--input', dest='input', default='./input', nargs='+',
                  help='Use this argument to specify an input folder\n\n')
options = parser.parse_args()

#OCR
find_and_combine_ocr_bbox(options.input)
print('OCR revise process ended')

#Head tail
print('arrow_head_tail finding....')
arrow_head_tail(args)
print('arrow_head_tail detection ended')

#final processing
print('start final processing....')
output= make_reaction_and_text_classifier(args,text_classifier)
print('final output save to '+ args.output_dir)

