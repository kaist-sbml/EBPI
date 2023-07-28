#!/usr/bin/env python
# coding: utf-8

# In[1]:
from text.text_classifier_model import text_classifier
from PaddleOCR.tools.infer.predict_system import predict
import utility
from arrow.arrow_total import arrow_head_tail
from pathway.chemical_reaction_output import make_reaction_and_text_classifier
from text.revise_ocr_bbox import revise_ocr_bbox

#OCR
# 여기는 고쳐야 할 곳
print('OCR finding....')
args = utility.parse_args()
predict(args)
print('OCR process ended')
print('OCR revise process start....')
revise_ocr_bbox(args)
print('OCR revise process ended')

#Head tail
print('arrow_head_tail finding....')
arrow_head_tail(args)
print('arrow_head_tail detection ended')

#final processing
print('start final processing....')
output= make_reaction_and_text_classifier(args,text_classifier)
print('final output save to '+ args.output_dir)

