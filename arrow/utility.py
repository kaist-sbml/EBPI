import argparse

def init_args():
    parser = argparse.ArgumentParser()
    
    #paddleocr parameters
    
    # params for prediction engine
    parser.add_argument("--use_gpu", default=True)
    parser.add_argument("--use_xpu", default=False)
    parser.add_argument("--use_npu", default=False)
    parser.add_argument("--ir_optim", default=True)
    parser.add_argument("--use_tensorrt", default=False)
    parser.add_argument("--min_subgraph_size", type=int, default=15)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--gpu_mem", type=int, default=500)

    # params for text detector
    parser.add_argument("--image_dir", type=str, default='basic_image')
    parser.add_argument("--page_num", type=int, default=0)
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_model_dir", type=str, default='PaddleOCR/inference/det/en_PP-OCRv3_det_infer/')
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default='max')
    parser.add_argument("--det_box_type", type=str, default='quad')

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.6)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.5)
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--use_dilation", default=False)
    parser.add_argument("--det_db_score_mode", type=str, default="fast")

    # EAST parmas
    parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
    parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
    parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)

    # SAST parmas
    parser.add_argument("--det_sast_score_thresh", type=float, default=0.5)
    parser.add_argument("--det_sast_nms_thresh", type=float, default=0.2)

    # PSE parmas
    parser.add_argument("--det_pse_thresh", type=float, default=0)
    parser.add_argument("--det_pse_box_thresh", type=float, default=0.85)
    parser.add_argument("--det_pse_min_area", type=float, default=16)
    parser.add_argument("--det_pse_scale", type=int, default=1)

    # FCE parmas
    parser.add_argument("--scales", type=list, default=[8, 16, 32])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--fourier_degree", type=int, default=5)

    # params for text recognizer
    parser.add_argument("--rec_algorithm", type=str, default='SVTR_LCNet')
    parser.add_argument("--rec_model_dir", type=str, default='PaddleOCR/inference/reg/ch_PP-OCRv3_rec_infer/')
    parser.add_argument("--rec_image_inverse", default=True)
    parser.add_argument("--rec_image_shape", type=str, default="3, 48, 320")
    parser.add_argument("--rec_batch_num", type=int, default=6)
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default="PaddleOCR/ppocr/utils/ppocr_keys_v1.txt")
    parser.add_argument("--use_space_char", default=True)
    parser.add_argument(
        "--vis_font_path", type=str, default="PaddleOCR/doc/fonts/simfang.ttf")
    parser.add_argument("--drop_score", type=float, default=0.5)

    # params for e2e
    parser.add_argument("--e2e_algorithm", type=str, default='PGNet')
    parser.add_argument("--e2e_model_dir", type=str)
    parser.add_argument("--e2e_limit_side_len", type=float, default=768)
    parser.add_argument("--e2e_limit_type", type=str, default='max')

    # PGNet parmas
    parser.add_argument("--e2e_pgnet_score_thresh", type=float, default=0.5)
    parser.add_argument(
        "--e2e_char_dict_path", type=str, default="PaddleOCR/ppocr/utils/ic15_dict.txt")
    parser.add_argument("--e2e_pgnet_valid_set", type=str, default='totaltext')
    parser.add_argument("--e2e_pgnet_mode", type=str, default='fast')

    # params for text classifier
    parser.add_argument("--use_angle_cls", default=False)
    parser.add_argument("--cls_model_dir", type=str, default='PaddleOCR/inference/cls/ch_ppocr_mobile_v2.0_cls_infer/')
    parser.add_argument("--cls_image_shape", type=str, default="3, 48, 192")
    parser.add_argument("--label_list", type=list, default=['0', '180'])
    parser.add_argument("--cls_batch_num", type=int, default=6)
    parser.add_argument("--cls_thresh", type=float, default=0.9)

    parser.add_argument("--enable_mkldnn", default=False)
    parser.add_argument("--cpu_threads", type=int, default=10)
    parser.add_argument("--use_pdserving", default=False)
    parser.add_argument("--warmup", default=False)

    # SR parmas
    parser.add_argument("--sr_model_dir", type=str)
    parser.add_argument("--sr_image_shape", type=str, default="3, 32, 128")
    parser.add_argument("--sr_batch_num", type=int, default=1)

    #
    parser.add_argument(
        "--draw_img_save_dir", type=str, default="basic_inference")
    parser.add_argument("--save_crop_res", default=False)
    parser.add_argument("--crop_res_save_dir", type=str, default="PaddleOCR/output")

    # multi-process
    parser.add_argument("--use_mp", default=False)
    parser.add_argument("--total_process_num", type=int, default=1)
    parser.add_argument("--process_id", type=int, default=0)

    parser.add_argument("--benchmark", default=False)
    parser.add_argument("--save_log_path", type=str, default="PaddleOCR/log_output/")

    parser.add_argument("--show_log", default=True)
    parser.add_argument("--use_onnx", default=False)
    
    #confidence score of object detection
    parser.add_argument("--threshold", type=float, default=0.9)
    
    #number of object detection classes. In this case, arrow+background
    parser.add_argument("--num_classes", type=int, default=2)
    
    #result file directory
    parser.add_argument("--output_dir", type=str, default='basic_inference')
    
    parser.add_argument("--gpu", type=str, default='cuda')
    
    #object detection parameters
    parser.add_argument("--checkpoint", type=str, default='arrow_detection/checkpoint/checkpoint.pickle')
    
    #text classifier parameters
    parser.add_argument("--checkpoint_bert", type=str, default='text_classifier/checkpoint/text_classifier_model.pickle')
    
    return parser

def parse_args():
    parser = init_args()
    return parser.parse_args()

if __name__ == '__main__':
    pass