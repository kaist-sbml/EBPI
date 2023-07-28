import argparse

def init_args():
    parser = argparse.ArgumentParser()
    
    #input file path
    parser.add_argument("--image_dir", type=str, default='input_file')

    #confidence score of object detection
    parser.add_argument("--threshold", type=float, default=0.9)
    
    #result file directory
    parser.add_argument("--output_dir", type=str, default='inference_results')
    
    parser.add_argument("--gpu", type=str, default='cuda')
    
    #object detection parameters
    parser.add_argument("--checkpoint", type=str, default='arrow/model/checkpoint.pickle')
    
    #text classifier parameters
    parser.add_argument("--checkpoint_bert", type=str, default='text/model/model_BioBERT.pickle')
    
    return parser

def parse_args():
    parser = init_args()
    return parser.parse_args()

if __name__ == '__main__':
    pass