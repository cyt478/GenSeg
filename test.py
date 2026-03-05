import argparse
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_classes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--img', help='Image file')
    parser.add_argument('--out', help='Path to save the output image')
    args = parser.parse_args()

    model = init_segmentor(args.config, args.checkpoint, device='cuda:0')
    result = inference_segmentor(model, args.img)
    show_result_pyplot(model, args.img, result, get_classes('cityscapes'), out_file=args.out)

if __name__ == '__main__':
    main()