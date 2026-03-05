from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
import mmcv

config_file = '../configs/genseg_r50_512x512_160k_cityscapes.py'
checkpoint_file = '../pretrained/genseg.pth'

model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

img = 'test.jpg'
result = inference_segmentor(model, img)
show_result_pyplot(model, img, result, out_file='result.jpg')