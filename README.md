# GenSeg: Generative Segmentation

This is the official implementation of GenSeg.

```bash
pip install -r requirements.txt
python tools/train.py --config configs/genseg_r50_512x512_160k_cityscapes.py
python tools/test.py --config configs/genseg_r50_512x512_160k_cityscapes.py --checkpoint pretrained/genseg.pth
```