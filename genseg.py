from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder

@SEGMENTORS.register_module()
class GenSeg(EncoderDecoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)