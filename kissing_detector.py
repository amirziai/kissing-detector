import torch
from torch import nn
import vggish
from conv import convnet_init
from typing import Optional


class KissingDetector(nn.Module):
    def __init__(self,
                 conv_model_name: Optional[str],
                 num_classes: int,
                 feature_extract: bool,
                 use_pretrained: bool = True,
                 use_vggish: bool = True):
        super(KissingDetector, self).__init__()
        conv_output_size = 0
        vggish_output_size = 0
        conv_input_size = 0
        conv = None
        vggish_model = None

        if conv_model_name:
            conv, conv_input_size, conv_output_size = convnet_init(conv_model_name,
                                                                   num_classes,
                                                                   feature_extract,
                                                                   use_pretrained)
        if use_vggish:
            vggish_model, vggish_output_size = vggish.vggish(feature_extract)

        if not conv and not vggish_model:
            raise ValueError("Use VGGish, Conv, or both")

        self.conv_input_size = conv_input_size
        self.conv = conv
        self.vggish = vggish_model
        self.combined = nn.Linear(vggish_output_size + conv_output_size, num_classes)

    def forward(self, audio: torch.Tensor, image: torch.Tensor):
        a = self.vggish(audio) if self.vggish else None
        c = self.conv(image) if self.conv else None

        if a and c:
            combined = torch.cat((c.view(c.size(0), -1), a.view(a.size(0), -1)), dim=1)
        else:
            combined = a if a else c

        return self.combined(combined)
