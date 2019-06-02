import torch
from torch import nn
import vggish
from conv import convnet_init


class KissingDetector(nn.Module):
    def __init__(self, model_name: str, num_classes: int, feature_extract: bool, use_pretrained: bool = True):
        super(KissingDetector, self).__init__()
        conv, conv_input_size, conv_output_size = convnet_init(model_name, num_classes, feature_extract,
                                                               use_pretrained=use_pretrained)
        vggish_model, vggish_output_size = vggish.vggish(feature_extract)
        self.conv_input_size = conv_input_size
        self.conv = conv
        self.vggish = vggish_model
        self.combined = nn.Linear(vggish_output_size + conv_output_size, num_classes)

    def forward(self, audio: torch.Tensor, image: torch.Tensor):
        a = self.vggish(audio)
        c = self.conv(image)
        combined = torch.cat((c.view(c.size(0), -1), a.view(a.size(0), -1)), dim=1)
        out = self.combined(combined)
        return out
