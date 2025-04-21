import torch
import torch.nn as nn
import torch.nn.functional as F


class SpikeNet_features(nn.Module):
    def __init__(self, spikenet_pt_model, pretrained=True):
        super(SpikeNet_features, self).__init__()

        self.spikenet_pt_model = spikenet_pt_model
        self.pretrained = pretrained

        # Load the model using torch.load()
        self.model = torch.load(self.spikenet_pt_model)
        self.pad_dict = {}

        # Extract the list of layers from the loaded model
        # layers = list(loaded_model.children())

        # Create a new model with the same layers and architecture
        # self.model = nn.Sequential(*layers)

        # Access the weights of the loaded model
        # model_weights = loaded_model.state_dict()

        # Initialize the weights of the layers if pretrained is False
        if not self.pretrained:
            self.initialize_weights()

    def forward(self, x):
        for name, layer in self.model.named_children():
            if isinstance(layer, nn.Conv2d):
                if layer.padding == "same":
                    layer.padding = 0
                    self.pad_dict[name] = self.calc_pad(x, layer)

                if layer.padding == "valid":
                    layer.padding = 0
                    self.pad_dict[name] = [0, 0, 0, 0]

                x = F.pad(x, self.pad_dict[name])

            x = layer(x)

        return x

    def calc_pad(self, current_data, layer):
        out_height = (current_data.shape[2] + layer.stride[0] - 1) // layer.stride[0]
        out_width = (current_data.shape[3] + layer.stride[1] - 1) // layer.stride[1]

        padding_height = max(
            0,
            (out_height - 1) * layer.stride[0]
            + layer.kernel_size[0]
            - current_data.shape[2],
        )
        padding_width = max(
            0,
            (out_width - 1) * layer.stride[1]
            + layer.kernel_size[1]
            - current_data.shape[3],
        )

        padding_top = padding_height // 2
        padding_bottom = padding_height - padding_top
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left

        return [padding_left, padding_right, padding_top, padding_bottom]

    def initialize_weights(self):
        # initialize the parameters
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def conv_info(self):
        kernel_sizes = []
        strides = []
        paddings = []

        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                kernel_sizes.append(m.kernel_size[0])
                strides.append(m.stride[0])
                paddings.append(3)

        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings

        return self.kernel_sizes, self.strides, self.paddings


def spikenet_features(spikenet_pt_model="spikenet_model_3.pt", pretrained=True):
    """Constructs a Spikenet model.
    Args:
        pretrained (bool): If True, returns the model pre-trained (based on the TensorFlow weights)
    """

    model = SpikeNet_features(
        spikenet_pt_model="spikenet_model_3.pt", pretrained=pretrained
    )

    # Randomly initialize the weights if not pretrained
    if not pretrained:
        # for param in loaded_model.parameters():
        #     param.requires_grad = True
        #     if param.dim() > 1:
        #         torch.nn.init.xavier_uniform_(param)
        raise NotImplementedError

    return model


class ReshapeLayer(nn.Module):
    def __init__(self, new_shape):
        super(ReshapeLayer, self).__init__()
        self.new_shape = new_shape

    def forward(self, x):
        return x.view(self.new_shape)


if __name__ == "__main__":
    spike_features = spikenet_features(
        spikenet_pt_model="spikenet_model_3.pt", pretrained=True
    )
    print(spike_features)
