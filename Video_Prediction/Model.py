import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib.colors import ListedColormap
from torchmetrics import classification as metrics

from Util import plot_image, get_device

batchnorm_momentum = 0.1
align_corners = False
rgb2classes = {
    (0, 0, 0): 0,  # Background (Schwarz)
    (0, 0, 255): 1,  # Human diver (Blau)
    (0, 255, 0): 2,  # Plant (Grün)
    (0, 255, 255): 3,  # Wreck or ruin (Sky)
    (255, 0, 0): 4,  # Robot (Rot)
    (255, 0, 255): 5,  # Reef or invertebrate (Pink)
    (255, 255, 0): 6,  # Fish or vertebrate (Gelb)
    (255, 255, 255): 7  # Sea-floor or rock (Weiß)
}
classColorMap = ListedColormap([(r / 255, g / 255, b / 255) for (r, g, b) in rgb2classes.keys()])


def get_model(model_path):
    if not os.path.exists(model_path):
        return MemoryError("Can not find model at path " + model_path)

    device = get_device()

    model = PIDNet(
        name="PIDNet-S",
        learning_rate=1e-6,
        out_channels=32,
        ppm_channels=96,
        head_channels=128
    )

    if device == torch.device("cuda"):
        model.load_state_dict(torch.load(model_path))

    else:
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    return model


def get_prediction(model, frame):
    with torch.no_grad():
        return model.predict(torch.Tensor(frame) / 255, 1024, 1024, plot=False)


class PIDNet(nn.Module):
    def __init__(self, name: str, learning_rate: float, in_channels=3, out_channels=32, ppm_channels=96,
                 head_channels=128, num_classes=len(rgb2classes)):
        super(PIDNet, self).__init__()

        self.name = name
        self.num_classes = num_classes

        self.relu = nn.ReLU(inplace=True)

        # I Branch
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
                momentum=batchnorm_momentum
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
                momentum=batchnorm_momentum
            ),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self.__make_layer(
            block=BasicBlock,
            in_channels=out_channels,
            out_channels=out_channels,
            blocks=2
        )

        self.layer2 = self.__make_layer(
            block=BasicBlock,
            in_channels=out_channels,
            out_channels=out_channels * 2,
            blocks=2,
            stride=2
        )

        self.layer3 = self.__make_layer(
            block=BasicBlock,
            in_channels=out_channels * 2,
            out_channels=out_channels * 4,
            blocks=3,
            stride=2
        )

        self.layer4 = self.__make_layer(
            block=BasicBlock,
            in_channels=out_channels * 4,
            out_channels=out_channels * 8,
            blocks=3,
            stride=2
        )

        self.layer5 = self.__make_layer(
            block=BottleneckBlock,
            in_channels=out_channels * 8,
            out_channels=out_channels * 8,
            blocks=2,
            stride=2
        )

        # P Branch
        self.compression3 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels * 4,
                out_channels=out_channels * 2,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=out_channels * 2,
                momentum=batchnorm_momentum
            ),
        )

        self.compression4 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels * 8,
                out_channels=out_channels * 2,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=out_channels * 2,
                momentum=batchnorm_momentum
            ),
        )

        self.pag3 = PagFM(
            in_channels=out_channels * 2,
            inter_channels=out_channels
        )

        self.pag4 = PagFM(
            in_channels=out_channels * 2,
            inter_channels=out_channels
        )

        self.layer3_ = self.__make_layer(
            block=BasicBlock,
            in_channels=out_channels * 2,
            out_channels=out_channels * 2,
            blocks=2

        )
        self.layer4_ = self.__make_layer(
            block=BasicBlock,
            in_channels=out_channels * 2,
            out_channels=out_channels * 2,
            blocks=2
        )

        self.layer5_ = self.__make_layer(
            block=BottleneckBlock,
            in_channels=out_channels * 2,
            out_channels=out_channels * 2,
            blocks=1
        )

        # D Branch
        self.layer3_d = self.__make_single_layer(
            block=BasicBlock,
            in_channels=out_channels * 2,
            out_channels=out_channels
        )

        self.layer4_d = self.__make_layer(
            block=BottleneckBlock,
            in_channels=out_channels,
            out_channels=out_channels,
            blocks=1
        )

        self.diff3 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels * 4,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
                momentum=batchnorm_momentum
            ),
        )

        self.diff4 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels * 8,
                out_channels=out_channels * 2,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=out_channels * 2,
                momentum=batchnorm_momentum
            ),
        )

        self.spp = PAPPM(
            in_channels=out_channels * 16,
            branch_channels=ppm_channels,
            out_channels=out_channels * 4
        )

        self.dfm = LightBag(
            in_channels=out_channels * 4,
            out_channels=out_channels * 4
        )

        self.layer5_d = self.__make_layer(
            block=BottleneckBlock,
            in_channels=out_channels * 2,
            out_channels=out_channels * 2,
            blocks=1
        )

        # Prediction Head
        self.seghead_p = SegmentHead(
            in_channels=out_channels * 2,
            inter_channels=head_channels,
            out_channels=num_classes
        )

        self.seghead_d = SegmentHead(
            in_channels=out_channels * 2,
            inter_channels=out_channels,
            out_channels=1
        )

        self.final_layer = SegmentHead(
            in_channels=out_channels * 4,
            inter_channels=head_channels,
            out_channels=num_classes
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    tensor=m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.semantic_loss_function = SemanticCrossEntropyLoss(
            ignore_label=-1,
            thres=0.9,
            min_kept=100_000,
            weight=None
        )

        self.boundary_loss_function = BoundaryLoss()
        self.learning_rate = learning_rate
        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)

    @staticmethod
    def __make_layer(block, in_channels, out_channels, blocks, stride=1):
        downsample = None

        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(
                    num_features=out_channels * block.expansion,
                    momentum=batchnorm_momentum
                ),
            )

        layers = []
        layers.append(
            block(in_channels, out_channels, stride, downsample)
        )

        in_channels = out_channels * block.expansion

        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(
                    block(in_channels, out_channels, stride=1, no_relu=True)
                )

            else:
                layers.append(
                    block(in_channels, out_channels, stride=1, no_relu=False)
                )

        return nn.Sequential(*layers)

    @staticmethod
    def __make_single_layer(block, in_channels, out_channels, stride=1):
        downsample = None

        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(
                    num_features=out_channels * block.expansion,
                    momentum=batchnorm_momentum
                ),
            )

        layer = block(in_channels, out_channels, stride, downsample, no_relu=True)

        return layer

    def forward(self, x):
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)

        x_ = self.layer3_(x)
        x_d = self.layer3_d(x)

        x = self.layer3(x)
        x = self.relu(x)

        x_ = self.pag3(x_, self.compression3(x))
        x_d = x_d + F.interpolate(
            input=self.diff3(x),
            size=[height_output, width_output],
            mode='bilinear',
            align_corners=align_corners
        )

        temp_p = x_

        x = self.layer4(x)
        x = self.relu(x)

        x_ = self.relu(x_)
        x_ = self.layer4_(x_)

        x_d = self.relu(x_d)
        x_d = self.layer4_d(x_d)

        x_ = self.pag4(x_, self.compression4(x))
        x_d = x_d + F.interpolate(
            input=self.diff4(x),
            size=[height_output, width_output],
            mode='bilinear',
            align_corners=align_corners
        )

        temp_d = x_d

        x_ = self.relu(x_)
        x_ = self.layer5_(x_)

        x_d = self.relu(x_d)
        x_d = self.layer5_d(x_d)

        x = F.interpolate(
            self.spp(self.layer5(x)),
            size=[height_output, width_output],
            mode='bilinear',
            align_corners=align_corners
        )

        x_ = self.final_layer(self.dfm(x_, x, x_d))

        x_extra_p = self.seghead_p(temp_p)
        x_extra_d = self.seghead_d(temp_d)

        return x_extra_p, x_, x_extra_d

    train_mode = 0
    test_mode = 1

    def __predict(self, image_tensor, output_height, output_width):
        image_tensor = image_tensor.permute(0, 3, 1, 2)

        prediction_start_ts = datetime.now()

        batch_p_branch_prediction_tensor, batch_i_branch_prediction_tensor, batch_d_branch_prediction_tensor = self.forward(
            image_tensor)

        prediction_stop_ts = datetime.now()
        prediction_time = (prediction_stop_ts - prediction_start_ts).total_seconds() * 1_000_000

        p_prediction_height, p_prediction_width = batch_i_branch_prediction_tensor.size(
            2), batch_i_branch_prediction_tensor.size(3)
        i_prediction_height, i_prediction_width = batch_i_branch_prediction_tensor.size(
            2), batch_i_branch_prediction_tensor.size(3)
        d_prediction_height, d_prediction_width = batch_i_branch_prediction_tensor.size(
            2), batch_i_branch_prediction_tensor.size(3)

        if p_prediction_height != output_height or p_prediction_height != output_width:
            batch_p_branch_prediction_tensor = F.interpolate(
                batch_p_branch_prediction_tensor,
                size=(output_height, output_width),
                mode='bilinear',
                align_corners=True
            )

        if i_prediction_height != output_height or i_prediction_height != output_width:
            batch_i_branch_prediction_tensor = F.interpolate(
                batch_i_branch_prediction_tensor,
                size=(output_height, output_width),
                mode='bilinear',
                align_corners=True
            )

        if d_prediction_height != output_height or d_prediction_height != output_width:
            batch_d_branch_prediction_tensor = F.interpolate(
                batch_d_branch_prediction_tensor,
                size=(output_height, output_width),
                mode='bilinear',
                align_corners=True
            )

        return batch_p_branch_prediction_tensor, batch_i_branch_prediction_tensor, batch_d_branch_prediction_tensor, prediction_time

    def predict(self, image_tensor, output_height, output_width, plot=False):
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(dim=0)

        _, prediction, _ = self.forward(
            image_tensor.permute(0, 3, 1, 2)
        )

        prediction = F.softmax(
            prediction.detach(),
            dim=1
        )

        prediction_height, prediction_width = prediction.size(1), prediction.size(2)
        if prediction_height != output_height or prediction_height != output_width:
            prediction = F.interpolate(
                prediction,
                size=(output_height, output_width),
                mode='bilinear',
                align_corners=True
            )

        prediction = torch.argmax(
            prediction.squeeze().permute(1, 2, 0),
            dim=2
        )

        if plot:
            plot_image(image_tensor.squeeze(), image_color=True)
            plot_image(prediction, image_color=False, image_cmap=classColorMap)
            plot_image(image_tensor.squeeze(), mask=prediction, mask_color=False, mask_cmap=classColorMap)

        return prediction

    def process_data(self, mode: int, batch_image_tensor, batch_image_label_tensor, batch_boundary_label_tensor,
                     print_out=False):
        batch_image_label_tensor = batch_image_label_tensor.permute(0, 3, 1, 2)
        label_height, label_width = batch_image_label_tensor.size(2), batch_image_label_tensor.size(3)

        batch_p_branch_prediction_tensor, batch_i_branch_prediction_tensor, batch_d_branch_prediction_tensor, prediction_time = self.__predict(
            batch_image_tensor, label_height, label_width)

        iou, pixel_accuracy = self.calc_metrics(batch_i_branch_prediction_tensor, batch_image_label_tensor)

        semantic_loss = self.semantic_loss_function(
            [batch_p_branch_prediction_tensor, batch_i_branch_prediction_tensor], batch_image_label_tensor)
        boundary_loss = self.boundary_loss_function(batch_d_branch_prediction_tensor, batch_boundary_label_tensor)

        filler = torch.ones_like(batch_image_label_tensor) * -1
        boundary_label = torch.where(
            torch.sigmoid(batch_d_branch_prediction_tensor) > 0.8,
            batch_image_label_tensor,
            filler
        )
        combined_loss = self.semantic_loss_function(batch_i_branch_prediction_tensor, boundary_label)

        full_loss = semantic_loss + boundary_loss + combined_loss
        full_loss = torch.unsqueeze(full_loss, 0).mean()

        if print_out:
            print(
                "\n=" * 100,
                "\nPixel_Accuracy:", pixel_accuracy,
                "\nSemantic Loss:", semantic_loss,
                "\nBoundary Loss:", boundary_loss,
                "\nCombined Loss:", combined_loss,
                "\nFull Loss:", full_loss
            )

        if mode == PIDNet.train_mode:
            self.optimiser.zero_grad(set_to_none=True)
            full_loss.backward()
            self.optimiser.step()

        return semantic_loss.mean().item(), boundary_loss.item(), combined_loss.item(), full_loss.item(), pixel_accuracy, iou, prediction_time

    def calc_metrics(self, batch_prediction_tensor, batch_image_label_tensor):
        batch_image_label_tensor = batch_image_label_tensor.argmax(dim=1)
        batch_prediction_tensor = F.softmax(batch_prediction_tensor, dim=1).detach()

        iou_metric = metrics.MulticlassJaccardIndex(num_classes=self.num_classes, average="weighted")
        pixel_acc_metric = metrics.MulticlassAccuracy(num_classes=self.num_classes, average="weighted",
                                                      multidim_average="samplewise")

        iou = iou_metric(batch_prediction_tensor, batch_image_label_tensor).item()
        pixel_acc = pixel_acc_metric(batch_prediction_tensor, batch_image_label_tensor).mean().item()

        return iou, pixel_acc

    def get_spec_string(self, override=""):
        if override != "":
            return override

        hyperParam_string = self.name + "-"
        hyperParam_string += str(self.semantic_loss_function).split("(")[0] + "-"
        hyperParam_string += str(self.optimiser).split(" ")[0] + "-"
        hyperParam_string += "lr" + str(self.learning_rate)

        return hyperParam_string


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()

        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            num_features=out_channels,
            momentum=batchnorm_momentum
        )

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(
            num_features=out_channels,
            momentum=batchnorm_momentum
        )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out

        return self.relu(out)


class BottleneckBlock(nn.Module):
    expansion = 2

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, no_relu=True):
        super(BottleneckBlock, self).__init__()

        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            num_features=out_channels,
            momentum=batchnorm_momentum
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(
            num_features=out_channels,
            momentum=batchnorm_momentum
        )

        self.conv3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels * self.expansion,
            kernel_size=1,
            bias=False
        )

        self.bn3 = nn.BatchNorm2d(
            out_channels * self.expansion,
            momentum=batchnorm_momentum
        )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out

        return self.relu(out)


class PagFM(nn.Module):
    def __init__(self, in_channels, inter_channels, after_relu=False, with_channel=False):
        super(PagFM, self).__init__()

        self.with_channel = with_channel
        self.after_relu = after_relu

        self.f_x = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=inter_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(inter_channels)
        )

        self.f_y = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=inter_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(inter_channels)
        )

        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(
                    in_channels=inter_channels,
                    out_channels=in_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(in_channels)
            )

        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        input_size = x.size()

        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        y_q = self.f_y(y)
        y_q = F.interpolate(
            input=y_q,
            size=[input_size[2], input_size[3]],
            mode='bilinear',
            align_corners=False
        )

        x_k = self.f_x(x)

        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))

        y = F.interpolate(
            input=y,
            size=[input_size[2], input_size[3]],
            mode='bilinear',
            align_corners=False
        )
        x = (1 - sim_map) * x + sim_map * y

        return x


class PAPPM(nn.Module):
    def __init__(self, in_channels, branch_channels, out_channels):
        super(PAPPM, self).__init__()

        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(
                num_features=in_channels,
                momentum=batchnorm_momentum
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=branch_channels,
                kernel_size=1,
                bias=False
            ),
        )

        self.scale1 = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.BatchNorm2d(
                num_features=in_channels,
                momentum=batchnorm_momentum
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=branch_channels,
                kernel_size=1,
                bias=False
            ),
        )

        self.scale2 = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=9,
                stride=4,
                padding=4
            ),
            nn.BatchNorm2d(
                num_features=in_channels,
                momentum=batchnorm_momentum
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=branch_channels,
                kernel_size=1,
                bias=False
            ),
        )

        self.scale3 = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=17,
                stride=8,
                padding=8
            ),
            nn.BatchNorm2d(
                num_features=in_channels,
                momentum=batchnorm_momentum
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=branch_channels,
                kernel_size=1,
                bias=False
            ),
        )

        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.BatchNorm2d(
                num_features=in_channels,
                momentum=batchnorm_momentum
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=branch_channels,
                kernel_size=1,
                bias=False
            ),
        )

        self.scale_process = nn.Sequential(
            nn.BatchNorm2d(
                num_features=branch_channels * 5,
                momentum=batchnorm_momentum
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=branch_channels * 5,
                out_channels=branch_channels * 4,
                kernel_size=3,
                padding=1,
                groups=4,
                bias=False
            ),
        )

        self.compression = nn.Sequential(
            nn.BatchNorm2d(
                num_features=branch_channels * 5,
                momentum=batchnorm_momentum
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=branch_channels * 5,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
        )

        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(
                num_features=in_channels,
                momentum=batchnorm_momentum
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]

        x_scale0 = self.scale0(x)

        scale_list = [
            x_scale0
        ]

        x_scale1 = F.interpolate(
            input=self.scale1(x),
            size=[height, width],
            mode='bilinear',
            align_corners=align_corners
        ) + x_scale0
        scale_list.append(x_scale1)

        x_scale2 = F.interpolate(
            input=self.scale2(x),
            size=[height, width],
            mode='bilinear',
            align_corners=align_corners
        ) + x_scale0
        scale_list.append(x_scale2)

        x_scale3 = F.interpolate(
            input=self.scale3(x),
            size=[height, width],
            mode='bilinear',
            align_corners=align_corners
        ) + x_scale0
        scale_list.append(x_scale3)

        x_scale4 = F.interpolate(
            input=self.scale4(x),
            size=[height, width],
            mode='bilinear',
            align_corners=align_corners
        ) + x_scale0
        scale_list.append(x_scale4)

        scale_out = self.scale_process(torch.cat(scale_list, 1))
        out = self.compression(torch.cat([x_scale0, scale_out], 1)) + self.shortcut(x)

        return out


class LightBag(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LightBag, self).__init__()

        self.conv_p = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

        self.conv_i = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)

        p_add = self.conv_p((1 - edge_att) * i + p)
        i_add = self.conv_i(i + edge_att * p)

        return p_add + i_add


class LightBagV2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LightBagV2, self).__init__()

        self.conv_p = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

        self.conv_i = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)

        p_add = self.conv_p((1 - edge_att) * i + p)
        i_add = self.conv_i(i + edge_att * p)

        return p_add + i_add


class SegmentHead(nn.Module):

    def __init__(self, in_channels, inter_channels, out_channels, scale_factor=None):
        super(SegmentHead, self).__init__()

        self.scale_factor = scale_factor

        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(
            num_features=in_channels,
            momentum=batchnorm_momentum
        )

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(
            num_features=inter_channels,
            momentum=batchnorm_momentum
        )

        self.conv2 = nn.Conv2d(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            bias=True
        )

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)

        out = self.bn2(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor

            out = F.interpolate(
                input=out,
                size=[height, width],
                mode='bilinear',
                align_corners=align_corners
            )

        return out


class SemanticCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7, min_kept=100_000, weight=None):
        super(SemanticCrossEntropyLoss, self).__init__()

        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
        )

    def _ce_forward(self, prediction, target):
        prediction = F.softmax(prediction, dim=1)
        return self.criterion(prediction, target)

    def _ohem_forward(self, prediction, target):
        prediction = self._ce_forward(prediction, target).contiguous().view(-1)

        return prediction.mean()

    def forward(self, prediction, target):
        if not (isinstance(prediction, list) or isinstance(prediction, tuple)):
            prediction = [prediction]

        balance_weights = [0.5, 0.5]
        if len(balance_weights) == len(prediction):
            functions = [self._ce_forward] * (len(balance_weights) - 1) + [self._ohem_forward]
            return sum([
                weight * func(x, target)
                for (weight, x, func) in zip(balance_weights, prediction, functions)
            ])

        elif len(prediction) == 1:
            return 0.5 * self._ohem_forward(prediction[0], target)

        else:
            raise ValueError("lengths of prediction and target are not identical!")


class BoundaryLoss(nn.Module):
    def __init__(self, coeff_bce=20.0):
        super(BoundaryLoss, self).__init__()

        self.coeff_bce = coeff_bce

    def forward(self, prediction, target):
        return self.coeff_bce * self.weighted_bce(prediction, target)

    @staticmethod
    def weighted_bce(prediction, target):
        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(1, -1)
        target = target.view(1, -1)

        pos_index = (target == 1)
        neg_index = (target == 0)

        weights = torch.zeros_like(prediction)
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weights[pos_index] = neg_num * 1.0 / sum_num
        weights[neg_index] = pos_num * 1.0 / sum_num

        loss = F.binary_cross_entropy_with_logits(prediction, target, weights, reduction='mean')

        return loss
