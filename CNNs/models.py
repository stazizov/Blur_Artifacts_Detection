import numpy as np
# from metrics import HistologicalTransforms
import torch
from torch.utils import model_zoo
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from dataclasses import dataclass
import os
import tqdm
import torchvision

@dataclass 
class config:
    dataset_directory = ""
    best_weights_path = "/content/best_model.pth"
    train_directory = os.path.join(dataset_directory, "Train_Data")
    test_directory = os.path.join(dataset_directory, "Test_Data")
    image_height = 128
    image_width = 128
    batch_size = 8
    num_epochs = 200
    lr = 0.001


class CPDBModel:
    '''
    a simple heuristic model which assigns a label on the image crop with 
    respect to threshold from the metric cpbd (the threshold is evaluated from the local statistics)
    of the training data
    '''

    @staticmethod
    def cpbd(image: np.ndarray):
        pass
    
    def __init__(self, percentile: float = 51) -> None:
        '''
        percentile - percentile (must be between 0 and 100 inclusive) which will be
        the anchor in evaluating the threshold
        '''
        self.threshold = 0.5
        # self.metric = HistologicalTransforms().cpbd
        self.percentile = percentile

    def fit(self, images):
        '''
        fits the model by calculating the metric and assigning the threshold
        to the initialized percentile
        '''
        metrics = np.array([self.metric(img) for img in images])
        self.threshold = np.percentile(metrics, self.percentile, out=0.5)
    
    def predict(self, img):
        res = self.cpbd(img)
        return 1 if res >= self.threshold else 0


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'first_conv': 'layer0.conv1', 'classifier': 'last_linear',
        **kwargs
    }

default_cfg = {
    'seresnet18':
        _cfg(url='https://www.dropbox.com/s/3o3nd8mfhxod7rq/seresnet18-4bb0ce65.pth?dl=1',
             interpolation='bicubic')
}


class AdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(self, output_size=1, pool_type='avg'):
        super(AdaptivePool2d, self).__init__()
        self.output_size = output_size
        self.pool_type = pool_type
        self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        return self.pool(x)

    def feat_mult(self):
        return 1

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'output_size=' + str(self.output_size) \
               + ', pool_type=' + self.pool_type + ')'


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.fc1 = nn.Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class SEResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEReSNet(nn.Module):
    def __init__(self, block=SEResNetBlock, layers=[2, 2, 2, 2], groups=1, reduction=16,
                 in_chans=3, inplanes=64, downsample_kernel_size=1,
                 downsample_padding=0, num_classes=1000, global_pool='avg'):
        super(SEReSNet, self).__init__()
        self.inplanes = inplanes
        self.num_classes = num_classes

        layer0_modules = [
            ('conv1', nn.Conv2d(
                        in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1', nn.BatchNorm2d(inplanes)),
            ('relu1', nn.ReLU(inplace=True)),
            ]
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = AdaptivePool2d(pool_type=global_pool)
        self.num_features = 512 * block.expansion
        self.last_linear = nn.Linear(self.num_features, num_classes)

        for m in self.modules():
            self._weight_init(m)
    
    def _weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(
            self.inplanes, planes, groups, reduction, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.last_linear

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        del self.last_linear
        if num_classes:
            self.last_linear = nn.Linear(self.num_features, num_classes)
        else:
            self.last_linear = None

    def forward_features(self, x, pool=True):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if pool:
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
        return x

    def logits(self, x):
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.logits(x)
        return x


def load_pretrained(model, default_cfg, num_classes=1000, in_chans=3, filter_fn=None):
    state_dict = model_zoo.load_url(default_cfg['url'])

    if in_chans == 1:
        conv1_name = default_cfg['first_conv']
        print('Converting first conv (%s) from 3 to 1 channel' % conv1_name)
        conv1_weight = state_dict[conv1_name + '.weight']
        state_dict[conv1_name + '.weight'] = conv1_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        raise AssertionError("Invalid in_chans for pretrained weights")

    strict = True
    classifier_name = default_cfg['classifier']

    if num_classes != default_cfg['num_classes']:
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    model.load_state_dict(state_dict, strict=strict)


def seresnet18(num_classes=1000, in_chans=3, pretrained=False, **kwargs):
    cfg = default_cfg['seresnet18']
    model = SEReSNet(SEResNetBlock, [2, 2, 2, 2], groups=1, reduction=16,
                  inplanes=64,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = cfg
    if pretrained:
        load_pretrained(model, cfg, num_classes, in_chans)
    
    model.reset_classifier(1)
    model.load_state_dict(torch.load("weights/seresnet18.pt", map_location=torch.device("cpu")))
    return model


class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        test_dataset,
        num_epochs,
        batch_size,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ) -> None:

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        
        # model 
        self.model = model
        self.sigmoid = nn.Sigmoid()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr = config.lr, 
        )
        self.criterion = nn.BCELoss()
        
        # for saving checkpoint
        self.best_accuracy = 0
        

        self.train_dataLoader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size = batch_size, 
            collate_fn = self.collate_fn,
        )

        self.test_dataLoader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size = config.batch_size,
            collate_fn = self.collate_fn,
        )
        
    
    def collate_fn(self, batch):
        images, labels = [], []
        for image, label in batch:
            image = torch.tensor(image / 255)
            image = image.permute((2, 0, 1)).float()
            if image.shape == torch.Size([3, 224, 224]):
                images.append(image)

                label = torch.tensor([label]).float()
                labels.append(label)

        images = torch.stack(images).to(self.device)
        labels = torch.stack(labels).to(self.device)
        return images, labels
    
    def measure_accuracy(self, outputs, labels, thrershold=0.5):
        outputs = (outputs > thrershold).float()
        num_correct = (outputs == labels).sum() / len(labels)
        return num_correct
        
    def train_epoch(self, current_epoch):
        # switch the model to train mode
        self.model.train()
        
        pbar = tqdm.notebook.tqdm(
            enumerate(self.train_dataLoader), 
            total = len(self.train_dataLoader),
            desc = f"Epoch(train) {current_epoch} "
        )
        
        running_loss = 0
        running_accuracy = 0
        
        for index, (images, labels) in pbar:
            outputs = self.model(images)
            loss = self.criterion(
                self.sigmoid(outputs),
                labels,
            )
            
            running_accuracy += self.measure_accuracy(outputs, labels).item()
            running_loss += loss.item()
            
            pbar.set_postfix(
                dict(
                    accuracy = round(running_accuracy/(index+1), 5),
                    loss = round(running_loss/(index+1), 5)
                )
            )
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    
    def test_epoch(self, current_epoch):
        # switch model to evaluation mode
        self.model.eval()
        
        pbar = tqdm.notebook.tqdm(
            enumerate(self.test_dataLoader), 
            total = len(self.test_dataLoader),
            desc = f"Epoch(test) {current_epoch} "
        )
        
        running_loss = 0
        running_accuracy = 0
        
        for index, (images, labels) in pbar:
            with torch.no_grad():
                outputs = self.model(images)

            running_loss += self.criterion(
                self.sigmoid(outputs),
                labels,
            ).item()

            running_accuracy += self.measure_accuracy(outputs, labels).item()

            pbar.set_postfix(
                dict(
                    accuracy = round(running_accuracy/(index + 1), 5),
                    loss = round(running_loss/(index + 1), 5)
                )
            )
        
        if running_accuracy / (index + 1) > self.best_accuracy:
            self.best_accuracy = running_accuracy / (index + 1)
            torch.save(self.model.state_dict(), config.best_weights_path)
            print(f"saved model weights at: {config.best_weights_path}")

        return outputs, labels
    
    def start(self):
        print(f"Start training using {self.device}")
        self.model.to(self.device)
        for epoch in range(self.num_epochs):
            self.train_epoch(epoch)
            self.test_epoch(epoch)


def mobilenetv3():
    model = torchvision.models.mobilenet_v3_small(weights=None)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features=576, out_features=128, bias=True),
        torch.nn.Hardswish(),
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=128, out_features=1, bias=True)
    )
    model.load_state_dict(torch.load("weights/mobilenet_model.pth", map_location=torch.device('cpu')))
    return model

if __name__ == "__main__":
    print(seresnet18())
    print(mobilenetv3())
