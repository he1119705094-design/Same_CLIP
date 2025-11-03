'''                                        
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.
                        
Licensed under the Apache License, Version 2.0 (the "License");       
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at                    
                                           
    http://www.apache.org/licenses/LICENSE-2.0
                                                      
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,    
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         
See the License for the specific language governing permissions and
limitations under the License.
'''
import torch

def load_weights(model, model_path):
    from torch import load
    dat = load(model_path, map_location='cpu')
    if 'model' in dat:
        if ('module._conv_stem.weight' in dat['model']) or \
           ('module.fc.fc1.weight' in dat['model']) or \
           ('module.fc.weight' in dat['model']):
            model.load_state_dict(
                {key[7:]: dat['model'][key] for key in dat['model']})
        else:
            model.load_state_dict(dat['model'])
    elif 'state_dict' in dat:
        model.load_state_dict(dat['state_dict'])
    elif 'net' in dat:
        model.load_state_dict(dat['net'])
    elif 'main.0.weight' in dat:
        model.load_state_dict(dat)
    elif '_fc.weight' in dat:
        model.load_state_dict(dat)
    elif 'conv1.weight' in dat:
        model.load_state_dict(dat)
    else:
        print(list(dat.keys()))
        assert False
    return model


def compare_parameters(model, state_dict):
        model_dict = model.state_dict()
        for key in model_dict:
            if key in state_dict:
                if not torch.equal(model_dict[key], state_dict[key]):
                    print(f"Parameter {key} has been updated.")
                else:
                    print(f"Parameter {key} has not changed.")


def create_architecture(name_arch, pretrained=False, num_classes=1, leaky=False,ckpt=None, use_proj=False, proj_ratio=None, dropout=0.5):
    print('Weights status:', pretrained)
    if name_arch == "res50nodown":
        from .resnet_mod import resnet50

        if pretrained:
            model = resnet50(pretrained=True, stride0=1, dropout=dropout,leaky=leaky, use_proj=use_proj, proj_ratio=proj_ratio).change_output(num_classes,use_proj=use_proj)
        else:
            model = resnet50(num_classes=num_classes, stride0=1, dropout=dropout,leaky=leaky, use_proj=use_proj, proj_ratio=proj_ratio)
        if ckpt is not None:
            load_weights(model, model_path=ckpt)
        
    elif name_arch == "res50nodown_dino":
        from .resnet_mod import resnet50
        model = resnet50(num_classes=num_classes, stride0=1, dropout=dropout,leaky=leaky, use_proj=use_proj)
        load_result = model.load_state_dict(torch.load(ckpt),strict=False ) 
        missing_keys = load_result.missing_keys
        unexpected_keys = load_result.unexpected_keys
    
    elif name_arch == "res50":
        from .resnet_mod import resnet50

        if pretrained:
            model = resnet50(pretrained=True, stride0=2,leaky=leaky, use_proj=use_proj).change_output(num_classes)
        else:
            model = resnet50(num_classes=num_classes, stride0=2,leaky=leaky, use_proj=use_proj)
    elif name_arch == "res18nodown":
        from .resnet_mod import resnet18
        
        if pretrained:
            model = resnet18(pretrained=True, stride0=1,dropout=dropout,leaky=leaky, use_proj=use_proj).change_output(num_classes)
        else:
            model = resnet18(num_classes=num_classes, stride0=1, dropout=dropout,leaky=leaky, use_proj=use_proj)

    else:
        assert False
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
