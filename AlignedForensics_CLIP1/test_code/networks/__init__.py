
def create_architecture(name_arch, pretrained=False, num_classes=1):
    if name_arch == "res50nodown":
        from .resnet_mod import resnet50

        if pretrained:
            model = resnet50(pretrained=True, stride0=1, dropout=0.5).change_output(num_classes)
        else:
            model = resnet50(num_classes=num_classes, stride0=1, dropout=0.5)
    elif name_arch == "res50":
        from .resnet_mod import resnet50

        if pretrained:
            model = resnet50(pretrained=True, stride0=2).change_output(num_classes)
        else:
            model = resnet50(num_classes=num_classes, stride0=2)
    elif name_arch.startswith('opencliplinear_'):
        from .openclipnet import OpenClipLinear
        model = OpenClipLinear(num_classes=num_classes, pretrain=name_arch[15:], normalize=True)
    elif name_arch.startswith('opencliplinearnext_'):
        from .openclipnet import OpenClipLinear
        model = OpenClipLinear(num_classes=num_classes, pretrain=name_arch[19:], normalize=True, next_to_last=True)
    else:
        assert False
    print(f"创建模型：{name_arch}")  # 新加上去的为了更清楚看到创建了一个什么模型
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
            model.load_state_dict(dat['model'])# 把已经训练好的模型参数（从 .pth 文件中加载出来的）赋值到你创建好的模型结构中。
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
    print(f"加载权重文件：{model_path}")  # 新加上去的看一下加载了什么
    print(f"权重文件内容keys：{dat.keys()}")  # 看一下权重文件里面的内容是什么就可以看到训练都训练了哪些参数
    for k in dat.keys():
        v = dat[k]
        print(f"key: {k}, type: {type(v)}")
        if isinstance(v, dict):
            print(f"  dict子key数量: {len(v)}")
            for i, (subk, subv) in enumerate(v.items()):
                if hasattr(subv, 'shape'):
                    print(f"    子key: {subk}, shape: {subv.shape}")
                else:
                    print(f"    子key: {subk}, type: {type(subv)}")
                if i >= 4:
                    print("    ...（只显示前5个）")
                    break
        elif hasattr(v, 'shape'):
            print(f"  shape: {v.shape}")
        else:
            print(f"  value内容: {str(v)[:100]}")
    return model
