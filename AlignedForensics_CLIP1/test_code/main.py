import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import os
import pandas
import numpy as np
import tqdm
import glob
import sys
import yaml
from PIL import Image
from PIL.ImageFile import ImageFile

from torchvision.transforms  import CenterCrop, Resize, Compose, InterpolationMode
from utils.processing import make_normalize             ##图像处理模块
from utils.fusion import apply_fusion                   ##融合模块
from networks import create_architecture, load_weights #create_architecture：创建模型。后者加载模型的参数
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 加载模型的，因为模型文件是yaml格式的，这个函数可以返回模型名，模型路径，网络架构，归一化类型，裁剪的尺寸。
def get_config(model_name, weights_dir='./weights'):
    with open(os.path.join(weights_dir, model_name, 'config.yaml')) as fid:
        data = yaml.load(fid, Loader=yaml.FullLoader)
    model_path = os.path.join(weights_dir, model_name, data['weights_file'])
    return data['model_name'], model_path, data['arch'], data['norm_type'], data['patch_size']

# 模型进行测试的主要函数。
def runnig_tests(input_csv, weights_dir, models_list, device, batch_size = 1):
    table = pandas.read_csv(input_csv)[['filename',]]
    rootdataset = os.path.dirname(os.path.abspath(input_csv))
    
    models_dict = dict()
    transform_dict = dict()
    print("Models:")
    for model_name in models_list:
        print(model_name, flush=True)
        _, model_path, arch, norm_type, patch_size = get_config(model_name, weights_dir=weights_dir)

        model = load_weights(create_architecture(arch), model_path)
        model = model.to(device).eval()

        transform = list()
        if patch_size is None:
            print('input none', flush=True)
            transform_key = 'none_%s' % norm_type
        elif patch_size=='Clip224':
            print('input resize:', 'Clip224', flush=True)
            transform.append(Resize(224, interpolation=InterpolationMode.BICUBIC))
            transform.append(CenterCrop((224, 224)))
            transform_key = 'Clip224_%s' % norm_type
        elif isinstance(patch_size, tuple) or isinstance(patch_size, list):
            print('input resize:', patch_size, flush=True)
            transform.append(Resize(*patch_size))
            transform.append(CenterCrop(patch_size[0]))
            transform_key = 'res%d_%s' % (patch_size[0], norm_type)
        elif patch_size > 0:
            print('input crop:', patch_size, flush=True)
            transform.append(CenterCrop(patch_size))
            transform_key = 'crop%d_%s' % (patch_size, norm_type)
        
        transform.append(make_normalize(norm_type))
        transform = Compose(transform)
        transform_dict[transform_key] = transform
        models_dict[model_name] = (transform_key, model)
        print(flush=True)

    ### test
    with torch.no_grad():
        
        do_models = list(models_dict.keys())
        do_transforms = set([models_dict[_][0] for _ in do_models])
        print(do_models)
        print(do_transforms)
        print(flush=True)
        
        print("Running the Tests")
        batch_img = {k: list() for k in transform_dict}
        batch_id = list()
        last_index = table.index[-1]
        for index in tqdm.tqdm(table.index, total=len(table)):
            filename = os.path.join(rootdataset, table.loc[index, 'filename'])
            img_name = os.path.splitext(os.path.basename(filename))[0]
            # for k in transform_dict:
            #     batch_img[k].append(transform_dict[k](Image.open(filename).convert('RGB')))
            # batch_id.append(index)
            try:
                img = Image.open(filename).convert('RGB')
                for k in transform_dict:
                    batch_img[k].append(transform_dict[k](img))
                batch_id.append(index)
            except Exception as e:
                print(f"[跳过] 无法读取图像: {filename}，错误：{e}")
                continue

            if (len(batch_id) >= batch_size) or (index==last_index):
                for k in do_transforms:
                    batch_img[k] = torch.stack(batch_img[k], 0)

                for model_name in do_models:
                    model = models_dict[model_name][1]
                    model.current_img_name = img_name
                    out_tens = model(batch_img[models_dict[model_name][0]].clone().to(device)).cpu().numpy()
                    #out_tens = models_dict[model_name][1](batch_img[models_dict[model_name][0]].clone().to(device)).cpu().numpy()
                    if out_tens.shape[1] == 1:
                        out_tens = out_tens[:, 0]
                    elif out_tens.shape[1] == 2:
                        out_tens = out_tens[:, 1] - out_tens[:, 0]
                    else:
                        assert False
                    
                    if len(out_tens.shape) > 1:
                        logit1 = np.mean(out_tens, (1, 2))
                    else:
                        logit1 = out_tens

                    for ii, logit in zip(batch_id, logit1):
                        table.loc[ii, model_name] = logit

                batch_img = {k: list() for k in transform_dict}
                batch_id = list()

            torch.cuda.empty_cache()
            assert len(batch_id)==0
        
    return table


if __name__ == "__main__":
    #
    # import argparse     #argparse 是 Python 中用于处理命令行参数的标准工具。我们通过它来让程序从命令行接受用户输入的参数。
    # parser = argparse.ArgumentParser()  #创建一个 ArgumentParser 对象 parser。
    #                                         #原因：ArgumentParser 是 argparse 模块的核心类，负责解析命令行输入的参数并生成一个 Namespace 对象。
    # parser.add_argument("--in_csv"     , '-i', type=str, help="The path of the input csv file with the list of images")
    #                                     #添加一个命令行参数 --in_csv，它也可以用 -i 来调用。它的值类型是字符串。
    #                                         #--in_csv / -i：参数名称，长短格式都可以；
    #                                         #type=str：用户必须提供一个字符串（CSV 文件路径）；
    #                                         #help="..."：提示信息，调用 python script.py -h 时会显示；
    # parser.add_argument("--out_csv"    , '-o', type=str, help="The path of the output csv file", default="./results.csv")
    #                                     #添加一个可选参数 --out_csv 或 -o，类型为字符串，默认值为 ./results.csv。
    #                                         #用户可以选择将结果输出到指定路径的 CSV 文件中。如果不写，默认输出到当前目录下的 results.csv 文件。
    # parser.add_argument("--weights_dir", '-w', type=str, help="The directory to the networks weights", default="./weights")
    #                                     #指定模型权重所在的目录（默认是 ./weights）。
    #                                         #why?模型推理/评分时需要加载权重文件。默认路径让用户在大多数情况下无需额外输入。
    # parser.add_argument("--models"     , '-m', type=str, help="List of models to test", default='ours,ours-sync')
    #                                     #指定要使用的模型列表，格式为逗号分隔的字符串。
    #                                         #程序支持多个模型，可以通过字符串分割后逐个处理。默认启用两个模型 ours 和 ours-sync。
    # parser.add_argument("--fusion"     , '-f', type=str, help="Fusion function", default=None)
    #                                     #指定融合方法（比如对多个模型输出做加权平均、最大值融合等），默认为 None。
    #                                         #模型可能产生多个评分或判断，有时需要融合成一个最终结果；不指定时就默认不融合。
    # parser.add_argument("--device"     , '-d', type=str, help="Torch device", default='cuda:0')
    #                                     #指定 PyTorch 的设备（如 GPU cuda:0 或 CPU cpu）。
    # args = vars(parser.parse_args())    #首先括号里面parser.parse_args()：解析命令行输入，返回一个包含所有参数的 Namespace 对象；
                                        #var()函数的作用：vars(...)：将 Namespace 对象转换为字典形式，方便通过键访问参数值，如：args["in_csv"]。
    #解释一下加了vars()和不加vars的区别：
    # 从这里开始
    args = {
        "in_csv": r"D:\Model\AlignedForensics-master\SingleImage\DALLE2\fake2\fake2.csv",
        "out_csv": r"D:\Model\AlignedForensics-master\SingleImage\DALLE2\fake2\fake2_score.csv",
        "device": "cuda:0",
        "weights_dir": "D:/Model/AlignedForensics-master_1/test_code/weights",
        "models": "ours,ours-sync",
        "fusion": None
    }

    # 如果 models 是字符串（非列表），转为列表
    if isinstance(args['models'], str):
        args['models'] = args['models'].split(',')

    # 到这里结束都是新加进去的

    # if args['models'] is None:
    #     args['models'] = os.listdir(args['weights_dir'])
    # else:
    #     args['models'] = args['models'].split(',')
    
    table = runnig_tests(args['in_csv'], args['weights_dir'], args['models'], args['device'])
    
    output_csv = args['out_csv']
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    table.to_csv(output_csv, index=False)  # save the results as csv file









"""
解释163行：
    作用：检查命令行参数中是否指定了 --models。
    背景：之前的 argparse 设置了默认值是 'ours,ours-sync'，所以实际上只有在你改了默认值为 None 才会进入这个分支。
    目的：如果用户没有手动传入模型名，那么就自动从权重目录里列出所有文件作为模型名（假设每个文件代表一个模型）。
解释164：
    作用：从权重目录中获取所有文件（模型权重），作为要测试的模型列表。
    说明：os.listdir(path) 返回目录下的所有文件/子文件夹名列表；
解释166行：
    作用：将命令行传入的模型名称字符串分割成列表。
    例子：
        如果你运行命令是 --models ours,ours-sync；
        这一行会将 'ours,ours-sync' 变为 ['ours', 'ours-sync']。
    目的：将逗号分隔的字符串变成 Python 列表，以便后面循环调用或遍历。
解释168：
    作用：调用主功能函数 runnig_tests()，对输入的图像进行模型推理/评估。
    返回值：一个结果表格，应该是 Pandas 的 DataFrame 类型。
解释170：
    作用：取出输出结果 CSV 文件路径，便于下面使用。
    目的：让变量更清晰，不直接操作 args['out_csv']。
解释171：
    作用：确保输出文件的目录存在，如果不存在就创建它。
    详细解释：
        os.path.abspath(output_csv)：获取绝对路径；
        os.path.dirname(...)：获取路径的目录部分；
        os.makedirs(..., exist_ok=True)：递归创建目录，exist_ok=True 表示“如果目录已经存在就不报错”。
    目的：防止写文件时因路径不存在而报错。
解释172：
    作用：将 DataFrame（模型测试结果）保存成 CSV 文件。
    参数说明：
        output_csv：保存路径；
        index=False：不保存行索引，只保存内容。
    目的：保存模型评估结果，便于后续分析或可视化。
"""
