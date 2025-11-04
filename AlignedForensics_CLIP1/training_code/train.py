import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import tqdm
#from utils import TrainingModel, create_dataloader, EarlyStopping
from training_code.utils import TrainingModel, create_dataloader, EarlyStopping
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
try:
    from tensorboardX import SummaryWriter
except:
    from torch.utils.tensorboard import SummaryWriter
import argparse
import torch
# from utils.training import add_training_arguments
# from utils.dataset import add_dataloader_arguments
# from utils.multimodal_training import MultimodalTrainingModel, add_multimodal_training_arguments
from training_code.utils.training import add_training_arguments
from training_code.utils.dataset import add_dataloader_arguments
from training_code.utils.multimodal_training import MultimodalTrainingModel, add_multimodal_training_arguments


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser = add_training_arguments(parser)         # 添加"trainging.py里面有参数的解释"训练用的参数
    parser = add_dataloader_arguments(parser)       # 添加"dataset里面有参数的解释"数据加载用的参数
    parser = add_multimodal_training_arguments(parser)  # 添加多模态训练参数
    parser.add_argument(
        "--num_epoches", type=int, default=1000, help="# of epoches at starting learning rate"
    )
    parser.add_argument(
        "--earlystop_epoch",
        type=int,
        default=5,
        help="Number of epochs without loss reduction before lowering the learning rate",
    )
    opt = parser.parse_args()
    # 设置随机种子。作用是设置 PyTorch 的随机种子，以保证实验的可重复性。
    # 详细解释：设置了随机种子后，随机数生成器的初始状态被固定了，所以每次生成的“随机数序列”都一模一样。这样就导致所有依赖随机数的操作也**“看起来不再随机”**了。
    torch.manual_seed(opt.seed)
    #Changed from val to valid
    # 加载训练、验证集
    valid_data_loader = create_dataloader(opt, subdir="valid", is_train=False)
    train_data_loader = create_dataloader(opt, subdir="train", is_train=True)
    print()
    print("# validation batches = %d" % len(valid_data_loader))
    print("#   training batches = %d" % len(train_data_loader))
    # 构建模型对象
    if getattr(opt, 'enable_multimodal', False):
        model = MultimodalTrainingModel(opt, subdir=opt.name)  # 多模态训练模型
    else:
        model = TrainingModel(opt, subdir=opt.name)  # 原始训练模型
    # 设置日志记录器
    writer = SummaryWriter(os.path.join(model.save_dir, "logs"))
    writer_loss_steps = len(train_data_loader) // 32# 这是计算一个“写入日志的步频率”（写日志的间隔步数）。len(train_data_loader) 是训练集中有多少个批次（batch），比如训练集有320个batch。除以32后，得到一个较小的数字，代表每训练这么多批次后写一次日志。
    # early stopping 初始化
    early_stopping = None
    start_epoch = model.total_steps // len(train_data_loader)# 假设每个 epoch 有 100 个 batch（len(train_data_loader) == 100），加载时 model.total_steps == 250（说明之前训练了 250 批次）start_epoch = 250 // 100 = 2，说明已经完成了 2 个 epoch，下一步从第 3 个 epoch 开始训练。


    # 主训练循环：for epoch in range(...)
    for epoch in range(start_epoch, opt.num_epoches+1):
        if  epoch > start_epoch:
            # Training
            pbar = tqdm.tqdm(train_data_loader)
            for data in pbar:# 循环遍历训练集每个 batch：
                loss = model.train_on_batch(data).item()# 调用 model.train_on_batch(data) 进行一次训练更新，返回当前 batch 的损失。
                pbar.set_description(f"Train loss: {loss:.4f}")# 更新进度条的显示信息，实时显示当前 batch 的 loss。
                total_steps = model.total_steps
                if total_steps % writer_loss_steps == 0:# 每隔一定步数（writer_loss_steps）记录训练损失到 TensorBoard，方便训练过程监控。
                    writer.add_scalar("train/loss", loss, total_steps)#

            # Save model
            if epoch % 5 == 0:
                model.save_networks(epoch)# 训练完一个 epoch 后保存模型快照，以便后续恢复或评估。

        # Validation
        print("Validation ...", flush=True)
        y_true, y_pred, y_path = model.predict(valid_data_loader)

        acc = balanced_accuracy_score(y_true, y_pred > 0.0)# 计算验证准确率 acc（这里用 balanced accuracy，考虑类别不平衡）。
        lr = model.get_learning_rate()                     # 记录当前学习率和验证准确率到 TensorBoard，方便监控。
        writer.add_scalar("lr", lr, model.total_steps)
        
        writer.add_scalar("valid/accuracy", acc, model.total_steps)
        
        # 多模态组件评估
        if hasattr(model, 'evaluate_multimodal_components'):
            component_metrics = model.evaluate_multimodal_components(valid_data_loader)
            for metric_name, metric_value in component_metrics.items():
                writer.add_scalar(f"valid/{metric_name}", metric_value, model.total_steps)
                print(f"{metric_name}: {metric_value:.4f}")
        
        print("After {} epoches: val acc = {}".format(epoch, acc), flush=True)# 打印验证结果。
        #如果想要看acc的话就看这个函数里面的代码balanced_accuracy_score是什么样的！

        # Early Stopping
        if early_stopping is None:# 首次初始化 early_stopping 对象，传入当前验证准确率作为初始分数，设置耐心（多少个 epoch 验证准确率没提升就触发操作）。
            early_stopping = EarlyStopping(
                init_score=acc, patience=opt.earlystop_epoch,
                delta=0.001, verbose=True,
            )
        else:# 后续每个 epoch 检查验证准确率：
            if early_stopping(acc):# 如果性能提升（early_stopping 返回 True），保存当前最佳模型权重。
                print('Save best model', flush=True)
                model.save_networks('best')
            if early_stopping.early_stop:# 触发早停时：
                cont_train = model.adjust_learning_rate()# 调用 model.adjust_learning_rate() 降低学习率（例如乘以 0.1）。
                if cont_train:# 如果还能继续训练（学习率没低于最小阈值），重置早停计数，继续训练。
                    print("Learning rate dropped by 10, continue training ...", flush=True)
                    early_stopping.reset_counter()
                else:
                    print("Early stopping.", flush=True)# 否则打印"Early stopping"，跳出训练循环，结束训练。
                    
                    # 创建可解释性可视化
                    if hasattr(model, 'create_interpretability_visualization'):
                        print("Creating interpretability visualizations...")
                        model.create_interpretability_visualization(
                            valid_data_loader, 
                            output_dir=os.path.join(model.save_dir, "interpretability")
                        )
                    break
