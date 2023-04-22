from evaluation import Evaluator
import torch


def train_epoch():
    pass

# validation 流程
def validate_epoch(args, val_loader, model, evaluator):
    model.eval
    evaluator.reset()
    with torch.no_grad():
        for (inputs, labels) in val_loader:
            preds = model(input) 
            evaluator.process(preds, labels)  # 把一个 batch 的结果更新到 evaluator 里面

    return evaluator


def main(args):

    # 加载配置参数
    # 加载数据
    # 加载模型

    # 然后确定总样本数和类别数
    num_validation_samples = 0
    num_classes = 0

    # 在进入 validation 流程前创建 evaluator 可以是全局对象或在主函数中创建并作为参数传入val流程
    evaluator = Evaluator(num_validation_samples, num_classes)

    for epoch in range(args.nepochs):

        train_epoch(...)

        if (epoch+1) % 5 == 0: 
            evaluator = validate_epoch(evaluator, ...) 
            metrics = evaluator.evaluate()  # 跑完一次validate_epoch就可以算一次map
            for key in metrics.keys():
                print(f"{key}: {round(metrics[key] * 100, 2)}") # 输出 map: 百分比数值