import os
import logging
import random
import numpy as np
import torch
import torch.nn.functional as F
from tabulate import tabulate
from scipy import stats


class AverageMeter(object):
    def __init__(self):
        self.value = 0
        self.value_avg = 0
        self.value_sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.value_avg = 0
        self.value_sum = 0
        self.count = 0

    def update(self, value, count):
        self.value = value
        self.value_sum += value * count
        self.count += count
        self.value_avg = self.value_sum / self.count


def ConfigLogging(file_path):
    # 创建一个 logger
    logger = logging.getLogger("save_option_results")
    logger.setLevel(logging.DEBUG)

    # 创建一个handler，用于写入日志文件
    fh = logging.FileHandler(filename=file_path, encoding='utf8')
    fh.setLevel(logging.DEBUG)

    # 再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # 定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_model(save_path, result, modality, model):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file_path = os.path.join(
        save_path,
        'MOSEI_{}_MAE-{}_Corr-{}.pth'.format(
            modality,
            result["MAE"],
            result["Corr"]
        )
    )
    torch.save(model.state_dict(), save_file_path)


def save_print_results(opt, logger, train_re, valid_re, test_re):
    if opt.datasetName in ['mosi', 'mosei']:
        results = [
            ["Train", train_re["MAE"], train_re["Corr"], train_re["Mult_acc_7"], train_re["Has0_acc_2"], train_re["Non0_acc_2"], train_re["Has0_F1_score"], train_re["Non0_F1_score"]],
            ["Valid", valid_re["MAE"], valid_re["Corr"], valid_re["Mult_acc_7"], valid_re["Has0_acc_2"], valid_re["Non0_acc_2"], valid_re["Has0_F1_score"], valid_re["Non0_F1_score"]],
            ["Test", test_re["MAE"], test_re["Corr"], test_re["Mult_acc_7"], test_re["Has0_acc_2"], test_re["Non0_acc_2"], test_re["Has0_F1_score"], test_re["Non0_F1_score"]]
        ]
        headers = ["Phase", "MAE", "Corr", "Acc-7", "Acc-2", "Acc-2-N0", "F1", "F1-N0"]

        table = '\n' + tabulate(results, headers, tablefmt="grid") + '\n'
        logger.info(table.replace('\n', '\n\n'))
    else:
        results = [
            ["Train", train_re["MAE"], train_re["Corr"], train_re["Mult_acc_2"], train_re["Mult_acc_3"], train_re["Mult_acc_5"], train_re["F1_score"]],
            ["Valid", valid_re["MAE"], valid_re["Corr"], valid_re["Mult_acc_2"], valid_re["Mult_acc_3"], valid_re["Mult_acc_5"], valid_re["F1_score"]],
            ["Test", test_re["MAE"], test_re["Corr"], test_re["Mult_acc_2"], test_re["Mult_acc_3"], test_re["Mult_acc_5"], test_re["F1_score"]]
        ]
        headers = ["Phase", "MAE", "Corr", "Acc-2", "Acc-3", "Acc-5", "F1"]

        table = '\n' + tabulate(results, headers, tablefmt="grid") + '\n'
        if logger is not None:
            logger.info(table.replace('\n', '\n\n'))
        else:
            print(table)


def calculate_ratio_senti(uni_senti, multi_senti, k=2.):
    ratio = {}
    for m in ['T', 'V', 'A']:
        uni_senti[m] = torch.exp(-1 * k * torch.pow(torch.abs(uni_senti[m] - multi_senti), 2))

    # 进行归一化
    for m in ['T', 'V', 'A']:
        ratio[m] = uni_senti[m] / (uni_senti['T'] + uni_senti['V'] + uni_senti['A'])
        ratio[m] = ratio[m].unsqueeze(-1)

    return ratio


def calculate_u_test(pred, label):
    pred, label = pred.squeeze().numpy(), label.squeeze().numpy()
    label_mean = np.mean(label)
    alpha = 0.05

    pred_mean = np.mean(pred)
    pred_std = np.std(pred, ddof=1)
    label_std = np.std(label, ddof=1)
    # standard_error = pred_std / np.sqrt(len(pred))
    standard_error = np.sqrt(label_std / len(label) + pred_std / len(pred))

    Z = (label_mean - pred_mean) / standard_error
    critical_value = stats.norm.ppf(1 - alpha)
    if Z >= critical_value:
        print("拒绝原假设，接受备择假设")
    else:
        print("无法拒绝原假设")
