import datetime
import logging
import os
import time

from torch.utils.tensorboard import SummaryWriter
# from train_msmo import train_msmo
from train_videosumm import train_videosumm
from config import *
from utils import *

logger = logging.getLogger()


# For SumMe/TVSum datasets
def main_videosumm(args):
    now = datetime.datetime.now()
    log_file = now.strftime('log_%Y-%m-%d-%H-%M-%S.log')
    # init_logger(args.model_dir, args.log_file)
    # 原来参数中的log文件名弃用，代之以加上时间信息的名称
    init_logger(args.model_dir, log_file)
    set_random_seed(args.seed)
    # var() 将参数中的每一个成员使用字典的形式进行返回
    dump_yaml(vars(args), '{}/args.yml'.format(args.model_dir))

    logger.info(vars(args))
    os.makedirs(args.model_dir, exist_ok=True)

    # 在logs/{dadaset}目录下增加tensorboard文件夹作为文件保存目录
    args.writer = SummaryWriter(os.path.join(args.model_dir, 'tensorboard'))

    # 数据集划分yaml文件加载
    split_path = '{}/{}/{}'.format(args.data_root, args.dataset, args.splits)
    split_yaml = load_yaml(split_path)

    f1_results = {}
    stats = AverageMeter('fscore')
    rho_stats = AverageMeter('rho')
    tau_stats = AverageMeter('tau')
    # 将fscore设置为AverageMeter的key, 后面每次update需要验证这个key是否在total和count中已存在。相当于是一个钥匙用来开锁的，如果不设置这个参数就无法更新
    # 调用stats.fscore时会自动调用__getattr__函数计算5个split的最大F1的平均值
    # 总的来说，AverageMeter就是一个用于对视频摘要中的每个split做统计并最终给出均值的工具类

    # enumerate()函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
    for split_idx, split in enumerate(split_yaml):
        logger.info(f'Start training on {split_path}: split {split_idx}')
        max_val_fscore, best_val_epoch, max_train_fscore, max_val_rho_value, max_val_tau_value = train_videosumm(args, split, split_idx)
        stats.update(fscore=max_val_fscore)
        rho_stats.update(rho=max_val_rho_value)
        tau_stats.update(tau=max_val_tau_value)

        f1_results[f'split{split_idx}'] = float(max_val_fscore)

    logger.info(f'Training done on {split_path}.')
    logger.info(f'F1_results: {f1_results}')
    logger.info(f'F1-score: {stats.fscore:.4f}\n\n')
    logger.info(f'spearman_rho: {rho_stats.rho:.4f}\n\n')
    logger.info(f'kendall_tau: {tau_stats.tau:.4f}\n\n')

def format_run_time(run_seconds):
    run_time_results = ""
    if run_seconds >= 86400:
        day = run_seconds // 86400
        run_seconds %= 86400
        run_time_results += str(day) + "d"

    if run_seconds >= 3600:
        hour = run_seconds // 3600
        run_seconds %= 3600
        run_time_results += str(hour) + "h"

    if run_seconds >= 60:
        minutes = run_seconds // 60
        run_seconds %= 60
        run_time_results += str(minutes) + "min"

    if run_seconds >= 0:
        run_time_results += str(run_seconds) + "s"

    return run_time_results


if __name__ == '__main__':
    start_time = int(time.time())
    args = get_arguments()
    if args.dataset in ['TVSum', 'SumMe']:
        main_videosumm(args)
    else:
        raise NotImplementedError

    end_time = int(time.time())
    run_time = end_time - start_time
    logger.info("run time: " + format_run_time(run_time))

