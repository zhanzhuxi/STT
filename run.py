# -*- coding: utf-8 -*-
import datetime
import linecache
import sys
import os
import time
# from send_email import EmailOP

# 服务器环境这句会报错
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_cmd_results(dst, splits, similarity, lambda_contrastive_inter, lambda_inter_sample_contrastive_loss):
    os.system("python train.py --dataset " + dst + " --splits " + splits + " --similarity " + str(similarity) +
              " --lambda_contrastive_inter " + str(lambda_contrastive_inter) +
              " --lambda_inter_sample_contrastive_loss " + str(lambda_inter_sample_contrastive_loss))


run_cmd_results("TVSum", "full_cross_domain_tvsum_few_shot.yml", 0.5, 0.1, 1)
run_cmd_results("SumMe", "full_cross_domain_summe_few_shot.yml", 0.5, 1, 1)
run_cmd_results("TVSum", "several_cross_domain_tvsum_few_shot.yml", 0.5, 0.1, 1)
run_cmd_results("SumMe", "several_cross_domain_summe_few_shot.yml", 0.5, 1, 1)
