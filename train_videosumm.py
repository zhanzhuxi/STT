import time

import torch.utils.data
from torch.nn.utils.rnn import pad_sequence

# from models_5_equal import *
from models_5 import *
from losses import *
from datasets import *
from utils import *

from bbox_helper import nms
from vsumm_helper import bbox2summary, get_summ_f1score

from scipy.stats import spearmanr, kendalltau, rankdata

logger = logging.getLogger()


def train_videosumm(args, split, split_idx):
    batch_time = AverageMeter('time')
    data_time = AverageMeter('time')

    model = Model_VideoSumm(args=args)
    model = model.to(args.device)
    calc_contrastive_loss = Dual_Contrastive_Loss().to(args.device)
    calc_inter_sample_contrastive_loss = SupConLoss_clear().to(args.device)

    parameters = [p for p in model.parameters() if p.requires_grad] + \
                    [p for p in calc_contrastive_loss.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs('{}/checkpoint'.format(args.model_dir), exist_ok=True)

    max_train_fscore = -1
    max_val_fscore = -1
    max_val_rho_value = -1
    max_val_tau_value = -1
    best_val_epoch = 0

    # model testing, load from checkpoint
    checkpoint_path = None
    if args.checkpoint and args.test:
        checkpoint_path = '{}/model_best_split{}.pt'.format(args.checkpoint, split_idx)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("load checkpoint from {}".format(checkpoint_path))
        model.load_state_dict(checkpoint['model_state_dict'])

    train_set = VideoSummDataset(keys=split['train_keys'], args=args)
    train_sampler = BalancedSampler(train_set, num_samples_per_class=2)
    batch_sampler = CustomBatchSampler(train_sampler, batch_size=4, drop_last=False)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=my_collate_fn
    )

    val_set = VideoSummDataset(keys=split['test_keys'], args=args)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                                drop_last=False, pin_memory=True,
                                                worker_init_fn=worker_init_fn, collate_fn=my_collate_fn)

    if args.test:
        val_fscore = evaluate_videosumm(model, val_loader, args, epoch=0)
        logger.info(f'F-score: {val_fscore:.4f}')
        return val_fscore, best_val_epoch, max_train_fscore, max_val_rho_value, max_val_tau_value


    logger.info('\n' + str(model))

    for epoch in range(args.start_epoch, args.max_epoch):
        model.train()
        stats = AverageMeter('loss', 'cls_loss', 'loc_loss', 'ctr_loss', 'inter_contrastive_loss', 'inter_sample_contrastive_loss')
        end = time.time()
        for k, (video_name, classes, video_list, text_list, mask_video_list,
                video_cls_label_list, video_loc_label_list, video_ctr_label_list,
                text_cls_label_list, text_loc_label_list, text_ctr_label_list,
                user_summary_list, n_frames_list, ratio_list, n_frame_per_seg_list, picks_list, change_points_list,
                supp_video_list, supp_text_list, selected_indices_list, supp_video_mask_list, supp_text_mask_list,
                supp_video_to_text_mask_list, supp_text_to_video_mask_list) in enumerate(train_loader):
            data_time.update(time=time.time() - end)

            batch_size = len(video_list)
            video = pad_sequence(video_list, batch_first=True)
            text = pad_sequence(text_list, batch_first=True)
            mask_video = pad_sequence(mask_video_list, batch_first=True)
            video_cls_label = pad_sequence(video_cls_label_list, batch_first=True)
            video_loc_label = pad_sequence(video_loc_label_list, batch_first=True)
            video_ctr_label = pad_sequence(video_ctr_label_list, batch_first=True)
            supp_video = pad_sequence(supp_video_list, batch_first=True)
            supp_text = pad_sequence(supp_text_list, batch_first=True)
            supp_video_mask = pad_sequence(supp_video_mask_list, batch_first=True)
            supp_text_mask = pad_sequence(supp_text_mask_list, batch_first=True)

            for i in range(len(supp_video_list)):
                selected_indices_list[i] = selected_indices_list[i].to(args.device)
                supp_video_to_text_mask_list[i] = supp_video_to_text_mask_list[i].to(args.device)
                supp_text_to_video_mask_list[i] = supp_text_to_video_mask_list[i].to(args.device)
            video, text = video.to(args.device), text.to(args.device)
            supp_video, supp_text = supp_video.to(args.device), supp_text.to(args.device)
            mask_video = mask_video.to(args.device)
            supp_video_mask = supp_video_mask.to(args.device)
            supp_text_mask = supp_text_mask.to(args.device)
            video_cls_label = video_cls_label.to(args.device) #[B, T]
            video_loc_label = video_loc_label.to(args.device) #[B, T, 2]
            video_ctr_label = video_ctr_label.to(args.device) #[B, T]
            video_pred_cls, video_pred_loc, video_pred_ctr, contrastive_pairs = \
                model(video=video, text=text, mask_video=mask_video,
                      video_label=video_cls_label, picks_list=picks_list,
                      supp_video=supp_video, supp_text=supp_text, selected_indices=selected_indices_list,
                      supp_video_mask=supp_video_mask, supp_text_mask=supp_text_mask,
                      supp_video_to_text_mask=supp_video_to_text_mask_list,
                      supp_text_to_video_mask=supp_text_to_video_mask_list)
            cls_loss = 10 * calc_cls_loss(video_pred_cls, video_cls_label.to(torch.long), mask=mask_video)
            loc_loss = calc_loc_loss(video_pred_loc, video_loc_label, video_cls_label)
            ctr_loss = calc_ctr_loss(video_pred_ctr, video_ctr_label, video_cls_label)
            inter_contrastive_loss = calc_contrastive_loss(contrastive_pairs)
            inter_contrastive_loss = inter_contrastive_loss * args.lambda_contrastive_inter
            inter_sample_contrastive_loss = calc_inter_sample_contrastive_loss(contrastive_pairs['cls_video'], classes)
            inter_sample_contrastive_loss = inter_sample_contrastive_loss * args.lambda_inter_sample_contrastive_loss
            loss = cls_loss + loc_loss + ctr_loss + inter_contrastive_loss + inter_sample_contrastive_loss

            optimizer.zero_grad()   # optimizer.zero_grad() 清空过往梯度
            loss.backward()         # loss.backward() 反向传播，计算当前梯度
            optimizer.step()        # optimizer.step() 根据梯度更新网络参数

            stats.update(loss=loss.item(), cls_loss=cls_loss.item(),
                         loc_loss=loc_loss.item(), ctr_loss=ctr_loss.item(),
                         inter_contrastive_loss=inter_contrastive_loss.item(),
                         inter_sample_contrastive_loss=inter_sample_contrastive_loss)
                         # )
            batch_time.update(time=time.time() - end)
            end = time.time()

            logger.info(f'[Train] Epoch: {epoch+1}/{args.max_epoch} Iter: {k+1} '
                        f'Time: {batch_time.time:.3f} Data: {data_time.time:.3f} '
                        f'Loss: {stats.cls_loss:.4f}/{stats.loc_loss:.4f}/{stats.ctr_loss:.4f}/{stats.inter_contrastive_loss:.4f}/{stats.inter_sample_contrastive_loss:.4f}/{stats.loss:.4f}')

        save_checkpoint = {
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'max_val_fscore': max_val_fscore,
            'max_train_fscore': max_train_fscore,
        }

        if (epoch + 1) % args.eval_freq == 0:
            val_fscore, rho_value, tau_value = evaluate_videosumm(model, val_loader, args, epoch=epoch)
            if max_val_fscore < val_fscore:
                max_val_fscore = val_fscore
                best_val_epoch = epoch + 1

            if max_val_rho_value < rho_value:
                max_val_rho_value = rho_value

            if max_val_tau_value < tau_value:
                max_val_tau_value = tau_value

            logger.info(
                f'[Eval]  Epoch: {epoch + 1}/{args.max_epoch} F-score: {val_fscore:.4f}/{max_val_fscore:.4f} rho: {rho_value:.4f}/{max_val_rho_value:.4f} tau: {tau_value:.4f}/{max_val_tau_value:.4f}\n')

            args.writer.add_scalar(f'Split{split_idx}/Val/max_fscore', max_val_fscore, max_val_rho_value,
                                   max_val_tau_value, epoch + 1)
            args.writer.add_scalar(f'Split{split_idx}/Val/fscore', val_fscore, max_val_rho_value, max_val_tau_value,
                                   epoch + 1)

        # add_scalar() 三个参数：标签名、值、步
        args.writer.add_scalar(f'Split{split_idx}/Train/loss', stats.loss, epoch+1)
        args.writer.add_scalar(f'Split{split_idx}/Train/cls_loss', stats.cls_loss, epoch+1)
        args.writer.add_scalar(f'Split{split_idx}/Train/loc_loss', stats.loc_loss, epoch+1)
        args.writer.add_scalar(f'Split{split_idx}/Train/ctr_loss', stats.ctr_loss, epoch+1)
        args.writer.add_scalar(f'Split{split_idx}/Train/inter_contrastive_loss', stats.inter_contrastive_loss, epoch+1)

    return max_val_fscore, best_val_epoch, max_train_fscore, max_val_rho_value, max_val_tau_value


@torch.no_grad()
def evaluate_videosumm(model, val_loader, args, epoch=None):
    model.eval()
    stats = AverageMeter('fscore')
    rho_stats = AverageMeter('rho')
    tau_stats = AverageMeter('tau')

    data_length = len(val_loader)
    with torch.no_grad():
        for k, (video_name, classes, video_list, text_list, mask_video_list,
                video_cls_label_list, video_loc_label_list, video_ctr_label_list,
                text_cls_label_list, text_loc_label_list, text_ctr_label_list,
                user_summary_list, n_frames_list, ratio_list, n_frame_per_seg_list, picks_list, change_points_list,
                supp_video_list, supp_text_list, selected_indices_list, supp_video_mask_list, supp_text_mask_list,
                supp_video_to_text_mask_list, supp_text_to_video_mask_list) in enumerate(val_loader):

            batch_size = len(video_list)

            video = pad_sequence(video_list, batch_first=True)
            text = pad_sequence(text_list, batch_first=True)

            mask_video = pad_sequence(mask_video_list, batch_first=True)
            video_cls_label = pad_sequence(video_cls_label_list, batch_first=True)
            video_loc_label = pad_sequence(video_loc_label_list, batch_first=True)
            video_ctr_label = pad_sequence(video_ctr_label_list, batch_first=True)

            supp_video = pad_sequence(supp_video_list, batch_first=True)
            supp_text = pad_sequence(supp_text_list, batch_first=True)
            supp_video_mask = pad_sequence(supp_video_mask_list, batch_first=True)
            supp_text_mask = pad_sequence(supp_text_mask_list, batch_first=True)

            for i in range(len(supp_video_list)):
                selected_indices_list[i] = selected_indices_list[i].to(args.device)
                supp_video_to_text_mask_list[i] = supp_video_to_text_mask_list[i].to(args.device)
                supp_text_to_video_mask_list[i] = supp_text_to_video_mask_list[i].to(args.device)

            video, text = video.to(args.device), text.to(args.device)
            supp_video, supp_text = supp_video.to(args.device), supp_text.to(args.device)
            mask_video = mask_video.to(args.device)
            supp_video_mask = supp_video_mask.to(args.device)
            supp_text_mask = supp_text_mask.to(args.device)

            video_cls_label = video_cls_label.to(args.device)   # [B, T]
            video_loc_label = video_loc_label.to(args.device)   # [B, T, 2]
            video_ctr_label = video_ctr_label.to(args.device)   # [B, T]

            pred_cls_batch, pred_bboxes_batch = \
                model.predict(video=video, text=text, mask_video=mask_video,
                              video_label=video_cls_label, picks_list=picks_list,
                              supp_video=supp_video, supp_text=supp_text, selected_indices=selected_indices_list,
                              supp_video_mask=supp_video_mask, supp_text_mask=supp_text_mask,
                              supp_video_to_text_mask=supp_video_to_text_mask_list,
                              supp_text_to_video_mask=supp_text_to_video_mask_list)     # [B, T], [B, T, 2]
            mask_video_bool = mask_video.cpu().numpy().astype(bool)
            for i in range(len(supp_video_list)):
                picks_list[i] = picks_list[i].cpu().numpy()

            for i in range(batch_size):
                video_length = np.sum(mask_video_bool[i])
                pred_cls = pred_cls_batch[i, mask_video_bool[i]]    # [T]
                pred_bboxes = np.clip(pred_bboxes_batch[i, mask_video_bool[i]], 0, video_length).round().astype(np.int32)   # [T, 2]
                pred_cls, pred_bboxes = nms(pred_cls, pred_bboxes, args.nms_thresh)
                pred_summ, pred_summ_upsampled, pred_score, pred_score_upsampled = bbox2summary(
                    video_length, pred_cls, pred_bboxes, change_points_list[i], n_frames_list[i], n_frame_per_seg_list[i], picks_list[i], proportion=ratio_list[i], seg_score_mode='mean')

                eval_metric = 'max' if args.dataset == 'SumMe' else 'avg'
                fscore = get_summ_f1score(pred_summ_upsampled, user_summary_list[i], eval_metric=eval_metric)

                stats.update(fscore=fscore)

                rho_coeff, tau_coeff = [], []
                for annot in range(len(user_summary_list[i])):
                    true_user_score = user_summary_list[i][annot]
                    curr_rho_coeff, _ = spearmanr(pred_score_upsampled, true_user_score)
                    curr_tau_coeff, _ = kendalltau(rankdata(pred_score_upsampled), rankdata(true_user_score))
                    rho_coeff.append(curr_rho_coeff)
                    tau_coeff.append(curr_tau_coeff)

                rho_coeff = np.array(rho_coeff).mean()  # mean over all user annotations
                rho_stats.update(rho=rho_coeff)
                tau_coeff = np.array(tau_coeff).mean()  # mean over all user annotations
                tau_stats.update(tau=tau_coeff)

            if (k + 1) % args.print_freq == 0:
                logger.info(f'[Eval]  Epoch: {epoch+1}/{args.max_epoch} Iter: {k+1}/{data_length} F-score: {stats.fscore:.4f}')
    return stats.fscore, rho_stats.rho, tau_stats.tau




