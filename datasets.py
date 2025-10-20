import random
import torch
import h5py
import numpy as np
import json
import math
from tqdm import tqdm

from torch.utils.data import Sampler
from collections import defaultdict
from torch.utils.data import BatchSampler

from bbox_helper import get_loc_label, get_ctr_label
from vsumm_helper import get_keyshot_summ

from torch.nn import functional as F


class VideoSummDataset(object):
    def __init__(self, keys, args=None):
        self.keys = keys
        self.video_dict = h5py.File('{}/{}/feature/eccv16_dataset_{}_google_pool5.h5'.format(args.data_root, args.dataset, args.dataset.lower()), 'r')

        text_feature_path = '{}/{}/feature/text_roberta.npy'.format(args.data_root, args.dataset)
        text_feature_dict = np.load(text_feature_path, allow_pickle=True).item()
        video_id_list = text_feature_dict.keys()

        self.text_dict = {}
        for video_id in video_id_list:
            self.text_dict[video_id] = torch.from_numpy(text_feature_dict[video_id]).to(torch.float32)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        video_name = key.split('/')[-1]
        video_file = self.video_dict[video_name]
        video = torch.from_numpy(video_file['features'][...].astype(np.float32))    # torch.Size([645, 1024])
        text = self.text_dict[video_name] # torch.Size([41, 768])
        gtscore = video_file['gtscore'][...].astype(np.float32)     # [T]
        change_points = video_file['change_points'][...].astype(np.int32)   # [S, 2], S: number of segments, each row stores indices of a segment
        n_frames = video_file['n_frames'][...].astype(np.int32)     # [N], N: number of frames, N = T * 15
        n_frame_per_seg = video_file['n_frame_per_seg'][...].astype(np.int32)   # [S], indicates number of frames in each segment
        picks = video_file['picks'][...].astype(np.int32)   # [T], posotions of subsampled frames in original video
        classes = video_file['classes'][...].astype(np.int32)  # 新增类别
        user_summary = np.zeros(0, dtype=np.float32)
        if 'user_summary' in video_file:
            user_summary = video_file['user_summary'][...].astype(np.float32)
        gtscore -= gtscore.min()
        gtscore /= gtscore.max()

        keyshot_summ, gtscore_upsampled = get_keyshot_summ(gtscore, change_points, n_frames, n_frame_per_seg, picks)
        target = keyshot_summ[::15]
        video_cls_label = target
        video_loc_label = get_loc_label(target)
        video_ctr_label = get_ctr_label(target, video_loc_label)
        video_cls_label = torch.from_numpy(video_cls_label)
        video_loc_label = torch.from_numpy(video_loc_label)
        video_ctr_label = torch.from_numpy(video_ctr_label)

        # 视频帧的分类标签（video_cls_label）映射到文本句子的分类标签（text_cls_label）
        # 确保每个句子根据其对应的视频帧区域的标签来确定其自身的标签。
        num_frame = video.shape[0]
        num_sentence = text.shape[0]    # 对应/data/caption每个txt文件中的句子数，这两个num数量相除之后的frame_sentence_ratio是个定值
        frame_sentence_ratio = int(math.ceil(num_frame / num_sentence))
        text_cls_label = np.zeros((num_sentence), dtype=bool)
        for j in range(num_sentence):
            start_frame = j * frame_sentence_ratio
            end_frame = min((j + 1) * frame_sentence_ratio, num_frame)
            if video_cls_label[start_frame: end_frame].any():
                text_cls_label[j] = True

        text_loc_label = get_loc_label(text_cls_label)
        text_ctr_label = get_ctr_label(text_cls_label, text_loc_label)

        text_cls_label = torch.from_numpy(text_cls_label)
        text_loc_label = torch.from_numpy(text_loc_label)
        text_ctr_label = torch.from_numpy(text_ctr_label)
        mask_video = torch.ones(num_frame, dtype=torch.long)

        ratio = 0.15

        support_candidate = []
        for single_video_name in self.video_dict:
            support_video_file = self.video_dict[single_video_name]
            support_classes = support_video_file['classes'][...].astype(np.int32)
            if single_video_name != video_name and support_classes == classes:
                support_candidate.append(support_video_file)
        random_index = random.randint(0, len(support_candidate) - 1)
        random_element = support_candidate[random_index]
        supp_video = torch.from_numpy(random_element['features'][...].astype(np.float32))
        supp_text = self.text_dict[random_element.name.replace('/', '')]
        supp_gtscore = random_element['gtscore'][...].astype(np.float32)
        supp_cps = random_element['change_points'][...].astype(np.int32)
        supp_n_frames = random_element['n_frames'][...].astype(np.int32)
        supp_nfps = random_element['n_frame_per_seg'][...].astype(np.int32)
        supp_picks = random_element['picks'][...].astype(np.int32)
        supp_gtscore -= supp_gtscore.min()
        supp_gtscore /= supp_gtscore.max()

        support_keyshot_summ, _ = get_keyshot_summ(supp_gtscore, supp_cps, supp_n_frames, supp_nfps, supp_picks)
        support_target = torch.from_numpy(support_keyshot_summ[::15])
        selected_indices = torch.where(support_target == 1)[0]
        supp_num_frame = supp_video.shape[0]
        if selected_indices[selected_indices.shape[0] - 1] != supp_num_frame - 1:
            selected_indices = torch.cat((selected_indices, torch.tensor([supp_num_frame - 1])))
        supp_text = supp_text[selected_indices]
        supp_num_text = supp_text.shape[0]
        supp_video_to_text_mask = torch.zeros((supp_num_frame, supp_num_text), dtype=torch.long)
        supp_text_to_video_mask = torch.zeros((supp_num_text, supp_num_frame), dtype=torch.long)

        start = 0
        for j in range(supp_num_text):
            start_frame = start
            end_frame = selected_indices[j]
            if start_frame != end_frame:
                start_frame = end_frame
                frame_similarity = 1.0
                tmp_current_frame = supp_video[end_frame, :]
                while frame_similarity >= 0.95:
                    start_frame = start_frame - 1
                    if start_frame < 0:
                        break
                    tmp_forward_frame = supp_video[start_frame, :]
                    frame_similarity = F.cosine_similarity(tmp_current_frame, tmp_forward_frame, dim=0).item()
                start_frame = start_frame + 1
                if end_frame - start_frame == 0:
                    supp_video_to_text_mask[end_frame, j] = 1
                    supp_text_to_video_mask[j, end_frame] = 1
                else:
                    supp_video_to_text_mask[start_frame: end_frame + 1, j] = 1
                    supp_text_to_video_mask[j, start_frame: end_frame + 1] = 1
            else:
                supp_video_to_text_mask[start_frame, j] = 1
                supp_text_to_video_mask[j, start_frame] = 1
            start = end_frame + 1
        # 单个模态内部全1
        supp_video_mask = torch.ones(supp_num_frame, dtype=torch.long)
        supp_text_mask = torch.ones(supp_num_text, dtype=torch.long)
        return video_name, classes, video, text, mask_video, video_cls_label, video_loc_label, video_ctr_label, text_cls_label, \
               text_loc_label, text_ctr_label, user_summary, n_frames, ratio, n_frame_per_seg, picks, change_points, \
               supp_video, supp_text, selected_indices, supp_video_mask, supp_text_mask, supp_video_to_text_mask, supp_text_to_video_mask


def worker_init_fn(worker_id):
    """
    Re-seed each worker process to preserve reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return


def my_collate_fn(batch):
    batched_output_list = []
    for i in range(len(batch[0])):
        batched_output = [item[i] for item in batch]
        batched_output_list.append(batched_output)
    return batched_output_list


# 新增
class BalancedSampler(Sampler):
    def __init__(self, dataset, num_samples_per_class=2):
        self.dataset = dataset
        self.num_samples_per_class = num_samples_per_class

        # 将数据按类别分组
        self.class_indices = defaultdict(list)
        for idx, (_, class_label, *_) in enumerate(self.dataset):
            self.class_indices[int(class_label)].append(idx)

    def __iter__(self):
        indices = []
        for class_label, class_indices in self.class_indices.items():
            if len(class_indices) >= self.num_samples_per_class:
                selected_indices = random.sample(class_indices, self.num_samples_per_class)
            else:
                selected_indices = random.choices(class_indices, k=self.num_samples_per_class)
            indices.extend(selected_indices)
        return iter(indices)

    def __len__(self):
        return len(self.dataset)


class CustomBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        return len(self.sampler) // self.batch_size
# 新增 end
