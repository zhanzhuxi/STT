import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from scipy import ndimage
from bbox_helper import offset2bbox


torch.autograd.set_detect_anomaly(True)


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 dims,
                 k_dims=None,
                 v_dims=None,
                 h_dims=None,
                 o_dims=None,
                 heads=8,
                 p=0.1,
                 bias=True):
        super(MultiHeadAttention, self).__init__()

        self._q_dims = dims
        self._k_dims = k_dims or dims
        self._v_dims = v_dims or dims
        self._h_dims = h_dims or dims
        self._o_dims = o_dims or dims
        self._heads = heads
        self._p = p
        self._bias = bias
        self._head_dims = self._h_dims // heads

        self.q = nn.Linear(self._q_dims, self._h_dims, bias=bias)
        self.k = nn.Linear(self._k_dims, self._h_dims, bias=bias)
        self.v = nn.Linear(self._v_dims, self._h_dims, bias=bias)
        self.m = nn.Linear(self._h_dims, self._o_dims, bias=bias)

        self.drop1 = nn.Dropout(p)
        self.drop2 = nn.Dropout(p)

        self.reset_parameters()

    def __repr__(self):
        return ('{}(q_dims={}, k_dims={}, v_dims={}, h_dims={}, o_dims={}, '
                'heads={}, p={}, bias={})'.format(self.__class__.__name__,
                                                  self._q_dims, self._k_dims,
                                                  self._v_dims, self._h_dims,
                                                  self._o_dims, self._heads,
                                                  self._p, self._bias))

    def reset_parameters(self):
        for m in (self.q, self.k, self.v, self.m):
            nn.init.xavier_normal_(m.weight, gain=1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, q, k=None, v=None, mask=None):
        v = v if torch.is_tensor(v) else k if torch.is_tensor(k) else q
        k = k if torch.is_tensor(k) else q

        q = self.q(q).transpose(0, 1).contiguous()
        k = self.k(k).transpose(0, 1).contiguous()
        v = self.v(v).transpose(0, 1).contiguous()

        b = q.size(1) * self._heads

        q = q.view(-1, b, self._head_dims).transpose(0, 1)
        k = k.view(-1, b, self._head_dims).transpose(0, 1)
        v = v.view(-1, b, self._head_dims).transpose(0, 1)

        att = torch.bmm(q, k.transpose(1, 2)) / self._head_dims**0.5

        if mask is not None:
            mask = torch.where(mask > 0, .0, float('-inf'))
            mask = mask.repeat_interleave(self._heads, dim=0)
            att += mask

        att = att.softmax(-1)

        if self.drop1 is not None:
            att = self.drop1(att)

        m = torch.bmm(att, v).transpose(0, 1).contiguous()
        m = m.view(m.size(0), -1, self._h_dims).transpose(0, 1)
        m = self.m(m)

        if self.drop2 is not None:
            m = self.drop2(m)

        return m


class FFN(nn.Module):
    def __init__(self, num_input, p=0.1, ratio=4):
        super().__init__()
        self.fc1 = nn.Linear(num_input, num_input * ratio)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(p)
        self.fc2 = nn.Linear(num_input * ratio, num_input)
        self.drop2 = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MultiWayTransformer(nn.Module):
    def __init__(self, num_hidden, dropout_attn=0.1):
        super().__init__()
        self.norm1_fused = nn.LayerNorm(num_hidden)
        self.attn_fusion = MultiHeadAttention(num_hidden, p=dropout_attn)

        self.norm2_video = nn.LayerNorm(num_hidden)
        self.ffn_video = FFN(num_hidden, p=dropout_attn, ratio=4)

        self.norm2_text = nn.LayerNorm(num_hidden)
        self.ffn_text = FFN(num_hidden, p=dropout_attn, ratio=4)
    
    def forward(self, fused, mask_fused, N_video, N_text):
        residual = fused

        fused = self.norm1_fused(fused)
        fused = self.attn_fusion(fused, fused, fused, mask=mask_fused)
        residual = residual + fused

        residual_video, residual_text = torch.split(residual, [N_video, N_text], dim=1)

        video = self.norm2_video(residual_video)
        video = self.ffn_video(video)
        residual_video = residual_video + video

        text = self.norm2_text(residual_text)
        text = self.ffn_text(text)
        residual_text = residual_text + text

        return residual_video, residual_text


# For SumMe/TVSum datasets
class Model_VideoSumm(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        num_input_video = args.num_input_video  # 视频帧维度1024
        num_input_text = args.num_input_text    # 文本维度
        num_hidden = args.num_hidden            # 隐藏层大小

        self.ratio = args.ratio                 # 控制难负样本的数量
        self.similarity = args.similarity            # 控制支持集和查询集之间相似性阈值

        # 下面的FC层用于将视频和文本的维度对齐
        # nn.Sequential()是一个序列容器，用于搭建神经网络的模块，按照被传入构造器的顺序添加到nn.Sequential()容器中。
        # 构造了一个序列，其中包含一个线性层和一个dropout层
        self.proj_fc_video = nn.Sequential(
                                nn.Linear(num_input_video, num_hidden, bias=True),
                                nn.Dropout(args.dropout_video),
                            )
        self.proj_fc_text = nn.Sequential(
                                nn.Linear(num_input_text, num_hidden, bias=True),
                                nn.Dropout(args.dropout_text),
                            )
        self.proj_supp_fc_video = nn.Sequential(
            nn.Linear(num_input_video, num_hidden, bias=True),
            nn.Dropout(args.dropout_video),
        )
        self.proj_supp_fc_text = nn.Sequential(
            nn.Linear(num_input_text, num_hidden, bias=True),
            nn.Dropout(args.dropout_text),
        )

        # 位置嵌入
        self.pos_embed_video = nn.Parameter(torch.zeros(1, 5000, num_hidden))
        self.pos_embed_text = nn.Parameter(torch.zeros(1, 5000, num_hidden))
        # 片段嵌入
        self.pos_embed_segment = nn.Parameter(torch.zeros(1, 5000, num_hidden))
        # 类别嵌入
        self.type_video = nn.Parameter(torch.zeros(1, 1, num_hidden))
        self.type_text = nn.Parameter(torch.zeros(1, 1, num_hidden))
        # token信息
        self.cls_token_video = nn.Parameter(torch.zeros(1, 1, num_hidden))
        self.cls_token_text = nn.Parameter(torch.zeros(1, 1, num_hidden))
        self.cls_token_video_supp = nn.Parameter(torch.zeros(1, 1, num_hidden))
        self.cls_token_text_supp = nn.Parameter(torch.zeros(1, 1, num_hidden))

        # 掩码
        self.cls_mask_video = torch.ones([1, 1])
        self.cls_mask_text = torch.ones([1, 1])
        self.cls_mask_video_supp = torch.ones([1, 1])
        self.cls_mask_text_supp = torch.ones([1, 1])

        self.multiway_list = nn.ModuleList([MultiWayTransformer(num_hidden, dropout_attn=args.dropout_attn)] * args.num_layers)

        # 归一化
        self.norm_video = nn.LayerNorm(num_hidden)
        self.norm_text = nn.LayerNorm(num_hidden)

        self.fc_supp_video = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(True),
            nn.Dropout(args.dropout_fc),
            nn.LayerNorm(num_hidden),
        )
        self.fc_video = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(True),
            nn.Dropout(args.dropout_fc),
            nn.LayerNorm(num_hidden),
        )
        # DSNet anchor-free所用沿袭下来的
        self.fc_video_cls = nn.Linear(num_hidden, 1)
        self.fc_video_loc = nn.Linear(num_hidden, 2)
        self.fc_video_ctr = nn.Linear(num_hidden, 1)

        # 文本专家
        self.fc_supp_text = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(True),
            nn.Dropout(args.dropout_fc),
            nn.LayerNorm(num_hidden),
        )
        self.fc_text = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(True),
            nn.Dropout(args.dropout_fc),
            nn.LayerNorm(num_hidden),
        )
        self.fc_text_cls = nn.Linear(num_hidden, 1)
        self.fc_text_loc = nn.Linear(num_hidden, 2)
        self.fc_text_ctr = nn.Linear(num_hidden, 1)

        # 抽象摘要专家
        self.abstract_summary = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(True),
            nn.Dropout(args.dropout_fc),
            nn.LayerNorm(num_hidden),
        )

        self.num_layers = args.num_layers

        # 正态分布则可视为不进行任何截断的截断正态分布，也即自变量的取值为负无穷到正无穷。
        # nn.init.trunc_normal_() 截断正态分布可以限制变量的取值范围
        nn.init.trunc_normal_(self.pos_embed_video, std=.02)
        nn.init.trunc_normal_(self.pos_embed_text, std=.02)
        nn.init.trunc_normal_(self.pos_embed_segment, std=.02)
        nn.init.trunc_normal_(self.type_video, std=.02)
        nn.init.trunc_normal_(self.type_text, std=.02)
        nn.init.trunc_normal_(self.cls_token_video, std=.02)
        nn.init.trunc_normal_(self.cls_token_text, std=.02)
        nn.init.trunc_normal_(self.cls_token_video_supp, std=.02)
        nn.init.trunc_normal_(self.cls_token_text_supp, std=.02)

        self.apply(self._init_weights)

    # 初始化权重
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # 构造双重对比损失需要的条件
    # score：预测的分数
    # embedding：模型中最后预测前得到的特征video/text, 不带CLS tokens头
    # mask：掩码——同模态全1，跨模态对应起来的为1否则为0
    # label；标签
    def select_contrastive_embedding(self, score, embedding, mask, label):
        B = score.shape[0]  # 批大小

        key_embedding_list = []
        nonkey_embedding_list = []
        for i in range(B):
            length = torch.sum(mask[i].to(torch.long))
            key_embedding_num = max(1, length // self.ratio)
            nonkey_embedding_num = max(1, length // self.ratio)

            # 利用人工标签，对模型计算得到的特征（最终预测之前）获取应为true标记的特征
            key_embedding_index = label[i].to(torch.bool)
            key_embedding = embedding[i, key_embedding_index]

            # iterations=4：指定膨胀操作的迭代次数。每次迭代都会扩展对象边界一个像素，因此迭代四次会扩展四个像素
            # 如此，获得图3(a)中Expanded Key-frame Label
            key_embedding_index_expand = ndimage.binary_dilation(label[i].cpu().detach().numpy(), iterations=4).astype(np.int32)
            key_embedding_index_expand = torch.from_numpy(key_embedding_index_expand)

            # 获取图3(a)中的Top-k Predictions
            score_i = score[i, :length]
            score_i = F.softmax(score_i, dim=-1)
            _, idx_DESC = score_i.sort(descending=True)

            # 获取图3(a)中的Masked Top-k Predictions，即选择那些不在扩展关键嵌入索引内的索引，作为非关键嵌入索引
            non_key_embedding_index = []
            for j in range(idx_DESC.shape[0]):
                if key_embedding_index_expand[idx_DESC[j]] == 0:
                    non_key_embedding_index.append(idx_DESC[j].item())
                if len(non_key_embedding_index) >= nonkey_embedding_num:    # 控制难负样本不超过8个
                    break

            nonkey_embedding = embedding[i, non_key_embedding_index]

            key_embedding_list.append(key_embedding)
            nonkey_embedding_list.append(nonkey_embedding)
        # print(key_embedding_list[1].shape)
        # print(nonkey_embedding_list[1].shape)
        return key_embedding_list, nonkey_embedding_list

    def forward(self, **kwargs):
        # 获取输入参数，下面的6个输入数据经过padding填充了0，与同批的最大帧数保持一致
        video = kwargs['video']             # torch.Size([4, 645, 1024])
        text = kwargs['text']               # torch.Size([4, 41, 768])
        mask_video = kwargs['mask_video']   # torch.Size([4, 645])
        video_label = kwargs['video_label'] # torch.Size([4, 645])
        picks = kwargs['picks_list']
        supp_video = kwargs['supp_video']
        supp_text = kwargs['supp_text']
        supp_video_mask = kwargs['supp_video_mask']
        supp_text_mask = kwargs['supp_text_mask']
        supp_video_to_text_mask = kwargs['supp_video_to_text_mask']
        supp_text_to_video_mask = kwargs['supp_text_to_video_mask']

        B = video.shape[0]  # 第一个维度表示的是batch的数量，即输入四个视频就是4

        # ********************************************************************全连接层调整输入参数的维度至统一
        original_query_video = video
        video = self.proj_fc_video(video)   # torch.Size([4, 645, 128])
        text = self.proj_fc_text(text)      # torch.Size([4, 41, 128])

        # ********************************************************************处理支持集
        supp_video = self.proj_supp_fc_video(supp_video)
        supp_text = self.proj_supp_fc_text(supp_text)
        residual_video = video
        residual_supp_video = supp_video
        residual_supp_text = supp_text

        # 为支持集每个视频和文本加上一个 CLS token头
        supp_video = torch.cat([self.cls_token_video_supp.expand(B, -1, -1), supp_video], dim=1)  # torch.Size([4, 646, 128])
        supp_text = torch.cat([self.cls_token_text_supp.expand(B, -1, -1), supp_text], dim=1)  # torch.Size([4, 42, 128])
        # 掩码也加上cls_tokens拼接
        supp_video_mask = torch.cat([self.cls_mask_video_supp.expand(B, -1).to(supp_video_mask), supp_video_mask], dim=1)
        supp_text_mask = torch.cat([self.cls_mask_text_supp.expand(B, -1).to(supp_text_mask), supp_text_mask], dim=1)
        # 支持集的特征有做CLS_tokens拼接
        B, N_supp_video, C = supp_video.shape
        B, N_supp_text, C = supp_text.shape
        supp_video = supp_video + self.pos_embed_video[:, :N_supp_video, :] + self.type_video
        # 加入片段嵌入，更明确地在数据特征中界定片段划分
        supp_text = supp_text + self.pos_embed_text[:, :N_supp_text, :] + self.type_text + self.pos_embed_segment[:, :N_supp_text, :]

        # 利用掩码自注意力处理支持集视频和文本之间的GT对立关系，获取到text模态的抽象摘要特征
        supp_mask_fused = torch.zeros((B, N_supp_video + N_supp_text, N_supp_video + N_supp_text), dtype=torch.long).to(supp_video_mask)
        for i in range(B):
            # 处理单个模态内部的掩码：全填充1即可，即利用mask_video和mask_text填了进来
            supp_mask_fused[i, :N_supp_video, :N_supp_video] = supp_video_mask[i].view(1, N_supp_video).expand(N_supp_video, -1) #[N_video, N_video]
            supp_mask_fused[i, N_supp_video:, N_supp_video:] = supp_text_mask[i].view(1, N_supp_text).expand(N_supp_text, -1) #[N_text, N_text]

            # 使用N_video_valid, N_text_valid的原因是train_videosumm.py中有pad_sequence操作做了0填充，需要使用有效的部分来做input embedding操作
            N_supp_video_valid, N_supp_text_valid = supp_video_to_text_mask[i].shape  # [N_video_valid, N_text_valid]
            # 处理跨模态的掩码，根据视频帧数和文本帧数之间的时间对应关系来设置，同时间范围内的置1，否则置0，利用video_to_text_mask_list和text_to_video_mask_list自带的对应关系填充之
            supp_mask_fused[i, 1:1 + N_supp_video_valid, 1 + N_supp_video:1 + N_supp_video + N_supp_text_valid] = supp_video_to_text_mask[i]
            supp_mask_fused[i, 1 + N_supp_video:1 + N_supp_video + N_supp_text_valid:, 1:1 + N_supp_video_valid] = supp_text_to_video_mask[i]
            # 这里就是利用frame行text列的掩码和片段嵌入做矩阵乘法。掩码的每一行都代表了帧与文本之间的对应关系，因此有对应关系的会被选中
            supp_pos_embed_segment_video = supp_video_to_text_mask[i].to(torch.float32) @ self.pos_embed_segment[0, :N_supp_text_valid, :]
            supp_video[i, 1:1 + N_supp_video_valid, :] = supp_video[i, 1:1 + N_supp_video_valid, :] + supp_pos_embed_segment_video

        supp_fused = torch.cat([supp_video, supp_text], dim=1)  # 到此，完成模型的input构造，即文中3.2节最后的X
        for i in range(self.num_layers):
            supp_video, supp_text = self.multiway_list[i](supp_fused, supp_mask_fused, N_supp_video, N_supp_text)
            supp_fused = torch.cat([supp_video, supp_text], dim=1)
        cls_video_supp, supp_video = torch.split(supp_video, [1, N_supp_video - 1], dim=1)
        cls_text_supp, supp_text = torch.split(supp_text, [1, N_supp_text - 1], dim=1)
        supp_video = self.norm_video(residual_supp_video + supp_video)  # 这里Norm待考虑去掉
        supp_text = self.norm_text(residual_supp_text + supp_text)
        # 专家模型分别对两个模态做处理，这里的专家模型fc_video和fc_text也就是DSNet中的预测部分，所以…………soso
        supp_video = self.fc_supp_video(supp_video)  # [batch_size, n_frames, 128]
        supp_text = self.fc_supp_text(supp_text)

        # *****************************************使用文本-文本方式选择下标
        cosine_eps = 1e-7
        video_to_text_mask_list = []
        text_to_video_mask_list = []
        mask_text_list = []
        text_list = []
        for i in range(B):
            query_valid = picks[i].shape[0]

            # 文本域先验摘要合并
            # *************因为经过位置编码和注意力的学习之后，经过padding的text特征的矩阵实际上已经可以完整表示了，不需要再做切片
            supp_video_query, supp_text_query = supp_video_to_text_mask[i].shape
            tmp_supp_text = supp_text[i].unsqueeze(0).unsqueeze(0)
            # 将经过学习的支持集text得到的类别抽象摘要调整到和查询集text同样的大小
            tmp_supp_text = F.interpolate(tmp_supp_text, size=(supp_text.shape[2], supp_text.shape[2]),
                                          mode='bilinear', align_corners=True)
            tmp_supp_text = tmp_supp_text.squeeze(0)
            tmp_query_text = text[i, : query_valid].unsqueeze(0)
            tmp_supp_norm_text = torch.norm(tmp_supp_text, 2, 2, True)
            tmp_query_norm_text = torch.norm(tmp_query_text, 2, 1, True)  # (input, 范数类型, 维度, 是否保持计算范数后的输出维度)
            similarity_text = torch.bmm(tmp_query_text, tmp_supp_text) / (
                        torch.bmm(tmp_query_norm_text, tmp_supp_norm_text) + cosine_eps)
            similarity_text = similarity_text.squeeze(0)
            prior_text = (similarity_text - similarity_text.min(1)[0].unsqueeze(1)) / (
                    similarity_text.max(1)[0].unsqueeze(1) - similarity_text.min(1)[0].unsqueeze(1) + cosine_eps)
            text_clone = text.clone()
            text_clone[i, : query_valid] = text_clone[i, : query_valid] + prior_text
            text = text_clone
            similarity_text = prior_text.mean(1)

            query_select_indices = torch.where(similarity_text >= self.similarity)[0]
            query_select_indices_number = query_select_indices.shape[0]

            picks[i] = torch.from_numpy(picks[i])
            picks[i] = picks[i].to(supp_video_mask)

            # 这里为了覆盖所有帧，将最后一帧始终选中
            if query_select_indices_number != 0:
                if query_select_indices[query_select_indices.shape[0] - 1] != query_valid - 1:
                    query_select_indices = torch.cat((query_select_indices, torch.tensor([query_valid - 1]).to(query_select_indices.device)))
            else:
                query_select_indices = torch.cat((query_select_indices, torch.tensor([query_valid - 1]).to(query_select_indices.device)))

            query_text = text[i, query_select_indices] 
            text_list.append(query_text)
            query_num_text = query_text.shape[0]

            video_to_text_mask = torch.zeros((query_valid, query_num_text), dtype=torch.long).to(supp_video_mask)
            text_to_video_mask = torch.zeros((query_num_text, query_valid), dtype=torch.long).to(supp_video_mask)
            if query_num_text != 1:
                start = 0
                for j in range(query_num_text):
                    start_frame = start
                    end_frame = query_select_indices[j].cpu()

                    if start_frame != end_frame:
                        video_to_text_mask[start_frame: end_frame + 1, j] = 1  # 切片最后的end_frame不包括，所以+1
                        text_to_video_mask[j, start_frame: end_frame + 1] = 1
                    else:
                        video_to_text_mask[start_frame, j] = 1
                        text_to_video_mask[j, start_frame] = 1
                    start = end_frame + 1
                mask_text = torch.ones(query_num_text, dtype=torch.long)
            else:
                mask_text = torch.zeros(query_num_text, dtype=torch.long)

            video_to_text_mask_list.append(video_to_text_mask)
            text_to_video_mask_list.append(text_to_video_mask)
            mask_text_list.append(mask_text)
        mask_text = pad_sequence(mask_text_list, batch_first=True).to(supp_video_mask)
        text = pad_sequence(text_list, batch_first=True).to(supp_video_mask)
        residual_text = text

        # prepend the [CLSV] and [CLST] tokens to the video and text feature sequences
        # ********************************************************************为每个视频和文本加上一个 CLS token头
        video = torch.cat([cls_video_supp, video], dim=1)  # torch.Size([4, 646, 128])
        text = torch.cat([cls_text_supp, text], dim=1)  # torch.Size([4, 42, 128])
        # 掩码也加上 CLS token头
        mask_video = torch.cat([self.cls_mask_video.expand(B, -1).to(mask_video), mask_video], dim=1) #[B, N_video]torch.Size([4, 646])
        mask_text = torch.cat([self.cls_mask_text.expand(B, -1).to(mask_text), mask_text], dim=1) #[B, N_text]torch.Size([4, 42])

        # ****************************************************************add positional embedding and segment embedding
        B, N_video, C = video.shape
        B, N_text, C = text.shape
        video = video + self.pos_embed_video[:, :N_video, :] + self.type_video      # torch.Size([4, 646, 128])
        text = text + self.pos_embed_text[:, :N_text, :] + self.type_text + self.pos_embed_segment[:, :N_text, :]

        # generate global attention mask with time correspondence
        mask_fused = torch.zeros((B, N_video+N_text, N_video+N_text), dtype=torch.long).to(mask_video) # [B, N_video+N_text, N_video+N_text]
        for i in range(B):
            # 处理单个模态内部的掩码：全填充1即可，即利用mask_video和mask_text填了进来
            mask_fused[i, :N_video, :N_video] = mask_video[i].view(1, N_video).expand(N_video, -1) #[N_video, N_video]
            mask_fused[i, N_video:, N_video:] = mask_text[i].view(1, N_text).expand(N_text, -1) #[N_text, N_text]

            # 使用N_video_valid, N_text_valid的原因是train_videosumm.py中有pad_sequence操作做了0填充，需要使用有效的部分来做input embedding操作
            N_video_valid, N_text_valid = video_to_text_mask_list[i].shape  # [N_video_valid, N_text_valid]
            # 处理跨模态的掩码，根据视频帧数和文本帧数之间的时间对应关系来设置，同时间范围内的置1，否则置0，利用video_to_text_mask_list和text_to_video_mask_list自带的对应关系填充之
            mask_fused[i, 1:1+N_video_valid, 1+N_video:1+N_video+N_text_valid] = video_to_text_mask_list[i]
            mask_fused[i, 1+N_video:1+N_video+N_text_valid:, 1:1+N_video_valid] = text_to_video_mask_list[i]
            pos_embed_segment_video = video_to_text_mask_list[i].to(torch.float32) @ self.pos_embed_segment[0, :N_text_valid, :]
            video[i, 1:1+N_video_valid, :] = video[i, 1:1+N_video_valid, :] + pos_embed_segment_video
            # 切片逗号用以区分维度，冒号用以在维度内划定范围

        # multiway transformer layers
        fused = torch.cat([video, text], dim=1)     # 到此，完成模型的input构造，即文中3.2节最后的X
        for i in range(self.num_layers):
            video, text = self.multiway_list[i](fused, mask_fused, N_video, N_text)
            fused = torch.cat([video, text], dim=1)
        cls_video, video = torch.split(video, [1, N_video-1], dim=1)
        cls_text, text = torch.split(text, [1, N_text-1], dim=1)

        # 预测前的准备：对多路transformer生成的结果做处理，将注意力+到原始特征中
        video = self.norm_video(residual_video + video)
        video = self.fc_video(video)    # [batch_size, n_frames, 128]

        # 开始预测
        pred_video_cls = self.fc_video_cls(video).squeeze(-1)               # [B, N]
        pred_video_loc = self.fc_video_loc(video).exp()                     # [B, N, 2]
        pred_video_ctr = self.fc_video_ctr(video).squeeze(-1).sigmoid()     # [B, N]

        # 对比对内部的键值对用于传递给损失函数计算双重对比损失
        contrastive_pairs = {
            'cls_video': cls_video,
            'cls_text': cls_text,
        }

        return pred_video_cls, pred_video_loc, pred_video_ctr, contrastive_pairs

    def predict(self, **kwargs):
        pred_video_cls, pred_video_loc, pred_video_ctr, contrastive_pairs = self(**kwargs)

        pred_video_cls = pred_video_cls.sigmoid() 
        pred_video_cls *= pred_video_ctr
        pred_video_cls /= pred_video_cls.max() + 1e-8

        pred_video_cls = pred_video_cls.cpu().numpy()
        pred_video_loc = pred_video_loc.cpu().numpy()

        pred_video_bboxes = offset2bbox(pred_video_loc)
        return pred_video_cls, pred_video_bboxes
