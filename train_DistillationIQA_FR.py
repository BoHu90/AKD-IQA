import sys
import time

import torch
import os
import random

from torch.optim.lr_scheduler import CosineAnnealingLR

from Util.OutputSaver import Saver
from Util.tools import convert_obj_score
from dataloaders.dataloader_LQ_HQ import DataLoader  # 对FR的训练，只需要使用到LQ和HQ
from models.DistillationIQA import DistillationIQANetT
from FR.option_train_DistillationIQA_FR import set_args, check_args
from scipy import stats
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
img_num = {
    'kadid10k': list(range(0, 10125)),
    'live': list(range(0, 29)),  # ref HR image
    'csiq': list(range(0, 30)),  # ref HR image
    'tid2013': list(range(0, 25)),
    'livec': list(range(0, 1162)),  # no-ref image
    'koniq-10k': list(range(0, 10073)),  # no-ref image
    'bid': list(range(0, 586)),  # no-ref image
}
folder_path = {
    'pipal': '/data/dataset/PIPAL',
    'live': '/data/dataset/LIVE/',
    'csiq': '/data/dataset/CSIQ/',
    'tid2013': '/data/dataset/tid2013/',
    'livec': '/data/dataset/LIVEC/',
    'koniq-10k': '/data/dataset/koniq-10k/',
    'bid': '/data/dataset/BID/',
    'kadid10k': '/data/dataset/kadid10k/'
}


class DistillationFRIQASolver(object):
    def __init__(self, config):
        # 加载配置、设置设备（GPU或CPU）、创建日志文件
        self.config = config
        self.device = torch.device('cuda' if config.gpu_ids is not None else 'cpu')
        # self.txt_log_path = os.path.join(config.log_checkpoint_dir, 'log_origin.txt')
        # with open(self.txt_log_path, "w+") as f:
        #     f.close()

        # 老师模型 model: return encode_diff_inner_feature, encode_lq_inner_feature, pred  # 返回两特征以及分数预测
        # TODO 4f修改1 换模型且无法加载预训练模型
        # TODO 融合模块修改1 换模型
        self.teacherNet = DistillationIQANetT(self_patch_num=config.self_patch_num,
                                              distillation_layer=config.distillation_layer)
        # TODO 取消加载预训练权重，并且改为bs32
        # if config.teacherNet_model_path:
        #     self.teacherNet._load_state_dict(torch.load(config.teacherNet_pretrained_path))
        self.teacherNet = self.teacherNet.to(self.device)
        self.teacherNet.train(True)  # 训练模式

        # lr,opt,loss,epoch
        self.lr = config.lr
        self.lr_ratio = 10
        self.feature_loss_ratio = 0.1
        # 使用不同的学习率对 特征提取器和其他参数 进行更新
        resnet_params = list(map(id, self.teacherNet.feature_extractor.parameters()))
        res_params = filter(lambda p: id(p) not in resnet_params, self.teacherNet.parameters())
        paras = [{'params': res_params, 'lr': self.lr * self.lr_ratio},
                 {'params': self.teacherNet.feature_extractor.parameters(), 'lr': self.lr}
                 ]
        # 学习率\损失函数的定义
        self.optimizer = torch.optim.Adam(paras, weight_decay=config.weight_decay)

        # TODO 更新学习率
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epochs)

        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.sml_loss = torch.nn.SmoothL1Loss()
        self.epochs = config.epochs

        # data
        # 训练数据：kadid10k|tid2013
        config.train_index = img_num[config.train_dataset]
        random.shuffle(config.train_index)
        train_loader = DataLoader(config.train_dataset, folder_path[config.train_dataset], config.train_index,
                                  config.patch_size, config.train_patch_num, batch_size=config.batch_size, istrain=True,
                                  self_patch_num=config.self_patch_num)
        # TODO 测试数据修改1
        # 测试数据： live|csiq|tid2013
        # test_loader_LIVE = DataLoader('live', folder_path['live'], img_num['live'], config.patch_size,
        #                               config.test_patch_num, istrain=False, self_patch_num=config.self_patch_num)
        # test_loader_CSIQ = DataLoader('csiq', folder_path['csiq'], img_num['csiq'], config.patch_size,
        #                               config.test_patch_num, istrain=False, self_patch_num=config.self_patch_num)
        test_loader = DataLoader(config.test_dataset, folder_path[config.test_dataset], img_num[config.test_dataset], config.patch_size,
                                      config.test_patch_num, istrain=False, self_patch_num=config.self_patch_num)
        # test_loader_TID = DataLoader('tid2013', folder_path['tid2013'], img_num['tid2013'], config.patch_size,
        #                              config.test_patch_num, istrain=False, self_patch_num=config.self_patch_num)

        # TODO 测试数据修改2
        self.train_data = train_loader.get_dataloader()
        # self.test_data_LIVE = test_loader_LIVE.get_dataloader()
        # self.test_data_CSIQ = test_loader_CSIQ.get_dataloader()
        self.test_data = test_loader.get_dataloader()
        # self.test_data_TID = test_loader_TID.get_dataloader()

    def train(self):
        # best_srcc_LIVE, best_srcc_CSIQ, best_srcc_TID = 0.0, 0.0, 0.0
        # best_plcc_LIVE, best_plcc_CSIQ, best_plcc_TID = 0.0, 0.0, 0.0
        # best_krcc_LIVE, best_krcc_CSIQ, best_krcc_TID = 0.0, 0.0, 0.0
        best_srcc, best_plcc, best_krcc = 0.0, 0.0, 0.0
        best_srcc_train = 0.0
        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC\tTest_KRCC')
        # NEW
        scaler = torch.cuda.amp.GradScaler()

        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            for LQ_patches, refHQ_patches, label in self.train_data:
                LQ_patches, refHQ_patches, label = LQ_patches.to(self.device), refHQ_patches.to(self.device), label.to(
                    self.device)
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    # 老师模型，输入LQ 和 HQ 图像
                    # 只根据分数标签进行监督，所以只拿预测分数
                    _, _, pred = self.teacherNet(LQ_patches, refHQ_patches)
                    # 使用l1损失
                    pred_scores = pred_scores + pred.cpu().tolist()
                    gt_scores = gt_scores + label.cpu().tolist()
                    loss = self.sml_loss(pred.squeeze(), label.float().detach())

                epoch_loss.append(loss.item())
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
            # TODO 测试数据修改3
            # test_LIVE_srcc, test_LIVE_plcc, test_LIVE_krcc = self.test(self.test_data_LIVE)
            # test_CSIQ_srcc, test_CSIQ_plcc, test_CSIQ_krcc = self.test(self.test_data_CSIQ)
            test_srcc, test_plcc, test_krcc = self.test(self.test_data)
            # test_TID_srcc, test_TID_plcc, test_TID_krcc = self.test(self.test_data_TID)
            if train_srcc > best_srcc_train :
                best_srcc_train = train_srcc
            # if test_LIVE_srcc > best_srcc_LIVE:
            #     torch.save(self.teacherNet.state_dict(),
            #                os.path.join(self.config.model_checkpoint_dir, 'FRIQA_kadid_SwinT_saved_model_bs16.pth'.format(t)))
            #     print("模型被更新:")

            # TODO 测试数据修改4
            # if test_LIVE_srcc + test_LIVE_plcc + test_LIVE_krcc > best_srcc_LIVE + best_plcc_LIVE + best_krcc_LIVE:
            #     best_srcc_LIVE, best_plcc_LIVE, best_krcc_LIVE = test_LIVE_srcc, test_LIVE_plcc, test_LIVE_krcc
            # print('%d:live\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f \n' %
            #       (
            #           t, sum(epoch_loss) / len(epoch_loss), train_srcc, test_LIVE_srcc, test_LIVE_plcc,
            #           test_LIVE_krcc))

            if test_srcc + test_plcc + test_krcc > best_srcc + best_plcc + best_krcc:
                best_srcc, best_plcc, best_krcc = test_srcc, test_plcc, test_krcc
                # TODO 文件名修改1
                torch.save(self.teacherNet.state_dict(),
                           os.path.join(self.config.model_checkpoint_dir, f'FRIQA_kadid_SwinT_saved_model_bs32_36_{self.config.test_dataset}_o.pth'.format(t)))
                print("模型被更新:")
            print('%d:%s\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f \n' %
                  (
                      t, config.test_dataset, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc,
                      test_krcc))
            #
            # if test_TID_srcc + test_TID_plcc + test_TID_krcc > best_srcc_TID + best_plcc_TID + best_krcc_TID:
            #     best_srcc_TID, best_plcc_TID, best_krcc_TID = test_TID_srcc, test_TID_plcc, test_TID_krcc
            # print('%d:tid\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f \n' %
            #       (t, sum(epoch_loss) / len(epoch_loss), train_srcc, test_TID_srcc, test_TID_plcc, test_TID_krcc))

            # torch.save(self.teacherNet.state_dict(),
            #            os.path.join(self.config.model_checkpoint_dir, 'FRIQA_{}_saved_model.pth'.format(t)))

            # # 更新优化器参数
            # self.lr = self.lr / pow(10, (t // self.config.update_opt_epoch))
            # if t > 5:
            #     self.lr_ratio = 1
            # resnet_params = list(map(id, self.teacherNet.feature_extractor.parameters()))
            # rest_params = filter(lambda p: id(p) not in resnet_params, self.teacherNet.parameters())
            # paras = [{'params': rest_params, 'lr': self.lr * self.lr_ratio},
            #          {'params': self.teacherNet.feature_extractor.parameters(), 'lr': self.lr}
            #          ]
            # self.optimizer = torch.optim.Adam(paras, weight_decay=self.config.weight_decay)

            # TODO 更新学习率
            self.scheduler.step()
        # TODO 测试数据修改5
        # print('Best live test SRCC %f, PLCC %f, KRCC %f\n' % (best_srcc_LIVE, best_plcc_LIVE, best_krcc_LIVE))
        print('Best %s test SRCC %f, PLCC %f, KRCC %f\n' % (config.test_dataset, best_srcc, best_plcc, best_krcc))
        # print('Best tid2013 test SRCC %f, PLCC %f, KRCC %f\n' % (best_srcc_TID, best_plcc_TID, best_krcc_TID))

    def test(self, test_data):
        self.teacherNet.train(False)  # 冻结网络
        test_pred_scores, test_gt_scores = [], []
        for LQ_patches, refHQ_patches, label in test_data:
            LQ_patches, refHQ_patches, label = LQ_patches.to(self.device), refHQ_patches.to(self.device), label.to(
                self.device)
            with torch.no_grad():
                _, _, pred = self.teacherNet(LQ_patches, refHQ_patches)
                test_pred_scores.append(float(pred.item()))
                test_gt_scores = test_gt_scores + label.cpu().tolist()
        if self.config.use_fitting_prcc_srcc:
            fitting_pred_scores = convert_obj_score(test_pred_scores, test_gt_scores)
        test_pred_scores = np.mean(np.reshape(np.array(test_pred_scores), (-1, self.config.test_patch_num)), axis=1)
        test_gt_scores = np.mean(np.reshape(np.array(test_gt_scores), (-1, self.config.test_patch_num)), axis=1)
        test_srcc, _ = stats.spearmanr(test_pred_scores, test_gt_scores)
        if self.config.use_fitting_prcc_srcc:
            test_plcc, _ = stats.pearsonr(fitting_pred_scores, test_gt_scores)
        else:
            test_plcc, _ = stats.pearsonr(test_pred_scores, test_gt_scores)
        test_krcc, _ = stats.stats.kendalltau(test_pred_scores, test_gt_scores)
        test_srcc, test_plcc, test_krcc = abs(test_srcc), abs(test_plcc), abs(test_krcc)
        self.teacherNet.train(True)
        return test_srcc, test_plcc, test_krcc


if __name__ == "__main__":
    # 保存控制台输出
    # TODO 文件名修改2
    config = set_args()
    saver = Saver(f'./MKD_logs/FRIQA_teacher/train_KadidFR_ZERO_SwinT_bs32_36_{config.test_dataset}_o.log', sys.stdout)
    sys.stdout = saver
    config = check_args(config)
    solver = DistillationFRIQASolver(config=config)

    t1 = time.time()
    solver.train()
    t2 = time.time()
    print(f'运行时间：{t2-t1}s')
