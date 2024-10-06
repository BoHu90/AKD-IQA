import sys

import torch
import os
import random

from Util.OutputSaver import Saver
from Util.tools import convert_obj_score
from dataloaders.dataloader_LQ_HQ_diff_content_HQ import DataLoader
from models.Student import Student
from scipy import stats
import numpy as np

from option_train_student import set_stu_args, check_stu_args

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
    'livec': '/data/dataset/ChallengeDB_release',
    'koniq-10k': '/data/dataset/koniq-10k/',
    'bid': '/data/dataset/BID/',
    'kadid10k': '/data/dataset/kadid10k/'
}


class DistillationIQASolver(object):
    def __init__(self, config):
        # 加载配置、设置设备（GPU或CPU）、创建日志文件
        self.config = config
        self.device = torch.device('cuda' if config.gpu_ids is not None else 'cpu')
        # 日志文件
        # self.txt_log_path = os.path.join(config.log_checkpoint_dir, 'log_origin.txt')
        # with open(self.txt_log_path, "w+") as f:
        #     f.close()
        # TODO 学生模型
        self.studentNet = Student(self_patch_num=config.self_patch_num)
        # 不加载学生权重
        if config.studentNet_model_path:
            self.studentNet._load_state_dict(torch.load(config.studentNet_model_path))
            print(f'load student model from {config.studentNet_model_path}')
        self.studentNet = self.studentNet.to(self.device)
        self.studentNet.train(True)  # 训练学生模型

        # lr,opt,loss,epoch
        self.lr = config.lr
        self.lr_ratio = 1
        resnet_params = list(map(id, self.studentNet.feature_extractor.parameters()))  # 特征提取器参数
        res_params = filter(lambda p: id(p) not in resnet_params, self.studentNet.parameters())  # 其他参数
        # 使用不同的学习率进行更新
        paras = [{'params': res_params, 'lr': self.lr * self.lr_ratio},
                 {'params': self.studentNet.feature_extractor.parameters(), 'lr': self.lr}
                 ]
        # 学习率\损失函数的定义
        self.optimizer = torch.optim.Adam(paras, weight_decay=config.weight_decay)
        self.l1_loss = torch.nn.L1Loss()
        self.epochs = config.epochs

        # data：使用的是kadid10k
        # 获取index
        sel_num = img_num[config.train_dataset]
        # random.shuffle(sel_num)
        # config.train_index = sel_num  # 做跨库
        # test_index = img_num[config.test_dataset]
        # TODO 做库内:
        config.train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
        test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]
        # 训练数据 train_dataset：kadid10k；
        train_loader = DataLoader(config.train_dataset, folder_path[config.train_dataset],
                                  config.ref_train_dataset_path, config.train_index, config.patch_size,
                                  config.train_patch_num, batch_size=config.batch_size, istrain=True,
                                  self_patch_num=config.self_patch_num)
        # TODO 测试数据注释1
        # 库内测试数据
        test_loader = DataLoader(config.test_dataset, folder_path[config.test_dataset], config.ref_test_dataset_path, test_index,
                                       config.patch_size, config.test_patch_num, istrain=False,  # 测试模式
                                       self_patch_num=config.self_patch_num)
        # 跨测试数据
        # test_loader_TID = DataLoader('tid2013', folder_path['tid2013'], config.ref_test_dataset_path,
        #                              img_num['tid2013'], config.patch_size, config.test_patch_num, istrain=False,
        #                              self_patch_num=config.self_patch_num)
        # test_loader_LIVE = DataLoader('live', folder_path['live'], config.ref_test_dataset_path,
        #                               img_num['live'], config.patch_size, config.test_patch_num, istrain=False,
        #                               self_patch_num=config.self_patch_num)
        # test_loader_CSIQ = DataLoader('csiq', folder_path['csiq'], config.ref_test_dataset_path,
        #                               img_num['csiq'], config.patch_size, config.test_patch_num, istrain=False,
        #                               self_patch_num=config.self_patch_num)
        # test_loader_Koniq = DataLoader('koniq-10k', folder_path['koniq-10k'], config.ref_test_dataset_path,
        #                                img_num['koniq-10k'], config.patch_size, config.test_patch_num, istrain=False,
        #                                self_patch_num=config.self_patch_num)

        self.train_data = train_loader.get_dataloader()
        # TODO 测试数据注释2
        # 库内测试数据
        self.test_data = test_loader.get_dataloader()
        # 跨库测试数据
        # self.test_data_kadid = test_loader_kadid.get_dataloader()
        # self.test_data_TID = test_loader_TID.get_dataloader()
        # self.test_data_LIVE = test_loader_LIVE.get_dataloader()
        # self.test_data_CSIQ = test_loader_CSIQ.get_dataloader()
        # self.test_data_Koniq = test_loader_Koniq.get_dataloader()

    def train(self):
        # TODO 测试数据注释3
        # 库内
        best_srcc = 0.0
        best_plcc = 0.0
        best_krcc = 0.0

        # best_srcc_kadid = 0.0
        # best_plcc_kadid = 0.0
        # best_krcc_kadid = 0.0

        # best_srcc_TID = 0.0
        # best_plcc_TID = 0.0
        # best_krcc_TID = 0.0


        # best_srcc_LIVE = 0.0
        # best_plcc_LIVE = 0.0
        # best_krcc_LIVE = 0.0
        #
        # best_srcc_CSIQ = 0.0
        # best_plcc_CSIQ = 0.0
        # best_krcc_CSIQ = 0.0

        # best_srcc_Koniq = 0.0
        # best_plcc_Koniq = 0.0
        # best_krcc_Koniq = 0.0

        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC\tTest_KRCC')

        # NEW
        scaler = torch.cuda.amp.GradScaler()

        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []  # gt: ground truth

            for LQ_patches, _, ref_patches, label in self.train_data:
                # 转设备
                LQ_patches, ref_patches, label = LQ_patches.to(self.device), ref_patches.to(self.device), label.to(self.device)
                # 提取清零
                self.optimizer.zero_grad()

                # 开启混合精度训练
                with torch.cuda.amp.autocast():
                    # t_encode_diff_inner_feature, t_decode_inner_feature, _ = self.teacherNet(LQ_patches, refHQ_patches)
                    # 学生模型输入的是 LQ 和 NAR 图像
                    pred = self.studentNet(LQ_patches, ref_patches)

                    # 存储预测分数与真实分数并计算损失
                    pred_scores = pred_scores + pred.cpu().tolist()
                    gt_scores = gt_scores + label.cpu().tolist()
                    loss = self.l1_loss(pred.squeeze(), label.float().detach())

                # 反向传播
                epoch_loss.append(loss.item())
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
            # TODO 测试数据注释4 测试、指标、打印
            # 每个epoch进行一次测试
            # 库内测试
            test_srcc, test_plcc, test_krcc = self.test(self.test_data)
            # test_kadid_srcc, test_kadid_plcc, test_kadid_krcc = self.test(self.test_data_kadid)
            # test_TID_srcc, test_TID_plcc, test_TID_krcc = self.test(self.test_data_TID)
            # test_LIVE_srcc, test_LIVE_plcc, test_LIVE_krcc = self.test(self.test_data_LIVE)
            # test_CSIQ_srcc, test_CSIQ_plcc, test_CSIQ_krcc = self.test(self.test_data_CSIQ)
            # test_Koniq_srcc, test_Koniq_plcc, test_Koniq_krcc = solver.test(solver.test_data_Koniq)

            # # TODO 保存学生模型注释
            # if test_TID_srcc + test_LIVE_srcc + test_CSIQ_srcc > best_srcc_TID + best_srcc_LIVE + best_srcc_CSIQ:
            #     torch.save(self.studentNet.state_dict(), os.path.join(self.config.model_checkpoint_dir,
            #                                                       f'Student_{config.train_dataset}_saved_model.pth'))
            #     print("学生模型更新：")
            # 库内
            if test_srcc + test_plcc + test_krcc > best_srcc + best_plcc + best_krcc:
                best_srcc, best_plcc, best_krcc = test_srcc, test_plcc, test_krcc
                torch.save(self.studentNet.state_dict(), os.path.join(self.config.model_checkpoint_dir,
                                                                      f'Student_{config.train_dataset}_saved_model_NoPre1.pth'))
                print("学生模型更新：")
            print('%d:%s\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f \n' %
                  (t, config.test_dataset, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc, test_krcc))

            # if test_kadid_srcc + test_kadid_plcc + test_kadid_krcc > best_srcc_kadid + best_plcc_kadid + best_krcc_kadid:
            #     best_srcc_kadid, best_plcc_kadid, best_krcc_kadid = test_kadid_srcc, test_kadid_plcc, test_kadid_krcc
            # print('%d:kadid\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f \n' %
            #       (
            #           t, sum(epoch_loss) / len(epoch_loss), train_srcc, test_kadid_srcc, test_kadid_plcc,
            #           test_kadid_krcc))

            # if test_TID_srcc + test_TID_plcc + test_TID_krcc > best_srcc_TID + best_plcc_TID + best_krcc_TID:
            #     best_srcc_TID, best_plcc_TID, best_krcc_TID = test_TID_srcc, test_TID_plcc, test_TID_krcc
            # print('%d:tid\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f \n' %
            #       (t, sum(epoch_loss) / len(epoch_loss), train_srcc, test_TID_srcc, test_TID_plcc, test_TID_krcc))

            # if test_LIVE_srcc + test_LIVE_plcc + test_LIVE_krcc > best_srcc_LIVE + best_plcc_LIVE + best_krcc_LIVE:
            #     best_srcc_LIVE, best_plcc_LIVE, best_krcc_LIVE = test_LIVE_srcc, test_LIVE_plcc, test_LIVE_krcc
            # print('%d:live\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f \n' %
            #       (
            #           t, sum(epoch_loss) / len(epoch_loss), train_srcc, test_LIVE_srcc, test_LIVE_plcc,
            #           test_LIVE_krcc))
            #
            # if test_CSIQ_srcc + test_CSIQ_plcc + test_CSIQ_krcc > best_srcc_CSIQ + best_plcc_CSIQ + best_krcc_CSIQ:
            #     best_srcc_CSIQ, best_plcc_CSIQ, best_krcc_CSIQ = test_CSIQ_srcc, test_CSIQ_plcc, test_CSIQ_krcc
            # print('%d:csiq\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f \n' %
            #       (
            #           t, sum(epoch_loss) / len(epoch_loss), train_srcc, test_CSIQ_srcc, test_CSIQ_plcc,
            #           test_CSIQ_krcc))

            # if test_Koniq_srcc + test_Koniq_plcc + test_Koniq_krcc > best_srcc_Koniq + best_plcc_Koniq + best_krcc_Koniq:
            #     best_srcc_Koniq, best_plcc_Koniq, best_krcc_Koniq = test_Koniq_srcc, test_Koniq_plcc, test_Koniq_krcc
            # print('%d:koniq\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f \n' %
            #       (t, sum(epoch_loss) / len(epoch_loss), train_srcc, test_Koniq_srcc, test_Koniq_plcc,
            #        test_Koniq_krcc))

            self.lr = self.lr / pow(10, (t // self.config.update_opt_epoch))
            if t > 20:
                self.lr_ratio = 1
            resnet_params = list(map(id, self.studentNet.feature_extractor.parameters()))
            rest_params = filter(lambda p: id(p) not in resnet_params, self.studentNet.parameters())
            paras = [{'params': rest_params, 'lr': self.lr * self.lr_ratio},
                     {'params': self.studentNet.feature_extractor.parameters(), 'lr': self.lr}
                     ]
            self.optimizer = torch.optim.Adam(paras, weight_decay=self.config.weight_decay)
        # 取所有epoch最佳的指标
        # 库内
        print('Best %s test SRCC %f, PLCC %f, KRCC %f\n' % (config.test_dataset, best_srcc, best_plcc, best_krcc))
        # print('Best kadid test SRCC %f, PLCC %f, KRCC %f\n' % (best_srcc_kadid, best_plcc_kadid, best_krcc_kadid))
        # print('Best tid2013 test SRCC %f, PLCC %f, KRCC %f\n' % (best_srcc_TID, best_plcc_TID, best_krcc_TID))
        # print('Best csiq test SRCC %f, PLCC %f, KRCC %f\n' % (best_srcc_CSIQ, best_plcc_CSIQ, best_krcc_CSIQ))
        # print('Best live test SRCC %f, PLCC %f, KRCC %f\n' % (best_srcc_LIVE, best_plcc_LIVE, best_krcc_LIVE))
        # print('Best koniq test SRCC %f, PLCC %f, KRCC %f\n' % (best_srcc_Koniq, best_plcc_Koniq, best_krcc_Koniq))

    def test(self, test_data):
        # 测试模式
        self.studentNet.train(False)
        test_pred_scores, test_gt_scores = [], []

        for LQ_patches, _, ref_patches, label in test_data:
            LQ_patches, ref_patches, label = LQ_patches.to(self.device), ref_patches.to(self.device), label.to(
                self.device)
            with torch.no_grad():
                # 测试时，输入LQ和NAR图像，得到预测分数
                pred = self.studentNet(LQ_patches, ref_patches)
                test_pred_scores.append(float(pred.item()))
                test_gt_scores = test_gt_scores + label.cpu().tolist()
        if self.config.use_fitting_prcc_srcc:
            fitting_pred_scores = convert_obj_score(test_pred_scores, test_gt_scores)
        # 取平均
        test_pred_scores = np.mean(np.reshape(np.array(test_pred_scores), (-1, self.config.test_patch_num)), axis=1)
        test_gt_scores = np.mean(np.reshape(np.array(test_gt_scores), (-1, self.config.test_patch_num)), axis=1)
        test_srcc, _ = stats.spearmanr(test_pred_scores, test_gt_scores)
        if self.config.use_fitting_prcc_srcc:
            test_plcc, _ = stats.pearsonr(fitting_pred_scores, test_gt_scores)
        else:
            test_plcc, _ = stats.pearsonr(test_pred_scores, test_gt_scores)
        test_krcc, _ = stats.stats.kendalltau(test_pred_scores, test_gt_scores)
        test_srcc, test_plcc, test_krcc = abs(test_srcc), abs(test_plcc), abs(test_krcc)
        self.studentNet.train(True)
        return test_srcc, test_plcc, test_krcc


if __name__ == "__main__":
    config = set_stu_args()
    # TODO 日志路径修改
    saver = Saver(f'./MKD_logs/Student/Student_{config.train_dataset}_{config.batch_size}_NoPre_in1.log', sys.stdout)
    sys.stdout = saver
    config = check_stu_args(config)
    solver = DistillationIQASolver(config=config)
    solver.train()
