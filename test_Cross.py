import sys

import torch
import os

from Util.OutputSaver import Saver
from Util.tools import convert_obj_score
from dataloaders.dataloader_LQ_HQ_diff_content_HQ import DataLoader
from option_train_DistillationIQA import set_args, check_args
from scipy import stats
import numpy as np
from models.DistillationIQA import DistillationIQANet

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
    'pipal': '/home/dataset/PIPAL',
    'live': '/home/dataset/LIVE/',
    'csiq': '/home/dataset/CSIQ/',
    'tid2013': '/home/dataset/tid2013/',
    'livec': '/home/dataset/LIVEC/',
    'koniq-10k': '/home/dataset/koniq-10k/',
    'bid': '/home/dataset/BID/',
    'kadid10k': '/home/dataset/kadid10k/'
}


class DistillationIQASolver(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if config.gpu_ids is not None else 'cpu')

        self.studentNet = DistillationIQANet(self_patch_num=config.self_patch_num,
                                              distillation_layer=config.distillation_layer)
        if config.studentNet_model_path:
            self.studentNet._load_state_dict(torch.load(config.studentNet_model_path))
            print(f"加载了{config.studentNet_model_path}")
        self.studentNet = self.studentNet.to(self.device)
        self.studentNet.train(True)

        # data
        test_loader_LIVE = DataLoader('live', folder_path['live'], config.ref_test_dataset_path, img_num['live'],
                                      config.patch_size, config.test_patch_num, istrain=False,
                                      self_patch_num=config.self_patch_num)
        test_loader_CSIQ = DataLoader('csiq', folder_path['csiq'], config.ref_test_dataset_path, img_num['csiq'],
                                      config.patch_size, config.test_patch_num, istrain=False,
                                      self_patch_num=config.self_patch_num)
        test_loader_TID = DataLoader('tid2013', folder_path['tid2013'], config.ref_test_dataset_path,
                                     img_num['tid2013'], config.patch_size, config.test_patch_num, istrain=False,
                                     self_patch_num=config.self_patch_num)
        # test_loader_Koniq = DataLoader('koniq-10k', folder_path['koniq-10k'], config.ref_test_dataset_path,
        #                                img_num['koniq-10k'], config.patch_size, config.test_patch_num, istrain=False,
        #                                self_patch_num=config.self_patch_num)

        self.test_data_LIVE = test_loader_LIVE.get_dataloader()
        self.test_data_CSIQ = test_loader_CSIQ.get_dataloader()
        self.test_data_TID = test_loader_TID.get_dataloader()
        # self.test_data_Koniq = test_loader_Koniq.get_dataloader()

    def test(self, test_data):
        self.studentNet.train(False)
        test_pred_scores, test_gt_scores = [], []
        for LQ_patches, _, ref_patches, label in test_data:
            LQ_patches, ref_patches, label = LQ_patches.to(self.device), ref_patches.to(self.device), label.to(
                self.device)
            with torch.no_grad():
                _, _, pred = self.studentNet(LQ_patches, ref_patches)
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
        self.studentNet.train(True)
        return test_srcc, test_plcc, test_krcc


if __name__ == "__main__":
    config = set_args()
    # 保存控制台输出
    # TODO 日志路径修改
    # saver = Saver('./MKD_logs/Cross/SwinT_tkadid32_bs16_1_kadid_cross.log', sys.stdout)
    saver = Saver(f'./MKD_logs/Test/SwinT_live_in.log', sys.stdout)
    sys.stdout = saver
    config = check_args(config)
    solver = DistillationIQASolver(config=config)
    fold_10_test_LIVE_srcc, fold_10_test_LIVE_plcc, fold_10_test_LIVE_krcc = [], [], []
    fold_10_test_CSIQ_srcc, fold_10_test_CSIQ_plcc, fold_10_test_CSIQ_krcc = [], [], []
    fold_10_test_TID_srcc, fold_10_test_TID_plcc, fold_10_test_TID_krcc = [], [], []
    fold_10_test_Koniq_srcc, fold_10_test_Koniq_plcc, fold_10_test_Koniq_krcc = [], [], []

    print('Round\tTest_SRCC\tTest_PLCC\tTest_KRCC')
    for i in range(3):
        test_LIVE_srcc, test_LIVE_plcc, test_LIVE_krcc = solver.test(solver.test_data_LIVE)
        print(f'{i + 1}\tLIVE\t{test_LIVE_srcc:.3f}\t{test_LIVE_plcc:.3f}\t{test_LIVE_krcc:.3f}\n')
        fold_10_test_LIVE_srcc.append(test_LIVE_srcc)
        fold_10_test_LIVE_plcc.append(test_LIVE_plcc)
        fold_10_test_LIVE_krcc.append(test_LIVE_krcc)

        test_CSIQ_srcc, test_CSIQ_plcc, test_CSIQ_krcc = solver.test(solver.test_data_CSIQ)
        print(f'{i + 1}\tCSIQ\t{test_CSIQ_srcc:.3f}\t{test_CSIQ_plcc:.3f}\t{test_CSIQ_krcc:.3f}\n')
        fold_10_test_CSIQ_srcc.append(test_CSIQ_srcc)
        fold_10_test_CSIQ_plcc.append(test_CSIQ_plcc)
        fold_10_test_CSIQ_krcc.append(test_CSIQ_krcc)

        test_TID_srcc, test_TID_plcc, test_TID_krcc = solver.test(solver.test_data_TID)
        print(f'{i + 1}\tTID\t{test_TID_srcc:.3f}\t{test_TID_plcc:.3f}\t{test_TID_krcc:.3f}\n')
        fold_10_test_TID_srcc.append(test_TID_srcc)
        fold_10_test_TID_plcc.append(test_TID_plcc)
        fold_10_test_TID_krcc.append(test_TID_krcc)

        # test_Koniq_srcc, test_Koniq_plcc, test_Koniq_krcc = solver.test(solver.test_data_Koniq)
        # print(f'{i + 1}\tKoniq\t{test_Koniq_srcc:.3f}\t{test_Koniq_plcc:.3f}\t{test_Koniq_krcc:.3f}\n')
        # fold_10_test_Koniq_srcc.append(test_Koniq_srcc)
        # fold_10_test_Koniq_plcc.append(test_Koniq_plcc)
        # fold_10_test_Koniq_krcc.append(test_Koniq_krcc)

    print('LIVE Test SRCC:{}, PLCC:{}, KRCC:{}\n'.format(np.mean(fold_10_test_LIVE_srcc),
                                                         np.mean(fold_10_test_LIVE_plcc),
                                                         np.mean(fold_10_test_LIVE_krcc)))
    print('CSIQ Test SRCC:{}, PLCC:{}, KRCC:{}\n'.format(np.mean(fold_10_test_CSIQ_srcc),
                                                         np.mean(fold_10_test_CSIQ_plcc),
                                                         np.mean(fold_10_test_CSIQ_krcc)))
    print('TID Test SRCC:{}, PLCC:{}, KRCC:{}\n'.format(np.mean(fold_10_test_TID_srcc),
                                                        np.mean(fold_10_test_TID_plcc),
                                                        np.mean(fold_10_test_TID_krcc)))
    # print('Koniq Test SRCC:{}, PLCC:{}, KRCC:{}\n'.format(np.mean(fold_10_test_Koniq_srcc),
    #                                                       np.mean(fold_10_test_Koniq_plcc),
    #                                                       np.mean(fold_10_test_Koniq_krcc)))
