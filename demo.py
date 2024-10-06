import torch
import os
from option_train_DistillationIQA import set_args, check_args
import numpy as np
from models.DistillationIQA import DistillationIQANet
from PIL import Image
import torchvision

img_num = {
        'kadid10k': list(range(0,10125)),
        'live': list(range(0, 29)),#ref HR image
        'csiq': list(range(0, 30)),#ref HR image
        'tid2013': list(range(0, 25)),
        'livec': list(range(0, 1162)),# no-ref image
        'koniq-10k': list(range(0, 10073)),# no-ref image
        'bid': list(range(0, 586)),# no-ref image
    }
folder_path = {
    'pipal': '/home/dataset/PIPAL',
    'live': '/home/dataset/LIVE/',
    'csiq': '/home/dataset/csiq/',
    'tid2013': '/home/dataset/tid2013/',
    'livec': '/home/dataset/LIVEC/',
    'koniq-10k': '/home/dataset/koniq10k/',
    'bid': '/home/dataset/BID/',
    'kadid10k': '/home/dataset/kadid10k/'
    }


class DistillationIQASolver(object):
    def __init__(self, config, lq_path, ref_path):
        self.config = config
        config.studentNet_model_path = './model_zoo/Student_Origin_tid2013_in_saved_model.pth'
        config.teacherNet_model_path = './model_zoo/FR_teacher_cross_dataset.pth'

        self.device = torch.device('cuda' if config.gpu_ids is not None else 'cpu')
        # print(self.device)
        # self.txt_log_path = os.path.join(config.log_checkpoint_dir,'log_origin.txt')
        # with open(self.txt_log_path,"w+") as f:
        #     f.close()
        
        # model teacherNet
        self.teacherNet = DistillationIQANet(self_patch_num=config.self_patch_num, distillation_layer=config.distillation_layer)
        if config.teacherNet_model_path:
            self.teacherNet._load_state_dict(torch.load(config.teacherNet_model_path))
            # 打印加载的权重路径
            print(f"Loaded weights from {config.teacherNet_model_path}")
        self.teacherNet = self.teacherNet.to(self.device)
        self.teacherNet.train(False)

        # model studentNet
        self.studentNet = DistillationIQANet(self_patch_num=config.self_patch_num, distillation_layer=config.distillation_layer)
        if config.studentNet_model_path:
            self.studentNet._load_state_dict(torch.load(config.studentNet_model_path))
            # 打印加载的权重路径
            print(f"Loaded weights from {config.studentNet_model_path}")
        self.studentNet = self.studentNet.to(self.device)
        self.studentNet.train(True)

        self.transform = torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop(size=self.config.patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))
                ])
        # data 打patch
        self.LQ_patches = self.preprocess(lq_path)
        self.ref_patches = self.preprocess(ref_path)
    
    def preprocess(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img= img.convert('RGB')
        patches = []
        for _ in range(self.config.self_patch_num):
            patch = self.transform(img)
            patches.append(patch.unsqueeze(0))
        patches = torch.cat(patches, 0)
        return patches.unsqueeze(0)

    def test(self):
        self.studentNet.train(False)
        LQ_patches, ref_patches = self.LQ_patches.to(self.device), self.ref_patches.to(self.device)
        with torch.no_grad():
            _, _, pred = self.studentNet(LQ_patches, ref_patches)
        return float(pred.item())

if __name__ == "__main__":
    config = set_args()
    config = check_args(config)

    lq_path = './imgs/i09_19_1.bmp'
    lq_path2 = './imgs/i09_19_2.bmp'
    lq_path3 = './imgs/i09_19_3.bmp'
    lq_path4 = './imgs/i09_19_4.bmp'
    lq_path5 = './imgs/i09_19_5.bmp'
    ref_path = '/home/dataset/DIV2K/val_HR/0801.png'
    label = 1.15686274509804
    solver = DistillationIQASolver(config=config, lq_path=lq_path, ref_path=ref_path)
    solver2 = DistillationIQASolver(config=config, lq_path=lq_path2, ref_path=ref_path)
    solver3 = DistillationIQASolver(config=config, lq_path=lq_path3, ref_path=ref_path)
    solver4 = DistillationIQASolver(config=config, lq_path=lq_path4, ref_path=ref_path)
    solver5 = DistillationIQASolver(config=config, lq_path=lq_path5, ref_path=ref_path)

    scores = []
    scores2 = []
    scores3 = []
    scores4 = []
    scores5 = []
    for _ in range(10):
        scores.append(solver.test())
        scores2.append(solver2.test())
        scores3.append(solver3.test())
        scores4.append(solver4.test())
        scores5.append(solver5.test())

    # 打印分数均值并且保留4位小数
    print(f"{np.mean(scores):.4f}\n{np.mean(scores2):.4f}\n{np.mean(scores3):.4f}\n{np.mean(scores4):.4f}\n{np.mean(scores5):.4f}")
# 5.3901
# 4.7942
# 4.4851
# 4.0004
# 3.4913
