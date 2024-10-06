import torch
import torchvision
import folders.folders_LQ_HQ_diff_content_HQ as folders

class DataLoader(object):
    """Dataset class for IQA databases"""

    def __init__(self, dataset, path, ref_path, img_indx, patch_size, patch_num, batch_size=1, istrain=True, self_patch_num=10, use_HQref = True):

        self.batch_size = batch_size
        self.istrain = istrain

        if (dataset == 'live') | (dataset == 'csiq') | (dataset == 'tid2013') | (dataset == 'livec') | (dataset == 'kadid10k') | (dataset == 'spaq'):
            # Train transforms
            if istrain:
                HQ_diff_content_transform = torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomVerticalFlip(),
                    torchvision.transforms.RandomRotation(degrees=180),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))
                ])
            # Test transforms
            else:
                HQ_diff_content_transform = torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))
                ])
        elif dataset == 'koniq-10k':
            if istrain:
                HQ_diff_content_transform = torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomVerticalFlip(),
                    torchvision.transforms.RandomRotation(degrees=180),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
            else:
                HQ_diff_content_transform = torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
        elif dataset == 'bid':
            if istrain:
                HQ_diff_content_transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((512, 512)),
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomVerticalFlip(),
                    torchvision.transforms.RandomRotation(degrees=180),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
            else:
                HQ_diff_content_transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((512, 512)),
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
        elif dataset == 'flive':
            if istrain:
                HQ_diff_content_transform = torchvision.transforms.Compose([
                    # torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.Resize((640, 640)),
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
            else:
                HQ_diff_content_transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((640, 640)),
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
        else:
             HQ_diff_content_transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
        transforms = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])


        if dataset == 'live':
            self.data = folders.LIVEFolder(
                root=path, HQ_diff_content_root=ref_path, index=img_indx, transform=transforms, HQ_diff_content_transform=HQ_diff_content_transform, patch_num=patch_num, patch_size = patch_size, self_patch_num=self_patch_num)
        elif dataset == 'csiq':
            self.data = folders.CSIQFolder(
                root=path, HQ_diff_content_root=ref_path, index=img_indx, transform=transforms, HQ_diff_content_transform=HQ_diff_content_transform, patch_num=patch_num, patch_size = patch_size, self_patch_num=self_patch_num)
        elif dataset == 'kadid10k':
            self.data = folders.Kadid10kFolder(
                root=path, HQ_diff_content_root=ref_path, index=img_indx, transform=transforms, HQ_diff_content_transform=HQ_diff_content_transform, patch_num=patch_num, patch_size = patch_size, self_patch_num=self_patch_num)
        elif dataset == 'tid2013':
            self.data = folders.TID2013Folder(
                root=path, HQ_diff_content_root=ref_path, index=img_indx, transform=transforms, HQ_diff_content_transform=HQ_diff_content_transform, patch_num=patch_num, patch_size = patch_size, self_patch_num=self_patch_num)
        elif dataset == 'koniq-10k':
            self.data = folders.Koniq_10kFolder(
                root=path, index=img_indx, transform=transforms, HQ_diff_content_transform=HQ_diff_content_transform, patch_num=patch_num, patch_size = patch_size, self_patch_num=self_patch_num)
        elif dataset == 'livec':
            self.data = folders.LIVEChallengeFolder(
                root=path, HQ_diff_content_root=ref_path, index=img_indx, transform=transforms, HQ_diff_content_transform=HQ_diff_content_transform, patch_num=patch_num, patch_size = patch_size, self_patch_num=self_patch_num)
        elif dataset == 'flive':
            self.data = folders.FLiveFolder(
                root=path, index=img_indx, transform=transforms, HQ_diff_content_transform=HQ_diff_content_transform,
                patch_size=patch_size, patch_num=patch_num, self_patch_num=self_patch_num)
        elif dataset == 'spaq':
            self.data = folders.SPAQFolder(
                root=path, index=img_indx, transform=transforms, HQ_diff_content_transform=HQ_diff_content_transform,
                patch_size=patch_size, patch_num=patch_num, self_patch_num=self_patch_num)

    def get_dataloader(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True, num_workers=8)
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=False, num_workers=8)
        return dataloader

# 测试spaq数据集
# 写main函数
if __name__ == '__main__':
    img_num = {
        'kadid10k': list(range(0, 10125)),
        'live': list(range(0, 29)),  # ref HR image
        'csiq': list(range(0, 30)),  # ref HR image
        'tid2013': list(range(0, 25)),
        'livec': list(range(0, 1162)),  # no-ref image
        'koniq-10k': list(range(0, 10073)),  # no-ref image
        'bid': list(range(0, 586)),  # no-ref imaged
        'flive': list(range(0, 39807)),  # no-ref imaged
        'spaq': list(range(0, 11125))

    }
    folder_path = {
        'pipal': '/data/dataset/PIPAL',
        'live': '/data/dataset/LIVE/',
        'csiq': '/data/dataset/CSIQ/',
        'tid2013': '/data/dataset/tid2013/',
        'livec': '/data/dataset/ChallengeDB_release',
        'koniq-10k': '/data/dataset/koniq-10k/',
        'bid': '/data/dataset/BID/',
        'kadid10k': '/data/dataset/kadid10k/',
        'flive': '/data/dataset/FLive/database',
        'spaq': '/data/dataset/SPAQ'
    }
    datas = 'spaq'
    loader = DataLoader(datas, folder_path[datas], '/data/dataset/DIV2K/train_HR_Sample/',
                        img_num[datas], 224, 1,
                        16, istrain=True, self_patch_num=10)
    data = loader.get_dataloader()
    # 从loader中依次取数据，并打印数据维度和尺寸
    for i, (LQ, LQ_path, ZLQ, label) in enumerate(data):
        print(LQ_path)
        print(label)
        break