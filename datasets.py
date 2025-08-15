import torch.utils.data  # 导入 PyTorch 数据工具模块
import scipy.io as sio  # 导入 SciPy 的 IO 模块并命名为 sio
import torchvision.transforms as transforms  # 导入 torchvision 的变换模块

# 定义训练数据集类，继承自 PyTorch 的 Dataset 类
class TrainData(torch.utils.data.Dataset):
    def __init__(self, img, target, transform=None, target_transform=None):
        self.img = img.float()  # 将图像数据转换为浮点型
        self.target = target.float()  # 将目标数据转换为浮点型
        self.transform = transform  # 图像变换
        self.target_transform = target_transform  # 目标变换

    def __getitem__(self, index):
        img, target = self.img[index], self.target[index]  # 获取指定索引的图像和目标
        if self.transform:
            img = self.transform(img)  # 如果有图像变换，则应用变换
        if self.target_transform:
            target = self.target_transform(target)  # 如果有目标变换，则应用变换

        return img, target  # 返回图像和目标

    def __len__(self):
        return len(self.img)  # 返回数据集的长度

# 定义数据类
class Data:
    def __init__(self, dataset, device):
        super(Data, self).__init__()

        data_path = "./data/" + dataset + "_dataset.mat"  # 构建数据路径
        if dataset == 'samson':
            self.P, self.L, self.col = 3, 156, 95  # 设置 samson 数据集的参数
        elif dataset == 'jasper':
            self.P, self.L, self.col = 4, 198, 100  # 设置 jasper 数据集的参数
        elif dataset == 'urban':
            self.P, self.L, self.col = 4, 162, 305  # urban 裁剪后 col=305
        elif dataset == 'apex':
            self.P, self.L, self.col = 4, 285, 110  # 设置 apex 数据集的参数
        elif dataset == 'dc':
            self.P, self.L, self.col = 6, 191, 290  # 设置 dc 数据集的参数
        elif dataset == 'moffett':
            self.P, self.L, self.col = 3, 184, 50  # 设置 moffett 数据集的参数

        data = sio.loadmat(data_path)  # 加载 .mat 文件数据
        if dataset == 'urban':
            # urban: Y [94249, 162], A [94249, 4]，先reshape再裁剪
            Y = data['Y'].T  # [94249, 162]
            A = data['A'].T  # [94249, 4]
            Y = Y.reshape(307, 307, 162)[1:-1, 1:-1, :]  # [305, 305, 162]
            A = A.reshape(307, 307, 4)[1:-1, 1:-1, :]    # [305, 305, 4]
            Y = Y.reshape(-1, 162)  # [93025, 162]
            A = A.reshape(-1, 4)    # [93025, 4]
            self.Y = torch.from_numpy(Y).to(device)
            self.A = torch.from_numpy(A).to(device)
        else:
            self.Y = torch.from_numpy(data['Y'].T).to(device)
            self.A = torch.from_numpy(data['A'].T).to(device)
        self.M = torch.from_numpy(data['M'])  # 将 M 数据转换为 PyTorch 张量
        # moffett数据集M1顺序调整为(3,2,1)
        if dataset == 'moffett':
            self.M1 = torch.from_numpy(data['M1'])[:, [2,0,1]]
        elif dataset == 'urban':
            self.M1 = torch.from_numpy(data['M1'])[:, [1, 0, 2, 3]]
        else:
            self.M1 = torch.from_numpy(data['M1'])

    def get(self, typ):
        if typ == "hs_img":
            return self.Y.float()  # 返回高光谱图像数据
        elif typ == "abd_map":
            return self.A.float()  # 返回丰度图数据
        elif typ == "end_mem":
            return self.M  # 返回端元数据
        elif typ == "init_weight":
            return self.M1  # 返回初始权重数据
        
    def get_P(self):
        return self.P  # 返回 P 参数
    
    def get_L(self):    
        return self.L  # 返回 L 参数
    
    def get_col(self):
        return self.col  # 返回 col 参数

    def get_loader(self, batch_size=1):
        train_dataset = TrainData(img=self.Y, target=self.A, transform=transforms.Compose([]))  # 创建训练数据集
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False)  # 创建数据加载器
        return train_loader  # 返回数据加载器
