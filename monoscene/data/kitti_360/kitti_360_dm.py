from torch.utils.data.dataloader import DataLoader
from monoscene.data.kitti_360.kitti_360_dataset import Kitti360Dataset
import pytorch_lightning as pl
from monoscene.data.kitti_360.collate import collate_fn
from monoscene.data.utils.torch_util import worker_init_fn


class Kitti360DataModule(pl.LightningDataModule):
    """
    1.相当于Kitti360Dataset的成员函数，Kitti360Dataset自身除了初始化没有其他作用
    2.可以合并 DM和Dataset模块
    """
    def __init__(self, root, sequences, n_scans, batch_size=4, num_workers=3):
        super().__init__()
        """
        1.重载init 初始化变量
        """
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sequences = sequences
        self.n_scans = n_scans

    def setup(self, stage=None):
        """
        1.调用kitti类初始化
        """
        self.ds = Kitti360Dataset(
            root=self.root, sequences=self.sequences, n_scans=self.n_scans
        )

    def dataloader(self):
        """
        1.调用dataloader加载数据
        """
        return DataLoader(
            self.ds,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )
