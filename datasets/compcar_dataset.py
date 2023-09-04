
import os
from PIL import Image
from PIL import ImageFile
from torch.utils.data import DataLoader
import torch.utils.data as data
from tqdm import tqdm


class CompCarsDataset(data.Dataset):
    def __init__(self, split_file, base_path, transform=None) -> None:
        super().__init__()
        
        self.transform = transform

        major_class = []
        with open("compcars_label_index.txt") as f:
            for line in f:
                major_class.append(line.strip())
        self.label2index = {label:index for index, label in enumerate(major_class)}

        self.compcars = []
        self.targets = []
        for line in tqdm(open(split_file)):
            info = line.strip().split(" ")
            file_names = [ views.split(".")[0] + ".jpg" for views in info[1:]]
            label = info[0]
            manufacturer, model = info[0].split("_")
            images = [os.path.join(base_path, manufacturer, model, file_name) for file_name in file_names]
            self.compcars.append(images)
            self.targets.append(self.label2index[label])        
            
    def __getitem__(self, index):
        
        imgs = [Image.open(path).convert('RGB') for path in self.compcars[index]]

        if self.transform: 
            imgs = [ self.transform(img) for img in imgs]
            
        return imgs, self.targets[index]
    
    def __len__(self):
        return len(self.targets)

