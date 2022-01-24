from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


image_path = ".../img_align_celeba/"
batch_size = 32
celebA_data = ImageFolder(image_path)
celebA_dataloader = DataLoader(celebA_data, batch_size, shuffle=True)



