from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

image_path = r"D:\research\research\pytorch_deep_learning_practice\train\ant\OIP.webp"
img = Image.open(image_path)
writer = SummaryWriter("log2")
# ToTensor
trans_totensor = transforms.ToTensor()
tensor_img = trans_totensor(img)  # 从PIL Image/numpy.ndarray转为tensor
writer.add_image("img", tensor_img)
ori = tensor_img[0][0][0]
print(ori, (ori - 0.5) / 0.5)  # 0.4314 -0.1373
# Normalize
# 标准化：x=(x-mean)/std deviation
# 因为我们的图片一般是三通道，这个函数要求提供每个通道的均值和标准差
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
norm_img = trans_norm(tensor_img)
print(norm_img[0][0][0])  # -0.1373
writer.add_image("newimage", norm_img)
# Resize
print(img.size)  # (474, 296)/(W,H)
trans_resize = transforms.Resize((512, 512))  # 改变H,W
resize_img = trans_resize(img)
print(resize_img.size)  # (512, 512)
# 因为我传入的是PIL，所以返回的类型是PIL，但是add_image要tensor/numpy.ndarray,所以要先转化
resize_img = trans_totensor(resize_img)
writer.add_image("resize", resize_img)
# Compose
trans_resize2 = transforms.Resize(
    300
)  # 只输入一个数时，就是把最小边变为这个大小，然后另一边等比例缩放
trans_compose = transforms.Compose(
    [trans_resize2, trans_totensor]
)  # 一个transform操作序列，也就是依次执行的操作
resize2_img = trans_compose(img)
print(resize2_img.shape)  # type: ignore ,torch.Size([3, 300, 480])/(C,H,W)
writer.add_image("resize2", resize2_img)
# RandomCrop
trans_randomcrop = transforms.RandomCrop((210, 100))
trans_compose = transforms.Compose([trans_randomcrop, trans_totensor])
for i in range(10):
    crop_img = trans_compose(img)
    writer.add_image("crop2", crop_img, i)
writer.close()
