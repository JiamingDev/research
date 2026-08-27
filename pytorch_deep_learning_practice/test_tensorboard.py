from torch.utils.tensorboard import SummaryWriter
import cv2
import os

print(os.getcwd())
# 主要不要把文件名设为库名一样，不然运行这个文件时又导入这个文件，导致报错
# --logdir=指日志文件所在目录，port：所用端口，防止和他人用同一台服务器时发生冲突
# 命令行tensorboard --logdir=logs --port=6007
writer = SummaryWriter("logs")  #'logs'：以后日志文件的写入位置，就是写到logs目录里
# 注意，每次运行会往这个目录里继续添加日志，而不是覆盖原先日志，所以每次运行程序要把存放目录改一下，或者删除原来日志，不然最终绘图会乱掉
# 或者换标题，那么会画出多个图像
for i in range(100):
    writer.add_scalar("y=x", i, i)  # 第一个参数：图像标题，第二，三个分别是y，x轴，
image_path = r"train\ant\OIP.webp"
img = cv2.imread(image_path)  # 返回的是numpy.ndarray
# opencv默认是BGR，tensorboard一般是按照RGB，所以这里要转换一下通道顺序
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
writer.add_image(
    "img", img, 1, dataformats="HWC"
)  # 参数：title，file,epoch，通道类型顺序
writer.close()  # 将缓冲区的剩余日志写到文件中并关闭文件，释放资源
