import argparse
from collections import OrderedDict
from torch.autograd import Variable
import torchvision.transforms as transforms
import utils.utils as util
from models.mprnet import *
import torchvision.utils as utils
from utils.dataset import *
from PIL import Image
from matplotlib import pyplot as plt
from imageio import imwrite

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="SKT AI Fellowship SKSAK")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default="paris", help='path of datasets files')
args = parser.parse_args()

def normalize(data):
    return data/255.

def main():
    # Build model
    print('Loading model ...\n')
    model = MPRNet()
    # model = net.cuda()

    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('datasets/DIV2K/LR/paris/origin/', '*.png'))
    files_source.sort()

    print('Model Test Start!')

    for f in files_source:
        # image
        Img = Image.open(f)
        # Img = Img.resize((int(Img.width//4), int(Img.height//4)))

        Img = normalize(np.asarray(Img, dtype=np.float32))
        Img = np.transpose(Img, (2, 0, 1))


        Img = np.expand_dims(Img, 0)
        input_img = torch.Tensor(Img)

        # input_img = Variable(input_img.cuda())
        # model.cuda()
        # out = model(input_img)

        img_resize = utils.make_grid(input_img.data, nrow=8, normalize=True, scale_each=True)

        img_resize = torch.clamp(img_resize * 255, 0, 255)
        img_resize = np.uint8(img_resize.cpu())
        filename = f.split('origin')[1]
        imwrite('datasets/DIV2K/LR/paris/resize/' + filename, np.transpose(img_resize, (1, 2, 0)))


        H = input_img.size(2)
        W = input_img.size(3)

        pimg1 = input_img[:, :, 0:int(H / 2), :]
        pimg2 = input_img[:, :, int(H / 2):H, :]

        # pimg1 = input_img[:, :, :, 0:int(W / 2)]
        # pimg2 = input_img[:, :, :, int(W / 2):W]


        img1 = pimg1[:, :, :, 0:int(W / 2)]
        img2 = pimg1[:, :, :, int(W / 2):W]
        img3 = pimg2[:, :, :, 0:int(W / 2)]
        img4 = pimg2[:, :, :, int(W / 2):W]

        # w1_img = utils.make_grid(img1.data, nrow=8, normalize=True, scale_each=True)
        # # img2 = utils.make_grid(img2.data, nrow=8, normalize=True, scale_each=True)
        #
        # plt.imshow(np.transpose(w1_img.cpu(), (1, 2, 0)))
        # plt.show()
        #
        # plt.imshow(np.transpose(img2.cpu(), (1, 2, 0)))
        # plt.show()

        model.feed_data(input_img, need_GT=False)
        model.test()
        final = model.get_current_visuals(need_GT=False)
        final = torch.clamp(final['SR'], 0., 1.)

        # model.feed_data(img1, need_GT=False)
        # model.test()
        # out1 = model.get_current_visuals(need_GT=False)
        # out1 = torch.clamp(out1['SR'], 0., 1.)
        #
        # model.feed_data(img2, need_GT=False)
        # model.test()
        # out2 = model.get_current_visuals(need_GT=False)
        # out2 = torch.clamp(out2['SR'], 0., 1.)
        #
        # model.feed_data(img3, need_GT=False)
        # model.test()
        # out3 = model.get_current_visuals(need_GT=False)
        # out3 = torch.clamp(out3['SR'], 0., 1.)
        #
        # model.feed_data(img4, need_GT=False)
        # model.test()
        # out4 = model.get_current_visuals(need_GT=False)
        # out4 = torch.clamp(out4['SR'], 0., 1.)
        #
        # recon_1 = torch.cat([out1, out2], 2)
        # recon_2 = torch.cat([out3, out4], 2)
        #
        # final = torch.cat([recon_1, recon_2], 1)

        # model.feed_data(img2, need_GT=False)
        # model.test()
        # # model.back_projection()
        # out2 = model.get_current_visuals(need_GT=False)
        # out2 = torch.clamp(out2['SR'], 0., 1.)
        #
        # out = torch.cat([out1, out2], 3)

        clean_img = utils.make_grid(final.data, nrow=8, normalize=True, scale_each=True)
        # clean_img = utils.make_grid(out.data, nrow=8, normalize=True, scale_each=True)

        plt.imshow(np.transpose(clean_img.cpu(), (1,2,0)))
        plt.show()

        result_img = torch.clamp(clean_img * 255, 0, 255)
        result_img = np.uint8(result_img.cpu())
        filename = f.split('origin')[1]
        imwrite('datasets/DIV2K/LR/paris/clean/' + filename, np.transpose(result_img, (1, 2, 0)))


if __name__ == "__main__":
    main()
