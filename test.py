import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict
from torch.autograd import Variable
import torchvision.transforms as transforms
from utils.utils import *
from models.sksak_sr2 import *
import torchvision.utils as utils
from PIL import Image
from matplotlib import pyplot as plt
from imageio import imwrite
import os
import glob


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="NTIRE DeBlur Challenge")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default="paris", help='path of datasets files')
opt = parser.parse_args()

def normalize(data):
    return data/255.

def main():
    # Build model
    print('Loading model ...\n')
    model = SKSAK()
    model = model.cuda()

    load = torch.load(os.path.join(opt.logdir, 'SKT_AI_line_20_0.0382.pth'))
    model.load_state_dict(load)
    model.eval()

    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data/test/000/', '*.png'))
    files_source.sort()

    print('Model Test Start!')

    for f in files_source:
        # image
        origin_img = Image.open(f)
        # Img = Img.resize((800, 480), resample=Image.BICUBIC)

        right = 80
        left = 0
        top = 0
        bottom = 0

        width, height = origin_img.size
        new_width = width + right + left
        new_height = height + top + bottom

        Img = Image.new(origin_img.mode, (new_width, new_height), (0,0,0))
        Img.paste(origin_img, (left, top))
        Img = normalize(np.asarray(Img, dtype=np.float32))
        Img = np.transpose(Img, (2, 0, 1))

        Img = np.expand_dims(Img, 0)
        input_img = torch.Tensor(Img)

        filename = f.split('/000/')[1]
        # imwrite('data/test/out/12epoch/' + filename, np.transpose(img_resize, (1, 2, 0)))

        with torch.no_grad():
            input_img = Variable(input_img.cuda())
            outimg = model(input_img)
            out = outimg[0]

        clean_img = utils.make_grid(out.data, nrow=8, normalize=True, scale_each=True)

        print(filename, 'finished!')
        result_img = torch.clamp(clean_img * 255, 0, 255)
        result_img = np.uint8(result_img.cpu())
        np_img = np.transpose(result_img, (1, 2, 0))

        result = Image.fromarray(np_img)
        result = result.crop((0,0,1440,960))

        result.save('data/test/final/line_daughter/' + filename, format='PNG')



if __name__ == "__main__":
    main()
