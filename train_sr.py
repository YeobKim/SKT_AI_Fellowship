import argparse
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.utils import *
from models.srnet import *
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from PIL import ImageFilter
from datetime import datetime
from utils.dataset_sr import *
from torchsummary import summary
from warmup_scheduler import GradualWarmupScheduler
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="SKT AI Fellowship SKSAK")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--resume_epochs", type=int, default=52, help="Number of training epochs When training resume")
parser.add_argument("--lr", type=float, default=2e-5, help="Initial learning rate")
parser.add_argument("--last_lr", type=float, default=0.5e-6, help="Last learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--patchsize", type=int, default=512, help='patch size of image')
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')

opt = parser.parse_args()


def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True, batchSize=opt.batchSize, patchSize=opt.patchsize).data
    # # if you want to validate
    # dataset_val = Dataset(train=False).data
    print("# of training samples: %d\n" % int(len(dataset_train)))

    toTensor = transforms.Compose([transforms.ToTensor()])
    toPILImg = transforms.ToPILImage()

    # Set Seeds
    # random.seed(1234)
    # np.random.seed(1234)
    # torch.manual_seed(1234)
    # torch.cuda.manual_seed_all(1234)

    # Build model
    net = SKSAK()
    # net.apply(weights_init_kaiming)
    criterion = nn.L1Loss()
    criterion2 = nn.MSELoss()

    # Move to GPU
    model = net.cuda()
    # summary(model, (3, 256, 256))
    criterion.cuda()
    criterion2.cuda()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr)

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=opt.last_lr)
    # scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=3, after_scheduler=scheduler_cos)
    scheduler.step()

    # training
    writer = SummaryWriter(opt.outf)
    step = 0

    # if you want to train resume
    load = torch.load(os.path.join(opt.logdir, 'SKT_AI_sr_52_0.029.pth'))
    model.load_state_dict(load)
    # optimizer.load_state_dict(load['optimizer'])

    start_time = datetime.now()
    print('Training Start!!')
    print(start_time)

    for epoch in range(opt.resume_epochs, opt.epochs):
        # setting lr
        if (epoch + 1) < 63:
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = opt.last_lr

        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %.8f' % current_lr)
        loss_val = 0
        # train
        for i, (degraded_train, gt_train) in enumerate(dataset_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            randomnum = random.randint(0, 2)

            # Make Edge Image.
            original_edge = torch.FloatTensor(degraded_train.shape[0], 3, opt.patchsize, opt.patchsize)

            for j in range(degraded_train.shape[0]):
                original_edge[j] = toTensor((toPILImg(gt_train[j]).filter(ImageFilter.FIND_EDGES)))

            if randomnum == 0:
                noise = torch.FloatTensor(degraded_train.shape[0], 1, degraded_train.shape[2], degraded_train.shape[3]).normal_(mean=0, std=5 / 255.)
                degraded_img = degraded_train + noise
            elif randomnum == 1:
                noise = torch.FloatTensor(degraded_train.shape[0], 1, degraded_train.shape[2], degraded_train.shape[3]).normal_(mean=0, std=10 / 255.)
                degraded_img = degraded_train + noise
            else:
                # noise = torch.FloatTensor(degraded_train.size()).normal_(mean=0, std=15 / 255.)
                noise = torch.FloatTensor(degraded_train.shape[0], 1, degraded_train.shape[2], degraded_train.shape[3]).normal_(mean=0, std=15 / 255.)
                degraded_img = degraded_train + noise


            degraded_img, gt_train = Variable(degraded_img.cuda()), Variable(gt_train.cuda())
            original_edge = Variable(original_edge.cuda())

            out_train = model(degraded_img)

            if epoch < 63:
                img_loss = np.sum([criterion(out_train[j], gt_train) for j in range(len(out_train)-1)])
                edge_loss = criterion(out_train[-1], original_edge)
            else:
                img_loss = np.sum([criterion2(out_train[j], gt_train) for j in range(len(out_train)-1)])
                edge_loss = criterion(out_train[-1], original_edge)

            loss = img_loss + 0.5 * edge_loss
            loss_val += loss.item()

            loss.backward()
            optimizer.step()

            # results
            model.eval()
            # out_train, stage2_img, stage1_img, edge = model(degraded_train)
            # out_train = torch.clamp(out_train, 0., 1.)
            out_train = model(degraded_img)
            out_img = torch.clamp(out_train[0], 0., 1.)
            edge_img = out_train[-1]
            psnr_train = batch_PSNR(out_img, gt_train, 1.)
            # i%100 == 0 -> each 100 epochs, print loss and psnr.
            if i % 500 == 0:
                print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                      (epoch + 1, i + 1, len(dataset_train), loss.item(), psnr_train))

            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if i % 2000 == 0:
                Img = utils.make_grid(gt_train[0].data, nrow=8, normalize=True, scale_each=True)
                Imgn = utils.make_grid(degraded_img[0].data, nrow=8, normalize=True, scale_each=True)
                edgeImg = utils.make_grid(edge_img[0].data, nrow=8, normalize=True, scale_each=True)
                Irecon = utils.make_grid(out_img[0].data, nrow=8, normalize=True, scale_each=True)
                # writer.add_image('clean image', Img, epoch)
                # writer.add_image('noisy image', Imgn, epoch)
                # writer.add_image('reconstructed image', Irecon, epoch)

                # Compare clean, degraded, restoration image
                fig = plt.figure()
                fig.suptitle('SKT_AI_ %d' % (epoch + 1))
                rows = 2
                cols = 2

                ax1 = fig.add_subplot(rows, cols, 1)
                ax1.imshow(np.transpose(Img.cpu(), (1, 2, 0)), cmap="gray")
                ax1.set_title('gt image')

                ax2 = fig.add_subplot(rows, cols, 2)
                ax2.imshow(np.transpose(Imgn.cpu(), (1, 2, 0)), cmap="gray")
                ax2.set_title('degraded image')

                ax3 = fig.add_subplot(rows, cols, 3)
                ax3.imshow(np.transpose(edgeImg.cpu(), (1, 2, 0)), cmap="gray")
                ax3.set_title('edge image')

                ax4 = fig.add_subplot(rows, cols, 4)
                ax4.imshow(np.transpose(Irecon.cpu(), (1, 2, 0)), cmap="gray")
                ax4.set_title('restoration image [%.4f %.4f]' % (loss.item(), psnr_train))

                plt.show()

            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1

        # the end of each epoch
        # model.eval()
        loss_val /= len(dataset_train)
        print("Average Loss : %.4f" % (loss_val))

        midtime = datetime.now() - start_time
        print(midtime)

        # torch.save({'state_dict': model.state_dict(),
        #             'optimizer': optimizer.state_dict()
        #             }, os.path.join(opt.outf, 'SKT_AI_' + str(epoch + 1) + "_" + str(round(loss_val, 4)) + '.pth'))
        torch.save(model.state_dict(),
                   os.path.join(opt.outf, 'SKT_AI_sr_' + str(epoch + 1) + "_" + str(round(loss_val, 4)) + '.pth'))

        scheduler.step()

    end_time = datetime.now()
    print('Training Finished!!')
    print(end_time)


if __name__ == "__main__":
    main()
