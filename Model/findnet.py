"""
FIND-Net: Fourier-Integrated Network with Dictionary Kernels for Metal Artifact Reduction

This implementation of FIND-Net extends the DICDNet framework for metal artifact reduction in CT images.
The architecture and certain components of this model are built upon the original DICDNet work, which is cited below.

Reference:
Wang, H., Li, Y., He, N., Ma, K., Meng, D., Zheng, Y. 
"DICDNet: Deep Interpretable Convolutional Dictionary Network for Metal Artifact Reduction in CT Images."
IEEE Trans. Med. Imaging, 41(4), 869–880, 2022.
DOI: 10.1109/TMI.2021.3127074

Modifications in FIND-Net include the integration of Fourier domain processing and trainable Gaussian filtering.
"""
import os
import os.path
import argparse
import numpy as np
import torch
import utils.save_image as save_img
from torch.utils.data import DataLoader
from Dataset.dataset import MARTrainDataset
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from datetime import date
from Model.findnet import FINDNet

parser = argparse.ArgumentParser(description="ACDNet_Test")
parser.add_argument("--data_path", type=str, default="your_path_to_dataset", help='txt path to training data')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--patchSize', type=int, default=512, help='the height / width of the input image to network')
parser.add_argument('--batchnum', type=int, default=702, help='the number of batch')
parser.add_argument("--save_path", type=str, default="save_results/", help='path to testing results')
parser.add_argument('--num_M', type=int, default=32, help='the number of feature maps')
parser.add_argument('--num_Q', type=int, default=32, help='the number of channel concatenation')
parser.add_argument('--T', type=int, default=3, help='the number of ResBlocks in every ProxNet')
parser.add_argument('--S', type=int, default=10, help='Stage number')
parser.add_argument('--etaM', type=float, default=1, help='stepsize for updating M')
parser.add_argument('--etaX', type=float, default=5, help='stepsize for updating X')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument('--log_dir', default='./logs/', help='tensorboard logs')
parser.add_argument('--model_dir', default=f'./models/false/', help='saving model')
parser.add_argument('--test_info', type=str, default='', help='test_info')
parser.add_argument('--save_imgs', action="store_true", help='save_imgs')
parser.add_argument('--checkpoint', type=str, default='checkpoint.pt', help='epoch_no')
parser.add_argument('--masked_eval', action="store_true", help='masked_eval')

opt = parser.parse_args()

today_date = date.today()

checkpoint_name = opt.checkpoint
opt.save_path = opt.save_path + str(today_date) + opt.test_info

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  " + path + "  ---")
    else:
        print("---  There exsits folder " + path + " !  ---")

mkdir(opt.save_path)
if opt.save_imgs:
    print(checkpoint_name.split('.')[0])
    out_dir = opt.save_path + f'/{opt.model_directory_name.split("/")[-1]}_Epoch{opt.epoch_no}/image/'
    out_hudir = opt.save_path + f'/{opt.model_directory_name.split("/")[-1]}_Epoch{opt.epoch_no}/hu/'
    mkdir(out_dir)
    mkdir(out_hudir)

    input_dir = opt.save_path + '/input/image/'
    input_hudir = opt.save_path + '/input/hu/'
    mkdir(input_dir)
    mkdir(input_hudir)

    gt_dir = opt.save_path + '/gt/image/'
    gt_hudir = opt.save_path+ '/gt/hu/'
    mkdir(gt_dir)
    mkdir(gt_hudir)

    li_dir = opt.save_path + '/li/image/'
    li_hudir = opt.save_path + '/li/hu/'
    mkdir(li_dir)
    mkdir(li_hudir)

    A_dir = opt.save_path + '/A/image/'
    A_hudir = opt.save_path + '/A/hu/'
    mkdir(A_dir)
    mkdir(A_hudir)



def normalized(X):
    maxX = np.max(X)
    minX = np.min(X)
    X = (X - minX) / (maxX - minX)
    return X

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

def image_get_minmax():
    return 0.0, 1.0

def normalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = data * 255.0
    data = data.astype(np.float32)
    data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)),0)
    return data

def tohu(X):
    # Convert from mm^-1 back to original units (assuming initial range and transformation)
    CT = (X * 5000) - 1500
    return CT


def calculate_metrics(gt, pred, max_val=3000):
    # Convert tensors to numpy arrays
    gt_np = gt.data.cpu().numpy().squeeze()
    pred_np = pred.data.cpu().numpy().squeeze()  

    # Calculate MAE
    mae = np.mean(np.abs(gt_np - pred_np))

    # Calculate SSIM
    ssim_val = ssim(gt_np, pred_np, data_range=max_val)
    
    # Calculate PSNR
    psnr_val = psnr(gt_np, pred_np, data_range=max_val)
    
    return mae, ssim_val, psnr_val


def main(datasets):
    data_loader = DataLoader(datasets, batch_size=opt.batchSize, shuffle=False)
    print('Loading model ...\n')
    model = FINDNet(opt).cuda()
    print("Model: ", opt.model_dir + checkpoint_name)
    model.load_state_dict(torch.load(opt.model_dir + checkpoint_name))
    model.eval()
    count = 0
    total_mae, total_ssim, total_psnr = 0, 0, 0
    total_mae_lst, total_ssim_lst, total_psnr_lst = [], [], []
    with open(opt.save_path + 'test_log_all.txt', 'a') as log_file:
                    # Redirecting print statements to the file
        print(f"file_name,mae,ssim,psnr", file=log_file)

    for imag_idx in range(1):
        print("imag_idx:", imag_idx)
        for ii, data in enumerate(data_loader):
            gt_filename, Xma, X, XLI, M = data
            gt_filename = gt_filename[0]
            Xma, X, XLI, M = Xma.cuda(), X.cuda(), XLI.cuda(), M.cuda()
            with torch.no_grad():
                if opt.use_gpu:
                    torch.cuda.synchronize()
                X0, ListX, ListA = model(Xma, XLI, M)
   
            Xgtclip = X / 255.0
            Xmaclip = Xma / 255.0
            XLIclip = XLI / 255.0
            Xoutclip = ListX[-1] / 255.0
            Aoutclip = ListA[-1] / 255.0

            Xoutnorm = Xoutclip
            Xgtnorm = Xgtclip
            Xmanorm = Xmaclip
            XLInorm = XLIclip
            ALInorm = Aoutclip

            Xouthu = tohu(Xoutclip)
            Xgthu = tohu(Xgtclip)
            Xmahu = tohu(Xmaclip)
            XLIhu = tohu(XLInorm)
            ALIhu = tohu(ALInorm)

            count += 1
            idx = os.path.splitext(gt_filename)[0]  # Use the image filename without extension as idx
            Xnorm = [Xoutnorm, Xmanorm, Xgtnorm, XLInorm, ALInorm]
            Xhu = [Xouthu, Xmahu, Xgthu, XLIhu, ALIhu]

            if opt.masked_eval: # This will calculate metrics only on the non-metal regions
                mae, ssim_val, psnr_val = calculate_metrics(Xgthu * M, Xouthu * M)
                Xnorm = [Xoutnorm, Xmanorm, Xgtnorm, XLInorm, ALInorm]
                Xhu = [Xouthu, Xmahu, Xgthu, XLIhu, ALIhu]
                if opt.save_imgs:
                    dir = [out_dir, input_dir, gt_dir, li_dir, A_dir]
                    hudir = [out_hudir, input_hudir, gt_hudir, li_hudir, A_hudir]
                    save_img.masked_imwrite(idx, dir, Xnorm, format='tiff', window_norm=True, mask=torch.tensor(M))
                    save_img.masked_imwrite(idx, hudir, Xhu, format='tiff', window_norm=True, mask=torch.tensor(M))
            else:
                mae, ssim_val, psnr_val = calculate_metrics(Xgthu, Xouthu)
                Xnorm = [Xoutnorm, Xmanorm, Xgtnorm, XLInorm, ALInorm]
                Xhu = [Xouthu, Xmahu, Xgthu, XLIhu, ALIhu]
                if opt.save_imgs:
                    dir = [out_dir, input_dir, gt_dir, li_dir, A_dir]
                    hudir = [out_hudir, input_hudir, gt_hudir, li_hudir, A_hudir]
                    save_img.imwrite(idx, dir, Xnorm, format='tiff', window_norm=True)
                    save_img.imwrite(idx, hudir, Xhu, format='tiff', window_norm=True)
            print(f'File: {gt_filename}  --> MAE: {mae}, SSIM: {ssim_val}, PSNR: {psnr_val}')

            total_mae += mae
            total_ssim += ssim_val
            total_psnr += psnr_val
            total_mae_lst.append(mae)
            total_ssim_lst.append(ssim_val)
            total_psnr_lst.append(psnr_val)
            
    # Open a file in append mode
    with open(opt.save_path + 'test_log.txt', 'a') as log_file:
        # Redirecting print statements to the file
        print("count:", count, file=log_file)
        print((opt.model_directory_name + "  Epoch" + str(opt.epoch_no)), file=log_file)
        print(f'MAE: {np.mean(total_mae_lst)}, SSIM: {np.mean(total_ssim_lst)}, PSNR: {np.mean(total_psnr_lst)}', file=log_file)
        print(100 * '*', file=log_file)

if __name__ == "__main__":
    test_dataset = MARTrainDataset(opt.data_path, opt.patchSize, int(opt.batchSize * opt.batchnum), mode='test', augment_mode = False)
    main(test_dataset)