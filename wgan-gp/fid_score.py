import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import inception_v3
import numpy as np
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
import warnings
warnings.filterwarnings('ignore')


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    
    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3
    
    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        inception = inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output 
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


def get_activations(images, model, batch_size=50, dims=2048, device='cpu'):
    """Calculates the activations of the pool_3 layer for all images.
    
    Params:
    -- images      : torch.Tensor of shape (N, 3, H, W)
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > images.shape[0]:
        print(f'Warning: batch size is bigger than the data size. '
              f'Setting batch size to data size')
        batch_size = images.shape[0]

    pred_arr = np.empty((images.shape[0], dims))

    for i in range(0, images.shape[0], batch_size):
        start = i
        end = i + batch_size

        batch = images[start:end]
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start:end] = pred

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(images, model, batch_size=50, dims=2048, device='cpu'):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : torch.Tensor of images
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(images, model, batch_size, dims, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_fid_given_paths(real_images, generated_images, batch_size=50, device='cpu', dims=2048):
    """Calculates the FID of two datasets.
    Params:
    -- real_images       : torch.Tensor of real images
    -- generated_images  : torch.Tensor of generated images  
    -- batch_size        : Batch size to use
    -- device            : Device to run on
    -- dims              : Dimensionality of Inception features to use

    Returns:
    -- fid_value : The FID value between the two image sets
    """
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)
    
    # Normalize images to [0, 1] range if they are in [-1, 1]
    if real_images.min() < 0:
        real_images = (real_images + 1) / 2
    if generated_images.min() < 0:
        generated_images = (generated_images + 1) / 2
    
    # Ensure images have 3 channels
    if real_images.shape[1] == 1:
        real_images = real_images.repeat(1, 3, 1, 1)
    if generated_images.shape[1] == 1:
        generated_images = generated_images.repeat(1, 3, 1, 1)
    
    # Calculate statistics for real images
    m1, s1 = calculate_activation_statistics(real_images, model, batch_size, dims, device)
    
    # Calculate statistics for generated images
    m2, s2 = calculate_activation_statistics(generated_images, model, batch_size, dims, device)
    
    # Calculate FID
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    
    return fid_value


class FIDEvaluator:
    """FID Evaluator class for easy FID calculation"""
    
    def __init__(self, device='cpu', dims=2048, batch_size=50):
        self.device = device
        self.dims = dims
        self.batch_size = batch_size
        
        # Initialize Inception model
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx]).to(device)
        
        # Store real image statistics
        self.real_mu = None
        self.real_sigma = None
        
    def precompute_real_statistics(self, real_images):
        """Precompute statistics for real images"""
        # Normalize images to [0, 1] range if they are in [-1, 1]
        if real_images.min() < 0:
            real_images = (real_images + 1) / 2
            
        # Ensure images have 3 channels
        if real_images.shape[1] == 1:
            real_images = real_images.repeat(1, 3, 1, 1)
            
        self.real_mu, self.real_sigma = calculate_activation_statistics(
            real_images, self.model, self.batch_size, self.dims, self.device
        )
        
    def calculate_fid(self, generated_images):
        """Calculate FID score for generated images"""
        if self.real_mu is None or self.real_sigma is None:
            raise ValueError("Real image statistics not computed. Call precompute_real_statistics first.")
        
        # Normalize images to [0, 1] range if they are in [-1, 1]
        if generated_images.min() < 0:
            generated_images = (generated_images + 1) / 2
            
        # Ensure images have 3 channels
        if generated_images.shape[1] == 1:
            generated_images = generated_images.repeat(1, 3, 1, 1)
            
        # Calculate statistics for generated images
        gen_mu, gen_sigma = calculate_activation_statistics(
            generated_images, self.model, self.batch_size, self.dims, self.device
        )
        
        # Calculate FID
        fid_score = calculate_frechet_distance(self.real_mu, self.real_sigma, gen_mu, gen_sigma)
        
        return fid_score 