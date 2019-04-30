import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

# CNN-based Variational Autoencoder

#torch.manual_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, height, width):
        super(VAE, self).__init__()

        self.height = height
        self.width = width
        self.hidden_channels = hidden_channels
        self.zdim = (self.height//32+1)*(self.width//32+1)
        print('Dim of z:',self.zdim)

        # Encoder Architecture
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels[0],
                               kernel_size=kernel_size, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(hidden_channels[0])
        self.conv2 = nn.Conv2d(in_channels=hidden_channels[0], out_channels=hidden_channels[1],
                               kernel_size=kernel_size, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(hidden_channels[1])
        self.conv3 = nn.Conv2d(in_channels=hidden_channels[1], out_channels=hidden_channels[2],
                               kernel_size=kernel_size, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(hidden_channels[2])
        self.conv4 = nn.Conv2d(in_channels=hidden_channels[2], out_channels=hidden_channels[3],
                               kernel_size=kernel_size, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(hidden_channels[3])
        # Convolution for mu and sigma
        self.conv5_mu = nn.Conv2d(in_channels=hidden_channels[3], out_channels=1,
                               kernel_size=kernel_size, padding=1, stride=2)
        self.bn5_mu = nn.BatchNorm2d(1)
        self.conv5_sig = nn.Conv2d(in_channels=hidden_channels[3], out_channels=1,
                               kernel_size=kernel_size, padding=1, stride=2)
        self.bn5_sig = nn.BatchNorm2d(1)

        self.relu = nn.ReLU(inplace=True)

        # Decoder Architecture
        self.deconv5 = nn.ConvTranspose2d(in_channels=1, out_channels=hidden_channels[3],
                                        kernel_size=kernel_size, stride=2, padding=1)
        self.dbn5 = nn.BatchNorm2d(hidden_channels[2])
        self.deconv4 = nn.ConvTranspose2d(in_channels=hidden_channels[3], out_channels=hidden_channels[2],
                                        kernel_size=kernel_size, stride=2, padding=1)
        self.dbn4 = nn.BatchNorm2d(hidden_channels[2])
        self.deconv3  = nn.ConvTranspose2d(in_channels=hidden_channels[2],  out_channels=hidden_channels[1],
                                           kernel_size=2, stride=2, padding=0)
        self.dbn3 = nn.BatchNorm2d(hidden_channels[1])
        self.deconv2 = nn.ConvTranspose2d(in_channels=hidden_channels[1], out_channels=hidden_channels[0],
                                          kernel_size=2, stride=2, padding=0)
        self.dbn2 = nn.BatchNorm2d(hidden_channels[0])
        self.deconv1 = nn.ConvTranspose2d(in_channels=hidden_channels[0], out_channels=input_channels,
                                          kernel_size=2, stride=2, padding=0)

        # Define the leaky relu activation function
        self.l_relu = nn.LeakyReLU(0.1)
        
        # Output Activation function
        self.sigmoid_output = nn.Sigmoid()

    def encode(self, x):
        # Encoding the input image to the mean and var of the latent distribution
        bs, _, _, _ = x.shape
        conv1 = self.conv1(x)
        conv1 = self.bn1(conv1)
        conv1 = self.l_relu(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.bn2(conv2)
        conv2 = self.l_relu(conv2)
        conv3 = self.conv3(conv2)
        conv3 = self.bn3(conv3)
        conv3 = self.l_relu(conv3)
        conv4 = self.conv4(conv3)
        conv4 = self.bn4(conv4)
        conv4 = self.l_relu(conv4)

        # calc mu and sigma
        conv5_mu = self.conv5_mu(conv4)
        conv5_mu = self.bn5_mu(conv5_mu)
        conv5_mu = self.l_relu(conv5_mu)
        #
        conv5_sig = self.conv5_sig(conv4)
        conv5_sig = self.bn5_sig(conv5_sig)
        conv5_sig = self.l_relu(conv5_sig)

        # flatten
        mu = conv5_mu.view((bs, -1))
        logvar = conv5_sig.view((bs, -1))
        
        #self.skip_values['conv1'] = conv1
        #self.skip_values['conv2'] = conv2
        #self.skip_values['conv3'] = conv3
        #self.skip_values['conv4'] = conv4
        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        # Decoding the image from the latent vector
        z = z.view((-1, 1, self.height//32+1, self.width//32+1))
        #
        z = self.deconv5(z)
        z = self.l_relu(z)
        ## Add skip connections
        #z = torch.cat([z, self.skip_values['conv3']])
        #
        z = self.deconv4(z)
        z = self.l_relu(z)
        ## Add skip connections
        #z = torch.cat([z, self.skip_values['conv3']])
        #
        z = self.deconv3(z)
        z = self.l_relu(z)
        ## Add skip connections
        #z = torch.cat([z, self.skip_values['conv3']])
        #
        z = self.deconv2(z)
        z = self.l_relu(z)
        ## Add skip connections
        #z = torch.cat([z, self.skip_values['conv3']])
        
        output = self.deconv1(z)
        output = self.sigmoid_output(output)
        
        return output

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

if __name__ == '__main__':
    # gradient check
    vaemodel = VAE(input_channels=12, hidden_channels=[16,32,64,64], kernel_size=3,
                   height=200, width=200).cuda()
    print('VAE model structure ----------------')
    print(vaemodel)
    loss_fn = torch.nn.MSELoss()

    input  = torch.randn(1, 12, 200, 200).cuda()
    target = torch.randn(1, 12, 200, 200).cuda()

    print('VAE number of parameters ----------------')
    for parameter in vaemodel.parameters():
        print(parameter.numel())
                     
    output,mu,logvar = vaemodel(input)
    #import pdb; pdb.set_trace()
    #res = torch.autograd.gradcheck(loss_fn, (output, target), raise_exception=True)
    print(res)
