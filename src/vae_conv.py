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
    def __init__(self, input_channels, hidden_channels, kernel_size, height, width, z_dimension):
        super(VAE, self).__init__()

        #hidden_channels = [16,32,64,64]
        self.height = height
        self.width = width
        self.z_dimension = z_dimension
        self.hidden_channels = hidden_channels

        # Encoder Architecture
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels[0],
                               kernel_size=kernel_size, padding=0, stride=2)
        self.bn1 = nn.BatchNorm2d(hidden_channels[0])
        self.conv2 = nn.Conv2d(in_channels=hidden_channels[0], out_channels=hidden_channels[1],
                               kernel_size=kernel_size, padding=0, stride=2)
        self.bn2 = nn.BatchNorm2d(hidden_channels[1])
        self.conv3 = nn.Conv2d(in_channels=hidden_channels[1], out_channels=hidden_channels[2],
                               kernel_size=kernel_size, padding=0, stride=2)
        self.bn3 = nn.BatchNorm2d(hidden_channels[2])
        self.conv4 = nn.Conv2d(in_channels=hidden_channels[2], out_channels=hidden_channels[3],
                               kernel_size=kernel_size, padding=0, stride=2)
        self.bn4 = nn.BatchNorm2d(hidden_channels[3])

        self.hidden = (self.height//16)*(self.width//16)*hidden_channels[3]
        print('Dim of in_features:',self.hidden)
        # Size of input features = HxWx2C
        #self.linear1 = nn.Linear(in_features=self.height//16*self.width//16*hidden_channels[3],
        #                         out_features=self.hidden)
        #self.bn_l = nn.BatchNorm1d(self.hidden)
        self.latent_mu = nn.Linear(in_features=self.hidden, out_features=self.z_dimension)
        self.latent_logvar = nn.Linear(in_features=self.hidden, out_features=self.z_dimension)
        self.relu = nn.ReLU(inplace=True)

        # Decoder Architecture
        self.linear1_decoder = nn.Linear(in_features=self.z_dimension,
                                         out_features=self.hidden)
        self.conv5 = nn.ConvTranspose2d(in_channels=hidden_channels[3], out_channels=hidden_channels[2],
                                        kernel_size=kernel_size, stride=2, padding=0)
        self.bn5 = nn.BatchNorm2d(hidden_channels[2])
        self.conv6  = nn.ConvTranspose2d(in_channels=hidden_channels[2],  out_channels=hidden_channels[1],
                                         kernel_size=kernel_size, stride=2, padding=0)
        self.bn6 = nn.BatchNorm2d(hidden_channels[1])
        self.conv7 = nn.ConvTranspose2d(in_channels=hidden_channels[1], out_channels=hidden_channels[0],
                                        kernel_size=kernel_size, stride=2, padding=0)
        self.bn7 = nn.BatchNorm2d(hidden_channels[0])
        self.output = nn.ConvTranspose2d(in_channels=hidden_channels[0], out_channels=input_channels,
                                         kernel_size=kernel_size, stride=2, padding=0)

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
        
        fl = conv4.view((bs, -1))
        
        mu = self.latent_mu(fl)
        logvar = self.latent_logvar(fl)
        
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
        z = self.linear1_decoder(z)
        z = self.l_relu(z)
        z = z.view((-1, self.hidden_channels[3], self.height//16, self.width//16))
        z = self.conv5(z)
        z = self.l_relu(z)
        ## Add skip connections
        #z = torch.cat([z, self.skip_values['conv3']])
    
        z = self.conv6(z)
        z = self.l_relu(z)
        ## Add skip connections
        #z = torch.cat([z, self.skip_values['conv2']])
        
        z = self.conv7(z)
        z = self.l_relu(z)
        ## Add skip connections
        # z = torch.cat([z, self.skip_values['conv1']])
        
        output = self.output(z)
        output = self.sigmoid_output(output)
        
        return output
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

if __name__ == '__main__':
    # gradient check
    vaemodel = VAE(input_channels=12, hidden_channels=[16,32,64,64], kernel_size=3,
                   height=200, width=200, z_dimension=100).cuda()
    print('VAE model structure ----------------')
    print(vaemodel)
    loss_fn = torch.nn.MSELoss()

    input = (torch.randn(1, 12, 200, 200)).cuda()
    #target = Variable(torch.randn(1, 32, 64, 32)).cuda()
                     
    output = vaemodel(input)
    import pdb; pdb.set_trace()
    #output = output[0][0]
    #res = torch.autograd.gradcheck(loss_fn, (output, target), raise_exception=True)
    #print(res)
