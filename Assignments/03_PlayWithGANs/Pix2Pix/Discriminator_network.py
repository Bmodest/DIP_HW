import torch.nn as nn

class DiscriminatorNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.netD = defineD_n_layers(3, 3, 64, 3)

    def forward(self, x):
        return self.netD(x)



def defineD_n_layers(input_nc, output_nc, ndf, n_layers):
    netD = nn.Sequential()

    netD.add_module('conv1', nn.Conv2d(input_nc + output_nc, ndf, kernel_size=4, stride=2, padding=1, bias=False))
    netD.add_module('leaky_relu1', nn.LeakyReLU(0.2, inplace=True))

    nf_mult = 1
    nf_mult_prev = 1
    for n in range(1, n_layers):
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        netD.add_module(f'conv{n+1}', nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=False))
        netD.add_module(f'batch_norm{n+1}', nn.BatchNorm2d(ndf * nf_mult))
        netD.add_module(f'leaky_relu{n+1}', nn.LeakyReLU(0.2, inplace=True))

    nf_mult_prev = nf_mult
    nf_mult = min(2 ** n_layers, 8)
    netD.add_module(f'conv{n_layers+1}', nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=False))
    netD.add_module(f'batch_norm{n_layers+1}', nn.BatchNorm2d(ndf * nf_mult))
    netD.add_module(f'leaky_relu{n_layers+1}', nn.LeakyReLU(0.2, inplace=True))
    
    netD.add_module(f'conv{n_layers+2}', nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1, bias=False))
 

    netD.add_module('sigmoid', nn.Sigmoid())

    return netD
