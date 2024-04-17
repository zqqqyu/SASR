import torch
from torch import nn
import model.common as common
import torch.nn.functional as F
from moco.builder import MoCo
#   ddf_spatial cl
#   spatial attention x[0]
#5-5 new encoder
from ddf import ddf
import model.module_util as mutil

def make_model(args):
    return BlindSR(args)


class DA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction):
        super(DA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size
        self.kernel_combine ='mul'
        

        self.kernel_channel = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 64 * self.kernel_size * self.kernel_size, bias=False)
        )
        
        self.kernel_spatial = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, self.kernel_size * self.kernel_size, 1)
        )
        
        self.conv = common.default_conv(channels_in, channels_out, 1)
        self.ca = CA_layer(channels_in, channels_out, reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

        self.relu = nn.LeakyReLU(0.1, True)
        #self.weight =

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
         param x[2]: degradation representation: B * C * w * h
        '''
        b, c, h, w = x[0].size()

        # branch 1
        kernel_channel = self.kernel_channel(x[1]).view(b, c, self.kernel_size, self.kernel_size) # b*g, c//g, k, k
        #kernel_channel = torch.ones(b, c, self.kernel_size, self.kernel_size).cuda()
        #print(x[2].size())
        kernel_spatial = self.kernel_spatial(x[2]).view(b, -1, h, w) # (b*g, -1, h//s, w//s)
        #print(kernel_spatial.size())
        out = ddf(x[0], kernel_channel, kernel_spatial,
                  self.kernel_size, 1, 1, self.kernel_combine)
        out = out.reshape(b, c, h, w)
        #out = self.relu(F.conv2d(x[0].view(1, -1, h, w), kernel, groups=b*c, padding=(self.kernel_size-1)//2))
        #out = self.conv(out.view(b, -1, h, w))  
        out = self.conv(out.view(b, -1, h, w))+x[0]
        
        out1 = self.ca(x)
        out2 = x[0] * self.sa(x[0])

        # branch 2
        out = out + out1 +out2

        return out


class CA_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels_in, channels_in//reduction, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_in // reduction, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        att = self.conv_du(x[1][:, :, None, None])

        return x[0] * att


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class DAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction):
        super(DAB, self).__init__()

        self.da_conv1 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.da_conv2 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)
       # self.conv3 = conv(64, 32, 1)
       # self.conv4 = conv(32, 64, 1)

        self.relu =  nn.LeakyReLU(0.1, True)
        
       # self.RDB1 = ResidualDenseBlock_5C(32, 16)
       # self.RDB2 = ResidualDenseBlock_5C(32, 16)
       # self.RDB3 = ResidualDenseBlock_5C(32, 16)
       # self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''

        out = self.relu(self.da_conv1(x))
        out = self.relu(self.conv1(out))
        out = self.relu(self.da_conv2([out, x[1], x[2]]))
        out = self.conv2(out) + x[0]

        #out1 = self.conv3(out)
        #out1 = self.RDB1(out1)
        #out1 = self.RDB2(out1)
       # out1 = self.RDB3(out1)
        
        #out1 = self.conv4(out1)

     #   return out1*0.2+out
        return out


class DAG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_blocks):
        super(DAG, self).__init__()
        self.n_blocks = n_blocks
        modules_body = [
            DAB(conv, n_feat, kernel_size, reduction) \
            for _ in range(n_blocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        res = x[0]
        for i in range(self.n_blocks):
            res = self.body[i]([res, x[1], x[2]])
        res = self.body[-1](res)
        res = res + x[0]

        return res


class SASR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SASR, self).__init__()

        self.n_groups = 5  #5
        n_blocks = 5   #5
        n_feats = 64
        kernel_size = 3
        reduction = 8
        scale = int(args.scale[0])

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0) 
        self.sub_mean = common.MeanShift(255.0, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(255.0, rgb_mean, rgb_std, 1)

        # head module
        modules_head = [conv(3, n_feats, kernel_size)]
        self.head = nn.Sequential(*modules_head)

        # compress
        self.compress = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(0.1, True)
        )
        


        # body
        modules_body = [
            DAG(common.default_conv, n_feats, kernel_size, reduction, n_blocks) \
            for _ in range(self.n_groups)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

        # tail
        modules_tail = [common.Upsampler(conv, scale, n_feats, act=False),
                        conv(n_feats, 3, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, k_v, fead):
        k_v = self.compress(k_v)

        # sub mean
        x = self.sub_mean(x)

        # head
        x = self.head(x)

        # body
        res = x
        for i in range(self.n_groups):
            res = self.body[i]([res, k_v, fead])
        res = self.body[-1](res)
        res = res + x

        # tail
        x = self.tail(res)

        # add mean
        x = self.add_mean(x)

        return x
        
        
        
class ResidualDenseBlock_5C(nn.Module):
    '''  Residual Dense Block '''
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x        

        
        

class LA_conv(nn.Module):
    def __init__(self, n_feat, kernel_size):
        super(LA_conv, self).__init__()
        
        self.kernel_size = kernel_size
        self.query_conv = nn.Linear(256, 64, bias=True)
        self.key_conv = nn.Conv2d(64, 64,1)
        self.value_conv = nn.Conv2d(64, 64,1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
         #       nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
         param x[2]: degradation representation: B * C * w * h
        '''
         
        b, c, h, w = x[0].size()
        proj_query = self.query_conv(x[1]).unsqueeze(dim=1)  # (B,1, C)
        proj_key = self.key_conv(x[0]).view((b,c,h*w)) # (B,c,(W*H))
        score = torch.matmul(proj_query, proj_key)#(b,1,(w*h))
        score = self.softmax(score)
        
        proj_value = self.value_conv(x[0])   # B , C,w,h
        
        out = proj_value*score.view(b,1,h,w)

        out = self.gamma * out + x[0]

      #  if self.activation is not None:
      #      out = self.activation(out)

        return out  # , attention



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        

        self.E1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            )
            
        self.E2 = nn.Sequential(  
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            )
            
        self.E3 = nn.Sequential( 
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
        )
        
       
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )
        
        self.d_u  = nn.Sequential(
               nn.Conv2d(256, 256, kernel_size=3, padding=1),
               nn.BatchNorm2d(256),
               nn.LeakyReLU(0.1, True),
               )
        
        self.u1 = nn.Sequential(
                nn.Conv2d(512, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128*4, 3, 1, 1),
                nn.BatchNorm2d(128*4),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(upscale_factor=2),        
            )      
            
        self.u2 = nn.Sequential(
                nn.Conv2d(256, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64*4, 3, 1, 1),
                nn.BatchNorm2d(64*4),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(upscale_factor=2),        
            )      
        self.out_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.LA = LA_conv(64,3)
        
        #self.up = nn.Upsample(scale_factor=12, mode='bilinear')
                
    def forward(self, x):
       
        fea1 = self.E1(x)  #64
        fea2 = self.E2(fea1)   #128
        fea3 = self.E3(fea2)    #256
        #print(fea3.shape)
        fea = self.avgpool(fea3).squeeze(-1).squeeze(-1)
        #fea_ = self.avgpool(fea3)
        out = self.mlp(fea)
       
        
       # fea33 = self.up(fea_)
        f4 = self.d_u(fea3)
        #f4 = self.d_u(fea3)     #256
        f5 = self.u1(torch.cat([fea3, f4], dim=1))   #128 
        f6 = self.u2(torch.cat([fea2, f5], dim=1))   #64
        f7 = self.out_conv(torch.cat([fea1, f6], dim=1)) #128
        
        
        local = self.LA([f7,fea])
        
        return fea, out, f7





'''
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.E = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)

        return fea, out
'''

class BlindSR(nn.Module):
    def __init__(self, args):
        super(BlindSR, self).__init__()

        # Generator
        self.G = SASR(args)

        # Encoder
        self.E = MoCo(base_encoder=Encoder)

    def forward(self, x):

        if self.training:
            x_query = x[:, 0, ...]                          # b, c, h, w
            x_key = x[:, 1, ...]                            # b, c, h, w

            # degradation-aware represenetion learning
            fea, logits, labels, fead = self.E(x_query, x_key)

            # degradation-aware SR
            sr = self.G(x_query, fea,fead)
           

            return sr, logits, labels
        else:
            # degradation-aware represenetion learning
            fea, fead = self.E(x, x)

            # degradation-aware SR
            sr = self.G(x, fea,fead)
          

            return sr
