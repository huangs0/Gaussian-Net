# Gaussian-Net
This Neural Network is still in development and currently is only in preview. No Copyright are granted for that. 

## Author 
Huang Songlin, Student, Department of Computer Science, Faculty of Engineering, HKU, in charge of experiment and coding

Zheng Guang, Vice Professor, Faculty of EEE, HQU, in charge of 

## Environment
All the environment used are based on my another repository, all software and hardware setting are remained the same

## Reference

Ghostnet: see https://github.com/huawei-noah/ghostnet

## Introduction
In CVPR2020, Huawei Noah's GhostNet hit the Computer Vision, which introduce a new approach that use cheap operation in place of slow convolution operation to generate more feature map while take place of far less computation resource and save the time for model calculation based on mobile device like Huawei P30 in paper. In the original paper, Ghost Module first use the pointwise convolution to generate 1/n feature map, then use deepwise convolution as cheap operation to generate the other (n-1)/n feature map, finally use concate operation to integrate both feature maps. However this is very similar to Deep-wise convolution developed in Mobilenet by Google. Even more, whether or not the learnable parameters in deepwise convolution is a 'CHEAP' operation is still unclarified.

So we try to seek other cheap operation and find Gaussian Matrix, which is widely used in filter, sharpen images and integrated in OpenCV or PIL. Similar as GhostNet, we use pointwise convolution to generate 1/n parameter and use Gaussian Matrix to generate other (n-1)/n feature maps. For those n >= 2, we use Gaussian Matrix with different sigma value. 

Here we also provide a sample python program to generate Gaussian Matrix with different sigma, see `./get_gaussian_matrix.py`

## Gaussian Module
```Python
class Gaussian_Module(nn.Module):
    def __init__(self, in_channel, out_channel, gaussian_operations):
        #gaussian_operations is a list containing gaussian_operations
        super(Gaussian_Module,self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.bn_before = F.normalize
        self.pointwiseconv = nn.Conv2d(in_channels=self.in_channel,  out_channels=int(self.out_channel/(len(gaussian_operations)+1)),
                                       kernel_size=(1,1), stride=(1,1))
        self.bn_after = nn.BatchNorm2d(int(self.out_channel/(len(gaussian_operations)+1)),
                                       eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.activefun = nn.ReLU(inplace=True)
        self.gaussian_operations = gaussian_operations

    def forward(self,x):
        x = self.bn_before(x)
        x = self.pointwiseconv(x)
        x = self.bn_after(x)
        x = self.activefun(x)
        for gaussian_operation in self.gaussian_operations:
            ghost = F.conv2d(x, gaussian_operation[int(self.out_channel/(len(self.gaussian_operations)+1))],
                             stride=(1, 1), padding=(1, 1), groups=int(self.out_channel/(len(self.gaussian_operations)+1)))
            x = torch.cat((x,ghost),dim=1)
        return nn.BatchNorm2d(self.out_channel)(x)
```

## Recent BenchMark
GaussianNet is mainly focus on speed especially on mobile or embedded device, so we first measure the flops, parameter size and training time per epoch

We use `thop` to measure flops and parameter size. In different batch_size, thop will report same parameter size for unknown reason

When Batch_Size = 1:

```Shell
BN_Avg_ResNet:
100.334567424flops 24.043924G

GaussianNet:
6.669074432flops 1.516884G
```

When Batch_Size = 9:
```Shell
BN_Avg_ResNet:
903.011106816flops 24.043924G 146.2273964881897s

GaussianNet:
60.021669888flops 1.516884G 72.2591495513916s
```
