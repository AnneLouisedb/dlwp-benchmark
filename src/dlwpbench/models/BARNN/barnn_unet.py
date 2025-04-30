import torch as th
import torch.nn as nn
import einops
from hydra.utils import instantiate
from utils import CylinderPad
from utils import HEALPixLayer,HEALPixPadding, BARNNHEALPixLayer
from typing import Any, Dict, Optional, Sequence, Union
import torch.nn.functional as Fx

class _BaseBARNN(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.tol = 1e-18 # numerial stability, see https://github.com/pyg-team/pytorch_geometric/issues/2559
    def kl(self, alpha):
        scale = self.weight.shape.numel() + (self.bias.shape.numel() if self.bias is not None else 0)
        mean_alpha = alpha
        var_alpha = alpha.pow(2)
        mean_beta = mean_alpha.mean(0, keepdim=True)
        var_beta = var_alpha.mean(0, keepdim=True)
        kl = scale * 0.5 * ((mean_alpha-mean_beta).pow(2)/var_beta + (var_alpha/var_beta) - 1 - th.log(var_alpha/var_beta))
        return kl

class Conv2dBARNN(nn.Conv2d, _BaseBARNN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1):
        super(Conv2dBARNN, self).__init__(in_channels, out_channels, kernel_size, stride,
                                        padding, dilation, groups, False)
       
    def forward(self, input, alpha):
        """
        Forward with all regularized connections and random activations (Bayesian mode). Typically used for train
        """
     
        W = self.weight
        
        convo = Fx.conv2d(input, W, self.bias, self.stride, self.padding, self.dilation, self.groups)

        mean = alpha * convo
        var = alpha.pow(2) * Fx.conv2d(input.pow(2), W.pow(2), self.bias, self.stride, self.padding, self.dilation, self.groups)
        return mean + th.sqrt(var+self.tol)*th.randn_like(mean)


class AttentionBlock(th.nn.Module):
    """Attention block This is similar to [transformer multi-head
    attention](https://arxiv.org/abs/1706.03762).

    Args:
        n_channels: the number of channels in the input
        n_heads:  the number of heads in multi-head attention
        d_k: the number of dimensions in each head
        n_groups: the number of groups for [group normalization][torch.nn.GroupNorm]

    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: Optional[int] = None, n_groups: int = 1):
        """ """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = th.nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = th.nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = th.nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k**-0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: th.Tensor):
        
        batch_size, n_channels, height, width = x.shape 

        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        #Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = th.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = th.einsum("bihd,bjhd->bijh", q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=1)
        # Multiply by values
        res = th.einsum("bijh,bjhd->bihd", attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        #Add skip connection
        res += x

        #Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res


class ModernUNet(th.nn.Module):
    """
    A ModernUNet implementation as by the PDE Refiner Paper.
    Quote from paper: 'We also experimented with adding attention layers in the residual blocks, which, however, did not improve performance noticeably.'
    
    - Dropout: Dropout rate in the residual block
    """

    def __init__(
        self, 
        constant_channels: int = 4,
        prescribed_channels: int = 0,
        prognostic_channels: int = 1,
        hidden_channels: list = [64, 128, 256, 1024],
        activation: th.nn.Module = th.nn.GELU(),
        context_size: int = 1,
        mesh: str = "equirectangular",
        attention: bool = False,
        norm: bool = False, 
        recurrent:bool = False,
        dropout: float = 0.1,
        **kwargs
    ):
        super(ModernUNet, self).__init__()
        if isinstance(activation, str): activation = eval(activation)

        self.context_size = context_size
        self.mesh = mesh
        self.recurrent = recurrent

        print("initialize backbone...")
        # Define the parameters manually
        backbone_model_params = {
            "constant_channels": 2,
            "prescribed_channels": 1,
            "prognostic_channels": 13,
            "hidden_channels": [128, 128, 256, 1024],
            'activation': th.nn.GELU(),
            'context_size': 1,
            'norm': True,
            'mesh': 'healpix',
            'attention': False,
            'recurrent': False,
            'dropout': 0.0}

        self.backbone = None
        
        # # Instantiate the model directly
        # backbone_model = MUNetHPX(** backbone_model_params).to(device='cuda:0')
        # checkpoint_backbone = th.load('/home/adboer/dlwp-benchmark/src/dlwpbench/outputs/PushForward/checkpoints/PushForward_best.ckpt')
        # backbone_model.load_state_dict(checkpoint_backbone["model_state_dict"])
        # self.backbone = backbone_model

       
        in_channels = constant_channels + (prescribed_channels+prognostic_channels)*context_size

        c_in = in_channels
        c_hidden = 256
        c_out = 512

        layers = []
        
        # First layer
        layers.append(HEALPixLayer(layer=nn.Conv2d, in_channels=c_in, out_channels=c_hidden, kernel_size=3, padding=1))
        layers.append(nn.ReLU())

        # Second layer
        layers.append(HEALPixLayer(layer=nn.Conv2d, in_channels=c_hidden, out_channels=c_out, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        
        # Global Average Pooling
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))  # Output size is (1, 1)

        layers.append(HEALPixLayer(layer=nn.Conv2d, in_channels=c_out, out_channels=31, kernel_size=3, padding=1))
        layers.append(nn.Sigmoid()) 

        # Convert to nn.Sequential
        self.posterior = nn.Sequential(*layers)
     
   
        self.encoder = ModernUNetEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            activation=activation,
            attention = attention,
            mesh=mesh,
            dropout = dropout)

        self.middle = MiddleBlock(
            in_channels=hidden_channels[-1], 
            attention = attention,
            norm = norm,
            activation=activation,
            mesh=mesh,
            dropout = dropout
           
            )
      

        self.decoder = ModernUNetDecoder(
            hidden_channels=hidden_channels,
            out_channels=prognostic_channels,
            activation=activation,
            attention = attention,
            mesh=mesh,
            recurrent=recurrent,
            dropout = dropout
        )

    def _prepare_inputs(
        self,
        constants: th.Tensor = None,
        prescribed: th.Tensor = None,
        prognostic: th.Tensor = None
    ) -> th.Tensor:
        """
        return: Tensor of shape [B, (T*C), H, W] containing constants, prescribed, and prognostic/output variables
        """
        tensors = []
        if constants is not None: tensors.append(constants[:, 0])
        if prescribed is not None: tensors.append(einops.rearrange(prescribed, "b t c h w -> b (t c) h w"))
        if prognostic is not None: tensors.append(einops.rearrange(prognostic, "b t c h w -> b (t c) h w"))
        return th.cat(tensors, dim=1)

    def forward(
        self,
        constants: th.Tensor = None,
        prescribed: th.Tensor = None,
        prognostic: th.Tensor = None
    ) -> th.Tensor:
        """
        ...
        """
        # Shapes of inputs, where (F) is the optional face dimension when using HEALPix data
        # constants: [B, 1, C, (F), H, W]
        # prescribed: [B, T, C, (F), H, W]
        # prognostic: [B, T, C, (F), H, W]

       
        if self.mesh == "healpix": B, _, _, F, _, _ = prognostic.shape
        outs = []

    
        for t in range(self.context_size, prognostic.shape[1]):
            # For each t I want to store the loss in an array, and see how the prediction skill decreases over time
            
            t_start = max(0, t-(self.context_size))

            
                

            if t == self.context_size:
                # Initial condition
                

                # Needs a different prognostic_t when there is a backbone
                if self.backbone:
                    # BackBone # TAKE THE LAST PROGNOSTIC AND STACK DOUBLING THE DIMENSIONS
                    print("FEED FORWARD TO BACKBONE..")
                    stacked_tensor = th.stack([prognostic_t[:, t_start:t], prognostic_t[:, t_start:t]], dim=1) 

                    assert stacked_tensor.shape[1] == 2

                    # get the current presrcibed  and the 
                    with th.no_grad():
                        prognostic_t = self.backbone(
                                constants=constants if not constants == None else None,
                                prescribed=prescribed[:, t_start:t] if not prescribed_input == None else None,
                                prognostic=stacked_tensor) # take the last prognostic to forecast
                    print('output backbone', prognostic_t.shape)

                else:
                    prognostic_t = prognostic[:, t_start:t]
                    x_t = self._prepare_inputs(
                        constants=constants,
                        prescribed=prescribed[:, t_start:t] if prescribed is not None else None,
                        prognostic=prognostic_t 
                    )

                 
            else:
                
                if self.backbone:
                    # BackBone # TAKE THE LAST PROGNOSTIC AND STACK DOUBLING THE DIMENSIONS
                    print("FEED FORWARD TO BACKBONE..")
                    stacked_tensor = th.stack([prognostic_t[:,-1], prognostic_t[:,-1]], dim=1) 

                    assert stacked_tensor.shape[1] == 2

                    # get the current presrcibed  and the 
                    with th.no_grad():
                        prognostic_t = self.backbone(
                                constants=constants if not constants == None else None,
                                prescribed=prescribed if not prescribed_input == None else None,
                                prognostic=stacked_tensor) # take the last prognostic to forecast
                    print('output backbone', prognostic_t.shape)
                    
                else:
                
                    # In case of context_size > 1, blend prognostic input with outputs from previous time steps
                    prognostic_t = th.cat(
                        tensors=[prognostic[:, t_start:self.context_size],        # Prognostic input before context_size
                                th.stack(outs, dim=1)[:, -self.context_size:]],  # Outputs since context_size
                        dim=1
                    )
                    x_t = self._prepare_inputs(
                        constants=constants,
                        prescribed=prescribed[:, t-self.context_size:t] if prescribed is not None else None,
                        prognostic=prognostic_t
                    )
                
            

          
            p = self.posterior(x_t)
            alphas = p / (1 - p)

            enc, alphas = self.encoder(x_t, alphas)
            

            enc2, alphas = self.middle(enc[-1], alphas)
            
            out = self.decoder(x=enc2, skips=enc[::-1], alphas = alphas) 

            if self.mesh == "healpix": out = einops.rearrange(out, "(b f) tc h w -> b tc f h w", b=B, f=F)

            out = prognostic_t[:, -1] + out

            outs.append(out)

            # compute kls
            self.kl=0

            self.kl = sum(
                [sum(self.encoder.kl_values) + sum(self.middle.kl_values) + sum(self.decoder.kl_values)])
                

        return th.stack(outs, dim=1)

   
    
class BARNNMUNetHPX(ModernUNet):

    def __init__(self,
        constant_channels: int = 2,
        prescribed_channels: int = 1,
        prognostic_channels: int = 4,
        hidden_channels: list = [64, 128, 256, 1024],
        activation: th.nn.Module = th.nn.GELU(),
        context_size: int = 1,
        mesh: str = "healpix",
        attention: bool = False,
        norm: bool = False, 
        recurrent: bool = False,
        dropout: float = 0.1,
        **kwargs
    ):
        super(BARNNMUNetHPX, self).__init__(
            constant_channels=constant_channels,
            prescribed_channels=prescribed_channels,
            prognostic_channels=prognostic_channels,
            hidden_channels=hidden_channels,
            activation=activation,
            context_size=context_size,
            mesh="healpix",
            attention = attention,
            norm=norm,
            recurrent=recurrent,
            dropout = dropout,
            kwargs=kwargs

        )
    
    def _prepare_inputs(
        self,
        constants: th.Tensor = None,
        prescribed: th.Tensor = None,
        prognostic: th.Tensor = None
    ) -> th.Tensor:
        """
        return: Tensor of shape [(B*F), (T*C), H, W] containing constants, prescribed, and prognostic/output variables
        """
        tensors = []
        
        if constants is not None: tensors.append(einops.rearrange(constants[:, 0], "b c f h w -> (b f) c h w"))
        if prescribed is not None: tensors.append(einops.rearrange(prescribed, "b t c f h w -> (b f) (t c) h w"))
        if prognostic is not None:
            if prognostic.ndim == 6:
               
                tensors.append(einops.rearrange(prognostic, "b t c f h w -> (b f) (t c) h w")) 
            elif prognostic.ndim == 5:
                tensors.append(einops.rearrange(prognostic, "b t c h w -> b (t c) h w"))

        out = th.cat(tensors, dim=1)
        
        return out
    
class ModernUNetEncoder(th.nn.Module):
    """Unet encoder as used in the pde-refiner paper.
    - Each downblock combines the ResiduaBlock and the AttentionBlock??"""

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: list = [64, 128, 256, 1024],
        activation: th.nn.Module = th.nn.GELU(),
        attention: bool = False,
        mesh: str = "healpix",
        dropout: float = 0.1,
        
    ):
        super(ModernUNetEncoder, self).__init__()
        self.layers = []

        
        channels = [in_channels] + hidden_channels


        for c_idx in range(len(channels[:-1])):
            layer = []
            c_in = channels[c_idx]
            c_out = channels[c_idx+1]

            # Apply downsampling prior to convolutions if not in top-most layer
            if c_idx > 0: 
                
                layer.append(BARNNHEALPixLayer(layer=Conv2dBARNN,  in_channels=c_in, out_channels=c_in,kernel_size=3, stride=2, padding=1))
                
            # Image projection
            if c_idx == 0:  

                layer.append(HEALPixLayer(layer=th.nn.Conv2d , in_channels=c_in, out_channels=c_out,kernel_size=3, padding=1))
                c_in = c_out

            for _ in range(2):
                # STORE THIS OUTPUT
                if mesh == "equirectangular":
                    #layer.append(CylinderPad(padding=1))
                    # (1) norm (2) activation (3) convolution
                    
                    layer.append(ResidualBlock(
                        in_channels=c_in, 
                        out_channels=c_out,
                        kernel_size= 3,
                        padding = 1,
                        dropout = dropout))

                    # (4) Attention
                    layer.append(self.attn)

                elif mesh == "healpix":
                    layer.append(BARNNHEALPixLayer(
                        layer=ResidualBlock,
                        in_channels=c_in,
                        out_channels=c_out,
                        kernel_size= 3,
                        padding = 0, 
                        mesh=mesh,
                        dropout = dropout
                        ))

                    self.attn =  th.nn.Identity()

                    layer.append(self.attn)

                  
                c_in = c_out

                                             
            self.layers.append(th.nn.Sequential(*layer))

        self.layers = th.nn.ModuleList(self.layers)

    def forward(self, x: th.Tensor, alphas = None) -> list:
        # Store intermediate model outputs (per layer) for skip connections
        outs = []
        self.kl_values = []

        for idx, layer in enumerate(self.layers):

            for sublayer in layer:
               
                try:
                    alpha = alphas[:, 0, None] 

                    alphas = alphas[:, 1:]
                    
                    x = sublayer(x, alpha = alpha)

                    self.kl_values.append(sublayer.kl(alpha).mean())

                      
                except:

                    x = sublayer(x)


                if not isinstance(sublayer, (th.nn.Identity)):
                    outs.append(x)
                  
                  
        return outs, alphas
    

class ModernUNetDecoder(th.nn.Module):

    def __init__(
        self,
        hidden_channels: list = [64, 128, 256, 1024],
        out_channels: int = 2,
        activation: th.nn.Module = th.nn.GELU(),
        attention: bool = False,
        mesh: str = "healpix",
        recurrent: bool = False,
        dropout: float = 0.1,
    ):
        super(ModernUNetDecoder, self).__init__()
        self.layers = []
        final_out = hidden_channels[0] # due to skip connections
        hidden_channels = hidden_channels[::-1]  # Invert as we go up in decoder, i.e., from bottom to top layers

        self.activation = activation
        self.recurrent = recurrent
        
        
       

        for c_idx in range(len(hidden_channels)):
            
            layer = []
           
            c_out = hidden_channels[c_idx] 
            c_in_ = 2*hidden_channels[c_idx] 

            if c_idx < len(hidden_channels) - 1:
                c_next = hidden_channels[c_idx + 1]
            else:
                c_next = hidden_channels[c_idx]
           
            for _ in range(2):

                if mesh == "equirectangular":
                    layer.append(ResidualBlock(
                        in_channels=c_in_ , 
                        out_channels=c_out,
                        kernel_size= 3,
                        padding = 1,
                        dropout = dropout))

                    self.attn =  th.nn.Identity()
                    layer.append(self.attn)
                        
                elif mesh == "healpix":
                    
                    layer.append(BARNNHEALPixLayer(
                        layer=ResidualBlock,
                        in_channels=c_in_, 
                        out_channels=c_out, 
                        kernel_size=3, 
                        padding=0, 
                        mesh = mesh,
                        dropout = dropout
                    ))

                    self.attn = th.nn.Identity()
                    layer.append(self.attn)
                   
                    
            layer.append(BARNNHEALPixLayer(
                        layer=ResidualBlock,
                        in_channels=c_out+c_next,
                        out_channels=c_next,
                        kernel_size=3, 
                        padding=0, 
                        mesh = mesh
                    ))

            # Apply upsampling if not in top-most layer
            if c_idx < len(hidden_channels)-1: 
                layer.append(th.nn.ConvTranspose2d(c_next, c_next, (4, 4), (2, 2), (1, 1))) 
             
            self.layers.append(th.nn.Sequential(*layer))

        self.layers = th.nn.ModuleList(self.layers)

        # zero module?
        self.output_layer = HEALPixLayer(layer=th.nn.Conv2d, in_channels=c_next, out_channels=out_channels, kernel_size=3, padding=1).zero_out() 

        self.final_norm = th.nn.GroupNorm(8, final_out)

    def forward(self, x: th.Tensor, skips: list, alphas = None) -> th.Tensor:

        self.kl_values = []

        # Loop through each decoder layer
        for l_idx, layer in enumerate(self.layers):

            for block_idx, submodule in enumerate(layer):
              
                if isinstance(submodule,(th.nn.ConvTranspose2d, th.nn.Identity)):
                    x = submodule(x)

                # Apply skip connection at the block level
                elif isinstance(submodule, (ResidualBlock,HEALPixLayer,BARNNHEALPixLayer)) and l_idx < len(skips):
                    s = skips.pop(0)  # Get corresponding skip connection
                   
                    x = th.cat([s, x], dim=1)  # Concatenate skip connection with current feature map

                    try:
                        alpha = alphas[:, 0, None] 
                        alphas = alphas[:, 1:]
                        
                        x  = submodule(x, alpha = alpha)

                    
                        self.kl_values.append(submodule.kl(alpha).mean())
                    
                    except:

                        x = submodule(x)
                   
                else:
                    KeyError

        
        return self.output_layer(self.activation(self.final_norm(x)))

   
# BLOCKS
    
def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class ResidualBlock(th.nn.Module):
    """Wide Residual Blocks used in modern Unet architectures.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        cond_channels (int): Number of channels in the conditioning vector.
        activation (str): Activation function to use.
        norm (bool): Whether to use normalization.
        n_groups (int): Number of groups for group normalization. 
        dropout (float): Dropout rate used in the resblock
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation =  th.nn.GELU(),
        norm: bool = False, 
        n_groups: int = 1,
        kernel_size = 3,
        padding = 1,
        dropout: float = 0.1,
        mesh = 'healpix'
        
    ):
        super().__init__()
       
        self.activation = activation
        self.mesh = mesh
        self.dropout = th.nn.Dropout(dropout)
        
        # padding already provided 
        if self.mesh == 'healpix':
            padding = ((kernel_size - 1)//2) #*dilation
            self.cylinder_pad = HEALPixPadding(padding=padding)

        else:
            self.cylinder_pad = CylinderPad(padding=padding)

        self.conv1 = Conv2dBARNN(in_channels, out_channels, kernel_size=kernel_size, padding=0) 

        self.conv2 = zero_module(th.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=0)) 
        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = th.nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)) 
        else:
            self.shortcut = th.nn.Identity()

        if norm:
            self.norm1 = th.nn.GroupNorm(n_groups, in_channels)
            self.norm2 = th.nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = th.nn.Identity()
            self.norm2 = th.nn.Identity()

    def forward(self, x: th.Tensor, alpha = None):
        # First convolution layer
        h = self.activation(self.norm1(x))
        h = self.cylinder_pad(h)
        h = self.conv1(h, alpha)

        # Second convolution layer
        h = self.activation(self.norm2(h))
        h = self.dropout(h)
        h = self.cylinder_pad(h)
        h = self.conv2(h)
        # Add the shortcut connection and return
        
        return h + self.shortcut(x)

    def kl(self, alpha):
        return self.conv1.kl(alpha)



class MiddleBlock(th.nn.Module):
    """Middle block It combines a `ResidualBlock`, `AttentionBlock`, followed by another
    `ResidualBlock`.

    This block is applied at the lowest resolution of the U-Net.

    Args:
        n_channels (int): Number of channels in the input and output.
        has_attn (bool, optional): Whether to use attention block. Defaults to False.
        activation (str): Activation function to use. Defaults to "gelu".
        norm (bool, optional): Whether to use normalization. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        attention: bool = False,
        activation = th.nn.GELU(),
        norm: bool = False,
        mesh = 'healpix',
        dropout: float = 0.1,
         
    ):
        super().__init__()
        self.res1 = ResidualBlock(
            in_channels,
            in_channels,
            activation=activation,
            norm = norm,
            mesh = mesh,
            dropout=dropout)
        
        self.attn = th.nn.Identity()

        self.res2 = ResidualBlock(
            in_channels,
            in_channels,
            activation=activation,
            norm=norm,
            mesh=mesh,
            dropout=dropout)

    def forward(self, x: th.Tensor, alphas = None) -> th.Tensor:
        self.kl_values = []

        alpha= alphas[:, 0, None] 
        alphas = alphas[:, 1:]

        x = self.res1(x, alpha = alpha)
        self.kl_values.append(self.res1.kl(alpha).mean())

        x = self.attn(x)

        alpha= alphas[:, 0, None] 
        alphas = alphas[:, 1:]
        x = self.res2(x, alpha = alpha)
        self.kl_values.append(self.res2.kl(alpha).mean())

        return x, alphas