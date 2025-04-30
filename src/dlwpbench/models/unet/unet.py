# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch as th
import einops
from hydra.utils import instantiate
from utils import CylinderPad
from utils import HEALPixLayer,HEALPixPadding
from typing import Any, Dict, Optional, Sequence, Union
from models.convlstm.convlstm import ConvLSTMCell


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
       
        in_channels = constant_channels + (prescribed_channels+prognostic_channels)*context_size
        
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
                prognostic_t = prognostic[:, t_start:t]
                x_t = self._prepare_inputs(
                    constants=constants,
                    prescribed=prescribed[:, t_start:t] if prescribed is not None else None,
                    prognostic=prognostic_t 
                )
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

            enc = self.encoder(x_t)

            enc2 = self.middle(enc[-1])

            out = self.decoder(x=enc2, skips=enc[::-1]) 

            if self.mesh == "healpix": out = einops.rearrange(out, "(b f) tc h w -> b tc f h w", b=B, f=F)

            out = prognostic_t[:, -1] + out

            outs.append(out)

        return th.stack(outs, dim=1)
    
class MUNetHPX(ModernUNet):

    def __init__(self,
        constant_channels: int = 4,
        prescribed_channels: int = 0,
        prognostic_channels: int = 1,
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
        super(MUNetHPX, self).__init__(
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
                layer.append(HEALPixLayer(layer=th.nn.Conv2d, in_channels=c_in, out_channels=c_in,kernel_size=3, stride=2, padding=1))
                
            # Image projection
            if c_idx == 0: 
                layer.append(HEALPixLayer(layer=th.nn.Conv2d, in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1))
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
                    layer.append(HEALPixLayer(
                        layer=ResidualBlock,
                        in_channels=c_in,
                        out_channels=c_out,
                        kernel_size= 3,
                        padding = 0, 
                        mesh=mesh,
                        dropout = dropout
                        ))

                    self.attn = AttentionBlock(c_out) if attention else th.nn.Identity()

                    layer.append(self.attn)

                  
                c_in = c_out

                                             
            self.layers.append(th.nn.Sequential(*layer))

        self.layers = th.nn.ModuleList(self.layers)

    def forward(self, x: th.Tensor) -> list:
        # Store intermediate model outputs (per layer) for skip connections
        outs = []
        for layer in self.layers:

            for sublayer in layer:
                x = sublayer(x)
                if not isinstance(sublayer, (AttentionBlock, th.nn.Identity)):
                    outs.append(x)
                            
        return outs
    
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

                    self.attn = AttentionBlock(c_out) if attention else th.nn.Identity()
                    layer.append(self.attn)
                        
                elif mesh == "healpix":
                    layer.append(HEALPixLayer(
                        layer=ResidualBlock,
                        in_channels=c_in_, 
                        out_channels=c_out, 
                        kernel_size=3, 
                        padding=0, 
                        mesh = mesh,
                        dropout = dropout
                    ))

                    self.attn = AttentionBlock(c_out) if attention else th.nn.Identity()
                    layer.append(self.attn)
                   
                    
            layer.append(HEALPixLayer(
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

    def forward(self, x: th.Tensor, skips: list) -> th.Tensor:

        # Loop through each decoder layer
        for l_idx, layer in enumerate(self.layers):

            for block_idx, submodule in enumerate(layer):
              
                if isinstance(submodule,(th.nn.ConvTranspose2d, AttentionBlock, th.nn.Identity)):
                    x = submodule(x)

                # Apply skip connection at the block level
                elif isinstance(submodule, (ResidualBlock,HEALPixLayer)) and l_idx < len(skips):
                    s = skips.pop(0)  # Get corresponding skip connection
                    x = th.cat([s, x], dim=1)  # Concatenate skip connection with current feature map

                    # Pass through the current submodule/block
                    x = submodule(x)
                else:
                    KeyError

        # Apply final normalization, activation, and output layer
        
        return self.output_layer(self.activation(self.final_norm(x)))

# BLOCKS
    
def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


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
        # Get shape
        # batch_size, n_channels, face, height, width = x.shape # I ADDED A FACE
        # # Change `x` to shape `[batch_size, seq, n_channels]`

        # # Reshape to [batch_size, face, seq_per_face, n_channels]
        # x_reshaped = x.view(batch_size, n_channels, face, -1).permute(0, 2, 3, 1)
        # seq_per_face = x_reshaped.shape[2]  # height * width

        # # Project QKV for all faces at once
        # qkv = self.projection(x_reshaped)  # [batch, face, seq, 3 * n_heads * d_k]
        # qkv = qkv.view(batch_size, face, seq_per_face, self.n_heads, 3 * self.d_k)
        # q, k, v = torch.chunk(qkv, 3, dim=-1)  # Each [batch, face, seq, heads, d_k]

        # # Scaled dot-product attention per face
        # attn = torch.einsum('bfihd,bfjhd->bfijh', q, k) * self.scale
        # attn = attn.softmax(dim=2)  # Softmax over sequence dimension

        # # Aggregate across sequence
        # res = torch.einsum('bfijh,bfjhd->bfihd', attn, v)

        # # Combine heads and project
        # res = res.view(batch_size, face, seq_per_face, self.n_heads * self.d_k)
        # res = self.output(res)  # [batch, face, seq, n_channels]

        # # Reshape back to original dimensions
        # res = res.permute(0, 3, 1, 2).view(batch_size, n_channels, face, height, width)
        # res += x  # Skip connection

        # # After face-specific attention
        # cross_face_q = res.mean(dim=2)  # [batch, face, n_channels]
        # cross_attn = torch.einsum('bfc,bgc->bfg', cross_face_q, cross_face_q) * self.scale
        # cross_attn = cross_attn.softmax(dim=-1)
        # res = torch.einsum('bfg,bgcsh->bfcsh', cross_attn, res)

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


class FourierResidualBlock(th.nn.Module):
    """Fourier Residual Block to be used in modern Unet architectures.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        modes1 (int): Number of modes in the first dimension.
        modes2 (int): Number of modes in the second dimension.
        activation (str): Activation function to use.
        norm (bool): Whether to use normalization.
        n_groups (int): Number of groups for group normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int = 16,
        modes2: int = 16,
        activation = th.nn.GELU(),
        norm: bool = False,
        n_groups: int = 1,
    ):
        super().__init__()
        self.activation = activation
        self.modes1 = modes1
        self.modes2 = modes2

        self.fourier1 = SpectralConv2d(in_channels, out_channels, modes1=self.modes1, modes2=self.modes2)
        self.conv1 = th.nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, padding_mode="zeros")
        self.fourier2 = SpectralConv2d(out_channels, out_channels, modes1=self.modes1, modes2=self.modes2)
        self.conv2 = th.nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, padding_mode="zeros")
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

    def forward(self, x: th.Tensor):
        # using pre-norms
        h = self.activation(self.norm1(x))
        x1 = self.fourier1(h)
        x2 = self.conv1(h)
        out = x1 + x2
        out = self.activation(self.norm2(out))
        x1 = self.fourier2(out)
        x2 = self.conv2(out)
        out = x1 + x2 + self.shortcut(x)
        return out


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

        self.conv1 = th.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0) 
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

    def forward(self, x: th.Tensor):
        # First convolution layer
        h = self.activation(self.norm1(x))
        h = self.cylinder_pad(h)
        h = self.conv1(h)

        
        # Second convolution layer
        h = self.activation(self.norm2(h))
        h = self.dropout(h)
        h = self.cylinder_pad(h)
        h = self.conv2(h)
        # Add the shortcut connection and return
        
        return h + self.shortcut(x)


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
        
        self.attn = AttentionBlock(in_channels) if attention else th.nn.Identity()

        self.res2 = ResidualBlock(
            in_channels,
            in_channels,
            activation=activation,
            norm=norm,
            mesh=mesh,
            dropout=dropout)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.res1(x)
        x = self.attn(x)
        x = self.res2(x)
        return x





class ConvNeXtLSTMBlock(th.nn.Module):
    def __init__(
            self,
            geometry_layer: th.nn.Module = HEALPixLayer,
            in_channels: int = 1,
            h_channels: int = 1,
            kernel_size: int = 7,
            dilation: int = 1,
            activation: th.nn.Module = None,
            dropout: float = 0.,
            enable_nhwc: bool = False,
            enable_healpixpad: bool = False):
        '''
        :param x_channels: Input channels
        :param h_channels: Latent state channels
        :param kernel_size: Convolution kernel size
        :param activation_fn: Output activation function
        '''
        super().__init__()
        h_channels = in_channels
        conv_channels = in_channels + h_channels
        #spatial mixing
        self.to_latent = th.nn.Sequential(
            geometry_layer(
                layer="torch.nn.Conv2d",
                in_channels=conv_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding="same",
                groups=in_channels,
                #enable_nhwc=enable_nhwc,
                #enable_healpixpad=enable_healpixpad
                ),
            geometry_layer(
                layer="torch.nn.GroupNorm",
                num_channels=in_channels,
                num_groups=1,
                affine=True,
                #enable_nhwc=enable_nhwc,
                #enable_healpixpad=enable_healpixpad
                ),
            geometry_layer(
                layer="torch.nn.Conv2d",
                in_channels=in_channels,
                out_channels=4*in_channels,
                kernel_size=1,
                #enable_nhwc=enable_nhwc,
                #enable_healpixpad=enable_healpixpad
                ),
            geometry_layer(
                layer="torch.nn.GroupNorm",
                num_channels=4*in_channels,
                num_groups=4, 
                affine=True,
                #enable_nhwc=enable_nhwc,
                #enable_healpixpad=enable_healpixpad
                )
        )
        #output activation
        self.to_output = th.nn.Sequential(
            geometry_layer(
                layer="torch.nn.Conv2d",
                in_channels=h_channels,
                out_channels=h_channels,
                kernel_size=1,
                #enable_nhwc=enable_nhwc,
                #enable_healpixpad=enable_healpixpad
                ),
            geometry_layer(
                layer="torch.nn.GroupNorm",
                num_channels=h_channels,
                num_groups=1, 
                affine=True,
                #enable_nhwc=enable_nhwc,
               #enable_healpixpad=enable_healpixpad
                ),
            
            th.nn.GELU()
        )
        #dropout
        self.dropout = th.nn.Dropout(dropout)

        # Latent states
        self.h = th.zeros(1)
        self.c = th.zeros(1)

    
    def forward(self, inputs: Sequence) -> Sequence:
        '''
        LSTM forward pass
        :param inputs: Input
        '''
        if inputs.shape != self.h.shape:
            self.h = th.zeros_like(inputs)
            self.c = th.zeros_like(inputs)

        # Spatial mixing
        z = th.cat((inputs, self.h), dim = 1) if inputs is not None else self.h
        z = self.to_latent(z)
        # LSTM activation
        f, i, g, o = einops.rearrange(z, 'b (gates c) h w -> gates b c h w', gates = 4) #forget gate, input gate, g, output gate
        cell = th.sigmoid(f) * self.c + th.sigmoid(i) * self.dropout(th.tanh(g))
        hidden = th.sigmoid(o) * self.to_output(self.c)
        
        self.h = hidden
        self.c = cell


        return hidden

    def reset(self):
        self.h = th.zeros_like(self.h)
        self.c = th.zeros_like(self.c)




############### UNET ######################





class UNet(th.nn.Module):
    """
    A UNet implementation.
    """

    def __init__(
        self, 
        constant_channels: int = 4,
        prescribed_channels: int = 0,
        prognostic_channels: int = 1,
        hidden_channels: list = [8, 16, 32],
        n_convolutions: int = 2,
        activation: th.nn.Module = th.nn.ReLU(),
        context_size: int = 1,
        mesh: str = "equirectangular",
        **kwargs
    ):
        super(UNet, self).__init__()
        if isinstance(activation, str): activation = eval(activation)

        self.context_size = context_size
        self.mesh = mesh
        in_channels = constant_channels + (prescribed_channels+prognostic_channels)*context_size

        self.encoder = UNetEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            n_convolutions=n_convolutions,
            activation=activation,
            mesh=mesh
        )
        self.decoder = UNetDecoder(
            hidden_channels=hidden_channels,
            out_channels=prognostic_channels,
            n_convolutions=n_convolutions,
            activation=activation,
            mesh=mesh
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
            
            t_start = max(0, t-(self.context_size))
            if t == self.context_size:
                # Initial condition
                prognostic_t = prognostic[:, t_start:t]
                x_t = self._prepare_inputs(
                    constants=constants,
                    prescribed=prescribed[:, t_start:t] if prescribed is not None else None,
                    prognostic=prognostic_t
                )
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

            # Forward input through model
            enc = self.encoder(x_t)
            out = self.decoder(x=enc[-1], skips=enc[::-1])
            if self.mesh == "healpix": out = einops.rearrange(out, "(b f) tc h w -> b tc f h w", b=B, f=F)
            out = prognostic_t[:, -1] + out
            outs.append(out)
        
        return th.stack(outs, dim=1)


class UNetHPX(UNet):

    def __init__(
        self,
        constant_channels: int = 4,
        prescribed_channels: int = 0,
        prognostic_channels: int = 1,
        hidden_channels: list = [8, 16, 32],
        n_convolutions: int = 2,
        activation: th.nn.Module = th.nn.ReLU(),
        context_size: int = 1,
        mesh: str = "healpix",
        **kwargs
    ):
        super(UNetHPX, self).__init__(
            constant_channels=constant_channels,
            prescribed_channels=prescribed_channels,
            prognostic_channels=prognostic_channels,
            hidden_channels=hidden_channels,
            n_convolutions=n_convolutions,
            activation=activation,
            context_size=context_size,
            mesh=mesh,
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
        if prognostic is not None: tensors.append(einops.rearrange(prognostic, "b t c f h w -> (b f) (t c) h w"))
        return th.cat(tensors, dim=1)


class UNetEncoder(th.nn.Module):

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: list = [8, 16, 32],
        n_convolutions: int = 2,
        activation: th.nn.Module = th.nn.ReLU(),
        mesh: str = "equirectangular"
    ):
        super(UNetEncoder, self).__init__()
        self.layers = []

        channels = [in_channels] + hidden_channels

        for c_idx in range(len(channels[:-1])):
            layer = []
            c_in = channels[c_idx]
            c_out = channels[c_idx+1]

            # Apply downsampling prior to convolutions if not in top-most layer
            if c_idx > 0: layer.append(th.nn.AvgPool2d(kernel_size=2, stride=2, padding=0))

            # Perform n convolutions (only half as many in bottom-most layer, since other half is done in decoder)
            n_convs = n_convolutions//2 if c_idx == len(hidden_channels)-1 else n_convolutions
            for n_conv in range(n_convs):
                if mesh == "equirectangular":
                    layer.append(CylinderPad(padding=1))
                    layer.append(th.nn.Conv2d(
                        in_channels=c_in if n_conv == 0 else c_out,
                        out_channels=c_out, 
                        kernel_size=3, 
                        padding=0
                    ))
                elif mesh == "healpix":
                    layer.append(HEALPixLayer(
                        layer=th.nn.Conv2d,
                        in_channels=c_in if n_conv == 0 else c_out,
                        out_channels=c_out,
                        kernel_size=3,
                        padding=1
                    ))
                
                # Activation function
                layer.append(activation)

            self.layers.append(th.nn.Sequential(*layer))

        self.layers = th.nn.ModuleList(self.layers)

    def forward(self, x: th.Tensor) -> list:
        # Store intermediate model outputs (per layer) for skip connections
        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)
        return outs


class UNetDecoder(th.nn.Module):

    def __init__(
        self,
        hidden_channels: list = [8, 16, 32],
        out_channels: int = 2,
        n_convolutions: int = 2,
        activation: th.nn.Module = th.nn.ReLU(),
        mesh: str = "equirectangular"
    ):
        super(UNetDecoder, self).__init__()
        self.layers = []
        hidden_channels = hidden_channels[::-1]  # Invert as we go up in decoder, i.e., from bottom to top layers

        for c_idx in range(len(hidden_channels)):
            layer = []
            c_in = hidden_channels[c_idx]
            c_out = hidden_channels[c_idx]

            # Perform n convolutions (only half as many in bottom-most layer, since other half is done in encoder)
            n_convs = n_convolutions//2 if c_idx == 0 else n_convolutions
            for n_conv in range(n_convs):
                c_in_ = c_in if c_idx == 0 else 2*hidden_channels[c_idx]  # Skip connection from encoder
                if mesh == "equirectangular":
                    layer.append(CylinderPad(padding=1))
                    layer.append(th.nn.Conv2d(
                        in_channels=c_in_ if n_conv == 0 else c_out,
                        out_channels=c_out,
                        kernel_size=3, 
                        padding=0
                    ))
                elif mesh == "healpix":
                    layer.append(HEALPixLayer(
                        layer=th.nn.Conv2d,
                        in_channels=c_in_ if n_conv == 0 else c_out,
                        out_channels=c_out,
                        kernel_size=3,
                        padding=1
                    ))

                # Activation function
                layer.append(activation)

            # Apply upsampling if not in top-most layer
            if c_idx < len(hidden_channels)-1:
                layer.append(th.nn.ConvTranspose2d(
                    in_channels=c_out,
                    out_channels=hidden_channels[c_idx+1],
                    kernel_size=2,
                    stride=2
                ))

            self.layers.append(th.nn.Sequential(*layer))

        self.layers = th.nn.ModuleList(self.layers)
        
        # Add linear output layer
        self.output_layer = th.nn.Conv2d(
            in_channels=c_out,
            out_channels=out_channels,
            kernel_size=1
        )

    def forward(self, x: th.Tensor, skips: list) -> th.Tensor:
        for l_idx, layer in enumerate(self.layers):
            x = th.cat([skips[l_idx], x], dim=1) if l_idx > 0 else x
            x = layer(x)
        return self.output_layer(x)


if __name__ == "__main__":

    # Demo
    in_channels = 1
    hidden_channels = [8, 16, 32]
    out_channels = 1
    n_convolutions = 2
    activation = th.nn.ReLU()
    context_size = 2
    mesh = "equirectangular"
    teacher_forcing_steps = 15
    
    model = UNet(
        name="model_name",
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        n_convolutions=n_convolutions,
        activation=activation,
        context_size=context_size,
        mesh=mesh
    )

    x = th.randn(4, 25, in_channels, 32, 64)  # B, T, C, H, W
    y_hat = model(x=x, teacher_forcing_steps=teacher_forcing_steps)

    
