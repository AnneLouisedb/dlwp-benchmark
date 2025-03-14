# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch as th
import einops
from hydra.utils import instantiate
from utils import CylinderPad
from utils import HEALPixLayer,HEALPixPadding
from typing import Any, Dict, Optional, Sequence, Union
from models.convlstm.convlstm import ConvLSTMCell
# File contains both a standard unet and modern unet with wide residual blocks


# Complex multiplication 2d
def batchmul2d(input, weights):
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    return th.einsum("bixy,ioxy->boxy", input, weights)

class SpectralConv2d(th.nn.Module):
    """2D Fourier layer. Does FFT, linear transform, and Inverse FFT.
    Implemented in a way to allow multi-gpu training.
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        modes1 (int): Number of Fourier modes to keep in the first spatial direction
        modes2 (int): Number of Fourier modes to keep in the second spatial direction
    [paper](https://arxiv.org/abs/2010.08895)
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = th.nn.Parameter(
            self.scale * th.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype=th.float32)
        )
        self.weights2 = th.nn.Parameter(
            self.scale * th.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype=th.float32)
        )

    def forward(self, x, x_dim=None, y_dim=None):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = th.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = th.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=th.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = batchmul2d(
            x_ft[:, :, : self.modes1, : self.modes2], th.view_as_complex(self.weights1)
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = batchmul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], th.view_as_complex(self.weights2)
        )

        # Return to physical space
        x = th.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x



class ModernUNet(th.nn.Module):
    """
    A ModernUNet implementation as by the PDE Refiner Paper.

    Quote from paper: 'We also experimented with adding attention layers in the residual blocks, which, however, did not improve performance noticeably.'
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
        norm: bool = False, # groupnorm in each residual block?
        recurrent:bool = False,
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
            mesh=mesh
        )
        self.middle = MiddleBlock(
            in_channels=hidden_channels[-1], 
            #attention = attention,
            norm = norm,
            activation=activation,
            mesh=mesh,
           
            )
      

        self.decoder = ModernUNetDecoder(
            hidden_channels=hidden_channels,
            out_channels=prognostic_channels,
            activation=activation,
            mesh=mesh,
            recurrent=recurrent
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
                    prognostic=prognostic_t # to forecast
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

            print('out shape', out.shape)
            print('prognostic', prognostic_t[:, -1].shape)

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
        norm: bool = False, # groupnorm in each residual block?
        recurrent: bool = False,
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
        # shape of prognostic?
        # torch.Size([16, 2, 3, 12, 8, 8])
        # constants torch.Size([192, 4, 8, 8]) -> 4
        # prescribed? torch.Size([192, 1, 8, 8]) -> 4 + 1 = 5
        # shape prognostic? torch.Size([16, 1, 3, 12, 8, 8])
        # rearrange? torch.Size([192, 3, 8, 8]) -> 5 + 3 = 8
        # inptu shape? torch.Size([192, 8, 8, 8])
        #######
        # torch.Size([16, 2, 3, 12, 8, 8])
        # constants torch.Size([192, 4, 8, 8])
        # prescribed? torch.Size([192, 4, 8, 8]) -> PRESCRIBED IS NOT ADDED?
        # shape prognostic? torch.Size([16, 1, 3, 12, 8, 8])
        # rearrange? torch.Size([192, 3, 8, 8])
        # inptu shape? torch.Size([192, 7, 8, 8])
        if constants is not None: tensors.append(einops.rearrange(constants[:, 0], "b c f h w -> (b f) c h w"))
        if prescribed is not None: tensors.append(einops.rearrange(prescribed, "b t c f h w -> (b f) (t c) h w"))
        if prognostic is not None:
            if prognostic.ndim == 6:
               
                tensors.append(einops.rearrange(prognostic, "b t c f h w -> (b f) (t c) h w")) 
            elif prognostic.ndim == 5:
                tensors.append(einops.rearrange(prognostic, "b t c h w -> b (t c) h w"))

        out = th.cat(tensors, dim=1)
        
        return out
    
    
            
            
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
        activation: th.nn.Module = th.nn.GELU(),
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
            # For each t I want to store the loss in an array, and see how the prediction skill decreases over time
          
            t_start = max(0, t-(self.context_size))
            if t == self.context_size:
                # Initial condition
                prognostic_t = prognostic[:, t_start:t]
                x_t = self._prepare_inputs(
                    constants=constants,
                    prescribed=prescribed[:, t_start:t] if prescribed is not None else None,
                    prognostic=prognostic_t # to forecast
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
        activation: th.nn.Module = th.nn.GELU(),
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
                        padding=1 
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
        activation: th.nn.Module = th.nn.GELU(),
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
    


class ModernUNetEncoder(th.nn.Module):
    """Unet encoder as used in the pde-refiner paper.
    - Each downblock combines the ResiduaBlock and the AttentionBlock??"""

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: list = [64, 128, 256, 1024],
        activation: th.nn.Module = th.nn.GELU(),
        attention: bool = False,
        mesh: str = "equirectangular"
    ):
        super(ModernUNetEncoder, self).__init__()
        self.layers = []
        self.attn = th.nn.Identity() # attention is not yet implemented

        channels = [in_channels] + hidden_channels


        for c_idx in range(len(channels[:-1])):
            layer = []
            c_in = channels[c_idx]
            c_out = channels[c_idx+1]

            # Apply downsampling prior to convolutions if not in top-most layer
            if c_idx > 0: layer.append(th.nn.Conv2d(c_in, c_in, (3, 3), (2, 2), (1, 1)))
            if c_idx == 0: layer.append(th.nn.Conv2d(c_in, c_in, (1, 1), (1, 1), (0, 0)))
            # STORE THIS OUTPUT
            if mesh == "equirectangular":
                #layer.append(CylinderPad(padding=1))
                # (1) norm (2) activation (3) convolution
                
                layer.append(ResidualBlock(
                    in_channels=c_in, 
                    out_channels=c_out,
                    kernel_size= 3,
                    padding = 1))
                
                
                  
                # (4) Attention
                layer.append(self.attn)

            elif mesh == "healpix":
                layer.append(HEALPixLayer(
                    layer=ResidualBlock,
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size= 3,
                    padding = 0, 
                    mesh=mesh
                    ))
                
                # STORE THIS OUTPUT
                
                layer.append(self.attn)
                    
                             
            self.layers.append(th.nn.Sequential(*layer))

        self.layers = th.nn.ModuleList(self.layers)

    def forward(self, x: th.Tensor) -> list:
        # Store intermediate model outputs (per layer) for skip connections
        outs = []
        for layer in self.layers:
            for sublayer in layer:
                x = sublayer(x)
                
                if isinstance(sublayer, (ResidualBlock, HEALPixLayer)):
                    outs.append(x)
                    


        return outs
    
class ModernUNetDecoder(th.nn.Module):

    def __init__(
        self,
        hidden_channels: list = [64, 128, 256, 1024],
        out_channels: int = 2,
        activation: th.nn.Module = th.nn.GELU(),
        attention: bool = False,
        mesh: str = "equirectangular",
        recurrent: bool = False
    ):
        super(ModernUNetDecoder, self).__init__()
        self.layers = []
        final_out = 2*hidden_channels[0] 
        hidden_channels = hidden_channels[::-1]  # Invert as we go up in decoder, i.e., from bottom to top layers
        self.attn = th.nn.Identity() # attention is not yet implemented
        self.activation = activation
        self.recurrent = recurrent

        

        for c_idx in range(len(hidden_channels)):
            layer = []
            c_in = hidden_channels[c_idx]
            c_out = hidden_channels[c_idx]

            c_in_ = c_in if c_idx == 0 else 2*hidden_channels[c_idx]  # Skip connection from encoder
            if mesh == "equirectangular":
                layer.append(ResidualBlock(
                    in_channels=c_in_ , 
                    out_channels=c_out,
                    kernel_size= 3,
                    padding = 1))
                layer.append(self.attn)
                    
            elif mesh == "healpix":
                layer.append(HEALPixLayer(
                    layer=ResidualBlock,
                    in_channels=c_in_,
                    out_channels=c_out,
                    kernel_size=3, 
                    padding=0, 
                    mesh = mesh
                ))
                layer.append(self.attn)

            if self.recurrent:
                self.lstm_block = ConvLSTMCell(
                batch_size=32*12,
                input_size=c_out,
                hidden_size=c_out,
                height=2, 
                width=2, 
                device='cuda:0',
                bias=True,
                mesh='healpix'
                )

                # clstm.append(
                # self.lstm_block = ConvNeXtLSTMBlock(
                #     geometry_layer=HEALPixLayer if mesh == "healpix" else th.nn.Identity,
                #     in_channels=c_out,
                #     h_channels=c_out,
                #     kernel_size=7,
                #     activation=activation,
                #     enable_healpixpad=(mesh == "healpix"))

                
                layer.append(self.lstm_block)

            if mesh == "healpix":
                if c_idx + 1 < len(hidden_channels):
                    c_out2 = 2*hidden_channels[c_idx+1]
                else:
                    c_out2 = 2*hidden_channels[c_idx]

                layer.append(HEALPixLayer(
                        layer=ResidualBlock,
                        in_channels=c_out,
                        out_channels=c_out2,
                        kernel_size=3, 
                        padding=0, # 1
                        mesh = mesh
                    ))

                
            # Apply upsampling if not in top-most layer
            if c_idx < len(hidden_channels)-1: 
                #layer.append(th.nn.ConvTranspose2d(c_out2, c_out2, (4, 4), (1, 1), (2, 2))) # old
                # size doubles!!
                layer.append(th.nn.ConvTranspose2d(c_out2, c_out2, (4, 4), (2, 2), (1, 1))) 
             
            self.layers.append(th.nn.Sequential(*layer))

        self.layers = th.nn.ModuleList(self.layers)

        
        # Add linear output layer
        self.output_layer = zero_module(th.nn.Conv2d(
            in_channels=c_out2,
            out_channels=out_channels,
            kernel_size=1 
        ))

        self.final_norm = th.nn.GroupNorm(8, final_out)

    def forward(self, x: th.Tensor, skips: list) -> th.Tensor:
           # Ensure that the skips are provided in reverse order (bottom to top)
            skips = skips[::-1]

            # Loop through each decoder layer
            for l_idx, layer in enumerate(self.layers):
                for block_idx, submodule in enumerate(layer):
                   
                    # Apply skip connection at the block level
                    if isinstance(submodule, ResidualBlock) and l_idx < len(skips):
                        s = skips[l_idx]  # Get corresponding skip connection
                        x = th.cat([s, x], dim=1)  # Concatenate skip connection with current feature map
                    
                    # Pass through the current submodule/block
                    x = submodule(x)
                    
                   
                    

            # Apply final normalization, activation, and output layer
            return self.output_layer(self.activation(self.final_norm(x)))
    
# BLOCKS
    
def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module

class AttentionBlockl(th.nn.Module):
    def __init__(
        self,
        in_channels: int):
        super().__init__()
   
        pass


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
        n_groups (int): Number of groups for group normalization. # CHECK DEFAULT?
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
        mesh = None
        
    ):
        super().__init__()
       
        self.activation = activation
        self.mesh = mesh
        

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
        mesh = None
         
    ):
        super().__init__()
        self.res1 = ResidualBlock(
            in_channels,
            in_channels,
            activation=activation,
            norm = norm,
            mesh = mesh)
        
        self.attn = th.nn.Identity() # AttentionBlock(in_channels) if attention else th.nn.Identity()

        self.res2 = ResidualBlock(
            in_channels,
            in_channels,
            activation=activation,
            norm =norm,
            mesh =mesh)

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
    
