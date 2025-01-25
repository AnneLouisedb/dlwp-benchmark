import torch as th
import einops
from utils import CylinderPad
from utils import HEALPixLayer

# Largely based on https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py
# MIT License


class ModernUNet(th.nn.Module):
    """
    use_scale_shift_norm (bool): Whether to use scale and shift approach to conditoning (also termed as `AdaGN`). Defaults to False.
    
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
        
        **kwargs
    ):
        super(ModernUNet, self).__init__()
        if isinstance(activation, str): activation = eval(activation)

        self.context_size = context_size
        self.mesh = mesh
        time_embed_dim = hidden_channels * 4
        self.time_embed = th.nn.Sequential(
            th.nn.Linear(hidden_channels, time_embed_dim),
            self.activation,
            th.nn.Linear(time_embed_dim, time_embed_dim),
        )
       
        in_channels = constant_channels + (prescribed_channels+prognostic_channels)*context_size
        
        self.encoder = ModernUNetEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            time_embed_dim = time_embed_dim,
            activation=activation,
            attention = attention,
            mesh=mesh
        )
        self.middle = MiddleBlock(
            in_channels=hidden_channels[-1], 
            #attention = attention,
            time_embed_dim = time_embed_dim,
            norm = norm,
            activation=activation,
            use_scale_shift_norm=use_scale_shift_norm
            #mesh = mesh
            )
      

        self.decoder = ModernUNetDecoder(
            hidden_channels=hidden_channels,
            out_channels=prognostic_channels,
            time_embed_dim = time_embed_dim,
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

            enc2 = self.middle(enc[-1])

            out = self.decoder(x=enc2, skips=enc[::-1]) 
            if self.mesh == "healpix": out = einops.rearrange(out, "(b f) tc h w -> b tc f h w", b=B, f=F)
            
            
            out = prognostic_t[:, -1] + out

            

            outs.append(out)

       
        return th.stack(outs, dim=1)


class ModernUNetEncoder(th.nn.Module):
    """Unet encoder as used in the pde-refiner paper.
    - Each downblock combines the ResiduaBlock and the AttentionBlock??"""

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: list = [64, 128, 256, 1024],
        time_embed_dim = 1024,
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
            #if c_idx > 0: layer.append(th.nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            if mesh == "equirectangular":
                #layer.append(CylinderPad(padding=1))
                # (1) norm (2) activation (3) convolution
                
                layer.append(ResidualBlock(
                    in_channels=c_in, 
                    out_channels=c_out,
                    cond_channels=time_embed_dim,
                    kernel_size= 3,
                    padding = 1))
                  
                # (4) Attention
                layer.append(self.attn)
                    
                             
                # elif mesh == "healpix":
                #     layer.append(HEALPixLayer(
                #         layer=ResidualBlock,
                #         in_channels=c_in if n_conv == 0 else c_out,
                #         out_channels=c_out))
                #     layer.append(self.attn)
                
              
            self.layers.append(th.nn.Sequential(*layer))

        self.layers = th.nn.ModuleList(self.layers)

    def forward(self, x: th.Tensor) -> list:
        # Store intermediate model outputs (per layer) for skip connections
        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)
        return outs
    
class ModernUNetDecoder(th.nn.Module):

    def __init__(
        self,
        hidden_channels: list = [64, 128, 256, 1024],
        out_channels: int = 2,
        activation: th.nn.Module = th.nn.GELU(),
        attention: bool = False,
        mesh: str = "equirectangular"
    ):
        super(ModernUNetDecoder, self).__init__()
        self.layers = []
        hidden_channels = hidden_channels[::-1]  # Invert as we go up in decoder, i.e., from bottom to top layers
        self.attn = th.nn.Identity() # attention is not yet implemented
        self.activation = activation
        

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
                    
                # elif mesh == "healpix":
                #     layer.append(HEALPixLayer(
                #         layer=th.nn.Conv2d,
                #         in_channels=c_in_ if n_conv == 0 else c_out,
                #         out_channels=c_out,
                #         kernel_size=3,
                #         padding=1
                #     ))

                
            # Apply upsampling if not in top-most layer
            if c_idx < len(hidden_channels)-1: 
                layer.append(th.nn.ConvTranspose2d(c_out, hidden_channels[c_idx+1], (4, 4), (2, 2), (1, 1)))
                # layer.append(th.nn.ConvTranspose2d(
                #     in_channels=c_out,
                #     out_channels=hidden_channels[c_idx+1],
                #     kernel_size=2,
                #     stride=2
                #     #kernel_size=3, # see pdf refiner paper
                # ))

            self.layers.append(th.nn.Sequential(*layer))

        self.layers = th.nn.ModuleList(self.layers)

        # Add linear output layer
        self.output_layer = zero_module(th.nn.Conv2d(
            in_channels=c_out,
            out_channels=out_channels,
            kernel_size=1 
        ))

        self.final_norm = th.nn.GroupNorm(8, c_out)

    def forward(self, x: th.Tensor, skips: list) -> th.Tensor:
        
        for l_idx, layer in enumerate(self.layers):
            x = th.cat([skips[l_idx], x], dim=1) if l_idx > 0 else x
            x = layer(x)
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


class ResidualBlock(th.nn.Module):
    """Wide Residual Blocks used in modern Unet architectures.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        cond_channels (int): Number of channels in the conditioning vector.
        activation (str): Activation function to use.
        norm (bool): Whether to use normalization.
        n_groups (int): Number of groups for group normalization.
        use_scale_shift_norm (bool): Whether to use scale and shift approach to conditoning (also termed as `AdaGN`).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        activation =  th.nn.GELU(),
        norm: bool = False,
        n_groups: int = 8,
        use_scale_shift_norm: bool = False,
        kernel_size = 3,
        padding = 1
        
    ):
        super().__init__()
       
        self.activation = activation
        self.use_scale_shift_norm = use_scale_shift_norm
        
        # padding already provided 
        self.cylinder_pad = CylinderPad(padding=padding)
        self.conv1 = th.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0) 
        self.conv2 = zero_module(th.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=0)) 
        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = th.nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1) ) 
        else:
            self.shortcut = th.nn.Identity()

        if norm:
            self.norm1 = th.nn.GroupNorm(n_groups, in_channels)
            self.norm2 = th.nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = th.nn.Identity()
            self.norm2 = th.nn.Identity()

        self.cond_emb = th.nn.Linear(cond_channels, 2 * out_channels if use_scale_shift_norm else out_channels)


    def forward(self, x: th.Tensor, emb: th.Tensor):
        # First convolution layer
        h = self.activation(self.norm1(x))
        h = self.cylinder_pad(h)
        h = self.conv1(h)

        emb_out = self.cond_emb(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = self.norm2(h) * (1 + scale) + shift  # where we do -1 or +1 doesn't matter
            h = self.activation(h)
            h = self.cylinder_pad(h)
            h = self.conv2(h)
        else:
            h = h + emb_out
            h = self.activation(self.norm2(h))
            h = self.cylinder_pad(h)
            h = self.conv2(h)
            # Add the shortcut connection and return
            
        return h + self.shortcut(x)


class MiddleBlock(ConditionedBlock):
    """Middle block It combines a `ResidualBlock`, `AttentionBlock`, followed by another
    `ResidualBlock`.

    This block is applied at the lowest resolution of the U-Net.

    Args:
        n_channels (int): Number of channels in the input and output.
        time_embed_dim (int): Number of channels in the conditioning vector.
        has_attn (bool, optional): Whether to use attention block. Defaults to False.
        activation (str): Activation function to use. Defaults to "gelu".
        norm (bool, optional): Whether to use normalization. Defaults to False.
        use_scale_shift_norm (bool, optional): Whether to use scale and shift approach to conditoning (also termed as `AdaGN`). Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        attention: bool = False,
        activation = th.nn.GELU(),
        norm: bool = False,
        use_scale_shift_norm: bool = False
         
    ):
        super().__init__()
        self.res1 = ResidualBlock(
            in_channels,
            in_channels,
            cond_channels=time_embed_dim,
            activation=activation,
            norm = norm,
            use_scale_shift_norm=use_scale_shift_norm)
        
        self.attn = th.nn.Identity() # AttentionBlock(in_channels) if attention else th.nn.Identity()

        self.res2 = ResidualBlock(
            in_channels,
            in_channels,
            cond_channels= time_embed_dim,
            activation=activation,
            norm = norm,
            use_scale_shift_norm=use_scale_shift_norm)

    def forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        x = self.res1(x, emb)
        x = self.attn(x)
        x = self.res2(x, emb)
        return x
