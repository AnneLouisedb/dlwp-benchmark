import torch as th
import einops
from utils import CylinderPad
from utils import HEALPixLayer
from abc import ABC, abstractmethod
import math 

def fourier_embedding(timesteps: th.Tensor, dim, max_period=10000):
    r"""Create sinusoidal timestep embeddings.

    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
        dim (int): the dimension of the output.
        max_period (int): controls the minimum frequency of the embeddings.
    Returns:
        embedding (torch.Tensor): [N $\times$ dim] Tensor of positional embeddings.
    """
    half = dim // 2
    
    freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding



# Largely based on https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py
# MIT License
class DiffusionModel(th.nn.Module):
    @abstractmethod
    def forward(self, constants, prescribed, prognostic, emb):
        pass
        

class ConditionedBlock(th.nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """Apply the module to `x` given `emb` embedding of time or others."""



class DiffModernUNet(DiffusionModel):
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
        use_scale_shift_norm=True,
        predict_diff = True,
        
        **kwargs
    ):
        super(DiffModernUNet, self).__init__()
        if isinstance(activation, str): activation = eval(activation)

        self.context_size = context_size
        self.mesh = mesh
        self.hidden_channels = list(hidden_channels)
        time_embed_dim = self.hidden_channels[0] * 4
        self.activation = activation
        self.predict_diff = predict_diff

        self.time_embed = th.nn.Sequential(
            th.nn.Linear(self.hidden_channels[0], time_embed_dim),
            self.activation,
            th.nn.Linear(time_embed_dim, time_embed_dim),
        )
       
        in_channels = constant_channels + (prescribed_channels+prognostic_channels)*context_size
        
        self.encoder = ModernUNetEncoder(
            in_channels=in_channels,
            hidden_channels=self.hidden_channels,
            time_embed_dim = time_embed_dim,
            activation=activation,
            attention = attention,
            mesh=mesh,
            use_scale_shift_norm=use_scale_shift_norm
        )
        self.middle = MiddleBlock(
            in_channels=self.hidden_channels[-1], 
            #attention = attention,
            time_embed_dim = time_embed_dim,
            norm = norm,
            activation=activation,
            use_scale_shift_norm=use_scale_shift_norm
            #mesh = mesh
            )
      

        self.decoder = ModernUNetDecoder(
            hidden_channels=self.hidden_channels,
            out_channels=prognostic_channels,
            time_embed_dim = time_embed_dim,
            activation=activation,
            mesh=mesh,
            use_scale_shift_norm=use_scale_shift_norm
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
    
    def single_forward(self, constants, prescribed, prognostic, y_noised, time: th.Tensor):
        # Apply the single diffusion forward function here, that takes the time step as an imput
        time_multiplier = 1000 / 3 #CHECK
        # multiply before passing to the fourier embeddings
        time = time*time_multiplier
        
        prognostic_t = th.cat([prognostic, y_noised], axis=1)

        x_t = self._prepare_inputs(
                constants=constants,
                prescribed=prescribed,
                prognostic=prognostic_t
            )

        fourier = fourier_embedding(time, self.hidden_channels[0]).to(time.device)
        time_emb = self.time_embed(fourier)

        enc = self.encoder(x_t, emb = time_emb) # this needs to be 16, 2, 3, 32, 64
    
        enc2 = self.middle(enc[-1], emb=time_emb)
        
        out = self.decoder(x=enc2, skips=enc[::-1], emb=time_emb) 

        return out
    
    def diffusion_forward(self, constants, prescribed, prognostic, noise_scheduler):
        # Apply all diffusion step and return the full output
        y_noised = th.randn_like(prognostic) 

        for k in range(3): #noise_scheduler.timesteps:
             
            print('prognostic shape?', prognostic)

            # Create a 1D tensor of length batch_size, filled with k
            k = th.full(
                (prognostic[0].shape[0],),  
                k,  # Fill value
                dtype=prognostic[0].dtype, 
                device=prognostic[0].device
            ) 
            print('prognostic shape?', prognostic)
            
            pred = self.single_forward(constants, prescribed, prognostic, y_noised, time = k)

            y_noised = noise_scheduler.step(pred, k, y_noised).prev_sample

        y = y_noised
        
        if self.predict_diff:
            print("do we get here??")
            print(y.shape)
            print(prognostic.shape)
            y = y  #+ prognostic[accum_idx][:, -1:] # * difference_weight, wight
        return y

  
    def forward(
        self,
        constants: th.Tensor = None,
        prescribed: th.Tensor = None,
        prognostic: th.Tensor = None,
        noise_scheduler = None,
           
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
        
        for t in range(self.context_size, prognostic.shape[1] //2):
        # [C, C, T1, T2, T3] -> prognostic time dimension
       
            t_start = max(0, t-(self.context_size))
            if t == self.context_size:
                # Initial condition
                prognostic_t = prognostic[:, t_start:t] 
                prescribed = prescribed[:, t_start:t] if prescribed is not None else None
                
            else:
               
                # In case of context_size > 1, blend prognostic input with outputs from previous time steps
                prognostic_t = th.cat(
                    tensors=[prognostic[:, t_start:self.context_size],        # Prognostic input before context_size 
                             th.stack(outs, dim=1)[:, -self.context_size:]],  # Outputs since context_size 
                    dim=1
                )
                prescribed = prescribed[:, t-self.context_size:t] if prescribed is not None else None
                
            out = self.diffusion_forward(constants, prescribed, prognostic, noise_scheduler)

            if self.mesh == "healpix": out = einops.rearrange(out, "(b f) tc h w -> b tc f h w", b=B, f=F)

            out = prognostic_t[:, -1] + out
        
            outs.append(out)

        return th.stack(outs, dim=1)
    
    def predict_next_solution(self, accum_idx, constants, prescribed, prognostic, target, noise_scheduler, time_multiplier):
            
            y_noised = th.randn_like(target[accum_idx] )
            print(y_noised.shape, 'shape of y_noised?') # torch.Size([16, 14, 3, 32, 64]) shape of y_noised?
            print('shape of target', len(target))
            # [:, -self.context_size:]
                           
            for k in noise_scheduler.timesteps:
                # Create a 1D tensor of length batch_size, filled with k
                time = th.full(
                    (target[accum_idx].shape[0],),  # 1D tensor with batch size length
                    k,  # Fill value
                    dtype=target[accum_idx].dtype, 
                    device=target[accum_idx].device
                ) 
                
                x_in = th.cat([prognostic[accum_idx], y_noised], axis=1)

                pred = self.forward(
                    constants=constants[accum_idx] if not constants == None else None,
                    prescribed=prescribed[accum_idx] if not prescribed == None else None,
                    prognostic=x_in,
                    time=time * time_multiplier)
                print("shape of prediction", pred.shape)
                
                y_noised = noise_scheduler.step(pred, k, y_noised).prev_sample

            y = y_noised
            
            if self.predict_diff:
                print("do we get here??")
                print(y.shape)
                print(prognostic.shape)
                y = y + prognostic[accum_idx][:, -1:] # * difference_weight, wight
            return y


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
        mesh: str = "equirectangular",
        use_scale_shift_norm=True,
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
                    padding = 1,
                    use_scale_shift_norm=use_scale_shift_norm))
                  
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

    def forward(self, x: th.Tensor, emb:th.Tensor) -> list:
        # Store intermediate model outputs (per layer) for skip connections
        outs = []
        for layer in self.layers:
            for module in layer:
                if isinstance(module, ConditionedBlock):
                    x = module(x, emb)
                else:
                    x = module(x)
                    
            outs.append(x)
        return outs
    
class ModernUNetDecoder(th.nn.Module):

    def __init__(
        self,
        hidden_channels: list = [64, 128, 256, 1024],
        out_channels: int = 2,
        time_embed_dim = 1024,
        activation: th.nn.Module = th.nn.GELU(),
        attention: bool = False,
        mesh: str = "equirectangular",
        use_scale_shift_norm=True,
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
                    cond_channels=time_embed_dim,
                    kernel_size= 3,
                    padding = 1,
                    use_scale_shift_norm=use_scale_shift_norm))
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

    def forward(self, x: th.Tensor, skips: list, emb: th.Tensor) -> th.Tensor:
        for l_idx, layer in enumerate(self.layers):
            x = th.cat([skips[l_idx], x], dim=1) if l_idx > 0 else x
            for module in layer:
                if isinstance(module, ConditionedBlock):
                    x = module(x, emb)
                else:
                    x = module(x)

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


class ResidualBlock(ConditionedBlock):
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
        use_scale_shift_norm: bool = True,
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
        use_scale_shift_norm: bool = True
         
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
