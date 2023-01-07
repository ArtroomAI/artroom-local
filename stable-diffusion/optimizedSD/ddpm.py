"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""
import itertools
import math
import torch

import k_diffusion as K
import numpy as np
import pytorch_lightning as pl

from einops import rearrange, repeat
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from tqdm.auto import trange, tqdm
from functools import partial
from omegaconf import ListConfig
from contextlib import contextmanager, nullcontext

from ldm.modules.ema import LitEma
from ldm.models.autoencoder import VQModelInterface
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import exists, default, instantiate_from_config


# from samplers import CompVisDenoiser

def disabled_train(self):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 timesteps=1000,
                 beta_schedule="linear",
                 ckpt_path=None,
                 ignore_keys=None,
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 ):
        super().__init__()
        if ignore_keys is None:
            ignore_keys = []
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)
        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))


class DDPMv2(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 make_it_fit=False,
                 ucg_training=None,
                 reset_ema=False,
                 reset_num_ema_updates=False,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0", "v"], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapperv2(unet_config, conditioning_key)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        self.make_it_fit = make_it_fit
        if reset_ema: assert exists(ckpt_path)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)
            if reset_ema:
                assert self.use_ema
                print(
                    f"Resetting ema to pure model weights. This is useful when restoring from an ema-only checkpoint.")
                self.model_ema = LitEma(self.model)
        if reset_num_ema_updates:
            print(" +++++++++++ WARNING: RESETTING NUM_EMA UPDATES TO ZERO +++++++++++ ")
            assert self.use_ema
            self.model_ema.reset_num_updates()

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = torch.nn.Parameter(self.logvar, requires_grad=True)

        self.ucg_training = ucg_training or dict()
        if self.ucg_training:
            self.ucg_prng = np.random.RandomState()

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        elif self.parameterization == "v":
            lvlb_weights = torch.ones_like(self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)))
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    @torch.no_grad()
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        if self.make_it_fit:
            n_params = len([name for name, _ in
                            itertools.chain(self.named_parameters(),
                                            self.named_buffers())])
            for name, param in tqdm(
                    itertools.chain(self.named_parameters(),
                                    self.named_buffers()),
                    desc="Fitting old weights to new weights",
                    total=n_params
            ):
                if not name in sd:
                    continue
                old_shape = sd[name].shape
                new_shape = param.shape
                assert len(old_shape) == len(new_shape)
                if len(new_shape) > 2:
                    # we only modify first two axes
                    assert new_shape[2:] == old_shape[2:]
                # assumes first axis corresponds to output dim
                if not new_shape == old_shape:
                    new_param = param.clone()
                    old_param = sd[name]
                    if len(new_shape) == 1:
                        for i in range(new_param.shape[0]):
                            new_param[i] = old_param[i % old_shape[0]]
                    elif len(new_shape) >= 2:
                        for i in range(new_param.shape[0]):
                            for j in range(new_param.shape[1]):
                                new_param[i, j] = old_param[i % old_shape[0], j % old_shape[1]]

                        n_used_old = torch.ones(old_shape[1])
                        for j in range(new_param.shape[1]):
                            n_used_old[j % old_shape[1]] += 1
                        n_used_new = torch.zeros(new_shape[1])
                        for j in range(new_param.shape[1]):
                            n_used_new[j] = n_used_old[j % old_shape[1]]

                        n_used_new = n_used_new[None, :]
                        while len(n_used_new.shape) < len(new_shape):
                            n_used_new = n_used_new.unsqueeze(-1)
                        new_param /= n_used_new

                    sd[name] = new_param

        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys:\n {missing}")
        if len(unexpected) > 0:
            print(f"\nUnexpected Keys:\n {unexpected}")

    def predict_start_from_z_and_v(self, x_t, t, v):
        # self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        # self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


class FirstStage(DDPM):
    """main class"""

    def __init__(self,
                 first_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__()
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        self.instantiate_first_stage(first_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                df = self.split_input_params["vqf"]
                self.split_input_params['original_image_size'] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                output_list = [self.first_stage_model.encode(z[:, :, :, :, i])
                               for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded

            else:
                return self.first_stage_model.encode(x)
        else:
            return self.first_stage_model.encode(x)


class CondStage(DDPM):
    """main class"""

    def __init__(self,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__()
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c


class DiffusionWrapperv2(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.sequential_cross_attn = diff_model_config.pop("sequential_crossattn", False)
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm', 'hybrid-adm', 'crossattn-adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None, c_adm=None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            if not self.sequential_cross_attn:
                cc = torch.cat(c_crossattn, 1)
            else:
                cc = c_crossattn
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'hybrid-adm':
            assert c_adm is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, y=c_adm)
        elif self.conditioning_key == 'crossattn-adm':
            assert c_adm is not None
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc, y=c_adm)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)

    def forward(self, x, t, cc):
        out = self.diffusion_model(x, t, context=cc)
        return out


class DiffusionWrapperOut(pl.LightningModule):
    def __init__(self, diff_model_config):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)

    def forward(self, h, emb, tp, hs, cc):
        return self.diffusion_model(h, emb, tp, hs, context=cc)


class CFGDenoiser(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale


class KDiffusionSampler:
    def __init__(self, m, sampler):
        self.model = m
        self.model_wrap = K.external.CompVisDenoiser(m)
        self.schedule = sampler

    def get_sampler_name(self):
        return self.schedule

    def sample_txt2img(self, S, conditioning, S_ddim_steps, batch_size, shape, verbose, unconditional_guidance_scale,
                       unconditional_conditioning, eta, x_T, callback_fn=None):
        sigmas = self.model_wrap.get_sigmas(S)
        model_wrap_cfg = CFGDenoiser(self.model_wrap)
        x0 = torch.randn_like(x_T)
        x_dec = x0 * sigmas[0]
        samples_ddim = K.sampling.__dict__[f'sample_{self.schedule}'](model_wrap_cfg, x_dec, sigmas,
                                                                      callback=callback_fn,
                                                                      extra_args={'cond': conditioning,
                                                                                  'uncond': unconditional_conditioning,
                                                                                  'cond_scale': unconditional_guidance_scale
                                                                                  },
                                                                      disable=False)
        return samples_ddim

    def sample(self, S, conditioning, S_ddim_steps, batch_size, shape, verbose, unconditional_guidance_scale,
               unconditional_conditioning, eta, x_T, mask=None, callback_fn=None):
        if S == S_ddim_steps:
            return self.sample_txt2img(S, conditioning, S_ddim_steps, batch_size, shape, verbose,
                                       unconditional_guidance_scale,
                                       unconditional_conditioning, eta, x_T, callback_fn)
        sigmas = self.model_wrap.get_sigmas(S_ddim_steps)
        noise = torch.randn_like(x_T) * sigmas[S_ddim_steps - S - 1]

        xi = x_T + noise

        if mask is not None and xi.shape[1] != 9:
            xi = ((1 - mask) * noise) + (mask * xi)

        sigma_sched = sigmas[S_ddim_steps - S - 1:]
        model_wrap_cfg = CFGDenoiser(self.model_wrap)
        samples = K.sampling.__dict__[f'sample_{self.schedule}'](model_wrap_cfg, xi, sigma_sched,
                                                                 extra_args={'cond': conditioning,
                                                                             'uncond': unconditional_conditioning,
                                                                             'cond_scale': unconditional_guidance_scale},
                                                                 disable=False)
        return samples


class UNet(DDPM):
    """main class"""

    def __init__(self,
                 unetConfigEncode,
                 unetConfigDecode,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 unet_bs=1,
                 scale_by_std=False,
                 *args, **kwargs):

        self.current_step = 0
        self.total_steps = 0
        self.parameterization = "eps"

        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.num_downs = 0
        self.cdevice = "cuda"
        self.unetConfigEncode = unetConfigEncode
        self.unetConfigDecode = unetConfigDecode
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None
        self.model1 = DiffusionWrapper(self.unetConfigEncode)
        self.model2 = DiffusionWrapperOut(self.unetConfigDecode)
        self.model1.eval()
        self.model2.eval()
        self.turbo = False
        self.unet_bs = unet_bs
        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.cdevice)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def apply_model(self, x_noisy, t, cond, return_ids=False):

        if not self.turbo:
            self.model1.to(self.cdevice)

        step = self.unet_bs
        h, emb, hs = self.model1(x_noisy[0:step], t[:step], cond[:step])
        bs = cond.shape[0]

        # assert bs%2 == 0
        lenhs = len(hs)

        for i in range(step, bs, step):
            h_temp, emb_temp, hs_temp = self.model1(x_noisy[i:i + step], t[i:i + step], cond[i:i + step])
            h = torch.cat((h, h_temp))
            emb = torch.cat((emb, emb_temp))
            for j in range(lenhs):
                hs[j] = torch.cat((hs[j], hs_temp[j]))

        if not self.turbo:
            self.model1.to("cpu")
            self.model2.to(self.cdevice)

        hs_temp = [hs[j][:step] for j in range(lenhs)]
        x_recon = self.model2(h[:step], emb[:step], x_noisy.dtype, hs_temp, cond[:step])

        for i in range(step, bs, step):
            hs_temp = [hs[j][i:i + step] for j in range(lenhs)]
            x_recon1 = self.model2(h[i:i + step], emb[i:i + step], x_noisy.dtype, hs_temp, cond[i:i + step])
            x_recon = torch.cat((x_recon, x_recon1))

        if not self.turbo:
            self.model2.to("cpu")

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def register_buffer1(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device(self.cdevice):
                attr = attr.to(torch.device(self.cdevice))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):

        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.num_timesteps, verbose=verbose)

        assert self.alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = lambda x: x.to(self.cdevice)
        self.register_buffer1('betas', to_torch(self.betas))
        self.register_buffer1('alphas_cumprod', to_torch(self.alphas_cumprod))
        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=self.alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta, verbose=verbose)
        self.register_buffer1('ddim_sigmas', ddim_sigmas)
        self.register_buffer1('ddim_alphas', ddim_alphas)
        self.register_buffer1('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer1('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))

    @torch.no_grad()
    def sample(self,
               S,
               conditioning,
               x0=None,
               shape=None,
               S_ddim_steps=None,
               seed=1234,
               callback=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               sampler="plms",
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               batch_size=None
               ):

        if self.turbo:
            self.model1.to(self.cdevice)
            self.model2.to(self.cdevice)
        if x0 is None:
            batch_size, b1, b2, b3 = shape
            img_shape = (1, b1, b2, b3)
            tens = []
            print("seeds used = ", [seed + s for s in range(batch_size)])
            for _ in range(batch_size):
                torch.manual_seed(seed)
                tens.append(torch.randn(img_shape, device=self.cdevice))
                seed += 1
            noise = torch.cat(tens)
            del tens
            try:
                self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=False)
            except Exception:
                self.make_schedule(ddim_num_steps=S + 1, ddim_eta=eta, verbose=False)

        x_latent = noise if x0 is None else x0
        # sampling
        if sampler == "plms":
            print(f'Data shape for PLMS sampling is {shape}')
            samples = self.plms_sampling(conditioning, batch_size, x_latent,
                                         callback=callback,
                                         img_callback=img_callback,
                                         quantize_denoised=quantize_x0,
                                         mask=mask, x0=x0,
                                         ddim_use_original_steps=False,
                                         noise_dropout=noise_dropout,
                                         temperature=temperature,
                                         score_corrector=score_corrector,
                                         corrector_kwargs=corrector_kwargs,
                                         log_every_t=log_every_t,
                                         unconditional_guidance_scale=unconditional_guidance_scale,
                                         unconditional_conditioning=unconditional_conditioning,
                                         )

        elif sampler == "ddim":
            samples = self.ddim_sampling(x_latent, conditioning, S,
                                         unconditional_guidance_scale=unconditional_guidance_scale,
                                         unconditional_conditioning=unconditional_conditioning,
                                         mask=mask, init_latent=x_T, use_original_steps=False)

        elif sampler == 'dpmpp_2m':
            model_wrap = K.external.CompVisDenoiser(self)
            sigmas = model_wrap.get_sigmas(S)
            # sigmas = K.sampling.get_sigmas_karras(n=S, sigma_min=0.1, sigma_max=10, device='cpu')
            if mask is not None:
                x0_noisy = x_T if x_T is not None else torch.randn_like(x_latent)
                x_latent = x0_noisy * mask + (1. - mask) * x_latent
            x_latent = x_latent * sigmas[0]
            model_wrap_cfg = CFGDenoiser(model_wrap)
            samples = self.sample_dpmpp_2m(model_wrap_cfg, x_latent, sigmas,
                                           extra_args={'cond': conditioning,
                                                       'uncond': unconditional_conditioning,
                                                       'cond_scale': unconditional_guidance_scale},
                                           disable=False)
        elif sampler == 'dpmpp_2s_ancestral':
            model_wrap = K.external.CompVisDenoiser(self)
            sigmas = model_wrap.get_sigmas(S)
            # sigmas = K.sampling.get_sigmas_karras(n=S, sigma_min=0.1, sigma_max=10, device='cpu')

            if mask is not None:
                x0_noisy = x_T if x_T is not None else torch.randn_like(x_latent)
                x_latent = x0_noisy * mask + (1. - mask) * x_latent
            x_latent = x_latent * sigmas[0]
            model_wrap_cfg = CFGDenoiser(model_wrap)
            samples = self.sample_dpmpp_2s_ancestral(model_wrap_cfg, x_latent, sigmas,
                                                     extra_args={'cond': conditioning,
                                                                 'uncond': unconditional_conditioning,
                                                                 'cond_scale': unconditional_guidance_scale},
                                                     disable=False)
        else:
            if sampler == 'dpm_a':
                sampler = KDiffusionSampler(self, 'dpm_2_ancestral')
            elif sampler == 'dpm_2':
                sampler = KDiffusionSampler(self, 'dpm_2')
            elif sampler == 'euler_a':
                sampler = KDiffusionSampler(self, 'euler_ancestral')
            elif sampler == 'euler':
                sampler = KDiffusionSampler(self, 'euler')
            elif sampler == 'heun':
                sampler = KDiffusionSampler(self, 'heun')
            elif sampler == 'lms':
                sampler = KDiffusionSampler(self, 'lms')

            samples = sampler.sample(S=S, conditioning=conditioning, batch_size=int(x_latent.shape[0]),
                                     shape=x_latent[0].shape, verbose=False, S_ddim_steps=S_ddim_steps,
                                     unconditional_guidance_scale=unconditional_guidance_scale, mask=mask,
                                     unconditional_conditioning=unconditional_conditioning, eta=0.0, x_T=x_latent)
        if self.turbo:
            self.model1.to("cpu")
            self.model2.to("cpu")

        return samples

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start)).to(x_start.device)
        return (extract_into_tensor(self.sqrt_alphas_cumprod.to(x_start.device), t.to(x_start.device),
                                    x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t.to(x_start.device),
                                    x_start.shape) * noise)

    @torch.no_grad()
    def plms_sampling(self, cond, b, img,
                      ddim_use_original_steps=False,
                      callback=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):

        device = self.betas.device
        timesteps = self.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running PLMS Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='PLMS Sampler', total=total_steps)
        old_eps = []

        for i, step in enumerate(iterator):
            self.current_step = i
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts).to(img.device)
                img = img_orig * mask + (1. - mask) * img
                del img_orig

            outs = self.p_sample_plms(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      old_eps=old_eps, t_next=ts_next)
            img, pred_x0, e_t = outs
            old_eps.append(e_t)
            if len(old_eps) >= 4:
                old_eps.pop(0)
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

        return img

    @torch.no_grad()
    def p_sample_plms(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, old_eps=None, t_next=None):
        b, *_, device = *x.shape, x.device

        def get_model_output(x, t):
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.apply_model(x, t, c)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            if score_corrector is not None:
                assert self.parameterization == "eps"
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

            return e_t

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas

        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        e_t = get_model_output(x, t)
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next = get_model_output(x_prev, t_next)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        return x_prev, pred_x0, e_t

    @torch.no_grad()
    def stochastic_encode(self, x0, t, seed, ddim_eta, ddim_steps, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        try:
            self.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
        except Exception:
            self.make_schedule(ddim_num_steps=ddim_steps + 1, ddim_eta=ddim_eta, verbose=False)

        sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)

        if noise is None:
            b0, b1, b2, b3 = x0.shape
            img_shape = (1, b1, b2, b3)
            tens = []
            print("seeds used = ", [seed + s for s in range(b0)])
            for _ in range(b0):
                torch.manual_seed(seed)
                tens.append(torch.randn(img_shape, device=x0.device))
                seed += 1
            noise = torch.cat(tens)
            del tens
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(self.ddim_sqrt_one_minus_alphas, t, x0.shape) * noise)

    @torch.no_grad()
    def add_noise(self, x0, t):
        sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
        noise = torch.randn(x0.shape, device=x0.device)

        # print(extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape),
        #       extract_into_tensor(self.ddim_sqrt_one_minus_alphas, t, x0.shape))
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(self.ddim_sqrt_one_minus_alphas, t, x0.shape) * noise)

    @torch.no_grad()
    def ddim_sampling(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
                      mask=None, init_latent=None, use_original_steps=False):

        timesteps = self.ddim_timesteps
        timesteps = timesteps[:t_start]
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            self.current_step = i
            x0 = init_latent if init_latent is not None else torch.randn_like(x_dec)
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            if mask is not None and x0.shape[1] != 9:
                # x0_noisy = self.add_noise(mask, torch.tensor([index] * x0.shape[0]).to(self.cdevice))
                x0_noisy = x0
                x_dec = x0_noisy * mask + (1. - mask) * x_dec

            x_dec = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                       unconditional_guidance_scale=unconditional_guidance_scale,
                                       unconditional_conditioning=unconditional_conditioning)

        if mask is not None:
            return x0 * mask + (1. - mask) * x_dec

        return x_dec

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev

    def append_zero(self, x):
        return torch.cat([x, x.new_zeros([1])])

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7., device='cpu'):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = torch.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return self.append_zero(sigmas).to(device)

    def get_sigmas_exponential(self, n, sigma_min, sigma_max, device='cpu'):
        """Constructs an exponential noise schedule."""
        sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
        return self.append_zero(sigmas)

    def get_sigmas_vp(self, n, beta_d=19.9, beta_min=0.1, eps_s=1e-3, device='cpu'):
        """Constructs a continuous VP noise schedule."""
        t = torch.linspace(1, eps_s, n, device=device)
        sigmas = torch.sqrt(torch.exp(beta_d * t ** 2 / 2 + beta_min * t) - 1)
        return self.append_zero(sigmas)

    def to_d(self, x, sigma, denoised):
        """Converts a denoiser output to a Karras ODE derivative."""
        return (x - denoised) / self.append_dims(sigma, x.ndim)

    def append_dims(self, x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
        return x[(...,) + (None,) * dims_to_append]

    def get_ancestral_step(self, sigma_from, sigma_to, eta=1.):
        """Calculates the noise level (sigma_down) to step down to and the amount
        of noise to add (sigma_up) when doing an ancestral sampling step."""
        if not eta:
            return sigma_to, 0.
        sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
        sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
        return sigma_down, sigma_up

    @torch.no_grad()
    def euler_sampling(self, x, sigmas, cond, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0.,
                       s_tmax=float('inf'), s_noise=1.):
        """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
        extra_args = {} if extra_args is None else extra_args
        s_in = x.new_ones([x.shape[0]]).half()
        for i in trange(len(sigmas) - 1, disable=disable):
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
            eps = torch.randn_like(x) * s_noise
            sigma_hat = (sigmas[i] * (gamma + 1)).half()
            if gamma > 0:
                x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
            denoised = self.apply_model(x, sigma_hat * s_in, cond)
            d = self.to_d(x, sigma_hat, denoised)
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
            dt = sigmas[i + 1] - sigma_hat
            # Euler method
            x = x + d * dt
        return x

    @torch.no_grad()
    def heun_sampling(self, x, sigmas, cond, unconditional_conditioning=None, unconditional_guidance_scale=1,
                      extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'),
                      s_noise=1.):
        """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
        extra_args = {} if extra_args is None else extra_args
        print(sigmas)
        # sigmas = self.get_sigmas_karras(steps, 0.01, 80, rho=7., device=x.device)
        print(x[0])
        x = x * sigmas[0]
        print("alu", x[0])
        s_in = x.new_ones([x.shape[0]])
        for i in trange(len(sigmas) - 1, disable=disable):
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
            eps = torch.randn_like(x) * s_noise
            sigma_hat = (sigmas[i] * (gamma + 1)).half()
            if gamma > 0:
                x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5

            # print(sigma_hat * s_in)
            # x, sigma_hat * s_in, cond
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([sigma_hat * s_in] * 2)
            c_in = torch.cat([unconditional_conditioning, cond])
            e_t_uncond, e_t = self.apply_model(x_in, t_in, c_in).chunk(2)
            denoised = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            # denoised = self.apply_model(x, sigma_hat * s_in, cond)
            d = self.to_d(x, sigma_hat, denoised)
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
            dt = sigmas[i + 1] - sigma_hat
            # if sigmas[i + 1] == 0:
            if 1:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                denoised_2 = self.apply_model(x_2, sigmas[i + 1] * s_in, cond)
                d_2 = self.to_d(x_2, sigmas[i + 1], denoised_2)
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
        return x

    @torch.no_grad()
    def sample_dpmpp_2s_ancestral(self, model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.,
                                  s_noise=1.):
        """Ancestral sampling with DPM-Solver++(2S) second-order steps."""
        extra_args = {} if extra_args is None else extra_args
        s_in = x.new_ones([x.shape[0]])
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

        for i in trange(len(sigmas) - 1, disable=disable):
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            sigma_down, sigma_up = self.get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
            if sigma_down == 0:
                # Euler method
                d = self.to_d(x, sigmas[i], denoised)
                dt = sigma_down - sigmas[i]
                x = x + d * dt
            else:
                # DPM-Solver-2++(2S)
                t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
                r = 1 / 2
                h = t_next - t
                s = t + r * h
                x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
                denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
            # Noise addition
            x = x + torch.randn_like(x) * s_noise * sigma_up
        return x

    @torch.no_grad()
    def sample_dpmpp_2m(self, model, x, sigmas, extra_args=None, callback=None, disable=None):
        """DPM-Solver++(2M)."""
        extra_args = {} if extra_args is None else extra_args
        s_in = x.new_ones([x.shape[0]])
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()
        old_denoised = None

        for i in trange(len(sigmas) - 1, disable=disable):
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            if old_denoised is None or sigmas[i + 1] == 0:
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
            else:
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
            old_denoised = denoised
        return x


class LatentDiffusion(DDPMv2):
    """main class"""

    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 force_null_conditioning=False,
                 *args, **kwargs):
        self.current_step = 0
        self.total_steps = 0

        self.force_null_conditioning = force_null_conditioning
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__' and not self.force_null_conditioning:
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        reset_ema = kwargs.pop("reset_ema", False)
        reset_num_ema_updates = kwargs.pop("reset_num_ema_updates", False)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)

        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True
            if reset_ema:
                assert self.use_ema
                print(
                    f"Resetting ema to pure model weights. This is useful when restoring from an ema-only checkpoint.")
                self.model_ema = LitEma(self.model)
        if reset_num_ema_updates:
            print(" +++++++++++ WARNING: RESETTING NUM_EMA UPDATES TO ZERO +++++++++++ ")
            assert self.use_ema
            self.model_ema.reset_num_updates()

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def register_buffer1(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device(self.cdevice):
                attr = attr.to(torch.device(self.cdevice))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):

        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.num_timesteps, verbose=verbose)

        assert self.alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = lambda x: x.to(self.cdevice)
        self.register_buffer1('betas', to_torch(self.betas))
        self.register_buffer1('alphas_cumprod', to_torch(self.alphas_cumprod))
        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=self.alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta, verbose=verbose)
        self.register_buffer1('ddim_sigmas', ddim_sigmas)
        self.register_buffer1('ddim_alphas', ddim_alphas)
        self.register_buffer1('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer1('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is expected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    @torch.no_grad()
    def get_unconditional_conditioning(self, batch_size, null_label=None):
        if null_label is not None:
            xc = null_label
            if isinstance(xc, ListConfig):
                xc = list(xc)
            if isinstance(xc, dict) or isinstance(xc, list):
                c = self.get_learned_conditioning(xc)
            else:
                if hasattr(xc, "to"):
                    xc = xc.to(self.device)
                c = self.get_learned_conditioning(xc)
        else:
            if self.cond_stage_key in ["class_label", "cls"]:
                xc = self.cond_stage_model.get_unconditional_conditioning(batch_size, device=self.device)
                return self.get_learned_conditioning(xc)
            else:
                raise NotImplementedError("todo")
        if isinstance(c, list):  # in case the encoder gives us a list
            for i in range(len(c)):
                c[i] = repeat(c[i], '1 ... -> b ...', b=batch_size).to(self.device)
        else:
            c = repeat(c, '1 ... -> b ...', b=batch_size).to(self.device)
        return c

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def stochastic_encode(self, x0, t, seed, ddim_eta, ddim_steps, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        try:
            self.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
        except Exception:
            self.make_schedule(ddim_num_steps=ddim_steps + 1, ddim_eta=ddim_eta, verbose=False)

        sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)

        if noise is None:
            b0, b1, b2, b3 = x0.shape
            img_shape = (1, b1, b2, b3)
            tens = []
            print("seeds used = ", [seed + s for s in range(b0)])
            for _ in range(b0):
                torch.manual_seed(seed)
                tens.append(torch.randn(img_shape, device=x0.device))
                seed += 1
            noise = torch.cat(tens)
            del tens
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(self.ddim_sqrt_one_minus_alphas, t, x0.shape) * noise)


    @torch.no_grad()
    def sample(self,
               S,
               conditioning,
               x0=None,
               shape=None,
               S_ddim_steps=None,
               seed=1234,
               callback=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               sampler="plms",
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               batch_size=None
               ):

        if x0 is None:
            batch_size, b1, b2, b3 = shape
            img_shape = (1, b1, b2, b3)
            tens = []
            print("seeds used = ", [seed + s for s in range(batch_size)])
            for _ in range(batch_size):
                torch.manual_seed(seed)
                tens.append(torch.randn(img_shape, device=self.cdevice))
                seed += 1
            noise = torch.cat(tens)
            del tens
            try:
                self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=False)
            except Exception:
                self.make_schedule(ddim_num_steps=S + 1, ddim_eta=eta, verbose=False)

        x_latent = noise if x0 is None else x0
        # sampling
        if sampler == "plms":
            print(f'Data shape for PLMS sampling is {shape}')
            samples = self.plms_sampling(conditioning, batch_size, x_latent,
                                         callback=callback,
                                         img_callback=img_callback,
                                         quantize_denoised=quantize_x0,
                                         mask=mask, x0=x0,
                                         ddim_use_original_steps=False,
                                         noise_dropout=noise_dropout,
                                         temperature=temperature,
                                         score_corrector=score_corrector,
                                         corrector_kwargs=corrector_kwargs,
                                         log_every_t=log_every_t,
                                         unconditional_guidance_scale=unconditional_guidance_scale,
                                         unconditional_conditioning=unconditional_conditioning,
                                         )

        elif sampler == "ddim":
            samples = self.ddim_sampling(x_latent, conditioning, S,
                                         unconditional_guidance_scale=unconditional_guidance_scale,
                                         unconditional_conditioning=unconditional_conditioning,
                                         mask=mask, init_latent=x_T, use_original_steps=False)

        elif sampler == 'dpmpp_2m':
            model_wrap = K.external.CompVisDenoiser(self)
            sigmas = model_wrap.get_sigmas(S)
            # sigmas = K.sampling.get_sigmas_karras(n=S, sigma_min=0.1, sigma_max=10, device='cpu')
            if mask is not None:
                x0_noisy = x_T if x_T is not None else torch.randn_like(x_latent)
                x_latent = x0_noisy * mask + (1. - mask) * x_latent
            x_latent = x_latent * sigmas[0]
            model_wrap_cfg = CFGDenoiser(model_wrap)
            samples = self.sample_dpmpp_2m(model_wrap_cfg, x_latent, sigmas,
                                           extra_args={'cond': conditioning,
                                                       'uncond': unconditional_conditioning,
                                                       'cond_scale': unconditional_guidance_scale},
                                           disable=False)
        elif sampler == 'dpmpp_2s_ancestral':
            model_wrap = K.external.CompVisDenoiser(self)
            sigmas = model_wrap.get_sigmas(S)
            # sigmas = K.sampling.get_sigmas_karras(n=S, sigma_min=0.1, sigma_max=10, device='cpu')

            if mask is not None:
                x0_noisy = x_T if x_T is not None else torch.randn_like(x_latent)
                x_latent = x0_noisy * mask + (1. - mask) * x_latent
            x_latent = x_latent * sigmas[0]
            model_wrap_cfg = CFGDenoiser(model_wrap)
            samples = self.sample_dpmpp_2s_ancestral(model_wrap_cfg, x_latent, sigmas,
                                                     extra_args={'cond': conditioning,
                                                                 'uncond': unconditional_conditioning,
                                                                 'cond_scale': unconditional_guidance_scale},
                                                     disable=False)
        else:
            if sampler == 'dpm_a':
                sampler = KDiffusionSampler(self, 'dpm_2_ancestral')
            elif sampler == 'dpm_2':
                sampler = KDiffusionSampler(self, 'dpm_2')
            elif sampler == 'euler_a':
                sampler = KDiffusionSampler(self, 'euler_ancestral')
            elif sampler == 'euler':
                sampler = KDiffusionSampler(self, 'euler')
            elif sampler == 'heun':
                sampler = KDiffusionSampler(self, 'heun')
            elif sampler == 'lms':
                sampler = KDiffusionSampler(self, 'lms')

            samples = sampler.sample(S=S, conditioning=conditioning, batch_size=int(x_latent.shape[0]),
                                     shape=x_latent[0].shape, verbose=False, S_ddim_steps=S_ddim_steps,
                                     unconditional_guidance_scale=unconditional_guidance_scale, mask=mask,
                                     unconditional_conditioning=unconditional_conditioning, eta=0.0, x_T=x_latent)

        return samples

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start)).to(x_start.device)
        return (extract_into_tensor(self.sqrt_alphas_cumprod.to(x_start.device), t.to(x_start.device),
                                    x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t.to(x_start.device),
                                    x_start.shape) * noise)

    @torch.no_grad()
    def plms_sampling(self, cond, b, img,
                      ddim_use_original_steps=False,
                      callback=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):

        device = self.betas.device
        timesteps = self.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running PLMS Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='PLMS Sampler', total=total_steps)
        old_eps = []

        for i, step in enumerate(iterator):
            self.current_step = i
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts).to(img.device)
                img = img_orig * mask + (1. - mask) * img
                del img_orig

            outs = self.p_sample_plms(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      old_eps=old_eps, t_next=ts_next)
            img, pred_x0, e_t = outs
            old_eps.append(e_t)
            if len(old_eps) >= 4:
                old_eps.pop(0)
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

        return img

    @torch.no_grad()
    def p_sample_plms(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, old_eps=None, t_next=None):
        b, *_, device = *x.shape, x.device

        def get_model_output(x, t):
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.apply_model(x, t, c)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            if score_corrector is not None:
                assert self.parameterization == "eps"
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

            return e_t

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas

        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

            if quantize_denoised:
                pred_x0, _, *_ = self.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        e_t = get_model_output(x, t)
        if self.parameterization == "v":
            e_t = self.predict_eps_from_z_and_v(x, t, e_t)

        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next = get_model_output(x_prev, t_next)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        return x_prev, pred_x0, e_t

    @torch.no_grad()
    def ddim_sampling(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
                      mask=None, init_latent=None, use_original_steps=False):

        timesteps = self.ddim_timesteps
        timesteps = timesteps[:t_start]
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            self.current_step = i
            x0 = init_latent if init_latent is not None else torch.randn_like(x_dec)
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            if mask is not None and x0.shape[1] != 9:
                # x0_noisy = self.add_noise(mask, torch.tensor([index] * x0.shape[0]).to(self.cdevice))
                x0_noisy = x0
                x_dec = x0_noisy * mask + (1. - mask) * x_dec

            x_dec = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                       unconditional_guidance_scale=unconditional_guidance_scale,
                                       unconditional_conditioning=unconditional_conditioning)

        if mask is not None:
            return x0 * mask + (1. - mask) * x_dec

        return x_dec

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            model_uncond, model_t = self.apply_model(x_in, t_in, c_in).chunk(2)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        # print(self.parameterization)
        # e_t = self.predict_eps_from_z_and_v(x, t, model_output)
        if self.parameterization.strip() == "v":
            e_t = self.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

        if quantize_denoised:
            pred_x0, _, *_ = self.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev

    def append_zero(self, x):
        return torch.cat([x, x.new_zeros([1])])

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7., device='cpu'):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = torch.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return self.append_zero(sigmas).to(device)

    def get_sigmas_exponential(self, n, sigma_min, sigma_max, device='cpu'):
        """Constructs an exponential noise schedule."""
        sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
        return self.append_zero(sigmas)

    def get_sigmas_vp(self, n, beta_d=19.9, beta_min=0.1, eps_s=1e-3, device='cpu'):
        """Constructs a continuous VP noise schedule."""
        t = torch.linspace(1, eps_s, n, device=device)
        sigmas = torch.sqrt(torch.exp(beta_d * t ** 2 / 2 + beta_min * t) - 1)
        return self.append_zero(sigmas)

    def to_d(self, x, sigma, denoised):
        """Converts a denoiser output to a Karras ODE derivative."""
        return (x - denoised) / self.append_dims(sigma, x.ndim)

    def append_dims(self, x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
        return x[(...,) + (None,) * dims_to_append]

    def get_ancestral_step(self, sigma_from, sigma_to, eta=1.):
        """Calculates the noise level (sigma_down) to step down to and the amount
        of noise to add (sigma_up) when doing an ancestral sampling step."""
        if not eta:
            return sigma_to, 0.
        sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
        sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
        return sigma_down, sigma_up

    @torch.no_grad()
    def euler_sampling(self, x, sigmas, cond, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0.,
                       s_tmax=float('inf'), s_noise=1.):
        """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
        extra_args = {} if extra_args is None else extra_args
        s_in = x.new_ones([x.shape[0]]).half()
        for i in trange(len(sigmas) - 1, disable=disable):
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
            eps = torch.randn_like(x) * s_noise
            sigma_hat = (sigmas[i] * (gamma + 1)).half()
            if gamma > 0:
                x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
            denoised = self.apply_model(x, sigma_hat * s_in, cond)
            d = self.to_d(x, sigma_hat, denoised)
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
            dt = sigmas[i + 1] - sigma_hat
            # Euler method
            x = x + d * dt
        return x

    @torch.no_grad()
    def heun_sampling(self, x, sigmas, cond, unconditional_conditioning=None, unconditional_guidance_scale=1,
                      extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'),
                      s_noise=1.):
        """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
        extra_args = {} if extra_args is None else extra_args
        print(sigmas)
        # sigmas = self.get_sigmas_karras(steps, 0.01, 80, rho=7., device=x.device)
        print(x[0])
        x = x * sigmas[0]
        print("alu", x[0])
        s_in = x.new_ones([x.shape[0]])
        for i in trange(len(sigmas) - 1, disable=disable):
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
            eps = torch.randn_like(x) * s_noise
            sigma_hat = (sigmas[i] * (gamma + 1)).half()
            if gamma > 0:
                x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5

            # print(sigma_hat * s_in)
            # x, sigma_hat * s_in, cond
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([sigma_hat * s_in] * 2)
            c_in = torch.cat([unconditional_conditioning, cond])
            e_t_uncond, e_t = self.apply_model(x_in, t_in, c_in).chunk(2)
            denoised = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            # denoised = self.apply_model(x, sigma_hat * s_in, cond)
            d = self.to_d(x, sigma_hat, denoised)
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
            dt = sigmas[i + 1] - sigma_hat
            # if sigmas[i + 1] == 0:
            if 1:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                denoised_2 = self.apply_model(x_2, sigmas[i + 1] * s_in, cond)
                d_2 = self.to_d(x_2, sigmas[i + 1], denoised_2)
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
        return x

    @torch.no_grad()
    def sample_dpmpp_2s_ancestral(self, model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.,
                                  s_noise=1.):
        """Ancestral sampling with DPM-Solver++(2S) second-order steps."""
        extra_args = {} if extra_args is None else extra_args
        s_in = x.new_ones([x.shape[0]])
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

        for i in trange(len(sigmas) - 1, disable=disable):
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            sigma_down, sigma_up = self.get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
            if sigma_down == 0:
                # Euler method
                d = self.to_d(x, sigmas[i], denoised)
                dt = sigma_down - sigmas[i]
                x = x + d * dt
            else:
                # DPM-Solver-2++(2S)
                t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
                r = 1 / 2
                h = t_next - t
                s = t + r * h
                x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
                denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
            # Noise addition
            x = x + torch.randn_like(x) * s_noise * sigma_up
        return x

    @torch.no_grad()
    def sample_dpmpp_2m(self, model, x, sigmas, extra_args=None, callback=None, disable=None):
        """DPM-Solver++(2M)."""
        extra_args = {} if extra_args is None else extra_args
        s_in = x.new_ones([x.shape[0]])
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()
        old_denoised = None

        for i in trange(len(sigmas) - 1, disable=disable):
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            if old_denoised is None or sigmas[i + 1] == 0:
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
            else:
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
            old_denoised = denoised
        return x
