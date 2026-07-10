import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from pathlib import Path
from contextlib import nullcontext

from corr import CorrBlock
from utils.utils import coords_grid

from fd_encoder import LargeMotionEncoder, SmallContextEncoder
from fd_decoder import (
    ConvexUpsampleMask8x,
    ConvexUpsampleMask4x,
    FeatureCorrelationBuilder,
    MotionRefinementBlock,
    DiffusionMotionUpdateBlock,
)
from fd_corr import CorrBlock_FD_Sp4
from MFlownet import MaskFlownet_S_Compat as base_model

LOGGER = logging.getLogger(__name__)

def autocast(enabled):
    if not enabled:
        return nullcontext()

    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_type == 'cuda':
        return torch.amp.autocast(device_type=device_type)

    return torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def ste_round(x):
    return torch.round(x) - x.detach() + x


class FlowDiffuser(nn.Module):
    def __init__(self, args):
        super().__init__()
        LOGGER.info("Initializing FlowPainter model")

        args.corr_levels = 4
        args.corr_radius = 4
        args.m_dim = 256
        args.c_dim = c_dim = 128
        args.iters_const6 = 6 
        
        self.args = args
        self.args.UpdateBlock = 'SKUpdateBlock6_Deep_nopoolres_AllDecoder'
        self.args.k_conv = [1, 15]
        self.args.PCUpdater_conv = [1, 7] 
        self._deform_conv_cuda_warned = False
        self.sp4 = True
        self.rad = 8

        self.base_model, self.base_model_device = self._load_flow_prior_model(args)

        self.fnet = LargeMotionEncoder(pretrained=True)
        self.cnet = SmallContextEncoder(pretrained=True)
        self.trans = FeatureCorrelationBuilder(args) 
        self.C_inp = nn.Conv2d(in_channels=c_dim, out_channels=c_dim, kernel_size=1)
        self.C_net = nn.Conv2d(in_channels=c_dim, out_channels=c_dim, kernel_size=1)
        self.update = MotionRefinementBlock(self.args, hidden_dim=c_dim)
        self.um8 = ConvexUpsampleMask8x(c_dim)
        self.um4 = ConvexUpsampleMask4x(c_dim)
        self.zero = nn.Parameter(torch.zeros(12), requires_grad=False)

        self.diffusion = True
        if self.diffusion:
            self.update_dfm = DiffusionMotionUpdateBlock(self.args, hidden_dim=c_dim)

            timesteps = 1000
            sampling_timesteps = 4
            recurr_itrs = 6
            LOGGER.info("Diffusion denoise steps: %d", sampling_timesteps)
            LOGGER.info("Recurrent refinement iterations: %d", recurr_itrs)

            self.ddim_n = sampling_timesteps
            self.recurr_itrs = recurr_itrs
            self.n_sc = 0.1
            self.scale = nn.Parameter(torch.ones(1) * 0.5, requires_grad=False)
            self.n_lambda = 0.2

            self.base_flow_weight = 0.3
            self.mask_threshold = 0.5
            self.base_guidance_scale = 1.0

            self.objective = 'pred_x0'  
            betas = cosine_beta_schedule(timesteps)
            alphas = 1. - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
            timesteps, = betas.shape
            self.num_timesteps = int(timesteps)

            self.sampling_timesteps = default(sampling_timesteps, timesteps)
            assert self.sampling_timesteps <= timesteps
            self.is_ddim_sampling = self.sampling_timesteps < timesteps
            self.ddim_sampling_eta = 1.
            self.self_condition = False

            self.register_buffer('betas', betas)
            self.register_buffer('alphas_cumprod', alphas_cumprod)
            self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
            self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
            self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
            self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
            self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
            self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

            posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
            self.register_buffer('posterior_variance', posterior_variance)
            self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
            self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
            self.register_buffer('posterior_mean_coef2',
                                (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def _load_flow_prior_model(self, args):
        """Load the auxiliary flow prior used for confidence-guided refinement.

        Args:
            args: Runtime namespace. ``args.base_model_ckpt`` may override the
                default relative checkpoint path.

        Returns:
            A tuple containing the initialized prior model and its default device.
        """
        prior_model = base_model()
        prior_device = torch.device('cpu')
        default_base_ckpt = Path(__file__).resolve().parents[1] / 'weights' / '100000_raft-sintel.pth'
        base_ckpt = Path(getattr(args, 'base_model_ckpt', default_base_ckpt)).expanduser()
        if not base_ckpt.is_absolute():
            base_ckpt = Path(__file__).resolve().parents[1] / base_ckpt
        try:
            checkpoint = torch.load(base_ckpt, map_location='cpu')
        except FileNotFoundError as exc:
            raise RuntimeError(f"Base flow checkpoint not found: {base_ckpt}") from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to load base flow checkpoint {base_ckpt}: {exc}") from exc

        model_dict = prior_model.state_dict()
        for checkpoint_key, value in checkpoint.items():
            model_key = checkpoint_key[7:]
            if model_key in model_dict:
                model_dict[model_key] = value
            else:
                LOGGER.debug("Ignoring unmatched base-flow checkpoint key: %s", model_key)
        prior_model.load_state_dict(model_dict)
        prior_model.to(prior_device)
        return prior_model, prior_device

    def set_base_model_device(self, device):
        self.base_model_device = torch.device(device)
        self.base_model.to(self.base_model_device)
        return self

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def up_sample_flow8(self, flow, mask):
        B, _, H, W = flow.shape
        flow = torch.nn.functional.unfold(8 * flow, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1])
        flow = flow.reshape(B, 2, 9, 1, 1, H, W)
        mask = mask.reshape(B, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)
        up_flow = torch.sum(flow * mask, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3).contiguous()
        up_flow = up_flow.reshape(B, 2, H * 8, W * 8)
        return up_flow

    def up_sample_flow4(self, flow, mask):
        B, _, H, W = flow.shape
        flow = torch.nn.functional.unfold(4 * flow, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1])
        flow = flow.reshape(B, 2, 9, 1, 1, H, W)
        mask = mask.reshape(B, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)
        up_flow = torch.sum(flow * mask, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3).contiguous()
        up_flow = up_flow.reshape(B, 2, H * 4, W * 4)
        return up_flow

    def initialize_flow8(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device).permute(0, 2, 3, 1).contiguous()
        coords1 = coords_grid(N, H//8, W//8, device=img.device).permute(0, 2, 3, 1).contiguous()
        return coords0, coords1

    def initialize_flow4(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//4, W//4, device=img.device).permute(0, 2, 3, 1).contiguous()
        coords1 = coords_grid(N, H//4, W//4, device=img.device).permute(0, 2, 3, 1).contiguous()
        return coords0, coords1

    def _cache_prior_pyramid(self, flow_prior, confidence_logits, image_shape):
        """Prepare prior flow and confidence maps at model working scales.

        Args:
            flow_prior: Full-resolution prior flow shaped ``[B, 2, H, W]``.
            confidence_logits: Prior confidence logits shaped ``[B, C, H, W]``.
            image_shape: Input image shape ``[B, 3, H, W]``.

        Returns:
            None.
        """
        _, _, height, width = image_shape
        self.prior_flow_stride8 = F.interpolate(
            flow_prior, size=(height // 8, width // 8), mode='bilinear', align_corners=True
        ) / 8.0
        self.prior_confidence_stride8 = torch.sigmoid(
            F.interpolate(confidence_logits, size=(height // 8, width // 8), mode='bilinear', align_corners=True)
        )
        self.prior_flow_stride4 = F.interpolate(
            flow_prior, size=(height // 4, width // 4), mode='bilinear', align_corners=True
        ) / 4.0
        self.prior_confidence_stride4 = torch.sigmoid(
            F.interpolate(confidence_logits, size=(height // 4, width // 4), mode='bilinear', align_corners=True)
        )

    def _initialize_with_confident_prior(self, coords0, coords1, prior_flow, prior_confidence):
        """Replace low-confidence coordinate guesses with trusted prior flow.

        Args:
            coords0: Reference coordinate grid shaped ``[B, 2, H, W]``.
            coords1: Current target coordinate grid shaped ``[B, 2, H, W]``.
            prior_flow: Prior flow at the same scale, shaped ``[B, 2, H, W]``.
            prior_confidence: Confidence map broadcastable to ``prior_flow``.

        Returns:
            Updated target coordinate grid shaped ``[B, 2, H, W]``.
        """
        trusted = (prior_confidence > self.mask_threshold).float()
        return coords1 * (1 - trusted) + (coords0 + prior_flow) * trusted

    def _apply_prior_guidance(self, coords1, coords0, prior_flow, prior_confidence, strength):
        """Nudge current flow toward the auxiliary prior without changing tensor shapes.

        Args:
            coords1: Current target coordinates shaped ``[B, 2, H, W]``.
            coords0: Reference coordinates shaped ``[B, 2, H, W]``.
            prior_flow: Prior flow at the same scale, shaped ``[B, 2, H, W]``.
            prior_confidence: Confidence map broadcastable to ``prior_flow``.
            strength: Scalar or tensor guidance strength.

        Returns:
            Updated target coordinates shaped ``[B, 2, H, W]``.
        """
        current_flow = coords1 - coords0
        confidence = prior_confidence.clamp(min=0, max=1)
        return coords1 + (prior_flow - current_flow) * confidence * strength

    def _train_dfm(self, feat_shape, flow_gt, net, inp8, coords0, coords1):
        b, c, h, w = feat_shape
        if len(flow_gt.shape) == 3:
            flow_gt = flow_gt.unsqueeze(0)
        flow_gt_sp8 = F.interpolate(flow_gt, (h, w), mode='bilinear', align_corners=True) / 8. 

        x_t, noises, t = self._prepare_targets(flow_gt_sp8) 
        x_t = x_t * self.norm_const
        coords1 = coords1 + x_t.float()

        flow_up_s = []
        for ii in range(self.recurr_itrs):  
            t_ii = (t - t / self.recurr_itrs * ii).int() 

            itr = ii
            first_step = False if itr != 0 else True

            coords1 = coords1.detach()
            corr = self.corr_fn(coords1)
            flow = coords1 - coords0

            if hasattr(self, 'prior_flow_stride8') and hasattr(self, 'prior_confidence_stride8'):
                with torch.no_grad():
                    guidance_strength = (1.0 - t_ii.float() / self.num_timesteps) * self.base_guidance_scale
                    coords1 = self._apply_prior_guidance(
                        coords1,
                        coords0,
                        self.prior_flow_stride8,
                        self.prior_confidence_stride8,
                        guidance_strength * self.base_flow_weight,
                    )
            
            with autocast(enabled=self.args.mixed_precision):
                dfm_params = [t_ii, self.update, ii, 0]  
                net, delta_flow = self.update_dfm(net, inp8, corr, flow, itr, first_step=first_step, dfm_params=dfm_params)  
                up_mask = self.um8(net)

            coords1 = coords1 + delta_flow
            flow = coords1 - coords0

            flow_up = self.up_sample_flow8(flow, up_mask)
            flow_up_s.append(flow_up)

        return flow_up_s, coords1, net

    def _prepare_targets(self, flow_gt):
        noise = torch.randn(flow_gt.shape, device=self.device)
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()

        x_start = flow_gt / self.norm_const
        x_start = x_start * self.scale 
        x_t = self._q_sample(x_start=x_start, t=t, noise=noise)
        x_t = torch.clamp(x_t, min=-1, max=1)  
        x_t = x_t * self.n_sc  
        return x_t, noise, t

    def _q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def _ddim_sample(self, feat_shape, net, inp, coords0, coords1_init, clip_denoised=True):
        batch, c, h, w = feat_shape
        shape = (batch, 2, h, w)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective  
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) 
        x_in = torch.randn(shape, device=self.device)

        flow_s = []
        x_start = None
        pred_s = None
        for i_ddim, time_s in enumerate(time_pairs):
            time, time_next = time_s
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            t_next = torch.full((batch,), time_next, device=self.device, dtype=torch.long)

            x_pred, inner_flow_s, pred_s = self._model_predictions(x_in, time_cond, net, inp, coords0, coords1_init, i_ddim, pred_s, t_next)
            flow_s = flow_s + inner_flow_s

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            x_t = x_in 
            x_pred = x_pred * self.scale
            x_pred = torch.clamp(x_pred, min=-1 * self.scale, max=self.scale)
            eps = (1 / (1 - alpha).sqrt()) * (x_t - alpha.sqrt() * x_pred)
            x_next = alpha_next.sqrt() * x_pred + (1 - alpha_next).sqrt() * eps
            x_in = x_next

        net, up_mask, coords1 = pred_s

        return coords1, net, flow_s

    def _model_predictions(self, x, t, net, inp8, coords0, coords1, i_ddim, pred_last=None, t_next=None):
        x_flow = torch.clamp(x, min=-1, max=1)
        x_flow = x_flow * self.n_sc
        x_flow = x_flow * self.norm_const

        if pred_last:
            net, _, coords1 = pred_last
            x_flow = x_flow * self.n_lambda

        coords1 = coords1 + x_flow.float()

        flow_s = []
        for ii in range(self.recurr_itrs):
            t_ii = (t - (t - 0) / self.recurr_itrs * ii).int()

            corr = self.corr_fn(coords1)
            flow = coords1 - coords0

            if hasattr(self, 'prior_flow_stride8') and hasattr(self, 'prior_confidence_stride8'):
                with torch.no_grad():
                    guidance_strength = (1.0 - t_ii.float() / self.num_timesteps) * self.base_guidance_scale
                    coords1 = self._apply_prior_guidance(
                        coords1,
                        coords0,
                        self.prior_flow_stride8,
                        self.prior_confidence_stride8,
                        guidance_strength * self.base_flow_weight,
                    )

            with autocast(enabled=self.args.mixed_precision):
                itr = ii
                first_step = False if itr != 0 else True
                dfm_params = [t_ii, self.update, ii, 0]
                net, delta_flow = self.update_dfm(net, inp8, corr, flow, itr, first_step=first_step, dfm_params=dfm_params)
                up_mask = self.um8(net)

            coords1 = coords1 + delta_flow

            flow = coords1 - coords0
            flow_up = self.up_sample_flow8(flow, up_mask)

            flow_s.append(flow_up)

        flow = coords1 - coords0 
        x_pred = flow / self.norm_const

        return x_pred, flow_s, [net, up_mask, coords1]    

    def _predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def forward(self, image1, image2, test_mode=False, iters=None, flow_gt=None, flow_init=None):

        self.base_model.eval()
        model_device = image1.device
        base_device = getattr(self, 'base_model_device', model_device)
        with torch.no_grad():
            try:
                flow_base, mask_base = self.base_model(
                    image1.to(base_device),
                    image2.to(base_device),
                )
            except NotImplementedError as exc:
                if base_device.type != 'cuda' or 'torchvision::deform_conv2d' not in str(exc):
                    raise
                if not self._deform_conv_cuda_warned:
                    LOGGER.warning(
                        "torchvision deform_conv2d CUDA kernel is unavailable; running FlowPainter base model on cpu"
                    )
                    self._deform_conv_cuda_warned = True
                self.set_base_model_device('cpu')
                base_device = self.base_model_device
                flow_base, mask_base = self.base_model(
                    image1.to(base_device),
                    image2.to(base_device),
                )
            if flow_base.device != model_device:
                flow_base = flow_base.to(model_device)
                mask_base = mask_base.to(model_device)

        self._cache_prior_pyramid(flow_base, mask_base, image1.shape)

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        with autocast(enabled=self.args.mixed_precision):
            fmap = self.fnet(torch.cat([image1, image2], dim=0))
            inp = self.cnet(image1)

        fmap, fmap4 = fmap
        inp, inp4 = inp
        fmap = fmap.float()
        fmap4 = fmap4.float()
        inp = inp.float()
        inp4 = inp4.float()

        fmap1_4, fmap2_4 = torch.chunk(fmap4, chunks=2, dim=0)
        fmap1_8, fmap2_8 = torch.chunk(fmap, chunks=2, dim=0)
        inp8 = self.C_inp(inp)
        net = self.C_net(inp)

        corr_fn = self.trans(fmap1_8, fmap2_8, inp8)

        coords0, coords1 = self.initialize_flow8(image1)
        coords0 = coords0.permute(0, 3, 1, 2).contiguous()
        coords1 = coords1.permute(0, 3, 1, 2).contiguous()

        if hasattr(self, 'prior_flow_stride8'):
            coords1 = self._initialize_with_confident_prior(
                coords0,
                coords1,
                self.prior_flow_stride8,
                self.prior_confidence_stride8,
            )

        flow_list = []
        if flow_init is not None: 
            if flow_init.shape[-2:] != coords1.shape[-2:]:
                flow_init = F.interpolate(flow_init, coords1.shape[-2:], mode='bilinear', align_corners=True) * 0.5
            coords1 = coords1 + flow_init

        if self.diffusion:
            self.corr_fn = corr_fn
            self.device = fmap1_8.device
            h, w = fmap1_8.shape[-2:] 
            self.norm_const = torch.as_tensor([w, h], dtype=torch.float, device=self.device).view(1, 2, 1, 1)

            if self.training:
                coords1 = coords1.detach()
                flow_up_s, coords1, net = self._train_dfm(fmap1_8.shape, flow_gt, net, inp8, coords0, coords1)
            else: 
                coords1, net, flow_up_s = self._ddim_sample(fmap1_8.shape, net, inp8, coords0, coords1)

            if self.sp4:
                flow4 = torch.nn.functional.interpolate(2 * (coords1 - coords0), scale_factor=2, mode='bilinear', align_corners=True)
                coords0, coords1 = self.initialize_flow4(image1)
                coords0 = coords0.permute(0, 3, 1, 2).contiguous()
                coords1 = coords1.permute(0, 3, 1, 2).contiguous()
                coords1 = coords1 + flow4

                if hasattr(self, 'prior_flow_stride4'):
                    coords1 = self._apply_prior_guidance(
                        coords1,
                        coords0,
                        self.prior_flow_stride4,
                        (self.prior_confidence_stride4 > self.mask_threshold).float(),
                        0.5,
                    )

                net = torch.nn.functional.interpolate(net, scale_factor=2, mode='bilinear', align_corners=True)
                coords1_rd = ste_round(coords1)
                
                corr_fn4 = CorrBlock_FD_Sp4(fmap1_4, fmap2_4, num_levels=self.args.corr_levels, radius=self.args.corr_radius, coords_init=coords1_rd, rad=self.rad)

                for itr in range(self.args.iters_const6):
                    coords1 = coords1.detach()
                    corr = corr_fn4(coords1 - coords1_rd + self.rad)

                    flow = coords1 - coords0

                    if hasattr(self, 'prior_flow_stride4'):
                        with torch.no_grad():
                            iter_weight = max(0, 1.0 - itr / self.args.iters_const6) * 0.5
                            coords1 = self._apply_prior_guidance(
                                coords1,
                                coords0,
                                self.prior_flow_stride4,
                                self.prior_confidence_stride4,
                                iter_weight,
                            )
                            flow = coords1 - coords0

                    with autocast(enabled=self.args.mixed_precision):
                        net, delta_flow = self.update(net, inp4, corr, flow, itr, sp4=True)
                        up_mask = self.um4(net)

                    coords1 = coords1 + delta_flow
                    flow_up = self.up_sample_flow4(coords1 - coords0, up_mask)
                    flow_up_s.append(flow_up)

            flow_list = flow_list + flow_up_s

            if test_mode:
                flow = coords1 - coords0
                return flow, flow_list[-1]

            return flow_list
