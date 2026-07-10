import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
import numpy as np


class TwinsFeaturePyramid(nn.Module):
    """Two-level Twins-SVT feature extractor.

    Args:
        model_name: Name passed to ``timm.create_model``.
        pretrained: Whether to load ImageNet-pretrained weights.

    Returns:
        Forward returns a tuple ``(stride8, stride4)`` with shapes
        ``[B, C8, H/8, W/8]`` and ``[B, C4, H/4, W/4]``.
    """

    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.svt = timm.create_model(model_name, pretrained=pretrained)

        del self.svt.head
        del self.svt.patch_embeds[2]
        del self.svt.patch_embeds[2]
        del self.svt.blocks[2]
        del self.svt.blocks[2]
        del self.svt.pos_block[2]
        del self.svt.pos_block[2]

    def forward(self, x, data=None, layer=2):
        batch_size = x.shape[0]
        stride4_features = None
        for stage_index, (embed, drop, blocks, pos_block) in enumerate(self._iter_stages()):
            if stage_index == layer - 1:
                stride4_features = self._extract_stride4_branch(
                    x,
                    batch_size,
                    embed,
                    drop,
                    blocks,
                    pos_block,
                    stage_index,
                )

            x = self._run_stage(x, batch_size, embed, drop, blocks, pos_block, stage_index)
            if stage_index == layer - 1:
                break

        return x, stride4_features

    def compute_params(self, layer=2):
        num = 0
        for i, (embed, drop, blocks, pos_blk) in enumerate(self._iter_stages()):

            for param in embed.parameters():
                num +=  np.prod(param.size())

            for param in drop.parameters():
                num +=  np.prod(param.size())

            for param in blocks.parameters():
                num +=  np.prod(param.size())

            for param in pos_blk.parameters():
                num +=  np.prod(param.size())

            if i == layer-1:
                break
        
        return num

    def _iter_stages(self):
        return zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)

    def _extract_stride4_branch(self, x, batch_size, embed, drop, blocks, pos_block, stage_index):
        original_patch_size = embed.patch_size
        embed.patch_size = (1, 1)
        embed.proj.stride = embed.patch_size
        stride4 = torch.nn.functional.pad(x, [1, 0, 1, 0], mode='constant', value=0)
        stride4, feature_size = embed(stride4)
        feature_size = (feature_size[0] - 1, feature_size[1] - 1)
        stride4 = self._apply_transformer_blocks(stride4, feature_size, drop, blocks, pos_block)
        if stage_index < len(self.svt.depths) - 1:
            stride4 = stride4.reshape(batch_size, *feature_size, -1).permute(0, 3, 1, 2).contiguous()
        embed.patch_size = original_patch_size
        embed.proj.stride = original_patch_size
        return stride4

    def _run_stage(self, x, batch_size, embed, drop, blocks, pos_block, stage_index):
        x, feature_size = embed(x)
        x = self._apply_transformer_blocks(x, feature_size, drop, blocks, pos_block)
        if stage_index < len(self.svt.depths) - 1:
            x = x.reshape(batch_size, *feature_size, -1).permute(0, 3, 1, 2).contiguous()
        return x

    @staticmethod
    def _apply_transformer_blocks(tokens, feature_size, drop, blocks, pos_block):
        tokens = drop(tokens)
        for block_index, block in enumerate(blocks):
            tokens = block(tokens, feature_size)
            if block_index == 0:
                tokens = pos_block(tokens, feature_size)
        return tokens


class LargeMotionEncoder(TwinsFeaturePyramid):
    """Large Twins-SVT image-pair encoder."""

    def __init__(self, pretrained=True):
        super().__init__('twins_svt_large', pretrained=pretrained)


class SmallContextEncoder(TwinsFeaturePyramid):
    """Small Twins-SVT context encoder."""

    def __init__(self, pretrained=True):
        super().__init__('twins_svt_small', pretrained=pretrained)


twins_svt_large = LargeMotionEncoder
twins_svt_small_context = SmallContextEncoder
