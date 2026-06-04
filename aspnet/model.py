import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from aspnet.modules.attention import *


class ASPNet(nn.Module):

    def __init__(self, args, adim, tdim, vdim, D_e, n_classes, pdim=0, depth=4, num_heads=4, mlp_ratio=1, drop_rate=0, attn_drop_rate=0, no_cuda=False):
        super(ASPNet, self).__init__()
        self.n_classes = n_classes
        self.D_e = D_e
        self.num_heads = num_heads
        D = 3 * D_e
        self.device = args.device
        self.no_cuda = no_cuda
        self.adim, self.tdim, self.vdim, self.pdim = adim, tdim, vdim, pdim
        self.out_dropout = args.drop_rate
        self.use_semantic_alignment = bool(getattr(args, 'use_semantic_alignment', False) and pdim > 0)

        self.a_in_proj = nn.Sequential(nn.Linear(self.adim, D_e))
        self.t_in_proj = nn.Sequential(nn.Linear(self.tdim, D_e))
        self.v_in_proj = nn.Sequential(nn.Linear(self.vdim, D_e))
        if self.use_semantic_alignment:
            self.p_in_proj = nn.Sequential(nn.Linear(self.pdim, D_e))
            spatial_layer = nn.TransformerEncoderLayer(
                d_model=D_e,
                nhead=num_heads,
                dim_feedforward=D_e * 4,
                dropout=drop_rate,
                batch_first=True,
                activation='gelu',
            )
            self.prompt_spatial = nn.TransformerEncoder(spatial_layer, num_layers=1)
            self.prompt_channel = nn.Sequential(
                nn.Linear(D_e * 2, D_e * 2),
                nn.GELU(),
                nn.Dropout(drop_rate),
                nn.Linear(D_e * 2, D_e),
            )
            self.prompt_gate = nn.Linear(D_e * 2, D_e)
            nn.init.xavier_uniform_(self.prompt_gate.weight)
            nn.init.zeros_(self.prompt_gate.bias)
            self.prompt_norm = nn.LayerNorm(D_e)
        self.dropout_a = nn.Dropout(args.drop_rate)
        self.dropout_t = nn.Dropout(args.drop_rate)
        self.dropout_v = nn.Dropout(args.drop_rate)
        self.dropout_p = nn.Dropout(args.drop_rate)

        self.block = Block(
                    dim=D_e,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    depth=depth,
                )
        self.proj1 = nn.Linear(D, D)
        self.nlp_head_a = nn.Linear(D_e, n_classes)
        self.nlp_head_t = nn.Linear(D_e, n_classes)
        self.nlp_head_v = nn.Linear(D_e, n_classes)
        self.nlp_head = nn.Linear(D, n_classes)

    def _condition_from_mask(self, input_mask):
        available = input_mask.sum(dim=(0, 1)) > 0
        condition = ''.join([name for flag, name in zip(available.tolist(), ['a', 't', 'v']) if flag])
        return condition or 'atv'

    def _align_prompt(self, x_a, x_t, x_v, input_mask, umask, prompt_features):
        if not self.use_semantic_alignment or prompt_features is None:
            return x_a, x_t, x_v

        B, seq_len, _ = x_a.shape
        prompt = self.dropout_p(self.p_in_proj(prompt_features))
        token_mask = input_mask.bool() & umask.unsqueeze(-1).bool()
        prompt_mask = umask.reshape(B * seq_len).bool()

        modality_tokens = torch.stack([x_a, x_t, x_v], dim=2).reshape(B * seq_len, 3, self.D_e)
        prompt_tokens = prompt.reshape(B * seq_len, 1, self.D_e)
        spatial_tokens = torch.cat([modality_tokens, prompt_tokens], dim=1)
        spatial_mask = torch.cat([token_mask.reshape(B * seq_len, 3), prompt_mask.unsqueeze(-1)], dim=1)
        spatial_mask[~spatial_mask.any(dim=1), -1] = True
        spatial_out = self.prompt_spatial(spatial_tokens, src_key_padding_mask=~spatial_mask)[:, :3, :]

        mask_float = token_mask.reshape(B * seq_len, 3, 1).float()
        context = (spatial_out * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
        channel_delta = self.prompt_channel(torch.cat([context, prompt.reshape(B * seq_len, self.D_e)], dim=-1))
        channel_out = spatial_out + channel_delta.unsqueeze(1)

        enhanced_tokens = self.prompt_norm(channel_out)
        prompt_for_gate = prompt.reshape(B * seq_len, 1, self.D_e).expand(-1, 3, -1)
        gate = torch.sigmoid(self.prompt_gate(torch.cat([modality_tokens, prompt_for_gate], dim=-1)))
        fused_tokens = modality_tokens + gate * (enhanced_tokens - modality_tokens)
        fused_tokens = fused_tokens * mask_float
        fused_tokens = fused_tokens.reshape(B, seq_len, 3, self.D_e)
        return fused_tokens[:, :, 0, :], fused_tokens[:, :, 1, :], fused_tokens[:, :, 2, :]

    def _fuse_available_modalities(self, x_a, x_t, x_v, input_mask, first_stage):
        condition = 'atv' if first_stage else self._condition_from_mask(input_mask)
        zeros = torch.zeros_like(x_a)
        x_joint = torch.cat([
            x_a if 'a' in condition else zeros,
            x_t if 't' in condition else zeros,
            x_v if 'v' in condition else zeros,
        ], dim=-1)
        return x_joint, condition

    def forward(self, inputfeats, input_features_mask=None, umask=None, first_stage=False, prompt_features=None):
        """
        inputfeats -> ?*[seqlen, batch, dim]
        qmask -> [batch, seqlen]
        umask -> [batch, seqlen]
        seq_lengths -> each conversation lens
        input_features_mask -> ?*[seqlen, batch, 3]
        """
        # print(inputfeats[:,:,:])
        # print(input_features_mask[:,:,1])
        weight_save = []
        # sequence modeling
        audio, text, video = inputfeats[:, :, :self.adim], inputfeats[:, :, self.adim:self.adim + self.tdim], \
        inputfeats[:, :, self.adim + self.tdim:]
        seq_len, B, C = audio.shape

        # --> [batch, seqlen, dim]
        audio, text, video = audio.permute(1, 0, 2), text.permute(1, 0, 2), video.permute(1, 0, 2)
        proj_a = self.dropout_a(self.a_in_proj(audio))
        proj_t = self.dropout_t(self.t_in_proj(text))
        proj_v = self.dropout_v(self.v_in_proj(video))
        # --> [batch, seqlen, 3]
        input_mask = torch.clone(input_features_mask.permute(1, 0, 2))
        input_mask[umask == 0] = 0
        # --> [batch, 3, seqlen] -> [batch, 3*seqlen]
        attn_mask = input_mask.transpose(1, 2).reshape(B, -1)

        if first_stage:
            # --> [batch, seqlen, dim]
            x_a = self.block(proj_a, first_stage, attn_mask, 'a')
            x_t = self.block(proj_t, first_stage, attn_mask, 't')
            x_v = self.block(proj_v, first_stage, attn_mask, 'v')
            out_a = self.nlp_head_a(x_a)
            out_t = self.nlp_head_t(x_t)
            out_v = self.nlp_head_v(x_v)
            x = torch.cat([x_a, x_t, x_v], dim=1)
        else:
            out_a = torch.rand((B, seq_len, self.n_classes), device=inputfeats.device)
            out_t = torch.rand((B, seq_len, self.n_classes), device=inputfeats.device)
            out_v = torch.rand((B, seq_len, self.n_classes), device=inputfeats.device)
            x_a, x_t, x_v = self._align_prompt(proj_a, proj_t, proj_v, input_mask, umask, prompt_features)
            x = torch.cat([x_a, x_t, x_v], dim=1)

        x[attn_mask == 0] = 0

        x_a, x_t, x_v = x[:, :seq_len, :], x[:, seq_len:2*seq_len, :], x[:, 2*seq_len:, :]
        x_joint, _ = self._fuse_available_modalities(x_a, x_t, x_v, input_mask, first_stage)
        res = x_joint
        u = F.relu(self.proj1(x_joint))
        u = F.dropout(u, p=self.out_dropout, training=self.training)
        hidden = u + res
        out = self.nlp_head(hidden)

        return hidden, out, out_a, out_t, out_v, np.array(weight_save)


PromptFusionNetwork = ASPNet
