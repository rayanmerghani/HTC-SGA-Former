# # Old data

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange

# # ============================================================================
# # CONVOLUTIONAL COMPONENTS
# # ============================================================================

# class Conv(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
#         super(Conv, self).__init__(
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
#                       dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
#         )

# class SeparableConvBN(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
#                   norm_layer=nn.BatchNorm2d):
#         super(SeparableConvBN, self).__init__(
#             nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
#                       padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
#                       groups=in_channels, bias=False),
#             norm_layer(in_channels),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         )

# class LightMlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features // 2
#         self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
#         self.act = act_layer()
#         self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
#         self.drop = nn.Dropout(drop, inplace=False)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x

# # ============================================================================
# # FIXED ENHANCED MGFA MODULE
# # ============================================================================

# class EnhancedMGFA_Module(nn.Module):
#     """
#     Fixed Enhanced Mask-guided Feature Attention for coronary vessels
#     """
#     def __init__(self, feature_dim=64, output_dim=64):
#         super().__init__()
        
#         # 1. Preliminary mask prediction
#         self.preliminary_mask = nn.Sequential(
#             nn.Conv2d(feature_dim, feature_dim//2, 3, padding=1),
#             nn.BatchNorm2d(feature_dim//2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(feature_dim//2, 1, 1)
#         )
        
#         # 2. Multi-scale context extraction - FIXED CHANNEL CALCULATION
#         # Each branch produces feature_dim//2 channels
#         self.context_conv1 = nn.Conv2d(feature_dim, feature_dim//2, 3, padding=1, dilation=1)
#         self.context_conv2 = nn.Conv2d(feature_dim, feature_dim//2, 3, padding=2, dilation=2)
#         self.context_conv3 = nn.Conv2d(feature_dim, feature_dim//2, 3, padding=4, dilation=4)
        
#         # 3. Context fusion (combines all 3 context branches)
#         self.context_fusion = nn.Sequential(
#             nn.Conv2d(feature_dim//2 * 3, feature_dim, 1),
#             nn.BatchNorm2d(feature_dim),
#             nn.ReLU(inplace=True)
#         )
        
#         # 4. Attention mechanism
#         self.attention = nn.Sequential(
#             nn.Conv2d(1, feature_dim//4, 3, padding=1),
#             nn.BatchNorm2d(feature_dim//4),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(feature_dim//4, feature_dim, 1),
#             nn.Sigmoid()
#         )
        
#         # 5. Feature fusion - FIXED: Now expects feature_dim * 2 channels
#         self.fusion = nn.Sequential(
#             nn.Conv2d(feature_dim * 2, output_dim, 3, padding=1),
#             nn.BatchNorm2d(output_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(0.1)
#         )
        
#         # 6. Final refinement
#         self.final_refine = nn.Sequential(
#             nn.Conv2d(output_dim, output_dim, 3, padding=1),
#             nn.BatchNorm2d(output_dim),
#             nn.ReLU(inplace=True)
#         )
        
#     def forward(self, features):
#         """
#         Args:
#             features: Input features [B, C, H, W]
#         Returns:
#             Refined features [B, C, H, W], preliminary mask [B, 1, H, W]
#         """
#         # Step 1: Generate preliminary mask
#         prelim_mask = self.preliminary_mask(features)
        
#         # Step 2: Extract multi-scale context
#         context1 = self.context_conv1(features)
#         context2 = self.context_conv2(features)
#         context3 = self.context_conv3(features)
        
#         # Combine contexts
#         context_combined = torch.cat([context1, context2, context3], dim=1)
#         context = self.context_fusion(context_combined)
        
#         # Step 3: Create attention from mask
#         attention = self.attention(prelim_mask)
        
#         # Step 4: Apply attention with residual
#         attended_features = features * (1 + attention)
        
#         # Step 5: Combine attended features with context
#         combined = torch.cat([attended_features, context], dim=1)  # Now [B, feature_dim*2, H, W]
#         fused = self.fusion(combined)
        
#         # Step 6: Final refinement
#         output = self.final_refine(fused)
        
#         return output, prelim_mask

# # ============================================================================
# # ENHANCED ULTRA EFFICIENT GUIDED ATTENTION
# # ============================================================================

# class EnhancedUltraEfficientGuidedAttention(nn.Module):
#     def __init__(self, dim=256, num_heads=8, qkv_bias=False, 
#                   window_size=8, reduction_ratio=2, use_enhanced_qkv=True):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // self.num_heads
#         self.scale = self.head_dim ** -0.5
#         self.ws = window_size
#         self.reduction_ratio = reduction_ratio
#         self.use_enhanced_qkv = use_enhanced_qkv
        
#         self.qkv_dim = dim // reduction_ratio
        
#         if use_enhanced_qkv:
#             self.qkv_conv3x3 = nn.Conv2d(dim, dim // 4, kernel_size=3, padding=1, bias=False)
#             self.qkv_conv1x1 = nn.Conv2d(dim, dim // 4, kernel_size=1, bias=False)
#             self.qkv_conv5x5 = nn.Conv2d(dim, dim // 8, kernel_size=5, padding=2, bias=False)
            
#             self.qkv_depthwise = nn.Conv2d(
#                 dim, dim, kernel_size=3, padding=1, groups=dim, bias=False
#             )
#             self.qkv_pointwise = nn.Conv2d(dim, dim // 8, kernel_size=1, bias=False)
#             self.dw_norm = nn.BatchNorm2d(dim)
#             self.dw_act = nn.SiLU(inplace=True)
            
#             total_multi_scale_channels = dim // 4 + dim // 4 + dim // 8 + dim // 8
            
#             self.qkv_fusion = nn.Sequential(
#                 nn.Conv2d(total_multi_scale_channels, dim // 2, kernel_size=1, bias=False),
#                 nn.GELU(),
#                 nn.Conv2d(dim // 2, dim // 2, kernel_size=1, bias=False),
#                 nn.GELU(),
#                 nn.Conv2d(dim // 2, 3 * self.qkv_dim, kernel_size=1, bias=qkv_bias)
#             )
            
#             self.qkv_residual = nn.Conv2d(dim, 3 * self.qkv_dim, kernel_size=1, bias=False)
#             # self.residual_weight = nn.Parameter(torch.tensor(0.1))
#             self.residual_weight = nn.Parameter(torch.tensor(0.5))
#         else:
#             self.qkv = Conv(dim, 3 * self.qkv_dim, kernel_size=1, bias=qkv_bias)
        
#         self.local_path = SeparableConvBN(dim, dim, kernel_size=3)
#         self.proj = SeparableConvBN(dim, dim, kernel_size=3)

#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        
#         coords_h = torch.arange(window_size)
#         coords_w = torch.arange(window_size)
#         coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
#         coords_flatten = torch.flatten(coords, 1)
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()
#         relative_coords[:, :, 0] += window_size - 1
#         relative_coords[:, :, 1] += window_size - 1
#         relative_coords[:, :, 0] *= 2 * window_size - 1
#         relative_position_index = relative_coords.sum(-1)
#         self.register_buffer("relative_position_index", relative_position_index)
        
#         nn.init.normal_(self.relative_position_bias_table, mean=0.0, std=.02)

#     def pad(self, x, ps):
#         _, _, H, W = x.size()
#         if W % ps != 0:
#             x = F.pad(x, (0, ps - W % ps), mode='reflect')
#         if H % ps != 0:
#             x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
#         return x

#     def forward(self, x, guidance_mask=None):
#         B, C, H, W = x.shape
        
#         local = self.local_path(x)
#         x_padded = self.pad(x, self.ws)
#         B, C, Hp, Wp = x_padded.shape
        
#         if self.use_enhanced_qkv:
#             feat_1x1 = self.qkv_conv1x1(x_padded)
#             feat_3x3 = self.qkv_conv3x3(x_padded)
#             feat_5x5 = self.qkv_conv5x5(x_padded)
            
#             feat_depthwise = self.qkv_depthwise(x_padded)
#             feat_depthwise = self.dw_norm(feat_depthwise)
#             feat_depthwise = self.dw_act(feat_depthwise)
#             feat_depthwise = self.qkv_pointwise(feat_depthwise)
            
#             multi_scale_features = torch.cat([feat_1x1, feat_3x3, feat_5x5, feat_depthwise], dim=1)
#             qkv_mlp = self.qkv_fusion(multi_scale_features)
#             qkv_res = self.qkv_residual(x_padded)
#             residual_weight = torch.sigmoid(self.residual_weight)
#             # residual_weight = torch.sigmoid(self.residual_weight) * 2  # Scale to [0, 2]
#             qkv = qkv_mlp + residual_weight * qkv_res
            
            
            
#         else:
#             qkv = self.qkv(x_padded)
        
#         q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', 
#                             h=self.num_heads, d=self.qkv_dim//self.num_heads, 
#                             hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)
        
#         # if guidance_mask is not None:
#         #     mask_padded = self.pad(guidance_mask, self.ws)
#         #     mask_windowed = rearrange(mask_padded, 'b c (hh ws1) (ww ws2) -> (b hh ww) (ws1 ws2) c',
#         #                               hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)
#         #     mask_windowed = mask_windowed.unsqueeze(1)
#         #     mask_spatial = mask_windowed.squeeze(-1)
#         #     mask_spatial = mask_spatial.unsqueeze(-1)
            
#         #     q = q * mask_spatial.unsqueeze(-1)
#         #     k = k * mask_spatial.unsqueeze(-1)
        
#         dots = (q @ k.transpose(-2, -1)) * self.scale
        
#         relative_position_bias = self.relative_position_bias_table[
#             self.relative_position_index.view(-1)].view(
#                 self.ws * self.ws, self.ws * self.ws, -1)
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
#         dots += relative_position_bias.unsqueeze(0)

#         attn = dots.softmax(dim=-1)
#         attn_out = attn @ v

#         attn_out = rearrange(attn_out, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', 
#                             h=self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, 
#                             ws1=self.ws, ws2=self.ws)
        
#         if attn_out.shape[1] != C:
#             attn_out = F.conv2d(attn_out, 
#                                 torch.eye(C, attn_out.shape[1]).view(C, attn_out.shape[1], 1, 1).to(attn_out.device))

#         attn_out = attn_out[:, :, :H, :W]
#         out = attn_out + local
#         out = self.proj(out)
#         # out = local
        
#         return out



# # ============================================================================
# # ENHANCED LIGHTBLOCK
# # ============================================================================

# class EnhancedLightBlockWithTransformer(nn.Module):
#     def __init__(self, dim=64, num_heads=8, mlp_ratio=2., qkv_bias=False, 
#                   drop=0., norm_layer=nn.BatchNorm2d, window_size=8,
#                   use_enhanced_qkv=True):
#         super().__init__()
        
#         self.norm1 = norm_layer(dim)
#         self.attn = EnhancedUltraEfficientGuidedAttention(
#             dim=dim, 
#             num_heads=num_heads, 
#             qkv_bias=qkv_bias, 
#             window_size=window_size, 
#             reduction_ratio=2,
#             use_enhanced_qkv=use_enhanced_qkv
#         )
#         self.drop_path = nn.Identity()
        
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = LightMlp(in_features=dim, hidden_features=mlp_hidden_dim, 
#                             out_features=dim, drop=drop)
#         self.norm2 = norm_layer(dim)
        
#     def forward(self, x, guidance_mask=None):
#         attn_out = self.attn(self.norm1(x), guidance_mask)
#         x = x + self.drop_path(attn_out)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x

# # ============================================================================
# # YOUR ORIGINAL ENCODER - NO CHANGES
# # ============================================================================

# class BaselineEncoder(nn.Module):
#     def __init__(self, in_channels=1):
#         super(BaselineEncoder, self).__init__()
        
#         def encoder_block(in_c, out_c):
#             return nn.Sequential(
#                 nn.Conv2d(in_c, 32, 3, padding=1),
#                 nn.BatchNorm2d(32),
#                 nn.ReLU(),
#                 nn.Conv2d(32, out_c, 3, padding=1),
#                 nn.BatchNorm2d(out_c),
#                 nn.ReLU(),
#                 nn.Dropout(0.1)
#             )
        
#         self.enc1 = encoder_block(in_channels, 48)
#         self.enc2 = encoder_block(48, 48)
#         self.enc3 = encoder_block(48, 48)
#         self.enc4 = encoder_block(48, 48)
        
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(48, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Dropout(0.1)
#         )
        
#         self._channels = [48, 48, 48, 128]
        
#     @property
#     def channels(self):
#         return self._channels
        
#     def forward(self, x):
#         res1 = self.enc1(x)
#         p1 = F.max_pool2d(res1, 2)
#         p1 = F.dropout(p1, 0.1, training=self.training)
        
#         res2 = self.enc2(p1)
#         p2 = F.max_pool2d(res2, 2)
#         p2 = F.dropout(p2, 0.1, training=self.training)
        
#         res3 = self.enc3(p2)
#         p3 = F.max_pool2d(res3, 2)
#         p3 = F.dropout(p3, 0.1, training=self.training)
        
#         res4 = self.enc4(p3)
#         p4 = F.max_pool2d(res4, 2)
#         p4 = F.dropout(p4, 0.1, training=self.training)
        
#         bottleneck = self.bottleneck(p4)
        
#         return res1, res2, res3, bottleneck

# # ============================================================================
# # EFFICIENT FEATURE FUSION
# # ============================================================================

# class ConvBNReLU(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
#         super(ConvBNReLU, self).__init__(
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
#                       dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
#             norm_layer(out_channels),
#             nn.ReLU6()
#         )

# class EfficientWF(nn.Module):
#     def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
#         super(EfficientWF, self).__init__()
#         self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
#         self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
#         self.eps = eps
#         self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)
        
#     def forward(self, x, res):
#         # x = F.interpolate(x, size=res.shape[-2:], mode='bilinear', align_corners=False)
#         x = F.interpolate(x, size=res.shape[-2:], mode='nearest')
#         weights = nn.ReLU()(self.weights)
#         fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
#         x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
#         x = self.post_conv(x)
#         return x

# # ============================================================================
# # ENHANCED DECODER WITH FIXED MGFA
# # ============================================================================

# class EnhancedDecoderWithMGFA(nn.Module):
#     def __init__(self, encoder_channels=(48, 48, 48, 128), decode_channels=64, 
#                   dropout=0.1, window_size=8, num_classes=1, use_enhanced_qkv=True):
#         super().__init__()
        
#         # Pre-conv to match dimensions (128 -> 64)
#         self.pre_conv = nn.Sequential(
#             nn.Conv2d(128, decode_channels, 1),
#             nn.BatchNorm2d(decode_channels),
#             nn.ReLU(inplace=True)
#         )
        
#         # Transformer blocks
#         self.b4 = EnhancedLightBlockWithTransformer(
#             dim=decode_channels, 
#             num_heads=8, 
#             window_size=window_size,
#             use_enhanced_qkv=use_enhanced_qkv,
#             drop=dropout
#         )
        
#         self.b3 = EnhancedLightBlockWithTransformer(
#             dim=decode_channels, 
#             num_heads=8, 
#             window_size=window_size,
#             use_enhanced_qkv=use_enhanced_qkv,
#             drop=dropout
#         )
        
#         self.b2 = EnhancedLightBlockWithTransformer(
#             dim=decode_channels, 
#             num_heads=8, 
#             window_size=window_size,
#             use_enhanced_qkv=use_enhanced_qkv,
#             drop=dropout
#         )
        
#         # Feature fusion modules (48 -> 64 channels)
#         self.p3 = EfficientWF(encoder_channels[-2], decode_channels)  # 48 -> 64
#         self.p2 = EfficientWF(encoder_channels[-3], decode_channels)  # 48 -> 64
#         self.p1 = EfficientWF(encoder_channels[-4], decode_channels)  # 48 -> 64
        
#         # Stage 1 conv
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(decode_channels, decode_channels, 3, padding=1),
#             nn.BatchNorm2d(decode_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(decode_channels, decode_channels, 3, padding=1),
#             nn.BatchNorm2d(decode_channels),
#             nn.ReLU(inplace=True)
#         )
#         # self.conv1 = nn.Conv2d(decode_channels, num_classes, 1)
        
#         # FIXED Enhanced MGFA module
#         self.mgfa_module = EnhancedMGFA_Module(
#             feature_dim=decode_channels,  # 64
#             output_dim=decode_channels    # 64
#         )
        
#         # # Auxiliary head for better training
#         # self.aux_head = nn.Conv2d(decode_channels, num_classes, 1)
        
#         # Final segmentation head
#         self.seg_head = nn.Sequential(
#             nn.Conv2d(decode_channels, decode_channels, 3, padding=1),
#             nn.BatchNorm2d(decode_channels),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=0.1),
#             nn.Conv2d(decode_channels, decode_channels, 3, padding=1),
#             nn.BatchNorm2d(decode_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(decode_channels, num_classes, 1)
#         )
#         # self.seg_head = nn.Conv2d(decode_channels, num_classes, 1)
        
#     def forward(self, res1, res2, res3, res4, h, w, guidance_mask=None):
#         # ========= STAGE 4 =========
#         x = self.pre_conv(res4)  # 128 -> 64
#         x = self.b4(x, guidance_mask)
        
#         # ========= STAGE 3 =========
#         x = self.p3(x, res3)  # 48 -> 64
#         x = self.b3(x, guidance_mask)
        
#         # ========= STAGE 2 =========
#         x = self.p2(x, res2)  # 48 -> 64
#         x = self.b2(x, guidance_mask)
        
#         # ========= STAGE 1 =========
#         x = self.p1(x, res1)  # 48 -> 64
#         x = self.conv1(x)
        
#         # Final upsampling
#         # x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
#         x = F.interpolate(x, size=(h, w), mode='nearest')
        
        
#         # # Generate auxiliary prediction
#         # aux_pred = self.aux_head(x)
        
#         # Apply FIXED enhanced MGFA
#         x_refined, prelim_mask = self.mgfa_module(x)
        
#         # Final segmentation
#         final_pred = self.seg_head(x_refined)
        
#         if self.training:
#             return final_pred, prelim_mask
#             # return x
#         else:
#             return final_pred
#             # return x

# # ============================================================================
# # MAIN MODEL
# # ============================================================================

# class BaselineWithEnhancedTransformerMGFA(nn.Module):
#     def __init__(self, in_channels=1, decode_channels=64, dropout=0.1, 
#                   window_size=8, num_classes=1, use_enhanced_qkv=True):
#         super().__init__()
        
#         # Your ORIGINAL encoder
#         self.backbone = BaselineEncoder(in_channels=in_channels)
#         encoder_channels = self.backbone.channels  # [48, 48, 48, 128]
        
#         # Enhanced decoder with fixed MGFA
#         self.decoder = EnhancedDecoderWithMGFA(
#             encoder_channels=encoder_channels,
#             decode_channels=decode_channels,
#             dropout=dropout,
#             window_size=window_size,
#             num_classes=num_classes,
#             use_enhanced_qkv=use_enhanced_qkv
#         )
        
#     def forward(self, x):
#         h, w = x.size()[-2:]
#         res1, res2, res3, res4 = self.backbone(x)
        
#         embedding = F.interpolate(res4, size=(128, 128), mode='bilinear', align_corners=False)
        
#         if self.training:
#             final_pred, prelim_mask = self.decoder(res1, res2, res3, res4, h, w)
#             # final_pred= self.decoder(res1, res2, res3, res4, h, w)
#             # return final_pred, embedding, prelim_mask, aux_pred
#             return final_pred, embedding
        
#         else:
#             final_pred = self.decoder(res1, res2, res3, res4, h, w)
#             return final_pred, embedding





# if __name__ == '__main__':
#     model = BaselineWithEnhancedTransformerMGFA()
#     total_params = sum([param.nelement() for param in model.parameters()])
#     print("Number of parameter: %.2fM" % (total_params/1e6))
    
#     # Test with sample input
#     res, res1 = model(torch.randn(1, 1, 256, 256))
#     print("Output shapes:", res.shape, res1.shape)
    
#     try:
#         from thop import profile
#         flops, params = profile(model, inputs=(torch.randn(1, 1, 256, 256),))
#         # Convert FLOPs to Giga FLOPs
#         giga_flops = flops / 1e9
#         print(f'FLOPs (G): {giga_flops:.2f}, Parameters (M): {params / 1e6:.2f}')
#     except:
#         print("FLOPs calculation skipped - thop might not be available")





import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ============================================================================
# CONVOLUTIONAL COMPONENTS
# ============================================================================

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                  norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )

class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                  norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

class LightMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features // 2
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ============================================================================
# FIXED ENHANCED MGFA MODULE
# ============================================================================

class EnhancedMGFA_Module(nn.Module):
    """
    Fixed Enhanced Mask-guided Feature Attention for coronary vessels
    """
    def __init__(self, feature_dim=64, output_dim=64):
        super().__init__()
        
        # 1. Preliminary mask prediction
        self.preliminary_mask = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim//2, 3, padding=1),
            nn.BatchNorm2d(feature_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim//2, 1, 1)
        )
        
        # 2. Multi-scale context extraction - FIXED CHANNEL CALCULATION
        # Each branch produces feature_dim//2 channels
        self.context_conv1 = nn.Conv2d(feature_dim, feature_dim//2, 3, padding=1, dilation=1)
        self.context_conv2 = nn.Conv2d(feature_dim, feature_dim//2, 3, padding=2, dilation=2)
        self.context_conv3 = nn.Conv2d(feature_dim, feature_dim//2, 3, padding=4, dilation=4)
        
        # 3. Context fusion (combines all 3 context branches)
        self.context_fusion = nn.Sequential(
            nn.Conv2d(feature_dim//2 * 3, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # 4. Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(1, feature_dim//4, 3, padding=1),
            nn.BatchNorm2d(feature_dim//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim//4, feature_dim, 1),
            nn.Sigmoid()
        )
        
        # 5. Feature fusion - FIXED: Now expects feature_dim * 2 channels
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 2, output_dim, 3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
        # 6. Final refinement
        self.final_refine = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, 3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features):
        """
        Args:
            features: Input features [B, C, H, W]
        Returns:
            Refined features [B, C, H, W], preliminary mask [B, 1, H, W]
        """
        # Step 1: Generate preliminary mask
        prelim_mask = self.preliminary_mask(features)
        
        # Step 2: Extract multi-scale context
        context1 = self.context_conv1(features)
        context2 = self.context_conv2(features)
        context3 = self.context_conv3(features)
        
        # Combine contexts
        context_combined = torch.cat([context1, context2, context3], dim=1)
        context = self.context_fusion(context_combined)
        
        # Step 3: Create attention from mask
        attention = self.attention(prelim_mask)
        
        # Step 4: Apply attention with residual
        attended_features = features * (1 + attention)
        
        # Step 5: Combine attended features with context
        combined = torch.cat([attended_features, context], dim=1)  # Now [B, feature_dim*2, H, W]
        fused = self.fusion(combined)
        
        # Step 6: Final refinement
        output = self.final_refine(fused)
        
        return output, prelim_mask

# ============================================================================
# ENHANCED ULTRA EFFICIENT GUIDED ATTENTION
# ============================================================================

# class EnhancedUltraEfficientGuidedAttention(nn.Module):
#     def __init__(self, dim=256, num_heads=8, qkv_bias=False, 
#                   window_size=8, reduction_ratio=2, use_enhanced_qkv=True):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // self.num_heads
#         self.scale = self.head_dim ** -0.5
#         self.ws = window_size
#         self.reduction_ratio = reduction_ratio
#         self.use_enhanced_qkv = use_enhanced_qkv
        
#         self.qkv_dim = dim // reduction_ratio
        
#         if use_enhanced_qkv:
#             self.qkv_conv3x3 = nn.Conv2d(dim, dim // 4, kernel_size=3, padding=1, bias=False)
#             self.qkv_conv1x1 = nn.Conv2d(dim, dim // 4, kernel_size=1, bias=False)
#             # self.qkv_conv1x1 = nn.Conv2d(dim, dim // 4, kernel_size=3, padding=1, bias=False)
#             # self.qkv_conv5x5 = nn.Conv2d(dim, dim // 8, kernel_size=5, padding=2, bias=False)
#             self.qkv_conv5x5 = nn.Conv2d(dim, dim // 8, kernel_size=3, padding=1, bias=False)
            
#             self.qkv_depthwise = nn.Conv2d(
#                 dim, dim, kernel_size=3, padding=1, groups=dim, bias=False
#             )
#             self.qkv_pointwise = nn.Conv2d(dim, dim // 8, kernel_size=1, bias=False)
#             self.dw_norm = nn.BatchNorm2d(dim)
#             # self.dw_act = nn.SiLU(inplace=True)
#             self.dw_act = nn.GELU()
#             # self.dw_act = nn.ReLU()
            
#             total_multi_scale_channels = dim // 4 + dim // 4 + dim // 8 + dim // 8
            
#             self.qkv_fusion = nn.Sequential(
#                 nn.Conv2d(total_multi_scale_channels, dim // 2, kernel_size=1, bias=False),
#                 # nn.Conv2d(total_multi_scale_channels, 3 * self.qkv_dim, kernel_size=3, padding=1, bias=False),
#                 nn.GELU(),
#                 # nn.ReLU(),
#                 # nn.Conv2d(dim // 2, dim // 2, kernel_size=1, bias=False),
#                 # nn.GELU(),
#                 # nn.ReLU(),
#                 nn.Conv2d(dim // 2, 3 * self.qkv_dim, kernel_size=1, bias=qkv_bias),
#             )
            
            
#             self.qkv_residual = nn.Conv2d(dim, 3 * self.qkv_dim, kernel_size=1, bias=False)
#             # self.residual_weight = nn.Parameter(torch.tensor(0.1))
#             self.residual_weight = nn.Parameter(torch.tensor(0.3))
#         else:
#             self.qkv = Conv(dim, 3 * self.qkv_dim, kernel_size=1, bias=qkv_bias)
        
#         # self.local_path = SeparableConvBN(dim, dim, kernel_size=1)
#         # self.local_path = ConvBNReLU(dim, dim, kernel_size=7)
#         self.local_path = Conv(dim, dim, kernel_size=3)
        
#         # self.proj = SeparableConvBN(dim, dim, kernel_size=3)
#         self.proj = Conv(dim, dim, kernel_size=3)

#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        
#         coords_h = torch.arange(window_size)
#         coords_w = torch.arange(window_size)
#         coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
#         coords_flatten = torch.flatten(coords, 1)
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()
#         relative_coords[:, :, 0] += window_size - 1
#         relative_coords[:, :, 1] += window_size - 1
#         relative_coords[:, :, 0] *= 2 * window_size - 1
#         relative_position_index = relative_coords.sum(-1)
#         self.register_buffer("relative_position_index", relative_position_index)
        
#         nn.init.normal_(self.relative_position_bias_table, mean=0.0, std=.07)

#     def pad(self, x, ps):
#         _, _, H, W = x.size()
#         if W % ps != 0:
#             x = F.pad(x, (0, ps - W % ps), mode='reflect')
#         if H % ps != 0:
#             x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
#         return x

#     def forward(self, x, guidance_mask=None):
#         B, C, H, W = x.shape
        
#         local = self.local_path(x)
#         x_padded = self.pad(x, self.ws)
#         B, C, Hp, Wp = x_padded.shape
        
#         if self.use_enhanced_qkv:
#             feat_1x1 = self.qkv_conv1x1(x_padded)
#             feat_3x3 = self.qkv_conv3x3(x_padded)
#             feat_5x5 = self.qkv_conv5x5(x_padded)
            
#             feat_depthwise = self.qkv_depthwise(x_padded)
#             feat_depthwise = self.dw_norm(feat_depthwise)
#             feat_depthwise = self.dw_act(feat_depthwise)
#             feat_depthwise = self.qkv_pointwise(feat_depthwise)
            
#             multi_scale_features = torch.cat([feat_1x1, feat_3x3, feat_5x5, feat_depthwise], dim=1)
#             qkv_mlp = self.qkv_fusion(multi_scale_features)
#             qkv_res = self.qkv_residual(x_padded)
#             residual_weight = torch.sigmoid(self.residual_weight)
#             # residual_weight = torch.sigmoid(self.residual_weight) * 2  # Scale to [0, 2]
#             qkv = qkv_mlp + residual_weight * qkv_res
            
            
            
#         else:
#             qkv = self.qkv(x_padded)
        
#         q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', 
#                             h=self.num_heads, d=self.qkv_dim//self.num_heads, 
#                             hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)
        
#         if guidance_mask is not None:
#             mask_padded = self.pad(guidance_mask, self.ws)
#             mask_windowed = rearrange(mask_padded, 'b c (hh ws1) (ww ws2) -> (b hh ww) (ws1 ws2) c',
#                                       hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)
#             mask_windowed = mask_windowed.unsqueeze(1)
#             mask_spatial = mask_windowed.squeeze(-1)
#             mask_spatial = mask_spatial.unsqueeze(-1)
            
#             q = q * mask_spatial.unsqueeze(-1)
#             k = k * mask_spatial.unsqueeze(-1)
        
#         dots = (q @ k.transpose(-2, -1)) * self.scale
        
#         relative_position_bias = self.relative_position_bias_table[
#             self.relative_position_index.view(-1)].view(
#                 self.ws * self.ws, self.ws * self.ws, -1)
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
#         dots += relative_position_bias.unsqueeze(0)

#         attn = dots.softmax(dim=-1)
#         attn_out = attn @ v

#         attn_out = rearrange(attn_out, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', 
#                             h=self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, 
#                             ws1=self.ws, ws2=self.ws)
        
#         if attn_out.shape[1] != C:
#             attn_out = F.conv2d(attn_out, 
#                                 torch.eye(C, attn_out.shape[1]).view(C, attn_out.shape[1], 1, 1).to(attn_out.device))

#         attn_out = attn_out[:, :, :H, :W]
#         out = attn_out + local
#         out = self.proj(out)
#         # out = local
        
#         return out

class EnhancedUltraEfficientGuidedAttention(nn.Module):
    def __init__(self, dim=256, num_heads=8, qkv_bias=False, 
                  window_size=8, reduction_ratio=2, use_enhanced_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.ws = window_size
        self.reduction_ratio = reduction_ratio
        self.use_enhanced_qkv = use_enhanced_qkv
        
        self.qkv_dim = dim // reduction_ratio
        
        if use_enhanced_qkv:
            self.qkv_conv3x3 = nn.Conv2d(dim, dim // 4, kernel_size=3, padding=1, bias=False)
            self.qkv_conv1x1 = nn.Conv2d(dim, dim // 4, kernel_size=1, bias=False)
            self.qkv_conv5x5 = nn.Conv2d(dim, dim // 8, kernel_size=3, padding=1, bias=False)
            
            self.qkv_depthwise = nn.Conv2d(
                dim, dim, kernel_size=3, padding=1, groups=dim, bias=False
            )
            self.qkv_pointwise = nn.Conv2d(dim, dim // 8, kernel_size=1, bias=False)
            self.dw_norm = nn.BatchNorm2d(dim)
            self.dw_act = nn.GELU()
            
            total_multi_scale_channels = dim // 4 + dim // 4 + dim // 8 + dim // 8
            
            # NEW: Add instance normalization for better cross-dataset generalization
            self.instance_norm = nn.InstanceNorm2d(total_multi_scale_channels, affine=True)
            
            self.qkv_fusion = nn.Sequential(
                nn.Conv2d(total_multi_scale_channels, dim // 2, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv2d(dim // 2, 3 * self.qkv_dim, kernel_size=1, bias=qkv_bias),
            )
            
            self.qkv_residual = nn.Conv2d(dim, 3 * self.qkv_dim, kernel_size=1, bias=False)
            self.residual_weight = nn.Parameter(torch.tensor(0.3))
            # self.residual_weight = nn.Parameter(torch.tensor(0.1))
        else:
            self.qkv = Conv(dim, 3 * self.qkv_dim, kernel_size=1, bias=qkv_bias)
   
        self.local_path = Conv(dim, dim, kernel_size=3)
        
        # NEW: Add dropout to projection for regularization
        self.proj_drop = nn.Dropout2d(0.3)
        self.proj = Conv(dim, dim, kernel_size=3)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        # NEW: Add learnable temperature scaling for attention
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)
        # self.temperature = nn.Parameter(torch.ones(1) * 0.6)
        
        nn.init.normal_(self.relative_position_bias_table, mean=0.0, std=.02)  # Reduced std from .07 to .02

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def forward(self, x, guidance_mask=None):
        B, C, H, W = x.shape
        
        local = self.local_path(x)
        x_padded = self.pad(x, self.ws)
        B, C, Hp, Wp = x_padded.shape
        
        if self.use_enhanced_qkv:
            feat_1x1 = self.qkv_conv1x1(x_padded)
            feat_3x3 = self.qkv_conv3x3(x_padded)
            feat_5x5 = self.qkv_conv5x5(x_padded)
            
            feat_depthwise = self.qkv_depthwise(x_padded)
            feat_depthwise = self.dw_norm(feat_depthwise)
            feat_depthwise = self.dw_act(feat_depthwise)
            feat_depthwise = self.qkv_pointwise(feat_depthwise)
            
            multi_scale_features = torch.cat([feat_1x1, feat_3x3, feat_5x5, feat_depthwise], dim=1)
            
            # NEW: Apply instance normalization for domain adaptation
            multi_scale_features = self.instance_norm(multi_scale_features)
            
            qkv_mlp = self.qkv_fusion(multi_scale_features)
            qkv_res = self.qkv_residual(x_padded)
            residual_weight = torch.sigmoid(self.residual_weight)
            qkv = qkv_mlp + residual_weight * qkv_res
            
        else:
            qkv = self.qkv(x_padded)
        
        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', 
                            h=self.num_heads, d=self.qkv_dim//self.num_heads, 
                            hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)
        
        # if guidance_mask is not None:
        #     mask_padded = self.pad(guidance_mask, self.ws)
        #     mask_windowed = rearrange(mask_padded, 'b c (hh ws1) (ww ws2) -> (b hh ww) (ws1 ws2) c',
        #                               hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)
        #     mask_windowed = mask_windowed.unsqueeze(1)
        #     mask_spatial = mask_windowed.squeeze(-1)
        #     mask_spatial = mask_spatial.unsqueeze(-1)
            
        #     q = q * mask_spatial.unsqueeze(-1)
        #     k = k * mask_spatial.unsqueeze(-1)
        
        # NEW: Apply temperature scaling to attention for smoother distributions
        dots = (q @ k.transpose(-2, -1)) * self.scale * self.temperature
        # dots = (q @ k.transpose(-2, -1)) * self.scale 
        
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn_out = attn @ v

        attn_out = rearrange(attn_out, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', 
                            h=self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, 
                            ws1=self.ws, ws2=self.ws)
        
        if attn_out.shape[1] != C:
            attn_out = F.conv2d(attn_out, 
                                torch.eye(C, attn_out.shape[1]).view(C, attn_out.shape[1], 1, 1).to(attn_out.device))

        attn_out = attn_out[:, :, :H, :W]
        out = attn_out + local
        
        # NEW: Apply dropout to projection
        out = self.proj_drop(out)
        out = self.proj(out)
        
        return out



# ============================================================================
# ENHANCED LIGHTBLOCK
# ============================================================================

class EnhancedLightBlockWithTransformer(nn.Module):
    def __init__(self, dim=64, num_heads=8, mlp_ratio=2., qkv_bias=False, 
                  drop=0., norm_layer=nn.BatchNorm2d, window_size=8,
                  use_enhanced_qkv=True):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = EnhancedUltraEfficientGuidedAttention(
            dim=dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            window_size=window_size, 
            reduction_ratio=2,
            use_enhanced_qkv=use_enhanced_qkv
        )
        self.drop_path = nn.Identity()
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LightMlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                            out_features=dim, drop=drop)
        self.norm2 = norm_layer(dim)
        
    def forward(self, x, guidance_mask=None):
        attn_out = self.attn(self.norm1(x), guidance_mask)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# ============================================================================
# YOUR ORIGINAL ENCODER - NO CHANGES
# ============================================================================

class BaselineEncoder(nn.Module):
    def __init__(self, in_channels=1):
        super(BaselineEncoder, self).__init__()
        
        def encoder_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        
        self.enc1 = encoder_block(in_channels, 48)
        self.enc2 = encoder_block(48, 48)
        self.enc3 = encoder_block(48, 48)
        self.enc4 = encoder_block(48, 48)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(48, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self._channels = [48, 48, 48, 128]
        
    @property
    def channels(self):
        return self._channels
        
    def forward(self, x):
        res1 = self.enc1(x)
        p1 = F.max_pool2d(res1, 2)
        p1 = F.dropout(p1, 0.1, training=self.training)
        
        res2 = self.enc2(p1)
        p2 = F.max_pool2d(res2, 2)
        p2 = F.dropout(p2, 0.1, training=self.training)
        
        res3 = self.enc3(p2)
        p3 = F.max_pool2d(res3, 2)
        p3 = F.dropout(p3, 0.1, training=self.training)
        
        res4 = self.enc4(p3)
        p4 = F.max_pool2d(res4, 2)
        p4 = F.dropout(p4, 0.1, training=self.training)
        
        bottleneck = self.bottleneck(p4)
        
        return res1, res2, res3, bottleneck

# ============================================================================
# EFFICIENT FEATURE FUSION
# ============================================================================

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class EfficientWF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(EfficientWF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)
        
    def forward(self, x, res):
        # x = F.interpolate(x, size=res.shape[-2:], mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=res.shape[-2:], mode='nearest')
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x

# ============================================================================
# ENHANCED DECODER WITH FIXED MGFA
# ============================================================================

class EnhancedDecoderWithMGFA(nn.Module):
    def __init__(self, encoder_channels=(48, 48, 48, 128), decode_channels=64, 
                  dropout=0.1, window_size=8, num_classes=1, use_enhanced_qkv=True):
        super().__init__()
        
        # Pre-conv to match dimensions (128 -> 64)
        self.pre_conv = nn.Sequential(
            nn.Conv2d(128, decode_channels, 1),
            nn.BatchNorm2d(decode_channels),
            nn.ReLU(inplace=True)
        )
        
        # Transformer blocks
        self.b4 = EnhancedLightBlockWithTransformer(
            dim=decode_channels, 
            num_heads=8, 
            window_size=window_size,
            use_enhanced_qkv=use_enhanced_qkv,
            drop=dropout
        )
        
        self.b3 = EnhancedLightBlockWithTransformer(
            dim=decode_channels, 
            num_heads=8, 
            window_size=window_size,
            use_enhanced_qkv=use_enhanced_qkv,
            drop=dropout
        )
        
        self.b2 = EnhancedLightBlockWithTransformer(
            dim=decode_channels, 
            num_heads=8, 
            window_size=window_size,
            use_enhanced_qkv=use_enhanced_qkv,
            drop=dropout
        )
        
        # Feature fusion modules (48 -> 64 channels)
        self.p3 = EfficientWF(encoder_channels[-2], decode_channels)  # 48 -> 64
        self.p2 = EfficientWF(encoder_channels[-3], decode_channels)  # 48 -> 64
        self.p1 = EfficientWF(encoder_channels[-4], decode_channels)  # 48 -> 64
        
        # Stage 1 conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(decode_channels, decode_channels, 3, padding=1),
            nn.BatchNorm2d(decode_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decode_channels, decode_channels, 3, padding=1),
            nn.BatchNorm2d(decode_channels),
            nn.ReLU(inplace=True)
        )
        # self.conv1 = nn.Conv2d(decode_channels, num_classes, 1)
        
        # FIXED Enhanced MGFA module
        self.mgfa_module = EnhancedMGFA_Module(
            feature_dim=decode_channels,  # 64
            output_dim=decode_channels    # 64
        )
        
        # # Auxiliary head for better training
        # self.aux_head = nn.Conv2d(decode_channels, num_classes, 1)
        
        # Final segmentation head
        # self.seg_head = nn.Sequential(
        #     nn.Conv2d(decode_channels, decode_channels, 3, padding=1),
        #     nn.BatchNorm2d(decode_channels),
        #     nn.ReLU(inplace=True),
        #     # nn.Dropout2d(p=0.1),
        #     nn.Dropout2d(p=0.5),
        #     nn.Conv2d(decode_channels, decode_channels, 3, padding=1),
        #     nn.BatchNorm2d(decode_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(decode_channels, num_classes, 1)
        # )
        
        # self.seg_head = nn.Sequential(
        #     # nn.Conv2d(decode_channels, decode_channels, 3, padding=1),
        #     nn.Conv2d(decode_channels, decode_channels, 1),
        #     nn.BatchNorm2d(decode_channels),
        #     nn.ReLU(inplace=True),
        #     # nn.GELU(),
        #     # nn.SiLU(),
        #     # nn.Dropout2d(p=0.4),
        #     nn.Dropout2d(p=0.2),
        #     # nn.Conv2d(decode_channels, decode_channels, 3, padding=1),
        #     # nn.Conv2d(decode_channels, decode_channels, 1),
        #     # nn.BatchNorm2d(decode_channels),
        #     # nn.ReLU(inplace=True),
        #     # nn.GELU(),
        #     # nn.SiLU(),
        #     # nn.Dropout2d(p=0.1),
        #     # nn.Dropout2d(p=0.4),
        #     nn.Conv2d(decode_channels, num_classes, 1),
        #     # nn.BatchNorm2d(num_classes),
        #     # nn.ReLU(inplace=True),
        #     nn.Dropout2d(p=0.2),
        # )
        
        self.seg_head = nn.Conv2d(decode_channels, num_classes, 1)
        
        # self.seg_head = nn.Sequential(
        #     nn.Conv2d(decode_channels, decode_channels, 3, padding=1),
        #     nn.BatchNorm2d(decode_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(p=0.1),
        #     nn.Conv2d(decode_channels, decode_channels, 3, padding=1),
        #     nn.BatchNorm2d(decode_channels),
        #     nn.ReLU(inplace=True),
        #     # Changed: Added dropout before final layer and adjusted final conv
        #     nn.Dropout2d(p=0.05),  # Light dropout before final classification
        #     nn.Conv2d(decode_channels, num_classes, kernel_size=1)  # Added kernel_size explicitly
        # )
        
    def forward(self, res1, res2, res3, res4, h, w, guidance_mask=None):
        # ========= STAGE 4 =========
        x = self.pre_conv(res4)  # 128 -> 64
        x = self.b4(x, guidance_mask)
        
        # ========= STAGE 3 =========
        x = self.p3(x, res3)  # 48 -> 64
        x = self.b3(x, guidance_mask)
        
        # ========= STAGE 2 =========
        x = self.p2(x, res2)  # 48 -> 64
        x = self.b2(x, guidance_mask)
        
        # ========= STAGE 1 =========
        x = self.p1(x, res1)  # 48 -> 64
        x = self.conv1(x)
        
        # Final upsampling
        # x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=(h, w), mode='nearest')
        
        
        # # Generate auxiliary prediction
        # aux_pred = self.aux_head(x)
        
        # Apply FIXED enhanced MGFA
        x_refined, prelim_mask = self.mgfa_module(x)
        
        # Final segmentation
        final_pred = self.seg_head(x_refined)
        
        if self.training:
            return final_pred, prelim_mask
            # return x
        else:
            return final_pred
            # return x

# ============================================================================
# MAIN MODEL
# ============================================================================

class BaselineWithEnhancedTransformerMGFA(nn.Module):
    def __init__(self, in_channels=1, decode_channels=64, dropout=0.1, 
                  window_size=8, num_classes=1, use_enhanced_qkv=True):
        super().__init__()
        
        # Your ORIGINAL encoder
        self.backbone = BaselineEncoder(in_channels=in_channels)
        encoder_channels = self.backbone.channels  # [48, 48, 48, 128]
        
        # Enhanced decoder with fixed MGFA
        self.decoder = EnhancedDecoderWithMGFA(
            encoder_channels=encoder_channels,
            decode_channels=decode_channels,
            dropout=dropout,
            window_size=window_size,
            num_classes=num_classes,
            use_enhanced_qkv=use_enhanced_qkv
        )
        
    def forward(self, x):
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        
        embedding = F.interpolate(res4, size=(128, 128), mode='bilinear', align_corners=False)
        
        if self.training:
            final_pred, prelim_mask = self.decoder(res1, res2, res3, res4, h, w)
            # final_pred= self.decoder(res1, res2, res3, res4, h, w)
            # return final_pred, embedding, prelim_mask, aux_pred
            return final_pred, embedding
        
        else:
            final_pred = self.decoder(res1, res2, res3, res4, h, w)
            return final_pred, embedding





if __name__ == '__main__':
    model = BaselineWithEnhancedTransformerMGFA()
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total_params/1e6))
    
    # Test with sample input
    res, res1 = model(torch.randn(1, 1, 256, 256))
    print("Output shapes:", res.shape, res1.shape)
    
    try:
        from thop import profile
        flops, params = profile(model, inputs=(torch.randn(1, 1, 256, 256),))
        # Convert FLOPs to Giga FLOPs
        giga_flops = flops / 1e9
        print(f'FLOPs (G): {giga_flops:.2f}, Parameters (M): {params / 1e6:.2f}')
    except:
        print("FLOPs calculation skipped - thop might not be available")