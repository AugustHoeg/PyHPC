import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Helper Functions for Patching ---
def tensor_to_patches(x, patch_d, patch_h, patch_w, verbose=False):
    """
    Converts a 5D tensor into a batch of 3D patches.
    x: (B, C, D, H, W)
    Returns: 
        patches: (B * n_patches_total, C, patch_d, patch_h, patch_w)
        n_patches_dims: (n_patches_d, n_patches_h, n_patches_w)
    """
    B, C, D, H, W = x.shape
    if verbose:
        print(f"[tensor_to_patches] Input tensor shape: {x.shape}, Target patch dims: D={patch_d}, H={patch_h}, W={patch_w}")
    if not (D % patch_d == 0 and H % patch_h == 0 and W % patch_w == 0):
        raise ValueError(f"Input dimensions ({D},{H},{W}) must be divisible by patch dimensions ({patch_d},{patch_h},{patch_w})")
    
    n_patches_d = D // patch_d
    n_patches_h = H // patch_h
    n_patches_w = W // patch_w
    if verbose:
        print(f"[tensor_to_patches] Num patches: D_n={n_patches_d}, H_n={n_patches_h}, W_n={n_patches_w}")

    # Reshape to (B, C, n_patches_d, patch_d, n_patches_h, patch_h, n_patches_w, patch_w)
    x_view = x.view(B, C, n_patches_d, patch_d, n_patches_h, patch_h, n_patches_w, patch_w)
    # Permute to (B, n_patches_d, n_patches_h, n_patches_w, C, patch_d, patch_h, patch_w)
    x_permuted = x_view.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    # Reshape to (B * n_total_patches, C, patch_d, patch_h, patch_w)
    n_total_patches = n_patches_d * n_patches_h * n_patches_w
    patches = x_permuted.view(B * n_total_patches, C, patch_d, patch_h, patch_w)
    if verbose:
        print(f"[tensor_to_patches] Output patches shape: {patches.shape}")
    return patches, (n_patches_d, n_patches_h, n_patches_w)

def patches_to_tensor(patches, batch_size, n_patches_dims, C_out, patch_d, patch_h, patch_w, verbose=False):
    """
    Reassembles a batch of 3D patches into a 5D tensor.
    patches: (B_total_patches, C_out, patch_d, patch_h, patch_w)
    n_patches_dims: (n_patches_d, n_patches_h, n_patches_w)
    """
    if verbose:
        print(f"[patches_to_tensor] Input patches shape: {patches.shape}, Target B={batch_size}, Num_patch_dims={n_patches_dims}")
    n_patches_d, n_patches_h, n_patches_w = n_patches_dims
    
    # Reshape to (B, num_patches_d, num_patches_h, num_patches_w, C_out, patch_d, patch_h, patch_w)
    x_view = patches.view(batch_size, n_patches_d, n_patches_h, n_patches_w, C_out, patch_d, patch_h, patch_w)
    # Permute to (B, C_out, num_patches_d, patch_d, num_patches_h, patch_h, num_patches_w, patch_w)
    x_permuted = x_view.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    # Reshape to (B, C_out, D, H, W)
    D_orig = n_patches_d * patch_d
    H_orig = n_patches_h * patch_h
    W_orig = n_patches_w * patch_w
    tensor = x_permuted.view(batch_size, C_out, D_orig, H_orig, W_orig)
    if verbose:
        print(f"[patches_to_tensor] Output tensor shape: {tensor.shape}")
    return tensor

# --- CNN Building Blocks ---
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, verbose=False):
        super().__init__()
        self.verbose = verbose
        if self.verbose:
            print(f"[CNNBlock init] In: {in_channels}, Out: {out_channels}")
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        if self.verbose:
            print(f"  [CNNBlock fwd] Input shape: {x.shape}")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        if self.verbose:
            print(f"  [CNNBlock fwd] Output shape: {x.shape}")
        return x

class BasePatchCNN(nn.Module):
    def __init__(self, in_channels, cnn_block_hidden_features, num_classes, verbose=False):
        super().__init__()
        self.verbose = verbose
        if self.verbose:
            print(f"[BasePatchCNN init] In channels: {in_channels}, Hidden features: {cnn_block_hidden_features}, Num classes: {num_classes}")
        # Using a single CNNBlock, can be made deeper if needed
        self.cnn_block = CNNBlock(in_channels, cnn_block_hidden_features, verbose=verbose)
        self.seg_head = nn.Conv3d(cnn_block_hidden_features, num_classes, kernel_size=1)

    def forward(self, x_patch):
        if self.verbose:
            print(f" [BasePatchCNN fwd] Input patch shape: {x_patch.shape}")
        features = self.cnn_block(x_patch)
        if self.verbose:
            print(f" [BasePatchCNN fwd] Features after CNNBlock shape: {features.shape}")
        logits = self.seg_head(features) # (B_patches, num_classes, patch_d, patch_h, patch_w)
        if self.verbose:
            print(f" [BasePatchCNN fwd] Output logits patch shape: {logits.shape}")
        return logits

class MultiStepSegmentationModel(nn.Module):
    def __init__(self, image_channels=1, initial_mask_channels=1, num_classes=1, 
                 base_cnn_hidden_features=16, 
                 patch_size_d=16, patch_size_h=32, patch_size_w=32, verbose=False): # Added verbose
        super().__init__()
        self.verbose = verbose
        if self.verbose: print(f"--- MultiStepSegmentationModel Init ---")
        self.image_channels = image_channels
        self.initial_mask_channels = initial_mask_channels
        self.num_classes = num_classes
        self.base_cnn_hidden_features = base_cnn_hidden_features
        
        self.patch_d = patch_size_d
        self.patch_h = patch_size_h
        self.patch_w = patch_size_w
        if self.verbose:
            print(f"  Config: ImgChannels={image_channels}, InitMaskChannels={initial_mask_channels}, NumClasses={num_classes}")
            print(f"  Config: BaseCNNFeatures={base_cnn_hidden_features}, PatchD={patch_size_d}, PatchH={patch_size_h}, PatchW={patch_size_w}")

        # Total channels for the (image + initial_mask) data stream
        self.model_feature_input_channels = self.image_channels + self.initial_mask_channels
        if self.verbose: print(f"  Derived: ModelFeatureInputChannels (img+init_mask): {self.model_feature_input_channels}")

        # Downsampling layers for the combined (image + initial_mask)
        self.downsample_4x = nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2), # To 2x downsampled
            nn.AvgPool3d(kernel_size=2, stride=2)  # To 4x downsampled
        )
        self.downsample_2x = nn.AvgPool3d(kernel_size=2, stride=2)

        # Upsampling layer for masks (logits)
        self.upsample_2x = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        # The single, shared BasePatchCNN
        # Input channels = channels from (image+initial_mask) + channels from previous stage's mask
        base_cnn_in_channels = self.model_feature_input_channels + self.num_classes
        if self.verbose: print(f"  Derived: BasePatchCNN input channels (features + prev_mask_classes): {base_cnn_in_channels}")
        self.base_patch_cnn = BasePatchCNN(
            in_channels=base_cnn_in_channels,
            cnn_block_hidden_features=self.base_cnn_hidden_features,
            num_classes=self.num_classes,
            verbose=verbose
        )
        if self.verbose: print(f"--- MultiStepSegmentationModel Init Complete ---")


    def _process_stage(self, stage_name, current_res_feature_data, prev_stage_upsampled_mask_logits, 
                       current_spatial_dims, B):
        """
        Helper function to process a refinement stage using patch-wise CNN.
        current_res_feature_data: (B, model_feature_input_channels, D, H, W) - e.g., x_down2 or x_orig
        prev_stage_upsampled_mask_logits: (B, num_classes, D, H, W)
        current_spatial_dims: (D, H, W) of the current_res_feature_data
        B: batch_size
        """
        if self.verbose:
            print(f"  --- Processing Stage: {stage_name} ---")
            print(f"    [{stage_name}] Input current_res_feature_data shape: {current_res_feature_data.shape}")
            print(f"    [{stage_name}] Input prev_stage_upsampled_mask_logits shape: {prev_stage_upsampled_mask_logits.shape}")
            print(f"    [{stage_name}] Target spatial_dims: {current_spatial_dims}, Patch Dims: ({self.patch_d},{self.patch_h},{self.patch_w})")

        # Patch the image/feature data
        feature_patches, n_patches_dims = tensor_to_patches(
            current_res_feature_data, self.patch_d, self.patch_h, self.patch_w, self.verbose
        )
        if self.verbose: print(f"    [{stage_name}] feature_patches shape: {feature_patches.shape}, n_patches_dims: {n_patches_dims}")
        
        # Patch the upsampled previous mask
        prev_mask_patches, _ = tensor_to_patches(
            prev_stage_upsampled_mask_logits, self.patch_d, self.patch_h, self.patch_w, self.verbose
        )
        if self.verbose: print(f"    [{stage_name}] prev_mask_patches shape: {prev_mask_patches.shape}")

        # Concatenate along channel dimension for CNN input
        # Each patch will have (model_feature_input_channels + num_classes) channels
        cnn_input_patches = torch.cat([feature_patches, prev_mask_patches], dim=1)
        if self.verbose: print(f"    [{stage_name}] cnn_input_patches (for BasePatchCNN) shape: {cnn_input_patches.shape}")

        # Process all patches through the BasePatchCNN
        output_mask_patches = self.base_patch_cnn(cnn_input_patches)
        if self.verbose: print(f"    [{stage_name}] output_mask_patches (from BasePatchCNN) shape: {output_mask_patches.shape}")

        # Reassemble patches into a full tensor
        # Output channels will be num_classes
        output_logits = patches_to_tensor(
            output_mask_patches, B, n_patches_dims, 
            self.num_classes, self.patch_d, self.patch_h, self.patch_w, self.verbose
        )
        if self.verbose: print(f"    [{stage_name}] Reassembled output_logits shape (before F.interpolate): {output_logits.shape}")
        
        # Ensure output spatial dimensions match input features just in case of rounding with patches
        # (shouldn't happen if inputs are divisible by patch sizes)
        if output_logits.shape[2:] != current_spatial_dims:
            if self.verbose: print(f"    [{stage_name}] Interpolating output_logits from {output_logits.shape[2:]} to {current_spatial_dims}")
            output_logits = F.interpolate(output_logits, size=current_spatial_dims, mode='trilinear', align_corners=False)
            if self.verbose: print(f"    [{stage_name}] Reassembled output_logits shape (after F.interpolate): {output_logits.shape}")
        
        if self.verbose: print(f"  --- Stage {stage_name} Complete ---")
        return output_logits

    def forward(self, image, initial_mask):
        if self.verbose: print(f"--- MultiStepSegmentationModel Forward Pass ---")
        # image: (B, image_channels, D_orig, H_orig, W_orig) e.g., (B, 1, 64, 128, 128)
        # initial_mask: (B, initial_mask_channels, D_orig, H_orig, W_orig) e.g., (B, 1, 64, 128, 128)
        if self.verbose:
            print(f"  Initial image shape: {image.shape}")
            print(f"  Initial_mask shape: {initial_mask.shape}")
        
        B = image.shape[0]
        orig_spatial_dims = image.shape[2:] # (D, H, W)
        if self.verbose: print(f"  Batch size B={B}, Original spatial_dims D,H,W={orig_spatial_dims}")

        # 0. Concatenate image and initial_mask
        # This x_input is the main feature stream derived from original inputs
        x_input = torch.cat([image, initial_mask], dim=1) # (B, model_feature_input_channels, D,H,W)
        if self.verbose: print(f"  Combined x_input (image+initial_mask) shape: {x_input.shape}")

        # 1. Prepare multi-resolution versions of the input features
        x_orig_res = x_input
        x_down2 = self.downsample_2x(x_input) # Spatial: D/2, H/2, W/2
        x_down4 = self.downsample_4x(x_input) # Spatial: D/4, H/4, W/4
        if self.verbose:
            print(f"  x_orig_res shape: {x_orig_res.shape}")
            print(f"  x_down2 (2x downsampled features) shape: {x_down2.shape}")
            print(f"  x_down4 (4x downsampled features) shape: {x_down4.shape}")
        
        spatial_dims_down2 = x_down2.shape[2:]
        spatial_dims_down4 = x_down4.shape[2:]

        # --- Stage 1: Coarse Segmentation (on 4x downsampled input) ---
        if self.verbose: print(f"--- Stage 1: Coarse Segmentation (D/4, H/4, W/4) ---")
        # x_down4 is already at the patch resolution (e.g., 16x32x32 if D_orig=64, H_orig=128, W_orig=128)
        # For the first stage, the "previous mask" is a zero tensor.
        # Its spatial dimensions must match x_down4.
        zero_prev_mask_coarse = torch.zeros(B, self.num_classes, *spatial_dims_down4, 
                                            device=x_input.device, dtype=x_input.dtype)
        if self.verbose: print(f"    [Stage 1] zero_prev_mask_coarse shape: {zero_prev_mask_coarse.shape}")
        
        # Concatenate features and the zero mask
        cnn_input_coarse = torch.cat([x_down4, zero_prev_mask_coarse], dim=1)
        if self.verbose: print(f"    [Stage 1] cnn_input_coarse (for BasePatchCNN) shape: {cnn_input_coarse.shape}")
        
        # The BasePatchCNN expects patch-sized inputs. x_down4 (and cnn_input_coarse) is one such "patch".
        logits_coarse = self.base_patch_cnn(cnn_input_coarse) # (B, num_classes, D/4, H/4, W/4)
        if self.verbose: print(f"    [Stage 1] logits_coarse shape: {logits_coarse.shape}")


        # --- Stage 2: First Refinement (on 2x downsampled input) ---
        if self.verbose: print(f"--- Stage 2: Refinement 1 (D/2, H/2, W/2) ---")
        # Upsample coarse mask logits to the 2x downsampled resolution
        upsampled_mask_from_coarse = self.upsample_2x(logits_coarse)
        if self.verbose: print(f"    [Stage 2] upsampled_mask_from_coarse (from Stage 1 logits, upsampled 2x) shape before F.interpolate: {upsampled_mask_from_coarse.shape}")
        
        # Ensure correct target size after upsampling
        if upsampled_mask_from_coarse.shape[2:] != spatial_dims_down2:
            if self.verbose: print(f"    [Stage 2] Interpolating upsampled_mask_from_coarse from {upsampled_mask_from_coarse.shape[2:]} to {spatial_dims_down2}")
            upsampled_mask_from_coarse = F.interpolate(upsampled_mask_from_coarse, size=spatial_dims_down2, mode='trilinear', align_corners=False)
            if self.verbose: print(f"    [Stage 2] upsampled_mask_from_coarse shape after F.interpolate: {upsampled_mask_from_coarse.shape}")
            
        logits_refine1 = self._process_stage(
            stage_name="Refine1 (D/2)",
            current_res_feature_data=x_down2, 
            prev_stage_upsampled_mask_logits=upsampled_mask_from_coarse,
            current_spatial_dims=spatial_dims_down2,
            B=B
        ) # (B, num_classes, D/2, H/2, W/2)
        if self.verbose: print(f"    [Stage 2] logits_refine1 shape: {logits_refine1.shape}")

        # --- Stage 3: Final Refinement (on original resolution input) ---
        if self.verbose: print(f" --- Stage 3: Final Refinement (D, H, W) ---")
        # Upsample refined mask1 logits to the original resolution
        upsampled_mask_from_refine1 = self.upsample_2x(logits_refine1)
        if self.verbose: print(f"    [Stage 3] upsampled_mask_from_refine1 (from Stage 2 logits, upsampled 2x) shape before F.interpolate: {upsampled_mask_from_refine1.shape}")
        
        # Ensure correct target size after upsampling
        if upsampled_mask_from_refine1.shape[2:] != orig_spatial_dims:
            if self.verbose: print(f"    [Stage 3] Interpolating upsampled_mask_from_refine1 from {upsampled_mask_from_refine1.shape[2:]} to {orig_spatial_dims}")
            upsampled_mask_from_refine1 = F.interpolate(upsampled_mask_from_refine1, size=orig_spatial_dims, mode='trilinear', align_corners=False)
            if self.verbose: print(f"    [Stage 3] upsampled_mask_from_refine1 shape after F.interpolate: {upsampled_mask_from_refine1.shape}")


        logits_final = self._process_stage(
            stage_name="Final (D)",
            current_res_feature_data=x_orig_res,
            prev_stage_upsampled_mask_logits=upsampled_mask_from_refine1,
            current_spatial_dims=orig_spatial_dims,
            B=B
        ) # (B, num_classes, D_orig, H_orig, W_orig)
        if self.verbose: print(f"    [Stage 3] logits_final shape: {logits_final.shape}")
        
        if self.verbose:
            print(f"--- MultiStepSegmentationModel Forward Pass Complete ---")
            print(f"  Returning logits_coarse: {logits_coarse.shape}")
            print(f"  Returning logits_refine1: {logits_refine1.shape}")
            print(f"  Returning logits_final: {logits_final.shape}")
        return logits_coarse, logits_refine1, logits_final


if __name__ == '__main__':
    # Example Usage:
    # Input dimensions as per user: image (128,128,64) which is (D=64, H=128, W=128) for PyTorch
    batch_size = 1
    D_orig, H_orig, W_orig = 64, 128, 128

    # Configurable parameters for the model
    img_channels = 1       # E.g., 1 for grayscale MRIs
    init_mask_channels = 1 # E.g., 1 for a binary initial estimate mask
    n_classes = 1          # E.g., 1 for binary segmentation (foreground vs background)
    cnn_features = 128      # Hidden features in the BasePatchCNN's CNNBlock
    
    # Patch dimensions (these must divide the D/4, H/4, W/4, D/2, H/2, W/2, and D,H,W dimensions appropriately)
    # The model is designed for patch_d, patch_h, patch_w to be D/4, H/4, W/4 of original.
    # So, if D_orig=64, H_orig=128, W_orig=128:
    # D_patch = 64/4 = 16
    # H_patch = 128/4 = 32
    # W_patch = 128/4 = 32
    p_d, p_h, p_w = D_orig//4, H_orig//4, W_orig//4
    
    print(f"--- Main Example ---")
    print(f"Using batch_size: {batch_size}")
    print(f"Using original dimensions: D={D_orig}, H={H_orig}, W={W_orig}")
    print(f"Using patch dimensions: D={p_d}, H={p_h}, W={p_w}")
    print(f"Using CNN features: {cnn_features}")

    # Create dummy input tensors
    dummy_image = torch.randn(batch_size, img_channels, D_orig, H_orig, W_orig)
    dummy_initial_mask = torch.randn(batch_size, init_mask_channels, D_orig, H_orig, W_orig) # Could be zeros if no prior

    # Initialize the model with verbose=True
    model = MultiStepSegmentationModel(
        image_channels=img_channels, 
        initial_mask_channels=init_mask_channels,
        num_classes=n_classes,
        base_cnn_hidden_features=cnn_features,
        patch_size_d=p_d, 
        patch_size_h=p_h, 
        patch_size_w=p_w,
        verbose=False  # Enable verbose output
    )

    print(f"Model initialized. Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Perform a forward pass
    print(f"--- Performing Forward Pass ---")
    print(f"  Input image shape: {dummy_image.shape}")
    print(f"  Input initial mask shape: {dummy_initial_mask.shape}")
    
    logits_c, logits_r1, logits_f = model(dummy_image, dummy_initial_mask)

    print(f"--- Final Output Logits Shapes (from main) ---")
    print(f"  Coarse logits shape  (D/4): {logits_c.shape}")   # Expected: (B, num_classes, D/4, H/4, W/4)
    print(f"  Refine1 logits shape (D/2): {logits_r1.shape}") # Expected: (B, num_classes, D/2, H/2, W/2)
    print(f"  Final logits shape   (D)  : {logits_f.shape}")   # Expected: (B, num_classes, D,   H,   W)

    # Test with non-default patch sizes (demonstrating constraints if not D/4, H/4, W/4 for all stages)
    # For this specific architecture, the patch size is implicitly D/4, H/4, W/4 due to x_down4 stage.
    # If we wanted truly arbitrary patch sizes smaller than D/4, H/4, W/4, 
    # Stage 1 would also need to use _process_stage.
    # The current design assumes patch_size effectively IS the size of x_down4.
    # print("\nNote: The current model design inherently uses a patch size equivalent to D_orig/4, H_orig/4, W_orig/4 for processing.")
    # print("The `patch_size_d/h/w` parameters should match these derived dimensions.")

    # Example: what if input dimensions are not divisible by 4 for pooling?
    # D_odd, H_odd, W_odd = 63, 127, 127 # AvgPool3d will floor.
    # dummy_image_odd = torch.randn(batch_size, img_channels, D_odd, H_odd, W_odd)
    # dummy_initial_mask_odd = torch.randn(batch_size, init_mask_channels, D_odd, H_odd, W_odd)
    # print(f"\nTesting with odd input dimensions: D={D_odd}, H={H_odd}, W={W_odd}")
    # try:
    #     model_no_verbose = MultiStepSegmentationModel(image_channels=img_channels, initial_mask_channels=init_mask_channels, num_classes=n_classes, base_cnn_hidden_features=cnn_features,patch_size_d=p_d, patch_size_h=p_h, patch_size_w=p_w, verbose=False)
    #     model_no_verbose(dummy_image_odd, dummy_initial_mask_odd)
    # except ValueError as e:
    #     print(f"Caught expected error for non-divisible dimensions: {e}")
    # This test highlights that input dimensions must be compatible with 2x and 4x AvgPooling
    # and that resulting dimensions must be divisible by patch sizes.
    # The tensor_to_patches function has an assertion for this.
    print(f"--- Main Example End ---")

