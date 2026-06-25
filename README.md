# HTC-SGA-Former: A Hybrid Transformer-CNN Network with Self-Guided Attention and a New Boundary-Weighted Adaptive Loss for Coronary DSA Vessel Segmentation
# Abstract
Accurate coronary Digital Subtraction Angiography (DSA) vessel segmentation is essential for computer-aided diagnosis and treatment planning of coronary artery disease (CAD). However, thin low-contrast vessels, background interference, and severe vessel-background class imbalance make reliable segmentation of weak distal branches and vessel boundaries challenging. Existing segmentation methods struggle to balance global contextual reasoning with preservation of weak distal vessels, vessel continuity, and fine boundaries. To address these limitations, we propose HTC-SGA Former, a lightweight hybrid Transformer-CNN framework for coronary DSA vessel segmentation. The framework employs a CNN encoder for local vessel morphology extraction and a Transformer decoder for contextual feature modeling. A Multi-Scale Global-Local Window Attention (MS-GLWA) block performs efficient global-local contextual modeling, while a Self-Guided Feature Attention (SGFA) module enhances weak-vessel responses. In addition, a Boundary-Weighted Adaptive Compound Loss (BWACL) emphasizes thin-vessel boundaries and adaptively balances vessel recovery and boundary refinement under severe class imbalance. Experimental results on private right and left coronary artery DSA subsets demonstrate that HTC-SGA Former consistently outperforms 14 state-of-the-art segmentation methods while maintaining a compact architecture with only 0.81M parameters. Moreover, BWACL improves performance over conventional binary cross-entropy and Dice losses across four widely used encoder-decoder architectures, demonstrating strong cross-backbone applicability. HTC-SGA Former provides accurate and parameter-efficient coronary DSA vessel segmentation by improving thin-vessel recovery, vessel continuity, and boundary localization through complementary global-local contextual modeling, vessel-focused refinement, and adaptive optimization. The proposed framework enables reliable and computationally efficient coronary vessel analysis, supporting quantitative image analysis and future computer-assisted cardiovascular interventions.  

# Architecture
<img width="7326" height="4500" alt="Fig 1" src="https://github.com/user-attachments/assets/7b032ecb-29be-4d04-9706-b567822e9fee" />
Fig. Comparison of medical image segmentation architectures: (a) pure CNN encoder-decoder (U-Net (Ronneberger
et al., 2015)); (b) hybrid CNN with Transformer bottleneck (e.g., TransUNet (Chen et al., 2021)); (c) pure Transformer
encoder-decoder (e.g., SwinUNet (Cao et al., 2022)); and (d) the HTC-SGA Former combining a lightweight CNN encoder
with a Transformer decoder. The lower panel compares the standard Transformer block with the proposed MS-GLWA
decoder block integrating global window-based self-attention and local convolution.

<img width="7780" height="4500" alt="mixed" src="https://github.com/user-attachments/assets/09d4a7ad-30fe-4691-b5ca-b8b6275946e6" />
Fig. Overview of the proposed HTC-SGA Former. (a) Overall architecture comprising a CNN-based encoder and a Transformer-based decoder incorporating
Efficient WF, MS-GLWA, and SGFA. (b) Structure of the proposed MS-GLWA block, which combines enhanced multi-scale window-based self-attention with
a local convolution branch. (c) Structure of the SGFA module for vessel-focused refinement through preliminary vessel-guided attention, multi-scale context
extraction, and feature fusion.

# Results
<img width="1742" height="595" alt="Untitled" src="https://github.com/user-attachments/assets/c16f3455-b25f-43ee-b528-986f9b896a85" />
Fig. Recall comparison of different loss functions across U-Net, Attention U-Net, U-Net++, and Attention U-Net++.
(a) Right coronary artery subset. (b) Left coronary artery subset. BWACL achieves the highest recall.
<img width="5475" height="3168" alt="Fig 9" src="https://github.com/user-attachments/assets/e9dcbaba-074a-4a39-8cf7-38cc8ed920d9" />
Fig. Qualitative comparison on the right coronary artery subset between HTC-SGA Former and state-of-the-art
segmentation methods. Yellow rectangles highlight challenging regions with thin distal branches, vessel discontinuities,
boundary ambiguity, and false-positive responses.
<img width="5493" height="3177" alt="Fig 10" src="https://github.com/user-attachments/assets/24aed326-e1a3-4271-b0e4-dc8442666050" />
Fig. Qualitative comparison on the left coronary artery subset between HTC-SGA Former and state-of-the-art
segmentation methods. Yellow rectangles highlight challenging regions with thin distal branches, complex bifurcations,
boundary localization, and false-positive responses.
