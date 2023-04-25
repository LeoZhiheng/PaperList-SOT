# PaperReading

整理近兩年閱讀的600+文章(三維目標檢測/跟蹤，二維目標檢測/跟蹤，深度/光流估計等)，施工中~~~~~~~~~~

(只更新頂會頂刊)  
2023.4.20 更新19篇 3D Object Detection (Multi-Frame Fusion, Radar), 3D Single Object Tracking  
2023.4.21 更新20篇 3D Object Detection (Radar), 3D Single Object Tracking, 2D Single Object Tracking   
2023.4.22 更新24篇  3D Object Detection (Radar), 2D Single Object Tracking  
2023.4.23 更新21篇  3D Object Detection (LiDAR Range Image), 2D Single / Multi Object Tracking   
2023.4.24 更新16篇  3D Object Detection (Weakly Supervised, Mono)...**突破100篇**:star2::fire:😄     
2023.4.25 更新12篇  3D Object Detection (Optical Flow)

## Paper List

- [3D Object Detection](https://github.com/LeoZhiheng/PaperReading/blob/main/README.md#3d-object-detection)   
  - [Multi-Frame Fusion](https://github.com/LeoZhiheng/PaperReading/blob/main/README.md#multi-frame-fusion)   
  - [Weakly Supervised](https://github.com/LeoZhiheng/PaperReading/blob/main/README.md#weakly-supervised)
  - [LiDAR Range Image](https://github.com/LeoZhiheng/PaperReading/blob/main/README.md#liDAR-range-image)  
  - [Mono](https://github.com/LeoZhiheng/PaperReading/blob/main/README.md#mono)  
  - [Radar](https://github.com/LeoZhiheng/PaperReading/blob/main/README.md#radar)   
- [Optical Flow](https://github.com/LeoZhiheng/PaperReading/blob/main/README.md#optical-flow)   
- [2D Object Tracking](https://github.com/LeoZhiheng/PaperReading/blob/main/README.md#2D-object-tracking) 
  - [2D Single Object Tracking](https://github.com/LeoZhiheng/PaperReading/blob/main/README.md#2D-single-object-tracking) 
  - [2D Multi Object Tracking](https://github.com/LeoZhiheng/PaperReading/blob/main/README.md#2D-multi-object-tracking) 
- [3D Object Tracking](https://github.com/LeoZhiheng/PaperReading/blob/main/README.md#3D-object-tracking) 
  - [3D Single Object Tracking](https://github.com/LeoZhiheng/PaperReading/blob/main/README.md#3D-single-object-tracking) 

## 3D Object Detection
### Multi Frame Fusion

[MGTANet: Encoding Sequential LiDAR Points Using Long Short-Term Motion-Guided Temporal Attention for 3D Object Detection](https://arxiv.org/abs/2212.00442) [2022 AAAI]

[SWFormer: Sparse Window Transformer for 3D Object Detection in Point Clouds](https://arxiv.org/abs/2210.07372) [2022 ECCV]  

[CenterFormer: Center-based Transformer for 3D Object Detection](https://arxiv.org/abs/2209.05588) [2022 ECCV]  

[MPPNet: Multi-Frame Feature Intertwining with Proxy Points for 3D Temporal Object Detection](https://arxiv.org/abs/2205.05979) [2022 ECCV]  

[3D-MAN: 3D Multi-frame Attention Network for Object Detection](https://arxiv.org/abs/2012.12395) [2021 CVPR]  

[Fast and Furious: Real Time End-to-End 3D Detection, Tracking and Motion Forecasting with a Single Convolutional Net](https://arxiv.org/abs/2012.12395) [2018 CVPR]

### Weakly Supervised

[Weakly Supervised 3D Object Detection from Point Clouds](https://arxiv.org/abs/2007.13970) [2022 ACM MM]     

### LiDAR Range Image  

[Fully Convolutional One-Stage 3D Object Detection on LiDAR Range Images](https://arxiv.org/abs/2205.13764) [2022 NIPS]            

[VISTA: Boosting 3D Object Detection via Dual Cross-VIew SpaTial Attention](https://arxiv.org/abs/2203.09704) [2022 CVPR]    

[LaserFlow: Efficient and Probabilistic Object Detection and Motion Forecasting](https://arxiv.org/abs/2003.05982) [2021 RAL]    

[To the Point : Efficient 3D Object Detection in the Range Image with Graph Convolution Kernels](https://arxiv.org/abs/2106.13381) [2021 CVPR]    

[RangeIoUDet: Range Image based Real-Time 3D Object Detector Optimized by Intersection over Union](https://ieeexplore.ieee.org/document/9578898) [2021 CVPR]    

[RangeDet: In Defense of Range View for LiDAR-based 3D Object Detection](https://arxiv.org/abs/2103.10039) [2021 ICCV]           

[RSN: Range Sparse Net for Efficient, Accurate LiDAR 3D Object Detection](https://arxiv.org/abs/2106.13365) [2021 CVPR]    

[RV-FuseNet: Range View Based Fusion of Time-Series LiDAR Data for Joint 3D Object Detection and Motion Forecasting](https://arxiv.org/abs/2005.10863) [2021 IROS]     

[Range Conditioned Dilated Convolutions for Scale Invariant 3D Object Detection](https://arxiv.org/abs/2005.09927) [2020 CoRL]      

[LaserNet: An Efficient Probabilistic 3D Object Detector for Autonomous Driving](https://arxiv.org/abs/1903.08701) [2019 CVPR]      

### Mono

[MonoDETR: Depth-aware Transformer for Monocular 3D Object Detection](https://arxiv.org/abs/2203.13310) [2022 CVPR]      

[MonoDTR: Monocular 3D Object Detection with Depth-Aware Transformer](https://arxiv.org/abs/2203.10981) [2022 CVPR]     

[Densely Constrained Depth Estimator for Monocular 3D Object Detection](https://arxiv.org/abs/2207.10047) [2022 ECCV]     

[Categorical Depth Distribution Network for Monocular 3D Object Detection](https://arxiv.org/abs/2103.01100) [2021 CVPR]    

[Delving into Localization Errors for Monocular 3D Object Detection](https://arxiv.org/abs/2103.16237) [2021 CVPR]    

[Orthographic Feature Transform for Monocular 3D Object Detection](https://arxiv.org/abs/1811.08188) [2020 BMVC]     

[MonoGRNet: A Geometric Reasoning Network for Monocular 3D Object Localization](https://arxiv.org/abs/1811.10247) [2019 AAAI]       

[Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving](https://arxiv.org/abs/1812.07179) [2019 CVPR]  

[Multi-Level Fusion based 3D Object Detection from Monocular Images](https://ieeexplore.ieee.org/document/8578347) [2018 CVPR]  


### Radar

[Modality-Agnostic Learning for Radar-Lidar Fusion in Vehicle Detection](https://ieeexplore.ieee.org/document/9879704) [Radar+LiDAR] [2022 CVPR]  

[RaLiBEV: Radar and LiDAR BEV Fusion Learning for Anchor Box Free Object Detection Systems](https://arxiv.org/abs/2211.06108) [Radar+LiDAR] [2022 TNNLS]  

[K-Radar: 4D Radar Object Detection for Autonomous Driving in Various Weather Conditions](https://arxiv.org/abs/2206.08171) [4D Radar] [2022 NIPS]  

[Robust Multimodal Vehicle Detection in Foggy Weather Using Complementary Lidar and Radar Signals](https://ieeexplore.ieee.org/document/9578621) [Radar+LiDAR] [CVPR 2021]   

[RPFA-Net: a 4D RaDAR Pillar Feature Attention Network for 3D Object Detection](https://ieeexplore.ieee.org/document/9564754) [4D Radar] [2021 ITSC]  

[Seeing Through Fog Without Seeing Fog: Deep Multimodal Sensor Fusion in Unseen Adverse Weather](https://arxiv.org/abs/1902.08913) [Radar+LiDAR+Camera] [2020 CVPR]  

[RadarNet: Exploiting Radar for Robust Perception of Dynamic Objects](https://arxiv.org/abs/2007.14366) [Radar+LiDAR] [2020 ECCV]  

## Optical Flow

[Learning Optical Flow with Kernel Patch Attention(https://openaccess.thecvf.com/content/CVPR2022/papers/Luo_Learning_Optical_Flow_With_Kernel_Patch_Attention_CVPR_2022_paper.pdf) [2022 CVPR]    

[SKFlow: Learning Optical Flow with Super Kernels](https://arxiv.org/abs/2205.14623) [2022 NIPS]         

[FlowFormer: A Transformer Architecture for Optical Flow](https://arxiv.org/abs/2203.16194) [2022 ECCV]       

[Global Matching with Overlapping Attention for Optical Flow Estimation](https://arxiv.org/abs/2203.11335) [2022 CVPR]    

[Deep Equilibrium Optical Flow Estimation](https://arxiv.org/abs/2204.08442) [2022 CVPR]    

[Imposing Consistency for Optical Flow Estimation](https://arxiv.org/abs/2204.07262) [2022 CVPR]     

[Learning Optical Flow with Adaptive Graph Reasoning](https://arxiv.org/abs/2202.03857) [2022 AAAI]     

[GMFlow: Learning Optical Flow via Global Matching](https://arxiv.org/abs/2111.13680) [2022 CVPR]    

[CRAFT: Cross-Attentional Flow Transformer for Robust Optical Flow](https://arxiv.org/abs/2203.16896) [2022 CVPR]  

[Learning Optical Flow from a Few Matches](https://arxiv.org/abs/2104.02166) [2021 CVPR]    

[Separable Flow: Learning Motion Cost Volumes for Optical Flow Estimation](https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Separable_Flow_Learning_Motion_Cost_Volumes_for_Optical_Flow_Estimation_ICCV_2021_paper.html) [2021 ICCV]     

[Learning to Estimate Hidden Motions with Global Motion Aggregation](https://arxiv.org/abs/2104.02409) [2021 ICCV]    

[RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039) [2020 ECCV]       

[A Fusion Approach for Multi-Frame Optical Flow Estimation](https://arxiv.org/abs/1810.10066) [2019 WACV]    

[Volumetric Correspondence Networks for Optical Flow](http://papers.neurips.cc/paper/8367-volumetric-correspondence-networks-for-optical-flow) [2019 NIPS]     

[PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume](https://arxiv.org/abs/1709.02371) [2018 CVPR]         

[LiteFlowNet: A Lightweight Convolutional Neural Network for Optical Flow Estimation](https://arxiv.org/abs/1805.07036) [2018 CVPR]    

[FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks](https://arxiv.org/abs/1612.01925) [2017 CVPR]    

## 2D Object Tracking
### 2D Single Object Tracking

[DropMAE: Masked Autoencoders with Spatial-Attention Dropout for Tracking Tasks](https://arxiv.org/abs/2304.00571) [2023 CVPR]   

[MixFormer: End-to-End Tracking with Iterative Mixed Attention](https://arxiv.org/abs/2203.11082) [2022 CVPR]    

[Towards Grand Unification of Object Tracking](https://arxiv.org/abs/2207.07078) [2022 ECCV]    

[Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework](https://arxiv.org/abs/2203.11991) [2022 ECCV]       

[Towards Sequence-Level Training for Visual Tracking](https://arxiv.org/abs/2208.05810) [2022 ECCV]    

[FEAR: Fast, Efficient, Accurate and Robust Visual Tracker arXiv:2112.07957v1](https://arxiv.org/abs/2112.07957) [2022 ECCV]     

[AiATrack: Attention in Attention for Transformer Visual Tracking](https://arxiv.org/abs/2207.09603) [2022 ECCV]    

[SparseTT: Visual Tracking with Sparse Transformers](https://arxiv.org/abs/2205.03776) [2022 IJCAI]    

[High-Performance Discriminative Tracking with Transformers](https://ieeexplore.ieee.org/document/9710969) [2021 ICCV]  

[Graph Attention Tracking](https://arxiv.org/abs/2011.11204) [2021 CVPR]        

[Learning Target Candidate Association to Keep Track of What Not to Track](https://arxiv.org/abs/2103.16556) [2021 ICCV]   

[SiamRCR: Reciprocal Classification and Regression for Visual Object Tracking](https://arxiv.org/abs/2105.11237V) [2021 IJCAI]    

[Transformer Meets Tracker: Exploiting Temporal Context for Robust Visual Tracking](https://arxiv.org/abs/2103.11681) [2021 CVPR]    

[Learn to Match: Automatic Matching Network Design for Visual Tracking](https://arxiv.org/abs/2108.00803) [2021 ICCV]    

[STMTrack: Template-free Visual Tracking with Space-time Memory Networks](https://arxiv.org/abs/2104.00324) [2021 CVPR]

[Saliency-Associated Object Tracking](https://arxiv.org/abs/2108.03637) [2021 ICCV]    

[MART: Motion-Aware Recurrent Neural Network for Robust Visual Tracking](https://ieeexplore.ieee.org/document/9423078) [2021 WACV]     

[Transformer Tracking](https://arxiv.org/abs/2103.15436) [2021 CVPR]     

[Learning Spatio-Temporal Transformer for Visual Tracking](https://arxiv.org/abs/2103.17154) [2021 ICCV]  

[High-Performance Long-Term Tracking with Meta-Updater](https://arxiv.org/abs/2004.00305) [2020 CVPR]     

[Siam R-CNN: Visual Tracking by Re-Detection](https://arxiv.org/abs/1911.12836) [2020 CVPR]    

[Optical Flow in Deep Visual Tracking](https://ojs.aaai.org/index.php/AAAI/article/view/6890) [2020 AAAI]     

[Siamese Box Adaptive Network for Visual Tracking](https://arxiv.org/abs/2003.06761) [2020 CVPR]    

[Tracking by Instance Detection: A Meta-Learning Approach](https://arxiv.org/abs/2004.00830) [2020 CVPR]        

[State-Aware Tracker for Real-Time Video Object Segmentation](https://arxiv.org/abs/2003.00482) [2020 CVPR]         

[Deformable Siamese Attention Networks for Visual Object Tracking](https://arxiv.org/abs/2004.06711) [2020 CVPR]    

[Tracking Objects as Points](https://arxiv.org/abs/2004.01177) [2020 ECCV]      

[Model-free Tracking with Deep Appearance and Motion Features Integration](https://arxiv.org/abs/1812.06418) [2019 WACV]   

[ATOM: Accurate Tracking by Overlap Maximization](https://arxiv.org/abs/1811.07628) [2019 CVPR]     

[Fast Online Object Tracking and Segmentation: A Unifying Approach](https://arxiv.org/abs/1901.01660) [2019 CVPR]      

[Deeper and Wider Siamese Networks for Real-Time Visual Tracking](https://arxiv.org/abs/1901.01660) [2019 CVPR]   

[SiamRPN: High Performance Visual Tracking with Siamese Region Proposal Network](https://arxiv.org/abs/1808.06048) [2018 CVPR]   

[Distractor-aware Siamese Networks for Visual Object Tracking](https://arxiv.org/abs/1808.06048) [2018 ECCV]  

[End-to-end Flow Correlation Tracking with Spatial-temporal Attention](https://arxiv.org/abs/1711.01124) [2018 CVPR]  

[GOTURN: Learning to Track at 100 FPS with Deep Regression Networks](https://arxiv.org/abs/1604.01802) [2016 ECCV]  

[Fully-Convolutional Siamese Networks for Object Tracking](https://arxiv.org/abs/1606.09549) [2016 ECCV]  

### 2D Multi Object Tracking  

[Multiple Object Tracking with Correlation Learning](https://arxiv.org/abs/2104.03541) [2021 CVPR]       

[Deep SORT: Simple online and realtime tracking with a deep association metric](https://arxiv.org/abs/1703.07402) [2018 ICIP]     

[SORT: Simple online and realtime tracking](https://arxiv.org/abs/1602.00763) [2016 ICIP]    

## 3D Object Tracking
### 3D Single Object Tracking  

[A Lightweight and Detector-Free 3D Single Object Tracker on Point Clouds](https://arxiv.org/abs/2203.04232) [2023 TITS]  

[Spatio-Temporal Contextual Learning for Single Object Tracking on Point Clouds](https://ieeexplore.ieee.org/document/10011208) [2023 TNNLS]  

[CMT: Context-Matching-Guided Transformer for 3D Tracking in Point Clouds](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820091.pdf) [2022 ECCV]   

[GLT-T: Global-Local Transformer Voting for 3D Single Object Tracking in Point Clouds](https://arxiv.org/abs/2211.10927) [2022 AAAI] 

[Implicit and Efficient Point Cloud Completion for 3D Single Object Tracking](https://arxiv.org/abs/2209.00522) [2022 RAL]   

[Exploiting More Information in Sparse Point Cloud for 3D Single Object Tracking](https://arxiv.org/abs/2210.00519) [2022 RAL] :star2::fire:  

[3D Single-Object Tracking with Spatial-Temporal Data Association](https://ieeexplore.ieee.org/document/9981905) [2022 IROS]  

[3D Siamese Transformer Network for Single Object Tracking on Point Clouds](https://arxiv.org/abs/2207.11995) [2022 ECCV]  

[Beyond 3D Siamese Tracking: A Motion-Centric Paradigm for 3D Single Object Tracking in Point Clouds](https://arxiv.org/abs/2203.01730) [2022 CVPR]  

[PTTR: Relational 3D Point Cloud Object Tracking with Transformer](https://arxiv.org/abs/2112.02857) [2022 CVPR]  

[Temporal-aware Siamese Tracker : Integrate Temporal Context for 3D Object Tracking](https://openaccess.thecvf.com/content/ACCV2022/html/Lan_Temporal-aware_Siamese_Tracker_Integrate_Temporal_Context_for_3D_Object_Tracking_ACCV_2022_paper.html) [2022 ACCV]  

[Graph-Based Point Tracker for 3D Object Tracking in Point Clouds](https://ojs.aaai.org/index.php/AAAI/article/view/20101) [2022 AAAI]  

[MLVSNet: Multi-level Voting Siamese Network for 3D Visual Tracking](https://ieeexplore.ieee.org/document/9710975) [2021 ICCV]  

[3D Siamese Voxel-to-BEV Tracker for Sparse Point Clouds](https://arxiv.org/abs/2111.04426) [2021 NIPS]  

[Box-Aware Feature Enhancement for Single Object Tracking on Point Clouds](https://arxiv.org/abs/2108.04728) [2021 ICCV]   

[PTT: Point-Track-Transformer Module for 3D Single Object Tracking in Point Clouds](https://arxiv.org/abs/2108.06455) [2021 IROS] :star2::fire:

[Model-free Vehicle Tracking and State Estimation in Point Cloud Sequences](https://arxiv.org/abs/2103.06028) [2021 IROS]  

[3D Object Tracking with Transformer](https://arxiv.org/abs/2110.14921) [2021 BMVC] :star2::fire:  

[P2B: Point-to-box network for 3D object tracking in point clouds](https://arxiv.org/abs/2005.13888) [2020 CVPR]    

[F-Siamese Tracker: A Frustum-based Double Siamese Network for 3D Single Object Tracking](https://arxiv.org/abs/2010.11510) [2020 IROS]    

[Leveraging Shape Completion for 3D Siamese Tracking](https://arxiv.org/abs/1903.01784) [2019 CVPR]   

[Efficient Bird Eye View Proposals for 3D Siamese Tracking](https://arxiv.org/abs/1903.10168) [2019 BMVC]  
