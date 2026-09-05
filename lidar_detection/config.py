from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class VoxelConfig:
    point_cloud_range: List[float] = field(default_factory=lambda: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
    voxel_size: List[float] = field(default_factory=lambda: [0.2, 0.2, 8.0])
    max_points_per_voxel: int = 100
    max_voxels: int = 16000

    @property
    def grid_size(self) -> List[int]:
        pc_range = self.point_cloud_range
        vsize = self.voxel_size
        return [
            round((pc_range[3] - pc_range[0]) / vsize[0]),  # X: 432
            round((pc_range[4] - pc_range[1]) / vsize[1]),  # Y: 496
            round((pc_range[5] - pc_range[2]) / vsize[2]),  # Z: 1
        ]


@dataclass
class ModelConfig:
    num_point_features: int = 4           # x, y, z, intensity
    pillar_feature_dim: int = 64
    backbone_out_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    num_classes: int = 4                   # Car, Bus, Motorcycle, Pedestrian
    class_names: List[str] = field(default_factory=lambda: ["Car", "Bus", "Motorcycle", "Pedestrian"])
    # Optional per-class inference score thresholds (tau_{c,3D} in Algorithm 1),
    # indexed by class_id / class_names order. None = use the caller's scalar
    # default (DetectionPipeline.run's score_threshold argument).
    class_score_thresholds: Optional[List[float]] = None


@dataclass
class AnchorConfig:
    # [length, width, height, z_center, rotation] per class, order must match
    # ModelConfig.class_names. Only consumed by the anchor head (PointPillars);
    # the CenterPoint head regresses size/rotation directly and does not use
    # these. Kept in sync anyway so the anchor head doesn't silently break.
    # z_center calibrated for nuScenes: LiDAR ~1.84m above ground,
    # so ground objects sit at roughly z = -1.84 + height/2
    anchor_sizes: List[List[float]] = field(default_factory=lambda: [
        [4.63,  1.97, 1.74, -1.00, 0.0],  # Car
        [10.50, 2.94, 3.47, -0.10, 0.0],  # Bus
        [2.11,  0.77, 1.47, -1.10, 0.0],  # Motorcycle
        [0.73,  0.67, 1.77, -0.95, 0.0],  # Pedestrian
    ])
    anchor_rotations: List[float] = field(default_factory=lambda: [0.0, 1.5707963])
    pos_iou_threshold: float = 0.6
    neg_iou_threshold: float = 0.45


@dataclass
class TrainConfig:
    batch_size: int = 6
    lr: float = 2e-4
    epochs: int = 80
    weight_decay: float = 0.01
    score_threshold: float = 0.5
    nms_iou_threshold: float = 0.5


@dataclass
class PipelineConfig:
    voxel: VoxelConfig = field(default_factory=VoxelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    anchor: AnchorConfig = field(default_factory=AnchorConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
