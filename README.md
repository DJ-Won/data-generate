# 3DGS 多摄平滑变焦视频生成器

该项目从标准 3D Gaussian Splatting PLY 场景直接渲染一条带轻微连续相机运动的变焦视频。变焦由真实的相机内参 `fx/fy` 改变实现，不使用后处理裁剪或 resize。默认配置已经指向 `3dscene/garden/scene.ply`。

## 运行

环境已按本项目验证为 conda `gs`：

```bash
cd /home/wdj/projects/data-generate
conda run -n gs python generate_zoom_video.py \
  --scene-config configs/zoom_video/scenes/garden.yaml \
  --camera-config configs/zoom_video/cameras/garden_reference_static.yaml \
  --color-config configs/zoom_video/colors/multicamera_default.yaml \
  --camera-json outputs/scene_traversal_fixed_intrinsics/garden/position_0008/lens_0013/camera.json
```

先只检查配置：

```bash
conda run -n gs python generate_zoom_video.py \
  --scene-config configs/zoom_video/scenes/garden.yaml \
  --camera-config configs/zoom_video/cameras/garden_reference_static.yaml \
  --color-config configs/zoom_video/colors/multicamera_default.yaml \
  --camera-json outputs/scene_traversal_fixed_intrinsics/garden/position_0008/lens_0013/camera.json \
  --validate-only
```

## 使用训练相机直接渲染

`render_dataset_cameras.py` 从标准 3DGS 数据目录自动读取根目录下的
`cameras.json`（也兼容单个 `camera.json`），并选择数值最大的
`point_cloud/iteration_*/point_cloud.ply`。启动时只需给出数据路径和输出根目录：

```bash
./scripts/render_dataset_cameras.sh \
  /data0/wdj/datasets/dl3dv-gs/3DGS/1K/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f \
  ./outputs/dl3dv_camera_renders
```

输出按原始图像名自然排序，并保持输入相机的宽高和 `fx/fy`：

```text
outputs/dl3dv_camera_renders/<scene_name>/
  position_0000/lens_0000/image.png
  position_0000/lens_0000/camera.json
  ...
  render_summary.json
```

完整的 `image.png + camera.json` 对在再次运行时会自动跳过，以支持断点续跑。
半成品会明确报错，可传 `--overwrite` 修复；调试时可传 `--max-cameras 1`。
该模式使用输入 JSON 中的原始 OpenCV/COLMAP camera-to-world 矩阵和 identity
场景根变换，并保留 PLY 中全部数值有限的高斯。

变焦任务的三个配置文件按职责严格分工：

- scene：`input/scene/output/render/video/scene_analysis`，其中 `render` 只包含设备、
  SH 阶数和抗锯齿；
- camera：`camera` 以及倍率曲线、分帧、镜头倍率区间、相机光心偏移；
- color：背景色、镜头切色、时序曝光跳变和各镜头颜色参数。

相机与颜色文件中的镜头通过 `name` 合并，名称缺失、重复、不一致或字段放错文件都会
在生成前报错。

`--camera-json` 接受场景遍历任务生成的单个 `camera.json`。程序使用其中完整的
`camera_to_world` 作为初始外参，并把其中的垂直 FOV 作为 camera YAML 中首个镜头
`zoom_min` 对应的内参；程序会自动换算内部使用的 1× FOV。主点会按 JSON 图像尺寸
归一化后映射到视频分辨率。near/far 仍由场景和相机 YAML 计算。
JSON 中的相机位置、旋转矩阵和 `scene_root_transform` 会被严格校验，根变换与
scene YAML 不一致时拒绝生成。JSON 中的 `scene_type` 会用于选择 object/interior
轨迹边界规则；旧 JSON 没有该字段时保持兼容，默认按 interior 处理。
validated 渲染入口可通过 `camera.traversal_pullback` 在写入 JSON 前沿原相机视线反向
移动机位，以降低末端高倍率造成的几何拉伸；实际位移、原始位置和边界限幅会记录在
`camera.traversal_pullback`，流水线摘要也会保存该配置。

`generate_zoom_video.py` 不再提供 `--config` 组合配置入口。scene、camera、color
三个拆分参数均为必填；`--camera-json` 可选，省略时按 camera YAML 的初始化策略执行。

garden 拆分配置默认输出到 `outputs/garden_zoom_ref5`，其 `overwrite: false`
会保护已有输出。

## 架构

- `zoomgen/config.py`：Pydantic 严格 schema、跨字段校验、输出保护和 resolved config。
- `zoomgen/scene.py`：PLY 字段验证、SH 阶数/opacity 表示识别、内存映射、分块过滤、鲁棒范围估计和分块 GPU 搬运。
- `zoomgen/camera.py`：自适应内参、原世界坐标相机矩阵、裁剪面、低频样条运动和物理模组光心偏移。
- `zoomgen/zoom.py`：确定性分帧、倍率曲线、镜头切换和镜头内曝光跳变状态。
- `zoomgen/renderer.py`：复用环境中的 CUDA `diff_gaussian_rasterization`。alpha 由白色预计算颜色在黑背景上直接栅格化得到。
- `zoomgen/color.py`：浮点颜色管线、EV 跳变、线性域镜头 crossfade、gamma、暗角和可复现噪声。
- `zoomgen/video.py`：向 conda 环境内 FFmpeg 的 raw RGB 管道编码，并用 OpenCV 回读验收。
- `zoomgen/pipeline.py`：自动取景、完整轨迹预检查、渲染、编码和元数据汇总。

## 常用配置

最常调整的是：

- `video.width/height/fps/total_frames`：输出规格；
- `camera.initial_view.azimuth_deg/elevation_deg`：自动视角方向；也可同时指定 `position/look_at`；
- `camera.auto_fit.*`：alpha 覆盖目标和背景连通域限制；
- `camera.traversal_pullback.*`：validated 渲染入口生成 camera.json 时的视线反向
  拉远比例（相对相机到场景中心距离）；室内相机会按鲁棒 AABB 自动限幅；
- `zoom.lenses[].zoom_min/zoom_max`：UW/W/L 物理倍率段；
- `zoom.lenses[].temporal_color_jump`：每个镜头内一次曝光跳变的强度和位置；
- `zoom.lenses[].color`：模组间基础色彩差异；
- `camera.motion` 与 `camera_center_offset_ratio`：连续微动和光心差异；
- `zoom.lens_switch`：硬切或线性域 crossfade；
- `output.overwrite`：已有输出保护。

比例分帧采用“最大余数法”，余数相同时按镜头配置顺序分配。每个非零倍率区间必须至少有两帧，以严格包含首尾倍率。显式分帧时每个 `frame_count` 都必须设置，且总和必须等于 `total_frames`。

## 输出

```text
zoom_video.mp4
frames/frame_000000.png ...
camera_trajectory.json
frame_metadata.json
generation_summary.json
resolved_config.yaml
```

按配置还可保存 `linear_frames/*.npy`（float16、gamma 前）和
`alpha_masks/*.png`。逐帧元数据包含镜头、倍率、内外参、裁剪面、光心偏移、
覆盖率、基础颜色参数、跳变位置/进度/EV/增益及有效噪声。汇总中包含原始/有效/
过滤高斯数、鲁棒 AABB/中心/半径、自动取景搜索历史、覆盖率统计和镜头全局区间。

## 色彩和数值约定

PLY 的标准 3DGS `opacity` 通常是 logit；若样本全部位于 [0,1]，程序按概率处理。
`scale_*` 按标准 log-scale 解码，`rot_*` 在加载时归一化，SH 布局与
Graphdeco rasterizer 一致。颜色调整在 float32 渲染值上执行；高光使用从 0.8
开始的 soft-shoulder，之后才 gamma 编码和 8-bit 量化。可将
`tone_mapping: clip` 改为硬截断。

多数公开 3DGS PLY 是从 gamma 编码照片训练的，SH 系数并非经过物理标定的传感器
线性辐照度。本工具按线性工作值执行曝光/颜色运算以满足一致的数据生成语义，但不能
恢复输入中不存在的物理线性响应。

## 资源与兼容性

garden PLY 约 3.64 GB、1541 万高斯。读取使用 mmap 和分块校验，不制造完整 CPU
副本；渲染前仍需将有效高斯（包括全部 SH）常驻 GPU。已在 24 GB RTX 4090 和
`gs` 环境的 CUDA rasterizer 上设计。当前 rasterizer 不提供 CPU 后端。

自动取景在最广角帧上二分搜索，并对完整轨迹逐帧做低分辨率 alpha 预检查。对于有孔洞、
非满屏形状或不合适的初始方位，约束可能互相冲突；程序会保留评分最佳结果并在
`generation_summary.json` 写入警告，不会无限搜索。

## 室内场景初始化与参考图对齐

旧版仅适用于物体级场景：它用鲁棒包围球半径把相机放到场景中心外侧，再沿视线调距离。
对房间、园林和建筑内部，这会把相机推到几何体外。现在
`camera.initialization.scene_type` 支持 `object`、`interior`、`manual`、
`reference_image` 和 `auto`；默认 garden 配置使用 `reference_image`。

- `object` 保留原来的外部观察与距离自适应逻辑；
- `interior` 在鲁棒 AABB 内建立占据栅格，搜索有净空的内部机位，只优化朝向和 FOV，
  不再向场景外后退；
- `manual` 直接接受 `position + look_at` 或 `position + yaw/pitch/roll`；
- `reference_image` 先粗搜机位/朝向/FOV，再对 Top-K 局部细化，并输出灰度、梯度和
  SSIM 结构损失；`auto` 根据是否提供参考图选择室内或物体流程。

garden PLY 的有效世界变换是 `rotation_euler_deg: [180, 0, 0]`。默认配置还写入了
公开 SuperSplat 场景的相机种子；搜索网格保证精确评估该种子，不会因偶数采样遗漏它。
所有姿态均为列向量的 camera-to-world 矩阵，相机轴采用 OpenCV/COLMAP：
`+x` 向右、`+y` 向下、`+z` 向前，图像原点位于左上角。根变换定义为
`p_world = translation + scale * R_euler * p_ply`。

参考图不会被非等比拉伸：程序先按可选 `crop_xywh` 裁剪，再自动去除纯色边框，
最后中心裁剪到预览宽高比。多张参考图可用
`reference.image_paths + relative_rotation_euler_deg` 提供相对旋转约束。

## 对齐诊断与人工选择

参考图模式总会写入：

```text
camera_alignment/reference_processed.png
camera_alignment/best_aligned.png
camera_alignment/overlay_reference_aligned.png
camera_alignment/difference_map.png
camera_alignment/contact_sheet.png
camera_alignment/candidate_poses.json
camera_alignment/alignment_report.json
```

若前两名损失差不超过 `ambiguity_score_margin`，或最佳相似度低于
`minimum_similarity_score`，程序以退出码 3 停止，不生成最终视频。查看
`contact_sheet.png` 后，把所选编号写入
`camera.initialization.reference_match.selected_candidate_index` 再运行；显式选择会被记录到
`resolved_config.yaml` 和 `generation_summary.json`。也可以改用 `manual` 模式填写已知姿态。

单张截图若裁剪、宽高比、后期处理或拍摄位置未知，通常存在多个相似解；此时增加第二张带
相对旋转的参考图，比单纯扩大随机搜索更可靠。输出的相似度是本工具内部结构分数，
用于候选排序和低置信度保护，不是跨数据集可比较的感知质量指标。

## 独立场景遍历任务

该任务不调用或改变变焦视频生成流程。使用独立入口和配置：

```bash
conda run -n gs python generate_scene_traversal.py \
  --scene-config configs/travel/scenes/apartment.yaml \
  --camera-config configs/travel/cameras/default.yaml \
  --color-config configs/travel/colors/default.yaml
```

三个文件按顶层字段严格分工：

- scene：`input/scene/output/scene_analysis`，其中 `scene.scene_type` 描述场景类型；
- camera：`initialization`，包含固定内参、外参遍历范围与 k/l；
- color：`render/image`，包含背景、SH/抗锯齿、gamma 和 PNG 编码。

加载器会拒绝跨文件放错的字段和缺少的必填部分。旧的组合式
`--config configs/garden/traversal.yaml` 仍保留兼容，但不能与三个新参数混用。

可先使用 `--validate-only`。程序会在 GPU 渲染前规划机位和镜头。两类场景由
`scene.scene_type` 选择：

- `interior`：机位位于变换后的鲁棒 AABB 内，并满足净空和最小间距约束；
- `object`：将场景视为航拍/外部观察目标。机位使用带随机相位的低差异序列分布在
  上半球壳层，让任意一段候选都尽量覆盖不同方位和仰角；视线始终朝向场景中心附近，
  只叠加 `initialization.object` 中的小幅 yaw/pitch/roll 扰动，且不会产生上仰镜头。

通用的 `yaw_search_range_deg`、`pitch_search_range_deg` 和
`roll_search_range_deg` 继续控制 interior。object 使用以下专属字段：

- `distance_range_radius_ratio`、`azimuth_range_deg`、`elevation_range_deg`：
  初始上半球壳层；
- `center_yaw_jitter_range_deg`、`center_pitch_jitter_range_deg`、
  `roll_jitter_range_deg`：相对中心视线的小幅扰动；
- `random_candidate_multiplier`：object 随机质量模式的 TOPIQ-NR 评分预算为
  `target_count_k * random_candidate_multiplier`；覆盖预检失败不消耗该预算。

object 的 yaw/pitch/roll 不叠加 `seed_pose` 中的旧角度，只使用中心视线和上述专属
jitter；`seed_pose` 在该分支仅提供可选初始位置和固定 FOV。

object 示例：

```bash
conda run -n gs python generate_scene_traversal.py \
  --scene-config configs/travel/scenes/oxford.yaml \
  --camera-config configs/travel/cameras/object.yaml \
  --color-config configs/travel/colors/default.yaml
```

每个 object 候选在完整分辨率渲染和 TOPIQ-NR 之前，先以
`object.coverage.preview_width/preview_height` 渲染 alpha。覆盖率定义为
`alpha > alpha_threshold` 的像素比例；若低于 `minimum_pixel_ratio`，相机保持
朝向不变并沿场景中心径向拉近。搜索先用最近的合法外部位置确定目标是否可达，再在
失败/成功距离之间进行有界搜索，最终保留满足覆盖率的最远机位，避免越过阈值后仍贴近
场景。低分辨率搜索最多评估 `max_iterations` 次，选定机位还会以最终输出分辨率复核；
若预览存在像素离散误差，最终分辨率同样执行有界搜索。拉近下界是该方向上鲁棒 AABB
的外边界加
`exterior_margin_radius_ratio * scene_radius`，因此相机不会进入鲁棒包围盒。
planned 模式无法满足覆盖率时会明确报错；random 模式会跳过该候选，避免继续执行昂贵
的完整渲染和评分，并从备用方向继续采样。`max_position_sampling_attempts` 控制可用于
覆盖预检的机位上限；候选按镜头编号优先排序，因此会先从所有不同机位各取一个镜头，
再考虑同一机位的额外镜头。每个 `camera.json` 都记录规划/最终位置、距离、覆盖率和
搜索历史。

object 的质量门控由 `initialization.traversal.strategy: random` 启用。渲染过程中
始终维护 TOPIQ-NR 分数最高的 k 组候选：新分数高于当前最低分时立即替换内存候选。
累计得到 k 个严格满足 `topiq_nr > topiq_nr_threshold_l` 的结果后提前结束；如果
评分预算耗尽仍不足 k 个达标结果，则不报错，统一保存已评分候选中最高的 k 组（默认
就是十组），并在终端打印回退警告、在 summary 中写入 `fallback_used: true`。
`termination_reason` 会区分 `threshold_complete`、`score_budget_exhausted` 和
`candidate_pool_exhausted`，`search_exhaustive` 表示是否真的穷尽了覆盖候选池。
只有当整个覆盖候选池中都
找不到 k 个满足覆盖率的视角时才会少于 k，此时 `fallback_incomplete: true` 会明确
标记这一物理/采样约束不足：

- `random.target_count_k`：最终保存数量 k；
- `random.topiq_nr_threshold_l`：TOPIQ-NR 严格下限；
- `random.device`：评分设备，通常为 `cuda`；
- `object.random_candidate_multiplier`：TOPIQ-NR 评分预算相对 k 的倍数。

```bash
conda run -n gs python generate_scene_traversal.py \
  --scene-config configs/travel/scenes/canyon_random.yaml \
  --camera-config configs/travel/cameras/object_random.yaml \
  --color-config configs/travel/colors/default.yaml
```

interior 的 random 行为保持不变：使用
`max_position_sampling_attempts * images_per_position_l` 形成候选，低于阈值的图像
不落盘，达到 k 后结束，最终不足 k 时仍会报错。其示例继续使用
`configs/travel/cameras/random.yaml`。

两种场景都保持分辨率、主点、fx/fy 和 FOV 固定，其中 FOV 使用
`reference_match.seed_pose.fov_y_deg`；object 覆盖拟合只改变相机位置。
`random_seed` 保证机位和姿态可复现，`preview_width/preview_height` 是最终输出
PNG 尺寸。当前相机配置使用 512x512；如需其他分辨率，同时修改两个字段即可。


interior 自动种子有两种等价写法：整个 `seed_pose: null`，或保留 seed_pose 并设置
`position: null`。程序在鲁棒 AABB 内评估候选中心周围满足净空条件的探测点数量，
优先选择有效机位密度高、远离边界且靠近场景中心的中心。选择过程由 `random_seed`
控制并可复现；显式 position 仍会严格执行 AABB 校验，不会被自动纠正。

整个 seed_pose 为 null 时，默认 yaw/pitch/roll 为 0、固定垂直 FOV 为 96°。自动位置
在 GPU 加载前解析，并写入 `resolved_traversal_config.yaml`、`traversal_plan.json`、
`traversal_summary.json` 和每个 `camera.json`，同时记录 position_source。
`--validate-only` 只检查 YAML schema，不扫描 PLY，因此不会提前解析自动位置。

```text
<root_directory>/<scene_name>/
  traversal_plan.json
  traversal_summary.json
  position_0000/
    lens_0000/
      image.png
      camera.json
```

每个 `camera.json` 记录对应 PNG 的 c2w、w2c、位置、视线、yaw/pitch/roll、FOV、
内参、near/far、场景净空和根变换。默认 `overwrite: false`，防止覆盖已有遍历数据。


每个视角还会在 `camera.json.geometry_quality` 中记录屏幕空间高斯质量指标，并在
`traversal_summary.json.captures[*].geometry_quality` 中汇总：

- `max_projected_area_px2`：单个高斯 3σ 投影椭圆的最大面积；
- `max_projected_major_axis_to_image_diagonal_ratio`：最大 3σ 长轴直径与图像对角线之比；
- `largest_single_gaussian_dominated_pixel_ratio`：被同一个高斯主导的最大像素比例；
- `top5_gaussians_dominated_pixel_ratio`：贡献最大的五个高斯合计主导的像素比例；
- `multi_layer_overlap_pixel_ratio` 和 `max_overlap_layer_count`：用于衡量多层虚影风险。

当前 CUDA rasterizer 不输出 per-pixel Gaussian ID。诊断先用原生 `radii` 选取屏幕
半径最大的 4096 个可见候选，再从中选择 `opacity × projected_area` 最大的 64 个
进行 2D 协方差 ownership 栅格化。JSON 中会写入 `is_approximate`、候选上限、
选择方法和是否发生候选截断，便于后续质量评价区分原生数据与诊断近似。
