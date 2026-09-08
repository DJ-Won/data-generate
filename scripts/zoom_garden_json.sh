CUDA_VISIBLE_DEVICES=0 python generate_zoom_video.py \
  --scene-config configs/zoom_video/scenes/dl3dv_001dccbc.yaml \
  --camera-config configs/zoom_video/cameras/x3.yaml \
  --color-config configs/zoom_video/colors/multicamera_low.yaml \
  --camera-json /data0/wdj/zooming/data-generate/outputs/dl3dv_camera_renders/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f/position_0000/lens_0000/camera.json