CUDA_VISIBLE_DEVICES=0 python generate_zoom_video.py \
  --scene-config configs/zoom_video/scenes/kelpies.yaml \
  --camera-config configs/zoom_video/cameras/x3.yaml \
  --color-config configs/zoom_video/colors/multicamera_default.yaml \
  --camera-json /home/wdj/projects/data-generate/outputs/kelpies_traversal/kelpies/position_0067/lens_0000/camera.json