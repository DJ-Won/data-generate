CUDA_VISIBLE_DEVICES=0 python generate_zoom_video.py \
  --scene-config configs/zoom_video/scenes/apartment.yaml \
  --camera-config configs/zoom_video/cameras/x3.yaml \
  --color-config configs/zoom_video/colors/multicamera_low.yaml \
  --camera-json /home/wdj/projects/data-generate/outputs/apartment_random/apartment/position_0142/lens_0005/camera.json