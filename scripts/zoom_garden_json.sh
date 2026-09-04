CUDA_VISIBLE_DEVICES=0 python generate_zoom_video.py \
  --scene-config configs/zoom_video/scenes/garden.yaml \
  --camera-config configs/zoom_video/cameras/garden_reference_static.yaml \
  --color-config configs/zoom_video/colors/multicamera_low.yaml \
  --camera-json /home/wdj/projects/data-generate/outputs/garden_traversal/garden/position_0000/lens_0001/camera.json