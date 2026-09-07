#!/usr/bin/env bash

exec "${PYTHON:-python}" generate_scene_traversal.py \
  --scene-config configs/travel/scenes/kelpies.yaml \
  --camera-config configs/travel/cameras/object_random.yaml \
  --color-config configs/travel/colors/default.yaml \
  "$@"
