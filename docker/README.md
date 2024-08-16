# Build Docker image
```
docker build -t semantic_segmentation_ros_img .
```

# Run Docker container
```
docker run -it \
    --gpus all \
    --name semantic_segmentation_ros_cont \
    --shm-size=18g \
    -p 6006:6006 \
    semantic_segmentation_ros_img /bin/bash
```

# Reconnect to the container
```
docker start semantic_segmentation_ros_cont
docker attach semantic_segmentation_ros_cont
```

# Enter the container from another terminal
```
docker exec -it semantic_segmentation_ros_cont /bin/bash
```

### Tensorboard
```
tensorboard --logdir=log/ --host=0.0.0.0
```