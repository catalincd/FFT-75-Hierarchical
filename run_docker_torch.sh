docker run -it \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --ipc=host \
    --shm-size 16G \
    --group-add video \
    --group-add render \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v .:/workspace \
    -w /workspace \
    -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    rocm/pytorch:latest \
    /bin/bash