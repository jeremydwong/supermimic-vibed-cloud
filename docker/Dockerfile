# SuperMimic runtime: Isaac Lab 3 (main) kit-less image + tmux/ffmpeg/compiler.
# Build context is the repository root: docker build -f docker/Dockerfile .
FROM isaac-lab-kitless:main-46d6cc3
USER root
RUN apt-get update && apt-get install -y --no-install-recommends tmux ffmpeg util-linux build-essential python3-dev openssh-server && rm -rf /var/lib/apt/lists/*
# runpodctl lets a RunPod pod stop itself when the time budget runs out (docker/start.sh).
RUN curl -sfL -o /usr/local/bin/runpodctl https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 && chmod +x /usr/local/bin/runpodctl
# Isaac Lab ships as editable installs under /workspace/isaaclab, which a RunPod volume mounted at
# /workspace would hide; move it to /opt and rewrite every path that points at the old location.
RUN mv /workspace/isaaclab /opt/isaaclab \
 && grep -rlI "/workspace/isaaclab" /opt/isaaclab-venv /opt/isaaclab | xargs -r sed -i "s#/workspace/isaaclab#/opt/isaaclab#g" \
 && python -c "import isaaclab, isaaclab_newton"
ENV ISAACLAB_PATH=/opt/isaaclab
# Register the NVIDIA EGL vendor (the driver library is injected by the NVIDIA container runtime);
# without it GLVND only knows Mesa and MuJoCo's offscreen renderer falls back to CPU rasterization.
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
# Code lives outside /workspace, which RunPod mounts its volume over. data/, logs/ and .cache/
# are symlinked to the volume by docker/start.sh; docker/run_container.sh still mounts the repo
# at /workspace/supermimic for local use.
COPY supermimic/ /opt/supermimic/supermimic/
COPY tools/ /opt/supermimic/tools/
COPY docker/ /opt/supermimic/docker/
COPY run_train.sh /opt/supermimic/
# Kernel caches stay on the container disk: on RunPod's network volume Triton fails with "Stale file handle".
ENV HOME=/root MPLCONFIGDIR=/tmp/mpl \
    WARP_CACHE_PATH=/root/.cache/warp TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor
WORKDIR /opt/supermimic
CMD ["bash", "/opt/supermimic/docker/start.sh"]
