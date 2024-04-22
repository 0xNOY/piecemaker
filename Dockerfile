FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND noninteractive

# Entrypoint
RUN mkdir /entrypoint.d
RUN { \
    echo "#!/bin/bash"; \
    echo "for f in /entrypoint.d/*.sh; do"; \
    echo "    [ ! -d \$f ] && source \$f"; \
    echo "done"; \
    echo "exec \"\$@\""; \
    } > /entrypoint.sh && chmod +x /entrypoint.sh
RUN touch /entrypoint.d/empty.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]

# Install basic packages
RUN apt update && apt install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt clean && rm -rf /var/lib/apt/lists/*

# PyTorch
RUN pip3 install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 xformers==0.0.25 --index-url https://download.pytorch.org/whl/cu121

COPY ./requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

COPY . /app
WORKDIR /app

RUN git submodule update --init --recursive

CMD ["python3", "main.py", "gradio"]

# Clean
RUN apt autoremove -y && apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
ENV DEBIAN_FRONTEND newt