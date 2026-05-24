FROM nvcr.io/nvidia/isaac-lab:2.3.2

# Make python/python3 available everywhere
RUN printf '#!/bin/bash\nexec /isaac-sim/python.sh "$@"\n' > /usr/local/bin/python && \
    chmod +x /usr/local/bin/python && \
    ln -sf /usr/local/bin/python /usr/local/bin/python3

# Allow git to operate on the mounted workspace (owned by host user)
RUN git config --global --add safe.directory /workspace/cf_lab && \
    git config --global --add safe.directory /workspace/isaaclab

RUN curl -fsSL https://claude.ai/install.sh | bash

# Enable bash completion for cf_lab training/play scripts.
RUN echo '[ -f /workspace/cf_lab/scripts/completion.bash ] && source /workspace/cf_lab/scripts/completion.bash' \
    >> /root/.bashrc
