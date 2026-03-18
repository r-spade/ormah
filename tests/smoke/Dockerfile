FROM ubuntu:22.04

# Minimal deps — just what a real user would have
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates bash git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (simulates real install)
RUN useradd -m -s /bin/bash testuser
USER testuser
WORKDIR /home/testuser

# Copy local source into the image
COPY --chown=testuser:testuser . /src/ormah

# Install ormah from local source using install.sh
# ORMAH_INSTALL_SOURCE overrides the default PyPI source
ENV ORMAH_INSTALL_SOURCE="/src/ormah[litellm]"
RUN bash /src/ormah/install.sh --no-setup

# Copy and set up the smoke test script
COPY --chown=testuser:testuser tests/smoke/run.sh /home/testuser/run.sh
RUN chmod +x /home/testuser/run.sh

CMD ["/home/testuser/run.sh"]
