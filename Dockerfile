# Multi-stage build for production optimization

# ============================================================================
# Stage 1: Base Dependencies
# ============================================================================
FROM python:3.11-slim AS base

# Set build-time arguments
ARG VELERO_VERSION=v1.14.1
ARG BUILD_ENV=production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VELERO_VERSION=${VELERO_VERSION} \
    BUILD_ENV=${BUILD_ENV}

# Set locale and timezone
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=UTC

# Install essential system packages
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ============================================================================
# Stage 2: Python Dependencies Builder
# ============================================================================
FROM base AS deps-builder

# Create temporary directory for dependencies
WORKDIR /deps

# Copy requirements files
COPY requirements-prod.txt requirements-dev.txt ./

# Install dependencies to user directory for easy copying
RUN pip install --user --no-cache-dir \
    -r $([ "${BUILD_ENV}" = "development" ] && echo "requirements-dev.txt" || echo "requirements-prod.txt")

# ============================================================================
# Stage 3: Velero Binary Builder
# ============================================================================
FROM base AS velero-builder

# Install Velero CLI binary
RUN set -eu; \
    echo "Installing Velero ${VELERO_VERSION} for linux-amd64..."; \
    \
    # Download and install Velero binary
    curl -sL "https://github.com/vmware-tanzu/velero/releases/download/${VELERO_VERSION}/velero-${VELERO_VERSION}-linux-amd64.tar.gz" | \
    tar -xzC /tmp; \
    \
    # Move binary to staging location
    mv "/tmp/velero-${VELERO_VERSION}-linux-amd64/velero" /tmp/velero-binary; \
    chmod +x /tmp/velero-binary; \
    \
    # Verify installation
    /tmp/velero-binary version --client-only; \
    \
    echo "Velero ${VELERO_VERSION} installation completed successfully"

# ============================================================================
# Stage 4: Application Builder
# ============================================================================
FROM base AS app-builder

# Set working directory
WORKDIR /app-build

# Copy source code for validation
COPY src/ ./src/

# Validate Python syntax only (dependencies not available in this stage)
RUN python -m py_compile src/main.py && \
    find src/ -name "*.py" -exec python -m py_compile {} \; && \
    echo "Application syntax validation successful"

# ============================================================================
# Stage 5: Production Runtime
# ============================================================================
FROM base AS production

# Application-specific environment variables
ENV VELERO_NAMESPACE=velero \
    VELERO_DEFAULT_BACKUP_TTL=720h \
    BACKUP_STORAGE_LOCATION=default \
    LOG_LEVEL=INFO \
    DEV_MODE=false \
    PYTHONPATH=/app/src

# Create application directory structure
WORKDIR /app
RUN mkdir -p /app/logs /app/config /app/data

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Copy Python dependencies from builder stage
COPY --from=deps-builder /root/.local /home/appuser/.local

# Copy Velero binary from builder stage
COPY --from=velero-builder /tmp/velero-binary /usr/local/bin/velero

# Copy validated application source code
COPY --from=app-builder /app-build/src/ ./src/

# Set proper ownership and permissions
RUN chown -R appuser:appuser /app /home/appuser/.local && \
    chmod +x /usr/local/bin/velero

# Update PATH to include user-installed packages
ENV PATH=/home/appuser/.local/bin:$PATH

# Switch to non-root user
USER appuser

# Verify installations
RUN python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')" && \
    python -c "import kubernetes; print(f'Kubernetes: {kubernetes.__version__}')" && \
    velero version --client-only

# Expose Streamlit default port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command to run the application
CMD ["streamlit", "run", "src/main.py", "--server.port=8501", "--server.address=0.0.0.0"]