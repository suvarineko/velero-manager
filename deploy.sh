docker build --platform linux/amd64 \
             --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
             --build-arg BUILD_VERSION=14.6-fixed \
             --build-arg VCS_REF=$(git rev-parse --short HEAD) \
             -t registry.apps.k8s.ose-prod.solution.sbt/r4c-development/velero-manager:v1.0.0-$(git rev-parse --short HEAD) \
             -f Dockerfile .

skopeo --override-os linux \
       --override-arch amd64 \
       copy --preserve-digests \
            --tls-verify=false \
            docker-daemon:registry.apps.k8s.ose-prod.solution.sbt/r4c-development/velero-manager:v1.0.0-$(git rev-parse --short HEAD) \
            docker://registry.apps.k8s.ose-prod.solution.sbt/r4c-development/velero-manager:v1.0.0-$(git rev-parse --short HEAD)

helm upgrade --install \
             velero-manager \
             helm/velero-manager \
             --create-namespace \
             -n velero-manager \
             -f helm/velero-manager/values.yaml \
             --set image.tag=v1.0.0-$(git rev-parse --short HEAD)
