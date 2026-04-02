# Docker

Container image and Docker Compose configuration for the photic forecast-gui
web application.

## Directory Contents

- `docker/Dockerfile` - Multi-stage build for the forecast-gui container image
- `docker/docker-compose.yaml` - Local development configuration
- `.dockerignore` - Files excluded from the Docker build context (lives at the
  repository root because Docker resolves it relative to the build context)

## Build the Container Image

All commands below assume the working directory is the repository root.

```bash
docker compose -f docker/docker-compose.yaml build
```

Or build directly with Docker:

```bash
docker build -f docker/Dockerfile -t photic:latest .
```

## Run Locally with Docker Compose

1. Create the forecast data directory:

```bash
mkdir -p data/forecasts
```

2. Start the forecast-gui service:

```bash
docker compose -f docker/docker-compose.yaml up
```

The application will be available at http://localhost:8080.

To run in the background:

```bash
docker compose -f docker/docker-compose.yaml up -d
```

To stop:

```bash
docker compose -f docker/docker-compose.yaml down
```

## Tag and Push to GitLab Container Registry

1. Log in to the GitLab Container Registry:

```bash
docker login registry.gitlab.com
```

2. Tag the image:

```bash
docker tag photic:latest registry.gitlab.com/ncsa-caps-rse/photic-k8s-gitops:dev
```

3. Push to the registry:

```bash
docker push registry.gitlab.com/ncsa-caps-rse/photic-k8s-gitops:dev
```

To push a specific version tag:

```bash
docker tag photic:latest registry.gitlab.com/ncsa-caps-rse/photic-k8s-gitops:v0.2.0
docker push registry.gitlab.com/ncsa-caps-rse/photic-k8s-gitops:v0.2.0
```
