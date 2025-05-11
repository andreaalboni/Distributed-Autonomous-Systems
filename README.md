# ROS2 Humble Docker Compose Setup

## Prerequisites

- Docker
- Docker Compose

## Setup Instructions

### 1. Set User Permissions (Optional but Recommended)

Before starting, you can set your host user ID and group ID to match the container:

```bash
export HOST_USER_ID=$(id -u)
export HOST_GROUP_ID=$(id -g)
```

### 2. Build the Image

```bash
docker-compose build
```

### 3. Start the Container

```bash
docker-compose up
```

### 4. Attach to a running container

```bash
docker exec -it das_ros2 /bin/bash
```

## to run GUI app in the docker container on wayland run at each boot of you PC

```bash
xhost +local:docker
```
