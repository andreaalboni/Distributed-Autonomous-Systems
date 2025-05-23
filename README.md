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
docker compose build
```

### 3. Start the Container

```bash
docker compose up
```

### 4. Attach to a running container

```bash
docker exec -it das_ros2 /bin/bash
```

in order to run GUI app in the docker container on wayland systems 
run at each boot outside of the docker container:

```bash
xhost +local:docker
```

### Latex
```bash
sudo apt-get install texlive-latex-base
sudo apt-get install latexmk
sudo apt-get install texlive-latex-recommended
sudo apt-get install texlive-full
```