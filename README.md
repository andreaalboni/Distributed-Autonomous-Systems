# Project istructions

To satisfy all python dependencies please follow the steps below:

1. Create virtual environment: `python -m venv venv`
2. activate it (on bash/Linux): `source venv/bin/activate`
3. Install packages: `pip install -r requirements.txt`

## Task 1

To execute the task 1.1 and 1.2 run:

```bash
python exam_project/task1/main.py
```

## Task 2.1

To execute the task 2.1 run:

```bash
python exam_project/task21/main.py 
```

## Task 2.2


### 1. Open and login to foxglove

With a chromium based browser visit [foxglove.dev](https://app.foxglove.dev/) or download from [here](https://foxglove.dev/download) the foxglove app.
Once the login is done, click `open connection...`, and open the foxglove WebSocket URL `ws://localhost:8765`.
In the top right, click on `layout`, then `Import from file...` and select the `DAS-foxglove-layout.json` file present in the root of the exam_project

### 2. Build the Image (Linux only)

In the folder containing the dockerfile run:

```bash
docker compose build
```

In order to run GUI app in the docker container on wayland systems run:

```bash
xhost +local:docker
```

### 3. Start the Container

```bash
docker compose up
```

### 4. Attach to a running container

```bash
docker exec -it das_ros2 /bin/bash
```

### 5. Run the simulation

To run the task 2.2

```bash
ros2 launch task22 launch.py
```

To change parameters change the values of the `PARAMETERS` dictionary in the launch.py file inside the folder `/exam_project/task22_ws/src/task22/launch_folder/` then a *colcon build* is required:

Inside `~/task22_ws`, run:

```bash
colcon build && source install/setup.bash && ros2 launch task22 launch.py
```
