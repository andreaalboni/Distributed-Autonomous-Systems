FROM ros:humble-ros-base

ENV DEV_NAME=user
ENV ROS_DISTRO=humble

##########################################################
# Create group and user
##########################################################
RUN groupadd -g 1000 ${DEV_NAME} && \
     useradd -d /home/${DEV_NAME} -s /bin/bash -m ${DEV_NAME} -u 1000 -g 1000 && \
     usermod -aG sudo ${DEV_NAME} && \
     echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
##########################################################

##########################################################
# Install ROS packages
##########################################################
RUN . /opt/ros/${ROS_DISTRO}/setup.sh
RUN apt-get update -q && \
     apt install -yq \
          ros-${ROS_DISTRO}-rqt-graph \
          ros-${ROS_DISTRO}-foxglove-bridge \
     && rm -rf /var/lib/apt/lists/*

##########################################################
# Install Utils
##########################################################
RUN apt update -q && \
        apt install -yq \
            gedit \
            iputils-ping \
            nano \
            net-tools \
            python3-pip \
            vim \
            xterm \
            python3-setuptools \
            python3-networkx \
            python3-numpy \
            python3-scipy \
            python3-matplotlib

# Install requirements
# COPY requirements.txt /home/${DEV_NAME}/
# RUN pip3 install -r /home/${DEV_NAME}/requirements.txt 

##########################################################
# Useful aliases
##########################################################
RUN echo 'PROMPT_DIRTRIM=1' >> /home/${DEV_NAME}/.bashrc
RUN echo 'export ROS_DOMAIN_ID=100' >> /home/${DEV_NAME}/.bashrc
##########################################################

WORKDIR /home/${DEV_NAME}
USER ${DEV_NAME}
