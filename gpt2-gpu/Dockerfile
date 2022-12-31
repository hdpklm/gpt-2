# FROM tensorflow/tensorflow:1.13.2-py3

# # nvidia-docker 1.0
# LABEL com.nvidia.volumes.needed="nvidia_driver"
# LABEL com.nvidia.cuda.version="${CUDA_VERSION}"

# # nvidia-container-runtime
# ENV NVIDIA_VISIBLE_DEVICES=all \
#     NVIDIA_DRIVER_CAPABILITIES=compute,utility \
#     NVIDIA_REQUIRE_CUDA="cuda>=8.0" \
#     LANG=C.UTF-8

# RUN mkdir /gpt-2
# WORKDIR /gpt-2
# ADD requirements.txt /gpt-2
# ADD download_model.py /gpt-2

# RUN pip3 install -r requirements.txt
# RUN python3 download_model.py 124M
# RUN python3 download_model.py 355M
# RUN python3 download_model.py 774M
# RUN python3 download_model.py 1558M

# ADD src /gpt-2/src
# RUN apt-get install nano
# RUN python3 src/app.py

# expose 5000