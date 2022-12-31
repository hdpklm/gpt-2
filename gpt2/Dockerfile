FROM tensorflow/tensorflow:1.13.2-py3
# RUN apt-get install nano -y
# RUN apt-get install curl -y

ENV LANG=C.UTF-8

RUN mkdir /gpt-2
WORKDIR /gpt-2


ADD requirements.txt /gpt-2
ADD download_model.py /gpt-2

RUN pip3 install -r requirements.txt
# RUN python3 download_model.py 124M
RUN python3 download_model.py 355M
# RUN python3 download_model.py 774M
# RUN python3 download_model.py 1558M

ADD ./src /gpt-2/src

EXPOSE 5000

ENTRYPOINT ["python3", "src/app.py"]



