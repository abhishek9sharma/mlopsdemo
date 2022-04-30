#FROM python:3.8
FROM tensorflow/tensorflow:latest-gpu-jupyter
EXPOSE 9001
EXPOSE 8888
EXPOSE 5000
RUN mkdir recsysapp
COPY ./ /recsysapp 
WORKDIR /recsysapp
RUN pip install --upgrade pip
#RUN cd mlcore && pip install -e .
RUN cd mlcore && pip install .
RUN cd .. 
RUN ls -lah
RUN chmod +x start.sh
RUN cd tests && pip install pytest && pytest && cd ..
CMD ["./start.sh"]
