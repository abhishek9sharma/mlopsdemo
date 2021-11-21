docker stop $(docker ps|grep recsysapp| awk '{print $1}')
docker container rm $(docker container ls -a|grep recsysapp| awk '{print $1}')
docker image rm $(docker images|grep recsysapp| awk '{print $3}')
docker build --tag recsysapp .
docker run  --env-file .envdocker \
            -p 9001:8888 \
            -p 5000:5000 \
            -v $PWD/notebooks:/recsysapp/notebooks \
            -v $PWD/mlcore:/recsysapp/mlcore \
            -v $PWD/models:/recsysapp/models \
            -v $PWD/data:/recsysapp/data \
            -v $PWD/logs:/recsysapp/logs \
            recsysapp
