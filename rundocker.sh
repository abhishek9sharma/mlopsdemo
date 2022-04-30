docker stop $(docker ps|grep mlapp| awk '{print $1}')
docker container rm $(docker container ls -a|grep mlapp| awk '{print $1}')
docker image rm $(docker images|grep mlapp| awk '{print $3}')
docker build --tag mlapp .
docker run  --env-file .envdocker \
            -p 9001:8888 \
            -p 5000:5000 \
            -v $PWD/notebooks:/mlapp/notebooks \
            -v $PWD/mlcore:/mlapp/mlcore \
            -v $PWD/models:/mlapp/models \
            -v $PWD/data:/mlapp/data \
            -v $PWD/logs:/mlapp/logs \
            mlapp
