# recsysdemo

A mini ML system which can train some prediction models for recommendation on a given user+item click interaction data set and autoamtically deploy models to be served by a simple python web API end point

### Steps to run the notebooks/code Used. Run in root mode on Unix Based System

- ~/recsysdemo$ chmod a+x rundocker.sh  && ./rundocker.sh 


- Open [http://localhost:9001/tree/notebooks](http://localhost:9001/tree/notebooks) in your browser to examine the notebooks  
- Open [http://localhost:5000/docs](http://localhost:5000/docs) in your browser to examine the prediction api. 
- If you system has CUDA + GPU configured properly you can try below command to startup

- ~/recsysdemo$ chmod a+x rundocker.sh  && ./rundocker_gpu.sh 

### Test API Request

        curl --location --request POST 'http://localhost:5000/predict' \
        --header 'Content-Type: application/json' \
        --data-raw '{
        "req_id":"1242765",
        "id":732810,
        "user_id":28349,
        "store_id":366,
        "device":"app_ios",
        "platform":"iOS App",
        "channel":"Direct",
        "created_at":"2021-02-03 23:47:27",
        "num_of_items_req":5
        }'