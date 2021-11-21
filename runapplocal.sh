rm -rf recsysappvenv
python3 -m pip install --user virtualenv
python3 -m virtualenv recsysappvenv
source recsysappvenv/bin/activate
cd mlcore && pip install -e .
cd ..
./start.sh