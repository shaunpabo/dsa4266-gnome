sudo apt install python3-pip -y
pip3 install -r requirements.txt
mkdir m6anet
cd m6anet
aws s3 sync --no-sign-request s3://sg-nex-data/data/processed_data/m6Anet/ .