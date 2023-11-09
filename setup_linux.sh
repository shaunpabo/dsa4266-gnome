sudo install python-pip
pip3 install pandas, scikit-learn, category-encoders, xgboost
mkdir m6anet
cd m6anet
aws s3 sync --no-sign-request s3://sg-nex-data/data/processed_data/m6Anet/ .