# Private data

Place the private cohort artifacts required for training here. The training
scripts expect an HDF5 tracing store, train/test label parquet files, and a
`PID,fold` split CSV. These files are intentionally excluded from version
control.

The public CTU-UHB cohort can be downloaded and converted with
`preprocessing/CTU/get_data.sh` followed by
`python preprocessing/CTU/preprocess_ctu.py`.
