# Private data

Place the private cohort HDF5 tracing stores and train/test label parquet files
here. Tracing stores contain `tracings` and `index`; labels are kept in the
train/test parquet sidecars. A `PID,fold` split CSV is also required.

The public CTU-UHB cohort can be downloaded and converted with
`preprocessing/CTU/get_data.sh` followed by
`python preprocessing/CTU/preprocess_ctu.py`.
