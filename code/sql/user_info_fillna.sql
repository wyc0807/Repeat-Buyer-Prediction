create table user_info_train_fillna
(
    row_id text DEFAULT NULL,
    user_id text DEFAULT NULL,
    age_range float DEFAULT NULL,
    gender float DEFAULT NULL
);
COPY user_info_train_fillna from 'C:\Users\yawen\Google Drive\DM project\features\user_age_gender_fillna_1212.csv' CSV HEADER;
