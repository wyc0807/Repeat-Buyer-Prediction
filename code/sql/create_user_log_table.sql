create table user_log
(
    user_id text DEFAULT NULL,
    item_id text DEFAULT NULL,
    cat_id text DEFAULT NULL,
    merchant_id text DEFAULT NULL,
    brand_id text DEFAULT NULL,
    time_stamp text DEFAULT NULL,
    action_type int DEFAULT NULL
);
COPY user_log from 'C:\Users\yawen\Google Drive\spatial temporal data analytics\urban computing\PurchasingBehavior\data\user_log_format1.csv' CSV HEADER;