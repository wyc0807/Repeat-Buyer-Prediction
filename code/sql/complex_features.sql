create table complex_features
(
    user_id text DEFAULT NULL,
    merchant_id text DEFAULT NULL,
    double11_click float DEFAULT NULL,
    double11_click_ratio float DEFAULT NULL,
    double11_purchase float DEFAULT NULL,
    double11_purchase_ratio float DEFAULT NULL,
    double11_favorite float DEFAULT NULL,
    double11_favorite_ratio float DEFAULT NULL,
    merchant_repeat_buyers_num float DEFAULT NULL,
    merchant_repeat_buyers_ratio float DEFAULT NULL,
    pca1 float DEFAULT NULL,
    pca2 float DEFAULT NULL,
    pca3 float DEFAULT NULL,
    pca4 float DEFAULT NULL,
    pca5 float DEFAULT NULL,
    pca6 float DEFAULT NULL,
    pca7 float DEFAULT NULL,
    pca8 float DEFAULT NULL,
    pca9 float DEFAULT NULL,
    pca10 float DEFAULT NULL,
    sim float DEFAULT NULL
);
COPY complex_features from 'C:\Users\yawen\Google Drive\DM project\features\complex_double11_fea.csv' CSV HEADER;