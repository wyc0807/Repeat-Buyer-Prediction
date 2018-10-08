create table train_ui_up_mp_um_double_pca_sim as
(
    select
    tu.*, 
    double11_click,
    double11_click_ratio,
    double11_purchase,
    double11_purchase_ratio,
    double11_favorite,
    double11_favorite_ratio,
    merchant_repeat_buyers_num,
    merchant_repeat_buyers_ratio,
    pca1,
    pca2,
    pca3,
    pca4,
    pca5,
    pca6,
    pca7,
    pca8,
    pca9,
    pca10,
    sim
    from train_ui_up_mp_um_profile as tu
    left join complex_features as cf
    on tu.user_id = cf.user_id and tu.merchant_id = cf.merchant_id
)