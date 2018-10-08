create table train_ui_up_mp_um_double_pca_sim_slope as
(
    select
    tu.*, 
    up_click_slope,
    up_cart_slope,
    up_purchase_slope,
    up_favorite_slope,
    mp_click_slope,
    mp_cart_slope,
    mp_purchase_slope,
    mp_favorite_slope,
    um_click_slope,
    um_cart_slope,
    um_purchase_slope,
    um_favorite_slope
    
    from train_ui_up_mp_um_double_pca_sim as tu
    left join slope_features as sf
    on tu.user_id = sf.user_id and tu.merchant_id = sf.merchant_id
)


