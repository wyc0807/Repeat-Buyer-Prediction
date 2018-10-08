create table train_ui_up_mp_um_profile as
(
    select
    tu.*, 
    um.count_click as um_count_click,
    um.count_cart as um_count_cart,
    um.count_purchase as um_count_purchase,
    um.count_favorite as um_count_favorite,
    
    um.count_click_may as um_count_click_may,
    um.count_click_jun as um_count_click_jun,
    um.count_click_jul as um_count_click_jul,
    um.count_click_aug as um_count_click_aug,
    um.count_click_sep as um_count_click_sep,
    um.count_click_oct as um_count_click_oct,
    um.count_click_nov as um_count_click_nov,
    
    um.count_cart_may as um_count_cart_may,
    um.count_cart_jun as um_count_cart_jun,
    um.count_cart_jul as um_count_cart_jul,
    um.count_cart_aug as um_count_cart_aug,
    um.count_cart_sep as um_count_cart_sep,
    um.count_cart_oct as um_count_cart_oct,
    um.count_cart_nov as um_count_cart_nov,
    
    um.count_purchase_may as um_count_purchase_may,
    um.count_purchase_jun as um_count_purchase_jun,
    um.count_purchase_jul as um_count_purchase_jul,
    um.count_purchase_aug as um_count_purchase_aug,
    um.count_purchase_sep as um_count_purchase_sep,
    um.count_purchase_oct as um_count_purchase_oct,
    um.count_purchase_nov as um_count_purchase_nov,
    
    um.count_favorite_may as um_count_favorite_may,
    um.count_favorite_jun as um_count_favorite_jun,
    um.count_favorite_jul as um_count_favorite_jul,
    um.count_favorite_aug as um_count_favorite_aug,
    um.count_favorite_sep as um_count_favorite_sep,
    um.count_favorite_oct as um_count_favorite_oct,
    um.count_favorite_nov as um_count_favorite_nov
    
    from train_ui_up_mp_profile as tu
    left join user_merchant_monthly_count as um
    on tu.user_id = um.user_id and tu.merchant_id = um.merchant_id
)