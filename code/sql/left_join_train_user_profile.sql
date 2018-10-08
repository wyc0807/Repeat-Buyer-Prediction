create table train_user_info_user_profile as
(
    select
    tu.*, 
    up.count_click as up_count_click,
    up.count_cart as up_count_cart,
    up.count_purchase as up_count_purchase,
    up.count_favorite as up_count_favorite,
    
    up.count_click_may as up_count_click_may,
    up.count_click_jun as up_count_click_jun,
    up.count_click_jul as up_count_click_jul,
    up.count_click_aug as up_count_click_aug,
    up.count_click_sep as up_count_click_sep,
    up.count_click_oct as up_count_click_oct,
    up.count_click_nov as up_count_click_nov,
    
    up.count_cart_may as up_count_cart_may,
    up.count_cart_jun as up_count_cart_jun,
    up.count_cart_jul as up_count_cart_jul,
    up.count_cart_aug as up_count_cart_aug,
    up.count_cart_sep as up_count_cart_sep,
    up.count_cart_oct as up_count_cart_oct,
    up.count_cart_nov as up_count_cart_nov,
    
    up.count_purchase_may as up_count_purchase_may,
    up.count_purchase_jun as up_count_purchase_jun,
    up.count_purchase_jul as up_count_purchase_jul,
    up.count_purchase_aug as up_count_purchase_aug,
    up.count_purchase_sep as up_count_purchase_sep,
    up.count_purchase_oct as up_count_purchase_oct,
    up.count_purchase_nov as up_count_purchase_nov,
    
    up.count_favorite_may as up_count_favorite_may,
    up.count_favorite_jun as up_count_favorite_jun,
    up.count_favorite_jul as up_count_favorite_jul,
    up.count_favorite_aug as up_count_favorite_aug,
    up.count_favorite_sep as up_count_favorite_sep,
    up.count_favorite_oct as up_count_favorite_oct,
    up.count_favorite_nov as up_count_favorite_nov
    
    from train_user_info as tu
    left join user_profile_monthly_count as up
    on tu.user_id = up.user_id
)