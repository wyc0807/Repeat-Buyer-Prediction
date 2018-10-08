CREATE TABLE user_profile_monthly_count AS
(
    SELECT 
    user_id,
    
    SUM(CASE WHEN action_type = 0 THEN 1 ELSE 0 END) AS count_click,
    SUM(CASE WHEN action_type = 1 THEN 1 ELSE 0 END) AS count_cart,
    SUM(CASE WHEN action_type = 2 THEN 1 ELSE 0 END) AS count_purchase,
    SUM(CASE WHEN action_type = 3 THEN 1 ELSE 0 END) AS count_favorite,
    
    SUM(CASE WHEN (action_type = 0) AND (time_month = '05') THEN 1 ELSE 0 END) AS count_click_may,
    SUM(CASE WHEN (action_type = 0) AND (time_month = '06') THEN 1 ELSE 0 END) AS count_click_jun,
    SUM(CASE WHEN (action_type = 0) AND (time_month = '07') THEN 1 ELSE 0 END) AS count_click_jul,
    SUM(CASE WHEN (action_type = 0) AND (time_month = '08') THEN 1 ELSE 0 END) AS count_click_aug,
    SUM(CASE WHEN (action_type = 0) AND (time_month = '09') THEN 1 ELSE 0 END) AS count_click_sep,
    SUM(CASE WHEN (action_type = 0) AND (time_month = '10') THEN 1 ELSE 0 END) AS count_click_oct,
    SUM(CASE WHEN (action_type = 0) AND (time_month = '11') THEN 1 ELSE 0 END) AS count_click_nov,
    
    SUM(CASE WHEN (action_type = 1) AND (time_month = '05') THEN 1 ELSE 0 END) AS count_cart_may,
    SUM(CASE WHEN (action_type = 1) AND (time_month = '06') THEN 1 ELSE 0 END) AS count_cart_jun,
    SUM(CASE WHEN (action_type = 1) AND (time_month = '07') THEN 1 ELSE 0 END) AS count_cart_jul,
    SUM(CASE WHEN (action_type = 1) AND (time_month = '08') THEN 1 ELSE 0 END) AS count_cart_aug,
    SUM(CASE WHEN (action_type = 1) AND (time_month = '09') THEN 1 ELSE 0 END) AS count_cart_sep,
    SUM(CASE WHEN (action_type = 1) AND (time_month = '10') THEN 1 ELSE 0 END) AS count_cart_oct,
    SUM(CASE WHEN (action_type = 1) AND (time_month = '11') THEN 1 ELSE 0 END) AS count_cart_nov,
    
    SUM(CASE WHEN (action_type = 2) AND (time_month = '05') THEN 1 ELSE 0 END) AS count_purchase_may,
    SUM(CASE WHEN (action_type = 2) AND (time_month = '06') THEN 1 ELSE 0 END) AS count_purchase_jun,
    SUM(CASE WHEN (action_type = 2) AND (time_month = '07') THEN 1 ELSE 0 END) AS count_purchase_jul,
    SUM(CASE WHEN (action_type = 2) AND (time_month = '08') THEN 1 ELSE 0 END) AS count_purchase_aug,
    SUM(CASE WHEN (action_type = 2) AND (time_month = '09') THEN 1 ELSE 0 END) AS count_purchase_sep,
    SUM(CASE WHEN (action_type = 2) AND (time_month = '10') THEN 1 ELSE 0 END) AS count_purchase_oct,
    SUM(CASE WHEN (action_type = 2) AND (time_month = '11') THEN 1 ELSE 0 END) AS count_purchase_nov,
    
    SUM(CASE WHEN (action_type = 3) AND (time_month = '05') THEN 1 ELSE 0 END) AS count_favorite_may,
    SUM(CASE WHEN (action_type = 3) AND (time_month = '06') THEN 1 ELSE 0 END) AS count_favorite_jun,
    SUM(CASE WHEN (action_type = 3) AND (time_month = '07') THEN 1 ELSE 0 END) AS count_favorite_jul,
    SUM(CASE WHEN (action_type = 3) AND (time_month = '08') THEN 1 ELSE 0 END) AS count_favorite_aug,
    SUM(CASE WHEN (action_type = 3) AND (time_month = '09') THEN 1 ELSE 0 END) AS count_favorite_sep,
    SUM(CASE WHEN (action_type = 3) AND (time_month = '10') THEN 1 ELSE 0 END) AS count_favorite_oct,
    SUM(CASE WHEN (action_type = 3) AND (time_month = '11') THEN 1 ELSE 0 END) AS count_favorite_nov
    
    FROM user_log
    GROUP BY user_id       
)