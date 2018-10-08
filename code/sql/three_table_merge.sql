create table train_data_expand as
(
    select t1.user_id, t2.age_range, t2.gender, t3.merchant_id, t3.item_id, t3.cat_id, t3.brand_id, t3.time_stamp, t3.action_type, t1.label
    from train_data t1 left join user_info t2 on t1.user_id = t2.user_id
    left join user_log t3 on (t1.user_id = t3.user_id and t1.merchant_id = t3.merchant_id)
)