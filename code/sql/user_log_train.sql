create table user_log_train as
(
    select tr.user_id, lg.item_id, lg.cat_id, tr.merchant_id, lg.brand_id, lg.time_stamp, lg.action_type, lg.time_month 
    from train_data as tr
    left join user_log as lg
    on tr.user_id = lg.user_id and tr.merchant_id = lg.merchant_id
)