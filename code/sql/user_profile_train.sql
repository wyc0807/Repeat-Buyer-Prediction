create table user_profile_train as
(
    select up.*
    from train_data as tr
    left join user_profile_monthly_count as up
    on tr.user_id = up.user_id
)