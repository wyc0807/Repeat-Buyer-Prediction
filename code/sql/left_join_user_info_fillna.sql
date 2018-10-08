create table train_ui_up_mp_um_double_pca_sim_slope_fillna as
(
    select
    tu.*, 
    age_range,
    gender
    from train_ui_up_mp_um_double_pca_sim_slope as tu
    left join user_info_train_fillna as ui
    on tu.user_id = ui.user_id
)