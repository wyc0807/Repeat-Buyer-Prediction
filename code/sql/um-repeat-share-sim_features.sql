CREATE TABLE train_data_with_user_info as
(
select train_data.user_id,user_info_filled.age_range,user_info_filled.gender, train_data.merchant_id,train_data.prob from train_data inner join user_info_filled on train_data.user_id=user_info_filled.user_id
)

create table user_info_log as
(
    select * from user_log inner join user_info on user_log.user_id=user_info.user_id
)



create table um_click as
(
select user_id, merchant_id,
sum(case when action_type between 0 and 1 and time_stamp between 500 and 535 then 1.0 else 0.0 end) as May_count,
sum(case when action_type between 0 and 3 and time_stamp between 500 and 535 then 1.0 else 0.0 end) as May_action_sum,
sum(case when action_type between 0 and 1 and time_stamp between 600 and 635 then 1.0 else 0.0 end) as June_count,
sum(case when action_type between 0 and 3 and time_stamp between 600 and 635 then 1.0 else 0.0 end) as June_action_sum,
sum(case when action_type between 0 and 1 and time_stamp between 700 and 735 then 1.0 else 0.0 end) as July_count,
sum(case when action_type between 0 and 3 and time_stamp between 700 and 735 then 1.0 else 0.0 end) as July_action_sum,
sum(case when action_type between 0 and 1 and time_stamp between 800 and 835 then 1.0 else 0.0 end) as Aug_count,
sum(case when action_type between 0 and 3 and time_stamp between 800 and 835 then 1.0 else 0.0 end) as Aug_action_sum,
sum(case when action_type between 0 and 1 and time_stamp between 900 and 935 then 1.0 else 0.0 end) as Sep_count,
sum(case when action_type between 0 and 3 and time_stamp between 900 and 935 then 1.0 else 0.0 end) as Sep_action_sum,
sum(case when action_type between 0 and 1 and time_stamp between 1000 and 1035 then 1.0 else 0.0 end) as Oct_count,
sum(case when action_type between 0 and 3 and time_stamp between 1000 and 1035 then 1.0 else 0.0 end) as Oct_action_sum,
sum(case when action_type between 0 and 1 and time_stamp between 1100 and 1135 then 1.0 else 0.0 end) as Nov_count,
sum(case when action_type between 0 and 3 and time_stamp between 1100 and 1135 then 1.0 else 0.0 end) as Nov_action_sum,
sum(case when action_type between 0 and 1 and time_stamp = 1111 then 1.0 else 0.0 end) as double11_count,
sum(case when action_type between 0 and 3 and time_stamp = 1111 then 1.0 else 0.0 end) as double11_action_sum,
sum(case when action_type between 0 and 1 then 1 else 0 end) as overall_count,
sum(case when action_type between 0 and 3 then 1.0 else 0.0 end) as overall_action_sum
from user_log group by user_id, merchant_id
)
create table um_click_ratio as
(
select user_id,merchant_id,
case when may_action_sum=0 then -1 else may_count/may_action_sum end as may_ratio,
case when june_action_sum=0 then -1 else june_count/june_action_sum end as june_ratio,
case when july_action_sum=0 then -1 else july_count/july_action_sum end as july_ratio,
case when aug_action_sum=0 then -1 else aug_count/aug_action_sum end as aug_ratio,
case when sep_action_sum=0 then -1 else sep_count/sep_action_sum end as sep_ratio,
case when oct_action_sum=0 then -1 else oct_count/oct_action_sum end as oct_ratio,
case when nov_action_sum=0 then -1 else nov_count/nov_action_sum end as nov_ratio,
case when double11_action_sum=0 then -1 else double11_count/double11e11_action_sum end as double11_ratio,
from um_click1
)

create table um_click_info as(select um_click.user_id,um_click.merchant_id,
um_click.may_count as may_click,um_click_ratio.may_ratio as may_click_ratio,
um_click.june_count as jun_click,um_click_ratio.june_ratio as jun_click_ratio,
um_click.july_count as jul_click,um_click_ratio.july_ratio as jul_click_ratio,
um_click.aug_count as aug_click,um_click_ratio.aug_ratio as aug_click_ratio,
um_click.sep_count as sep_click,um_click_ratio.sep_ratio as sep_click_ratio,
um_click.oct_count as oct_click,um_click_ratio.oct_ratio as oct_click_ratio,
um_click.nov_count as nov_click,um_click_ratio.nov_ratio as nov_click_ratio,
um_click.double11_count as double11_click,um_click_ratio.double11_ratio as double11_click_ratio,
um_click.overall_count as overall_click, um_click_ratio.overall_ratio as overall_click_ratio
from um_click, um_click_ratio where um_click.user_id=um_click_ratio.user_id and um_click.merchant_id=um_click_ratio.merchant_id
)

create table um_purchase as
(
select user_id, merchant_id,
sum(case when action_type=2 and time_stamp between 500 and 535 then 1 else 0.0 end) as May_count,
sum(case when action_type between 0 and 3 and time_stamp between 500 and 535 then 1.0 else 0.0 end) as May_action_sum,
sum(case when action_type=2 and time_stamp between 600 and 635 then 1.0 else 0.0 end) as June_count,
sum(case when action_type between 0 and 3 and time_stamp between 600 and 635 then 1.0 else 0.0 end) as June_action_sum,
sum(case when action_type=2 and time_stamp between 700 and 735 then 1.0 else 0.0 end) as July_count,
sum(case when action_type between 0 and 3 and time_stamp between 700 and 735 then 1.0 else 0.0 end) as July_action_sum,
sum(case when action_type=2 and time_stamp between 800 and 835 then 1.0 else 0.0 end) as Aug_count,
sum(case when action_type between 0 and 3 and time_stamp between 800 and 835 then 1.0 else 0.0 end) as Aug_action_sum,
sum(case when action_type=2 and time_stamp between 900 and 935 then 1.0 else 0.0 end) as Sep_count,
sum(case when action_type between 0 and 3 and time_stamp between 900 and 935 then 1.0 else 0.0 end) as Sep_action_sum,
sum(case when action_type=2 and time_stamp between 1000 and 1035 then 1.0 else 0.0 end) as Oct_count,
sum(case when action_type between 0 and 3 and time_stamp between 1000 and 1035 then 1.0 else 0.0 end) as Oct_action_sum,
sum(case when action_type=2 and time_stamp between 1100 and 1135 then 1.0 else 0.0 end) as Nov_count,
sum(case when action_type between 0 and 3 and time_stamp between 1100 and 1135 then 1.0 else 0.0 end) as Nov_action_sum,
sum(case when action_type=2 and time_stamp = 1111 then 1.0 else 0.0 end) as double11_count,
sum(case when action_type between 0 and 3 and time_stamp = 1111 then 1.0 else 0.0 end) as double11_action_sum,
sum(case when action_type=2 then 1 else 0 end) as overall_count,
sum(case when action_type between 0 and 3 then 1.0 else 0.0 end) as overall_action_sum
from user_log group by user_id, merchant_id
)
create table um_purchase_ratio as(
select user_id,merchant_id,
case when may_action_sum=0 then NULL else may_count/may_action_sum end as may_ratio,
case when june_action_sum=0 then NULL else june_count/june_action_sum end as june_ratio,
case when july_action_sum=0 then NULL else july_count/july_action_sum end as july_ratio,
case when aug_action_sum=0 then NULL else aug_count/aug_action_sum end as aug_ratio,
case when sep_action_sum=0 then NULL else sep_count/sep_action_sum end as sep_ratio,
case when oct_action_sum=0 then NULL else oct_count/oct_action_sum end as oct_ratio,
case when nov_action_sum=0 then NULL else nov_count/nov_action_sum end as nov_ratio,
case when double11_action_sum=0 then NULL else double11_count/double11_action_sum end as double11_ratio,
case when overall_action_sum=0 then NULL else overall_count/overall_action_sum end as overall_ratio 
from um_purchase)
create table um_purchase_info as(select um_purchase.user_id,um_purchase.merchant_id,
um_purchase.may_count as may_purchase,um_purchase_ratio.may_ratio as may_purchase_ratio,
um_purchase.june_count as jun_purchase,um_purchase_ratio.june_ratio as jun_purchase_ratio,
um_purchase.july_count as jul_purchase,um_purchase_ratio.july_ratio as jul_purchase_ratio,
um_purchase.aug_count as aug_purchase,um_purchase_ratio.aug_ratio as aug_purchase_ratio,
um_purchase.sep_count as sep_purchase,um_purchase_ratio.sep_ratio as sep_purchase_ratio,
um_purchase.oct_count as oct_purchase,um_purchase_ratio.oct_ratio as oct_purchase_ratio,
um_purchase.nov_count as nov_purchase,um_purchase_ratio.nov_ratio as nov_purchase_ratio,
um_purchase.double11_count as double11_purchase,um_purchase_ratio.double11_ratio as double11_purchase_ratio,
um_purchase.overall_count as overall_purchase, um_purchase_ratio.overall_ratio as overall_purchase_ratio
from um_purchase, um_purchase_ratio where um_purchase.user_id=um_purchase_ratio.user_id and um_purchase.merchant_id=um_purchase_ratio.merchant_id
)



create table um_favorite as
(
select user_id, merchant_id,
sum(case when action_type=3 and time_stamp between 500 and 535 then 1.0 else 0.0 end) as May_count,
sum(case when action_type between 0 and 3 and time_stamp between 500 and 535 then 1.0 else 0.0 end) as May_action_sum,
sum(case when action_type=3 and time_stamp between 600 and 635 then 1.0 else 0.0 end) as June_count,
sum(case when action_type between 0 and 3 and time_stamp between 600 and 635 then 1.0 else 0.0 end) as June_action_sum,
sum(case when action_type=3 and time_stamp between 700 and 735 then 1.0 else 0.0 end) as July_count,
sum(case when action_type between 0 and 3 and time_stamp between 700 and 735 then 1.0 else 0.0 end) as July_action_sum,
sum(case when action_type=3 and time_stamp between 800 and 835 then 1.0 else 0.0 end) as Aug_count,
sum(case when action_type between 0 and 3 and time_stamp between 800 and 835 then 1.0 else 0.0 end) as Aug_action_sum,
sum(case when action_type=3 and time_stamp between 900 and 935 then 1.0 else 0.0 end) as Sep_count,
sum(case when action_type between 0 and 3 and time_stamp between 900 and 935 then 1.0 else 0.0 end) as Sep_action_sum,
sum(case when action_type=3 and time_stamp between 1000 and 1035 then 1.0 else 0.0 end) as Oct_count,
sum(case when action_type between 0 and 3 and time_stamp between 1000 and 1035 then 1.0 else 0.0 end) as Oct_action_sum,
sum(case when action_type=3 and time_stamp between 1100 and 1135 then 1.0 else 0.0 end) as Nov_count,
sum(case when action_type between 0 and 3 and time_stamp between 1100 and 1135 then 1.0 else 0.0 end) as Nov_action_sum,
sum(case when action_type=3 and time_stamp = 1111 then 1.0 else 0.0 end) as double11_count,
sum(case when action_type between 0 and 3 and time_stamp = 1111 then 1.0 else 0.0 end) as double11_action_sum,
sum(case when action_type=3 then 1 else 0 end) as overall_count,
sum(case when action_type between 0 and 3 then 1.0 else 0.0 end) as overall_action_sum
from user_log group by user_id, merchant_id
)
create table um_favorite_ratio as
(
select user_id,merchant_id,
case when may_action_sum=0 then -1 else may_count/may_action_sum as may_ratio,
case when june_action_sum=0 then -1 else june_count/june_action_sum as june_ratio,
case when july_action_sum=0 then -1 else july_count/july_action_sum as july_ratio,
case when aug_action_sum=0 then -1 else aug_count/aug_action_sum as aug_ratio,
case when sep_action_sum=0 then -1 else sep_count/sep_action_sum as sep_ratio,
case when oct_action_sum=0 then -1 else oct_count/oct_action_sum as oct_ratio,
case when nov_action_sum=0 then -1 else nov_count/nov_action_sum as nov_ratio,
case when double11_action_sum=0 then -1 else double11_count/double11_action_sum as double11_ratio
case when overall_action_sum=0 then -1 else overall_count/overall_action_sum as double11_ratio 
from um_favorite
)

create table um_favorite_info as(select um_favorite.user_id,um_favorite.merchant_id,
um_favorite.may_count as may_favorite,um_favorite_ratio.may_ratio as may_favorite_ratio,
um_favorite.june_count as jun_favorite,um_favorite_ratio.june_ratio as jun_favorite_ratio,
um_favorite.july_count as jul_favorite,um_favorite_ratio.july_ratio as jul_favorite_ratio,
um_favorite.aug_count as aug_favorite,um_favorite_ratio.aug_ratio as aug_favorite_ratio,
um_favorite.sep_count as sep_favorite,um_favorite_ratio.sep_ratio as sep_favorite_ratio,
um_favorite.oct_count as oct_favorite,um_favorite_ratio.oct_ratio as oct_favorite_ratio,
um_favorite.nov_count as nov_favorite,um_favorite_ratio.nov_ratio as nov_favorite_ratio,
um_favorite.double11_count as double11_favorite,um_favorite_ratio.double11_ratio as double11_favorite_ratio,
um_favorite.overall_count as overall_favorite, um_favorite_ratio.overall_ratio as overall_favorite_ratio
from um_favorite, um_favorite_ratio where um_favorite.user_id=um_favorite_ratio.user_id and um_favorite.merchant_id=um_favorite_ratio.merchant_id
)

create table all_info as(
select um_click_info.*,um_purchase_info.may_purchase,um_purchase_info.may_purchase_ratio,um_purchase_info.jun_purchase,um_purchase_info.jun_purchase_ratio,
um_purchase_info.jul_purchase,um_purchase_info.jul_purchase_ratio,um_purchase_info.aug_purchase,um_purchase_info.aug_purchase_ratio,
um_purchase_info.sep,um_purchase_info.sep_purchase_ratio,um_purchase_info.oct,um_purchase_info.oct_purchase_ratio,um_purchase_info.nov,um_purchase_info.nov_purchase_ratio,
um_purchase_info.double11_purchase,um_purchase_info.double11_purchase_ratio,um_purchase_info.overall_purchase,um_purchase_info.overall_purchase_ratio,
um_favorite_info.may_favorite,um_favorite_info.may_favorite_ratio,um_favorite_info.jun_favorite,um_favorite_info.jun_favorite_ratio,
um_favorite_info.jul_favorite,um_favorite_info.jul_favorite_ratio,um_favorite_info.aug_favorite,um_favorite_info.aug_favorite_ratio,
um_favorite_info.sep,um_favorite_info.sep_favorite_ratio,um_favorite_info.oct,um_favorite_info.oct_favorite_ratio,um_favorite_info.nov,um_favorite_info.nov_favorite_ratio,
um_favorite_info.double11_favorite,um_favorite_info.double11_favorite_ratio,um_favorite_info.overall_favorite,um_favorite_info.overall_favorite_ratio
from um_click_info,um_purchase_info,um_favorite_info 
where um_click_info.user_id=um_purchase_info.user_id and um_purchase_info.user_id=um_favorite_info.user_id 
and um_click_info.merchant_id=um_purchase_info.merchant_id and um_purchase_info.merchant_id=um_favorite_info.merchant_id
)

create train_user_merchant as(
select all_info.* from all_info inner join train_data on all_info.user_id=train_data.user_id and all_info.merchant_id=train_data.merchant_id
)

create table statistical_info_user_merchant_count as (select
label, 
round(avg(may_click),4) as avg1,
round(avg(jun_click),4) as avg2,
round(avg(jul_click),4) as avg3,
round(avg(aug_click),4) as avg4,
round(avg(sep_click),4) as avg5,
round(avg(oct_click),4) as avg6,
round(avg(nov_click),4) as avg7,
round(avg(double11_click),4) as avg8,
round(avg(overall_click),4) as avg9,
round(avg(may_purchase),10) as avg11,
round(avg(jun_purchase),10) as avg12,
round(avg(jul_purchase),10) as avg13,
round(avg(aug_purchase),10) as avg14,
round(avg(sep_purchase),10) as avg15,
round(avg(oct_purchase),10) as avg16,
round(avg(nov_purchase),10) as avg17,
round(avg(double11_purchase),10) as avg18,
round(avg(overall_purchase),10) as avg19,
round(avg(may_favorite),10) as avg21,
round(avg(jun_favorite),10) as avg22,
round(avg(jul_favorite),10) as avg23,
round(avg(aug_favorite),10) as avg24,
round(avg(sep_favorite),10) as avg25,
round(avg(oct_favorite),10) as avg26,
round(avg(nov_favorite),10) as avg27,
round(avg(double11_favorite),4) as avg28,
round(avg(overall_favorite),4) as avg29,

round(avg(may_click),4) as avg1,
round(avg(jun_click),4) as avg2,
round(avg(jul_click),4) as avg3,
round(avg(aug_click),4) as avg4,
round(avg(sep_click),4) as avg5,
round(avg(oct_click),4) as avg6,
round(avg(nov_click),4) as avg7,
round(avg(double11_click),4) as avg8,
round(avg(overall_click),4) as avg9,
round(avg(may_purchase),10) as avg11,
round(avg(jun_purchase),10) as avg12,
round(avg(jul_purchase),10) as avg13,
round(avg(aug_purchase),10) as avg14,
round(avg(sep_purchase),10) as avg15,
round(avg(oct_purchase),10) as avg16,
round(avg(nov_purchase),10) as avg17,
round(avg(double11_purchase),10) as avg18,
round(avg(overall_purchase),10) as avg19,
round(avg(may_favorite),10) as avg21,
round(avg(jun_favorite),10) as avg22,
round(avg(jul_favorite),10) as avg23,
round(avg(aug_favorite),10) as avg24,
round(avg(sep_favorite),10) as avg25,
round(avg(oct_favorite),10) as avg26,
round(avg(nov_favorite),10) as avg27,
round(avg(double11_favorite),4) as avg28,
round(avg(overall_favorite),4) as avg29,

max(may_click) as max1,
max(jun_click) as max2,
max(jul_click) as max3,
max(aug_click) as max4,
max(sep_click) as max5,
max(oct_click) as max6,
max(nov_click) as max7,
max(double11_click) as max8,
max(overall_click) as max9,
max(may_purchase) as max11,
max(jun_purchase) as max12,
max(jul_purchase) as max13,
max(aug_purchase) as max14,
max(sep_purchase) as max15,
max(oct_purchase) as max16,
max(nov_purchase) as max17,
max(double11_purchase) as max18,
max(overall_purchase) as max19,
max(may_favorite) as max21,
max(jun_favorite) as max22,
max(jul_favorite) as max23,
max(aug_favorite) as max24,
max(sep_favorite) as max25,
max(oct_favorite) as max26,
max(nov_favorite) as max27,
max(double11_favorite) as max28,
max(overall_favorite) as max29,
min(may_click) as min1,
min(jun_click) as min2,
min(jul_click) as min3,
min(aug_click) as min4,
min(sep_click) as min5,
min(oct_click) as min6,
min(nov_click) as min7,
min(double11_click) as min8,
min(overall_click) as min9,
min(may_purchase) as min11,
min(jun_purchase) as min12,
min(jul_purchase) as min13,
min(aug_purchase) as min14,
min(sep_purchase) as min15,
min(oct_purchase) as min16,
min(nov_purchase) as min17,
min(double11_purchase) as min18,
min(overall_purchase) as min19,
min(may_favorite) as min21,
min(jun_favorite) as min22,
min(jul_favorite) as min23,
min(aug_favorite) as min24,
min(sep_favorite) as min25,
min(oct_favorite) as min26,
min(nov_favorite) as min27,
min(double11_favorite) as min28,
min(overall_favorite) as min29
from train_user_merchant group by label
)

#market share features
#Nm:
create table Nm as (
select merchant_id, count(action_type) from user_log where action_type=2 group by merchant_id
)
#Nb:
create table Nb as (
select brand_id, count(action_type) from user_log where action_type=2 group by brand_id
)
#Nmb:
create table Nmb as (
select merchant_id, brand_id, count(action_type) from user_log where action_type=2 group by merchant_id, brand_id
)
#Um:
create table Um as (
select merchant_id, count(distinct user_id) from user_log where action_type=2 group by merchant_id
)
#Ub:
create table Ub as (
select brand_id, count(distinct user_id) from user_log where action_type=2 group by brand_id
)
#Umb:
create table Umb as(
select merchant_id, brand_id, count(distinct user_id) from user_log where action_type=2 group by merchant_id,brand_id
)
create table brand_share as
(
    select nmb.merchant_id,nmb.brand_id, 
    CAST(nmb.purchase as float)/CAST(nb.purchase as float) as mb_share, 
    CAST(umb.users as float)/CAST(ub.users as float) as mu_share,
    CAST(nmb.purchase as float)/CAST(nm.purchase as float) as bm_share,
    CAST(umb.users as float)/CAST(um.users as float) as um_share
    from nmb,nb ,umb,ub,nm,um 
    where nmb.brand_id=nb.brand_id and umb.brand_id=ub.brand_id and nmb.merchant_id=nm.merchant_id and 
    nmb.merchant_id=um.merchant_id and nmb.merchant_id=umb.merchant_id and nmb.brand_id=umb.brand_id
)

#u-m similarity features
create table umb_purchase as(
select user_id,merchant_id,brand_id,sum(case when action_type=2 then 1 else 0 end) as purchase from user_log
    where action_type=2 group by user_id,merchant_id,brand_id
)

create table simil as(
select umb_purchase.*,
cast(brand_share.mb_share as numeric(10,4)),
round(cast(purchase as numeric(10,4))*cast(brand_share.mb_share as numeric(10,4)),4) as score 
from umb_purchase inner join brand_share on umb_purchase.merchant_id=brand_share.merchant_id and umb_purchase.brand_id=brand_share.brand_id
)
create table um_sim_fea as(
select user_id,merchant_id,sum(score) from simil group by user_id,merchant_id
)


create table train_log as(
select train_data.label,train_data.user_id,train_data.merchant_id,user_log.cat_id,user_log.brand_id from train_data inner join user_log
on train_data.user_id=user_log.user_id and train_data.merchant_id=user_log.merchant_id and action_type=2 and time_stamp=1111
)

create table repeat_um as
(
select user_id,merchant_id,count(distinct time_stamp) as days from user_log where action_type=2 group by user_id,merchant_id
)
create table repeat_buyers_num_merchant as
(
select merchant_id,sum(case when days>1 then 1 else 0 end) as repeat_num from repeat_um where days>1 group by merchant_id
)

create table repeat_features as
(
select repeat_buyers_num_merchant.merchant_id,repeat_buyers_num_merchant.repeat_num,
round(cast(repeat_buyers_num_merchant.repeat_num as numeric(10,4))/cast(buyers_num_merchant.count as numeric(10,4)),4) as repeat_buyers_ratio
from repeat_buyers_num_merchant,buyers_num_merchant where repeat_buyers_num_merchant.merchant_id=buyers_num_merchant.merchant_id
)

create table train_complex_features as (
select train_data.*,repeat_features.repeat_num as merchant_repeat_buyers_num, repeat_features.repeat_buyers_ratio as merchant_repeat_buyers_ratio,
um_similarity_feature.similarity as um_similarity from train_data,repeat_features,um_similarity_feature
where train_data.user_id=um_similarity_feature.user_id and train_data.merchant_id=um_similarity_feature.merchant_id and train_data.merchant_id=repeat_features.merchant_id
)


create table complex_double11_fea as 
(
    select double11_fea.*,complex_features.merchant_repeat_buyers_num,complex_features.merchant_repeat_buyers_ratio,complex_features.pca1,complex_features.pca2,
    complex_features.pca3,complex_features.pca4,complex_features.pca5,complex_features.pca6,complex_features.pca7,complex_features.pca8,complex_features.pca9,
    complex_features.pca10,complex_features.sim from double11_fea inner join complex_features 
    on double11_fea.user_id=complex_features.user_id and double11_fea.merchant_id=complex_features.merchant_id
)
