UPDATE user_log
SET time_month = SUBSTRING(time_stamp FROM 1 FOR 2);