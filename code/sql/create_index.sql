CREATE INDEX index_id
    ON user_log USING btree
    (user_id COLLATE pg_catalog."default", item_id COLLATE pg_catalog."default", merchant_id COLLATE pg_catalog."default", brand_id COLLATE pg_catalog."default", time_stamp COLLATE pg_catalog."default")
    TABLESPACE pg_default;