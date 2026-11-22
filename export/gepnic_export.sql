-- bid-screens export from scrape_hub for ONE GePNIC portal project and ONE tender-ID year (pkey range scan of records; read-only).
-- Run on the hub's database server at quiet hours, one portal-year at a time, by export/gepnic_export.py:
--   psql -d scrape_hub -v q=tenders|bids -v project=gepnic-wbtenders -v lo=2019 -v hi=2020 -f export/gepnic_export.sql
-- Columns match the GeM export so that the screens run unchanged. Same rules as the pilot export (bet 3 pilot), plus:
--   tenders: title (first 200 chars), org_id, category. Times stay text ('DD-Mon-YYYY HH:MI AM', IST); the exporter parses
--   them (an impossible date such as 31-Jun becomes empty).
--   bids: status (raw GePNIC status text) and stage: q = passed the technical stage (any Finance/AOC status, or an
--         Admitted/Accepted technical status), d = rejected at the fee / pre-qualification / technical stage, u = other.
-- item_id = a stable hash of (org_id, tender_category): GePNIC has no item class. tech = 1 when the bid reached the finance
-- stage (pilot rule), else 2. price = the OCR amount (doc-ocr-gepnic-awards) where the chart was read; else NULL.
\set ON_ERROR_STOP on
set default_transaction_read_only = on;
set statement_timeout = '15min';
set lock_timeout = '5s';
select :'q' = 'tenders' as q_t, :'q' = 'bids' as q_b \gset

\if :q_t
copy (
  select r.record_id as tender_id, left(r.record_id, 4)::int as yr,
         left(r.data->'details'->'fields'->>'published_date', 20) as published,
         left(r.data->'details'->'fields'->>'bid_submission_end_date', 20) as closing,
         'B' as ref_type, r.data->'list'->>'stage' as status,
         abs(hashtext(coalesce(r.data->>'org_id','') || '|' || coalesce(r.data->'details'->'fields'->>'tender_category',''))) as item_id,
         coalesce(r.data->'details'->'fields'->>'tender_category','') || ' | ' || left(coalesce(r.data->'list'->>'title',''), 80) as item_key,
         abs(hashtext(split_part(r.data->'list'->>'organisation_chain', '||', 1) || '||' || split_part(r.data->'list'->>'organisation_chain', '||', 2))) as buyer_id,
         split_part(r.data->'list'->>'organisation_chain', '||', 1) as ministry,
         r.data->'list'->>'organisation_chain' as organisation, :'project' as state,
         jsonb_array_length(coalesce(r.data->'details'->'bids','[]')) as n_received,
         null::int as n_tech, null::int as n_priced,
         case when r.data->'details'->'fields'->>'tender_value' ~ '^[0-9]+(\.[0-9]+)?$'
              then (r.data->'details'->'fields'->>'tender_value')::float8 end as award_value,
         left(coalesce(r.data->'list'->>'title',''), 200) as title,
         r.data->>'org_id' as org_id,
         coalesce(r.data->'details'->'fields'->>'tender_category','') as category
  from records r
  where r.project = :'project' and r.record_id >= :'lo' and r.record_id < :'hi'
    and jsonb_array_length(coalesce(r.data->'details'->'bids','[]')) >= 1
) to stdout with csv header;
\endif

\if :q_b
copy (
  with ocr as (
    select split_part(o.record_id, '/', 2) as tender_id, lower(regexp_replace(b->>'name', '[^A-Za-z0-9]', '', 'g')) as k,
           min((b->>'amount')::numeric) as amount
    from records o, jsonb_array_elements(o.data->'bidders') b
    where o.project = 'doc-ocr-gepnic-awards' and o.data->>'status' = 'found'
      and o.record_id >= replace(:'project', 'gepnic-', '') || '/' || :'lo'
      and o.record_id <  replace(:'project', 'gepnic-', '') || '/' || :'hi'
      and (b->>'amount') ~ '^[0-9]+(\.[0-9]+)?$'
    group by 1, 2)
  select r.record_id as tender_id,
         abs(hashtext(lower(regexp_replace(b->>'bidder_name', '[^A-Za-z0-9]', '', 'g')))) as seller_id,
         lower(regexp_replace(b->>'bidder_name', '[^A-Za-z0-9]', '', 'g')) as seller_key,
         case when b->>'status' ~* 'finance|financial|aoc' then 1 else 2 end as tech,
         left(b->>'submitted_date', 20) as submitted_at,
         ocr.amount::float8 as price,
         nullif(substring(b->>'remarks' from '^L([0-9]{1,4})$'), '')::int as rank_n,
         null::boolean as mse,
         left(b->>'status', 60) as status,
         case when b->>'status' ~* '^(admitted|accepted)-' or (b->>'status' ~* '^(rejected|not admitted)-(finance|aoc)$') then 'q'
              when b->>'status' ~* '^(rejected|not admitted)-' then 'd' else 'u' end as stage
  from records r cross join lateral jsonb_array_elements(r.data->'details'->'bids') b
  left join ocr on ocr.tender_id = r.record_id
              and ocr.k = lower(regexp_replace(b->>'bidder_name', '[^A-Za-z0-9]', '', 'g'))
  where r.project = :'project' and r.record_id >= :'lo' and r.record_id < :'hi'
) to stdout with csv header;
\endif
