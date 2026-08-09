SET taxonomy_file = '$taxonomy_file';
SET ancestor_map_file = '$ancestor_map_file';
SET spatial_data_file = '$spatial_data_file';
SET output_file = '$output_file';
SET sample_cap = $sample_cap;

CREATE TABLE taxonomy as SELECT * from read_csv_auto(taxonomy_file);
CREATE TABLE ancestor_map AS SELECT * from read_csv_auto(ancestor_map_file);
CREATE TABLE spatial_data as SELECT latitude, longitude, taxon_id from read_parquet(spatial_data_file);

CREATE TABLE expanded AS 
SELECT
    s.latitude,
    s.longitude,
    a.ancestor_id AS taxon_id
FROM spatial_data s
JOIN ancestor_map a
ON s.taxon_id = a.taxon_id;

CREATE TABLE sampled AS
SELECT latitude, longitude, taxon_id
FROM (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY taxon_id
            ORDER BY RANDOM()
        ) AS row_num
    FROM expanded
)
WHERE row_num <= sample_cap;

COPY sampled to output_file (FORMAT_PARQUET);
