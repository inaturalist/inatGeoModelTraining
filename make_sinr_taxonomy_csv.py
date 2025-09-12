import click
import pandas as pd

from lib.data.data_loader import load_sinr_dataset_from_parquet

@click.command
@click.option("--train_data_file", required=True)
@click.option("--train_meta_file", required=True)
@click.option("--out_file", required=True)
def make_taxonomy(train_data_file, train_meta_file, out_file):
    
    locs, class_ids, unique_taxa = load_sinr_dataset_from_parquet(train_data_file)
    
    taxon_ids = pd.DataFrame({ "taxon_id": unique_taxa })
    taxon_ids.reset_index(inplace=True)
    taxon_ids.rename({"index": "leaf_class_id"}, axis=1, inplace=True)
    
    print("loading training meta")
    train_meta = pd.read_json(train_meta_file)


    taxonomy = pd.merge(
        taxon_ids,
        train_meta,
        left_on="taxon_id",
        right_on="taxon_id",
    )
    
    taxonomy.to_csv(out_file, index=False)
    

if __name__ == "__main__":
    make_taxonomy()

