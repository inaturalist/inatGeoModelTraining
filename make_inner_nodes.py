import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm

class Taxon:
    
    def is_leaf(self):
        return false

    def __init__(self, tax_row):
        self.name = tax_row["name"]
        self.taxon_id = int(tax_row.taxon_id)

        # convert from pandas nans to python None
        if tax_row.parent_taxon_id is None or np.isnan(tax_row.parent_taxon_id):
            self.parent_taxon_id = None
        else:
            self.parent_taxon_id = int(tax_row.parent_taxon_id)

        if tax_row.spatial_class_id is None or np.isnan(tax_row.spatial_class_id):
            self.spatial_class_id = None
        else:
            self.spatial_class_id = int(tax_row.spatial_class_id)
            
        if tax_row.leaf_class_id is None or np.isnan(tax_row.leaf_class_id):
            self.leaf_class_id = None
        else:
            self.leaf_class_id = int(tax_row.leaf_class_id)

    def __str__(self):
        return "taxon id {} ({}) - parent taxon id {}".format(
            self.taxon_id,
            self.name,
            self.parent_taxon_id,
        )

    def is_leaf(self):
        if self.leaf_class_id is None:
            return False
        else:
            return True

class Taxonomy:
    
    def __init__(self, tax_path):
        self.all_taxa = []
        self.taxon_id_to_taxon = {}
        self.taxon_id_to_children = defaultdict(list)
        
        # convert from pandas to python objects
        tax_df = pd.read_csv(tax_path)
        for _, taxon_row in tax_df.iterrows():
            taxon = Taxon(taxon_row)
            self.all_taxa.append(taxon)

        # setup lookup dictionaries
        for taxon in self.all_taxa:
            self.taxon_id_to_taxon[taxon.taxon_id] = taxon
            self.taxon_id_to_children[taxon.parent_taxon_id].append(taxon)

def get_all_children(taxonomy, taxon_id):
    all_children = []
    taxon = taxonomy.taxon_id_to_taxon[taxon_id]
    if taxon.is_leaf():
        all_children.append(taxon_id)
    else:
        all_children.append(taxon_id)
        for child in taxonomy.taxon_id_to_children[taxon_id]:
            all_children.extend(get_all_children(taxonomy, child.taxon_id))

    return all_children


def main():
    print("loading taxonomy")
    taxonomy = Taxonomy("/data-ssd/alex/datasets/vision-export-20240218060015-aka-2.12/taxonomy.csv")

    all_spatial_data = []
    print("loading spatial export")
    spatial_data = pd.read_csv(
        "/data-ssd/alex/datasets/vision-export-20240218060015-aka-2.12/spatial_data.csv"
    )

    print("expanding spatial data to aggregate for inner nodes")
    all_taxon_ids = list(map(lambda x: x.taxon_id, taxonomy.all_taxa))
    np.random.shuffle(all_taxon_ids)
    print(all_taxon_ids)
    for taxon_id in tqdm(all_taxon_ids):
        taxon = taxonomy.taxon_id_to_taxon[taxon_id]

        if taxon.spatial_class_id is None:
            continue
    
    
        children_taxon_ids = set(get_all_children(taxonomy, taxon.taxon_id))
        children_obs = spatial_data[spatial_data.taxon_id.isin(children_taxon_ids)]
        for i in range(len(children_obs)):
            child_row = children_obs.iloc[i]
            all_spatial_data.append({
                "observation_id": child_row.observation_id,
                "latitude": child_row.latitude,
                "longitude": child_row.longitude,
                "observed_on": child_row.longitude,
                "spatial_class_id": taxon.spatial_class_id,
                "taxon_id": taxon.taxon_id,
                "community": child_row.community,
                "captive": child_row.captive,
            })

    print("making dataframe")
    all_spatial_data_df = pd.DataFrame(all_spatial_data)
    print("saving dataframe to csv")
    all_spatial_data_df.to_csv(
        "/data-ssd/alex/datasets/vision-export-20240218060015-aka-2.12/spatial_data_including_inner.csv",
        index=False
    )


if __name__ == "__main__":
    main()




