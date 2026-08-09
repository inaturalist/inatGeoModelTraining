import click
import pandas as pd
from tqdm.auto import tqdm

@click.command()
@click.option("--taxonomy_file", type=str, required=True)
@click.option("--output_file", type=str, required=True)
def make_ancestor_map(taxonomy_file, output_file):
    """
    Generates a flat (taxon_id, ancestor_id) mapping from a taxonomy file,
    including self-links, for use in DuckDB joins.
    Reads and writes CSV files.
    """
    tax = pd.read_csv(taxonomy_file)

    parent_map = tax.set_index("taxon_id")[
        "parent_taxon_id"
    ].dropna().astype(int).to_dict()

    ancestor_map = {}
    def get_ancestors(taxon_id):
        if taxon_id in ancestor_map:
            return ancestor_map[taxon_id]
        ancestors = []

        child_taxon_id = taxon_id
        while child_taxon_id in parent_map:
            child_taxon_id = parent_map[child_taxon_id]
            ancestors.append(child_taxon_id)
        ancestor_map[taxon_id] = ancestors
        return ancestors
    
    all_taxa = tax["taxon_id"].dropna().astype(int).unique()
    for taxon_id in tqdm(all_taxa):
        _ = get_ancestors(taxon_id)
  
    rows = [] 
    for taxon_id, ancestors in ancestor_map.items():
        for ancestor_id in ancestors:
             rows.append({
                "taxon_id": taxon_id,
                "ancestor_id": ancestor_id,
            })
    df = pd.DataFrame(rows)

    # include self
    self_rows = [
        { "taxon_id": taxon_id, "ancestor_id": taxon_id }
        for taxon_id in ancestor_map.keys()
    ]
    df = pd.concat([
        df, pd.DataFrame(self_rows)
    ], ignore_index=True)
    
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    make_ancestor_map()
