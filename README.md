# Geo Model Training

## Data Prep

For people outside of iNaturalist, your best bet will be to train using the dataset that was included with the sinr paper. You can download it here: https://github.com/elijahcole/sinr/tree/main/data

For iNat internal folks who want to use our internal exports, look for the spatial_data.csv file in the export. You'll want to convert it to h5 for the training, try duckdb. You'll also need the taxonomy.csv file.

If you want to train with just elevation, generate the elevation data. Intructions TBD.
If you want to train with all bioclim covariates, the instructions are in the sinr dataset download instructions above.
If you want to train without covariates, you're all set.

## Config File

Copy config.yml.example to config.yml and fill in stuff. Keep the training config vars the same if you want to reproduce the sinr paper results.

## Train

$ `python train.py <path_to_config.yml>`

## Eval

These evals are designed to mostly match the evaluations done for the SINR paper.

$ `python sinr_paper_evals.py <path_to_config.yml>`

## Export

`export_coreml_geomodel.py` converts to coreml for iOS deployment, `export_tflite_geomodel.py` converts to tflite for android deployment.

### Tests

$ `pytest` runs 'em.

## Todos

- [ ] write script to convert sinr dataset to an inat-style dataset so we don't need separate dataloaders
- [ ] write instructions to create the elevation npy file


