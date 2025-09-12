from pathlib import Path
import sys
sys.path.append('..')

import h3
import h3pandas
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from lib.models.tf_gp_model import TFGeoPriorModel
from lib.models.encoders import CoordEncoder

def hash_func_geomodel(obj: TFGeoPriorModel) -> str:
    return obj.model_path

def hash_func_encoder(obj: CoordEncoder) -> str:
    return obj.encoding_strategy + str(obj.raster)

my_hash_funcs = {
    CoordEncoder: hash_func_encoder,
    TFGeoPriorModel: hash_func_geomodel,
}

@st.cache_resource()
def make_model(model_path):
    print(f"loading model at {model_path}")
    return TFGeoPriorModel(model_path)

@st.cache_resource
def make_raster(raster_path):
    print(f"loading raster at {raster_path}")
    return np.load(raster_path).astype(np.float32)

@st.cache_resource
def make_encoder(input_encoding, raster):
    print(f"making encoder with {input_encoding} and raster with shape {raster.shape}")
    return CoordEncoder(input_encoding, raster)

@st.cache_data
def make_taxonomy(tax_path):
    print(f"loading taxonomy at {tax_path}")
    return pd.read_csv(tax_path)        

@st.cache_data
def make_spatial_train(spatial_train_file):
    print(f"loading spatial train data at {spatial_train_file}")
    return pd.read_parquet(spatial_train_file)

@st.cache_data
def make_dfh3():
    print(f"making dfh3")
    res0_cells = h3.get_res0_indexes()

    all_cells = set()
    for res0 in res0_cells:
        children = h3.h3_to_children(res0, 5)
        all_cells.update(children)
    h3_05_all = pd.Series(
        list(all_cells), name="h3_05"
    )
    df = pd.DataFrame({"h3_05": h3_05_all})
    df.set_index("h3_05", inplace=True)
    dfh3 = df.h3.h3_to_geo()
    dfh3["lng"] = dfh3.geometry.x
    dfh3["lat"] = dfh3.geometry.y
    _ = dfh3.pop("geometry")
    dfh3.reset_index(inplace=True)
    return dfh3


@st.cache_data(hash_funcs=my_hash_funcs)
def make_geomodel_feats(model, encoder, _dfh3):
    print(f"making geomodel feats for {model}")
    locs = _dfh3[["lng", "lat"]].values
    encoded_locs = encoder.encode(locs)
    return model.get_loc_emb(encoded_locs).numpy()
    

@st.cache_data(hash_funcs=my_hash_funcs)
def get_predictions_for_class_id(class_of_interest, model, geomodel_feats):
    return model.eval_one_class_from_feats(
        geomodel_feats,
        class_of_interest=class_of_interest
    )
       

#dataset_2_20_dir = Path(
#    "/data-ssd/alex/datasets/vision-export-20250119070012-aka-2.20"
#)
#model = make_model(
#    "/home/alex/export_eval_models/model_exports/tf_geoprior_2_20_r6_elevation.h5"
#)
#raster = make_raster("/home/alex/sinr/data/env/elev_scaled.npy")
#encoder = make_encoder("sinusoidal", raster)
#tax_2_20 = make_taxonomy(dataset_2_20_dir / "taxonomy.csv")
#spatial_train_h3 = make_spatial_train(
#    dataset_2_20_dir / "spatial_data_h3.parquet"
#)
#dfh3 = make_dfh3()
#geomodel_feats = make_geomodel_feats(model, encoder, dfh3)


models = {
    #"2_20_grid_elevation": {
    #    "dataset_dir": "/data-ssd/alex/datasets/vision-export-20250119070012-aka-2.20",
    #    "model_path": "/home/alex/export_eval_models/model_exports/tf_geoprior_2_20_r6_elevation.h5",
    #    "raster": "/home/alex/sinr/data/env/elev_scaled.npy",
    #    "input_encoding": "sinusoidal",
    #},
    "2_21_grid_elevation": {
        "dataset_dir": "/data-ssd/alex/datasets/vision-export-20250216070017-aka-2.21",
        "model_path": "/home/alex/export_eval_models/model_exports/tf_geoprior_2_21_r6_elevation.h5",
        "raster": "/home/alex/sinr/data/env/elev_scaled.npy",
        "input_encoding": "sinusoidal",
    },
    #"2_21_sinr_elevation": {
    #    "dataset_dir": "/data-ssd/alex/datasets/vision-export-20250216070017-aka-2.21",
    #    "model_path": "/home/alex/inatGeoModelTraining/models/saved_model_2_21_coords_elev.h5",
    #    "raster": "/home/alex/sinr/data/env/elev_scaled.npy",
    #    "input_encoding": "sinusoidal",
    #},
}

model_name = st.selectbox("Select a model", models.keys())
dataset_dir = Path(models[model_name]["dataset_dir"])

model = make_model(models[model_name]["model_path"])
tax = make_taxonomy((dataset_dir / "taxonomy.csv").as_posix())
raster = make_raster(models[model_name]["raster"])
encoder = make_encoder(models[model_name]["input_encoding"], raster)
spatial_train_h3 = make_spatial_train((dataset_dir / "spatial_data_h3_trimmed.parquet").as_posix())
dfh3 = make_dfh3()
geomodel_feats = make_geomodel_feats(model, encoder, dfh3)

range_csv_dir = dataset_dir / "taxon_range_csvs"
available_taxon_ids = set(int(p.stem) for p in range_csv_dir.glob("*.csv"))

# american pika only
available_taxon_ids = [43188]

valid_taxonomy = tax[tax["taxon_id"].isin(available_taxon_ids)].copy()
valid_taxonomy = valid_taxonomy.sort_values("name")
taxon_display = {
    f"{row['taxon_id']} – {row['name']}": row["taxon_id"]
    for _, row in valid_taxonomy.iterrows()
}
selected_label = st.selectbox("Select a taxon", list(taxon_display.keys()))
selected_taxon = taxon_display[selected_label]

taxon_obs = spatial_train_h3[
    (spatial_train_h3["taxon_id"] == selected_taxon) &
    (spatial_train_h3["community"] == 1)
]
obs_with_coords = pd.merge(taxon_obs, dfh3, how="inner", on="h3_05")
training_layer = pdk.Layer(
    "ScatterplotLayer",
    data=obs_with_coords,
    get_position=["lng", "lat"],
    get_radius=5_000,
    get_fill_color=[255, 255, 0, 180],
    pickable=False,
)

def make_predictions_map_layer(taxon_id, model, geomodel_feats, taxonomy):
    print("making predictions map layer")
    class_of_interest = int(taxonomy[taxonomy.taxon_id==taxon_id].iloc[0].leaf_class_id)
    preds = get_predictions_for_class_id(
        class_of_interest, model, geomodel_feats
    )
    dfh3["preds"] = preds[0]

    # filter training data to this taxon
    spatial_train_class_h3 = spatial_train_h3[(spatial_train_h3.taxon_id==taxon_id)&(spatial_train_h3.community==1)]

    # make lpt_r threshold
    presence_preds = dfh3[dfh3['h3_05'].isin(spatial_train_class_h3['h3_05'])]
    #lpt_r_threshold = np.percentile(presence_preds.preds, 5)
    #dfh3['binary_lpt_r'] = (dfh3['preds'] >= lpt_r_threshold).astype(int)

    # fixed threshold - pre-28c8
    #threshold = 0.013629647
    # threshold - post-28c8
    threshold = 0.0032035173
    
    # threshold for 2.20 (pre-28c8)
    #threshold = 0.010270023
    dfh3['binary_lpt_r'] = (dfh3['preds'] >= threshold).astype(int)

    predicted_cells = set(dfh3[dfh3['binary_lpt_r'] == 1]['h3_05'])

    geojson_polys_predicted = h3.h3_set_to_multi_polygon(predicted_cells, geo_json=True)

    predicted_feats = []

    for poly in geojson_polys_predicted:
        outer = poly[0]
        holes = poly[1:] if len(poly) > 1 else []
        coords = [outer] + holes  # GeoJSON format
        predicted_feats.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": coords
            },
            "properties": {"name": "Predicted Range"}
        })

    geojson_predicted = {
        "type": "FeatureCollection",
        "features": predicted_feats
    }

    return pdk.Layer(
        "GeoJsonLayer",
        data=geojson_predicted,
        stroked=False,
        filled=True,
        get_fill_color=[0, 200, 100, 120],  # green-ish
        get_line_color=[0, 0, 0],
        line_width_min_pixels=1,
        pickable=True,
    )

def make_expert_map_layer(taxon_id):
    print("making expert map layer")
    known_range = pd.read_csv(
        f"/data-ssd/alex/datasets/vision-export-20250216070017-aka-2.21/taxon_range_csvs/{taxon_id}.csv",
        names=["asdf"],
    )
    known_range_geojson_polys = h3.h3_set_to_multi_polygon(known_range.asdf.values, geo_json=True)
    expert_range_features = []

    for poly in known_range_geojson_polys:
        outer = poly[0]
        holes = poly[1:] if len(poly) > 1 else []
        coords = [outer] + holes  # GeoJSON format
        expert_range_features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": coords
            },
            "properties": {"name": "Expert Range"}
        })

    expert_range_features = {
        "type": "FeatureCollection",
        "features": expert_range_features
    }
    return pdk.Layer(
        "GeoJsonLayer",
        data=expert_range_features,
        stroked=True,
        filled=False,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
        pickable=True,
    )

expert_layer = make_expert_map_layer(selected_taxon)
preds_layer = make_predictions_map_layer(selected_taxon, model, geomodel_feats, tax)


layers = []
if st.checkbox("Show Predictions", value=True):
    layers.append(preds_layer)
if st.checkbox("Show Expert Range", value=True):
    layers.append(expert_layer)
if st.checkbox("Show Training Data", value=True):
    layers.append(training_layer)

view_state = pdk.ViewState(
    latitude=37.5,
    longitude=-90,
    zoom=1,
    bearing=0,
    pitch=0,
)

deck = pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    map_style="mapbox://styles/mapbox/light-v11"
)

st.pydeck_chart(deck)
