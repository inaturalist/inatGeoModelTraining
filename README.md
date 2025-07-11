# Geo Model Training

#### Input normalization

- Our geo model takes latitude, longitude, and elevation as inputs.
- The latitudes and longitudes are normalized to (-1, 1) to make it easier for ML algorithms to work with them. 
  - normLat = latitude / 90.0
  - normLng = longitude / 180.0
- We want latitude and longitude to wrap the earth (locations at the poles should be close to each other, locations on either side of the international dateline should be close to each other), so for each location pair lat, lng we map to `[sin(pi * lng), sin(pi * lat), cos(pi * lng), cos(pi * lat)]`
  - see https://arxiv.org/pdf/1906.05272 for discussion
- At training time, we lookup elevations for training observation locations from bioclim elevation data
- We normalize elevation to (-1, 1), again for ML
  - if elevation > 0, normElev = elevation / 5705.63
  - if elevation < 0, normElev = elevation / 32768.0
  - These are just the max and min values in the bioclim elevation map
- Two implementation challenges for inference to be aware of
  - If you’re getting elevation from a different source, such as an iPhone barometer, you may get different elevations than we used at training time, since bioclim maps the world to a single continuous surface. So if you’re underground or in the air, ymmv
  - If you’re over the ocean, then you may get wildly different results: bioclim provides an ocean mask for areas in the ocean, with pixels set to -32768. This is what we trained with, but your barometer probably wouldn’t report this when you’re over the ocean. So the geo model may overestimate the likelihood of land taxa and underestimate the likelihood of marine taxa when you’re in the ocean

#### Converting to SINR

We are converting from discretized & gridded training to following [the SINR approach](https://arxiv.org/abs/2306.02564).


#### Tests

$ `pytest` runs 'em.

