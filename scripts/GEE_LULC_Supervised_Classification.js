// ============================================================
// SUPERVISED LAND USE / LAND COVER CLASSIFICATION
// Google Earth Engine (GEE)
// ============================================================
//
// Study year: 2024
//
// Classification approach:
// 1. Sentinel-2 spectral bands + Random Forest
// 2. Google Satellite Embeddings + K-Nearest Neighbour (KNN)
// 3. Accuracy assessment using an independent validation split
//
// Land-cover classes:
// 0 = Urban
// 1 = Bare land
// 2 = Vegetation
// 3 = Water
//
// IMPORTANT:
// The study-area boundary and training samples were created as
// imported FeatureCollections in Google Earth Engine.
//
// Imported variables used:
//   geometry    = study-area boundary
//   urban       = urban training samples
//   bare        = bare-land training samples
//   vegetation  = vegetation training samples
//   water       = water training samples
//
// These imported GEE assets are not embedded in this GitHub
// script. The script documents the analytical workflow used
// for the classification.
// ============================================================


// ============================================================
// 1. STUDY AREA AND ANALYSIS PERIOD
// ============================================================

Map.centerObject(geometry);

var year = 2024;

var startDate = ee.Date.fromYMD(year, 1, 1);
var endDate = startDate.advance(1, 'year');


// ============================================================
// 2. SENTINEL-2 DATA PREPARATION
// ============================================================

// Load Sentinel-2 Surface Reflectance imagery.

var s2 = ee.ImageCollection(
  'COPERNICUS/S2_SR_HARMONIZED'
);

// Filter imagery based on:
// - cloud percentage
// - analysis period
// - study area

var filtered = s2
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
  .filter(ee.Filter.date(startDate, endDate))
  .filter(ee.Filter.bounds(geometry));


// ============================================================
// 3. CLOUD MASKING USING CLOUD SCORE+
// ============================================================

// Load Google Cloud Score+ data.

var csPlus = ee.ImageCollection(
  'GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED'
);

var csPlusBands = csPlus.first().bandNames();


// Link Cloud Score+ bands with Sentinel-2 imagery.

var filteredS2WithCs = filtered.linkCollection(
  csPlus,
  csPlusBands
);


// Function to mask pixels with low Cloud Score+ values.

function maskLowQA(image) {

  var qaBand = 'cs';
  var clearThreshold = 0.5;

  var mask = image
    .select(qaBand)
    .gte(clearThreshold);

  return image.updateMask(mask);
}


// Apply cloud masking and retain Sentinel-2 bands.

var filteredMasked = filteredS2WithCs
  .map(maskLowQA)
  .select('B.*');


// Create annual median composite.

var composite = filteredMasked.median();


// ============================================================
// 4. VISUALISE SENTINEL-2 COMPOSITE
// ============================================================

var rgbVis = {
  min: 0.0,
  max: 3000,
  bands: ['B4', 'B3', 'B2']
};

Map.addLayer(
  composite.clip(geometry),
  rgbVis,
  'Sentinel-2 Composite'
);


// ============================================================
// 5. TRAINING DATA
// ============================================================
//
// Training samples were created in Google Earth Engine and
// assigned the following land-cover labels:
//
// Urban       = 0
// Bare land   = 1
// Vegetation  = 2
// Water       = 3
//
// Imported FeatureCollections:
// urban, bare, vegetation, water
// ============================================================

var gcps = urban
  .merge(bare)
  .merge(water)
  .merge(vegetation);


// Extract Sentinel-2 pixel values at training locations.

var training = composite.sampleRegions({

  collection: gcps,

  properties: ['landcover'],

  scale: 10,

  tileScale: 16

});

print(
  'Sentinel-2 Training Samples',
  training
);


// ============================================================
// 6. RANDOM FOREST CLASSIFICATION
// ============================================================

var rfClassifier = ee.Classifier
  .smileRandomForest(50)
  .train({

    features: training,

    classProperty: 'landcover',

    inputProperties: composite.bandNames()

  });


// Classify Sentinel-2 composite.

var rfClassified = composite.classify(
  rfClassifier
);


// Classification colour palette.
//
// Urban       = 0
// Bare land   = 1
// Vegetation  = 2
// Water       = 3

var palette = [
  '#cc6d8f',
  '#ffc107',
  '#004d40',
  '#1e88e5'
];


Map.addLayer(
  rfClassified.clip(geometry),
  {
    min: 0,
    max: 3,
    palette: palette
  },
  'LULC Classification - Random Forest'
);


// ============================================================
// 7. GOOGLE SATELLITE EMBEDDINGS
// ============================================================
//
// Satellite Embeddings are used as an alternative set of
// input features for supervised land-cover classification.
// ============================================================

var embeddings = ee.ImageCollection(
  'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
);


// Filter embeddings to the analysis period and study area.

var filteredEmbeddings = embeddings
  .filter(ee.Filter.date(startDate, endDate))
  .filter(ee.Filter.bounds(geometry));


// Create the embedding image.

var embeddingsImage = filteredEmbeddings.mosaic();

print(
  'Satellite Embedding Image',
  embeddingsImage
);


// ============================================================
// 8. TRAINING DATA FOR SATELLITE EMBEDDINGS
// ============================================================

var embeddingTraining = embeddingsImage.sampleRegions({

  collection: gcps,

  properties: ['landcover'],

  scale: 10

});

print(
  'Embedding Training Samples',
  embeddingTraining.first()
);


// ============================================================
// 9. K-NEAREST NEIGHBOUR CLASSIFICATION
// ============================================================

var knnClassifier = ee.Classifier
  .smileKNN()
  .train({

    features: embeddingTraining,

    classProperty: 'landcover',

    inputProperties: embeddingsImage.bandNames()

  });


// Classify the Satellite Embeddings image.

var embeddingClassified = embeddingsImage.classify(
  knnClassifier
);


// Display classification.

Map.addLayer(
  embeddingClassified.clip(geometry),
  {
    min: 0,
    max: 3,
    palette: palette
  },
  'LULC Classification - Satellite Embeddings'
);


// ============================================================
// 10. EXPORT CLASSIFIED IMAGE
// ============================================================
//
// This export points to the original Google Earth Engine
// project asset location used during the analysis.
//
// The asset path is retained here to document the workflow.
// ============================================================

var exportFolder =
  'projects/sharan07/assets/Assignment3/';

var classifiedExportImage =
  'embeddings_classification';

var classifiedExportImagePath =
  exportFolder + classifiedExportImage;


Export.image.toAsset({

  image: embeddingClassified.clip(geometry),

  description:
    'Classified_Image_Export_Asset',

  assetId:
    classifiedExportImagePath,

  region:
    geometry,

  scale:
    10,

  pyramidingPolicy:
    'MODE',

  maxPixels:
    1e10

});


// ============================================================
// 11. ACCURACY ASSESSMENT
// ============================================================
//
// The labelled training samples are randomly divided into:
// - training subset
// - validation subset
//
// The original workflow uses a 60% / 40% split.
// ============================================================


// Add a random value to each training sample.

var gcpsRandom = gcps.randomColumn(
  'random'
);


// Training subset.

var trainingGcp = gcpsRandom.filter(
  ee.Filter.lt('random', 0.6)
);


// Validation subset.

var validationGcp = gcpsRandom.filter(
  ee.Filter.gte('random', 0.6)
);


// ============================================================
// 12. TRAIN RANDOM FOREST USING TRAINING SUBSET
// ============================================================

var validationTraining = composite.sampleRegions({

  collection: trainingGcp,

  properties: ['landcover'],

  scale: 10,

  tileScale: 16

});


var validationClassifier = ee.Classifier
  .smileRandomForest(50)
  .train({

    features: validationTraining,

    classProperty: 'landcover',

    inputProperties: composite.bandNames()

  });


// Classify the composite.

var validationClassified = composite.classify(
  validationClassifier
);


// Display classification.

Map.addLayer(
  validationClassified.clip(geometry),
  {
    min: 0,
    max: 3,
    palette: palette
  },
  'LULC Classification - Accuracy Assessment'
);


// ============================================================
// 13. VALIDATION DATASET
// ============================================================

var test = validationClassified.sampleRegions({

  collection: validationGcp,

  properties: ['landcover'],

  tileScale: 16,

  scale: 10

});


// ============================================================
// 14. CONFUSION MATRIX AND ACCURACY METRICS
// ============================================================

var testConfusionMatrix =
  test.errorMatrix(
    'landcover',
    'classification'
  );


print(
  'Confusion Matrix',
  testConfusionMatrix
);


print(
  'Overall Accuracy',
  testConfusionMatrix.accuracy()
);


print(
  'Producer Accuracy',
  testConfusionMatrix.producersAccuracy()
);


print(
  'Consumer Accuracy',
  testConfusionMatrix.consumersAccuracy()
);


print(
  'F-Score',
  testConfusionMatrix.fscore()
);


// ============================================================
// END OF WORKFLOW
// ============================================================
//
// Workflow summary:
//
// Sentinel-2 imagery
//        ↓
// Cloud Score+ masking
//        ↓
// Annual median composite
//        ↓
// Training samples
//        ↓
// Random Forest classification
//        ↓
// Satellite Embeddings
//        ↓
// KNN classification
//        ↓
// Accuracy assessment
//        ↓
// Confusion matrix + accuracy metrics
//
// ============================================================
