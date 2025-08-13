//  Step 1: Jim Corbett AOI 
var corbett = ee.Geometry.Polygon([
  [[78.580, 29.620],
   [78.580, 29.240],
   [79.260, 29.240],
   [79.260, 29.620]]
]);

Map.centerObject(corbett, 11);
Map.addLayer(corbett, {color: 'red'}, 'Jim Corbett AOI');

//  Step 2: Sentinel-2 (Oct 2023 - Mar 2024) 
var s2 = ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(corbett)
            .filterDate('2023-10-01', '2024-03-31')
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
            .select(['B2','B3','B4','B8','B11']) // only required bands
            .median()
            .clip(corbett);

//  Step 3: Indices 
var ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI');
var ndwi = s2.normalizedDifference(['B3', 'B8']).rename('NDWI');
var ndbi = s2.normalizedDifference(['B11', 'B8']).rename('NDBI');

var image = s2.addBands(ndvi).addBands(ndwi).addBands(ndbi);
var bands = ['B2','B3','B4','B8','B11','NDVI','NDWI','NDBI'];

// Step 4: Training Polygons (sample, replace with better ones) 
// Forest (dense green)
var forest = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Polygon([[[78.9435, 29.5450],[78.9440, 29.5450],[78.9440, 29.5445],[78.9435, 29.5445]]]), {'landcover': 0}),
    ee.Feature(ee.Geometry.Polygon([[[78.9880, 29.5355],[78.9885, 29.5355],[78.9885, 29.5350],[78.9880, 29.5350]]]), {'landcover': 0}),
    ee.Feature(ee.Geometry.Polygon([[[78.9120, 29.5000],[78.9125, 29.5000],[78.9125, 29.4995],[78.9120, 29.4995]]]), {'landcover': 0})
]);

// Grassland
var grassland = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Polygon([[[78.8650, 29.4700],[78.8655, 29.4700],[78.8655, 29.4695],[78.8650, 29.4695]]]), {'landcover': 1}),
    ee.Feature(ee.Geometry.Polygon([[[78.8890, 29.4505],[78.8895, 29.4505],[78.8895, 29.4500],[78.8890, 29.4500]]]), {'landcover': 1}),
    ee.Feature(ee.Geometry.Polygon([[[78.9200, 29.4600],[78.9205, 29.4600],[78.9205, 29.4595],[78.9200, 29.4595]]]), {'landcover': 1})
]);

// Water
var water = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Polygon([[[78.7880, 29.5200],[78.7885, 29.5200],[78.7885, 29.5195],[78.7880, 29.5195]]]), {'landcover': 2}),
    ee.Feature(ee.Geometry.Polygon([[[78.8000, 29.5050],[78.8005, 29.5050],[78.8005, 29.5045],[78.8000, 29.5045]]]), {'landcover': 2}),
    ee.Feature(ee.Geometry.Polygon([[[78.7800, 29.5400],[78.7805, 29.5400],[78.7805, 29.5395],[78.7800, 29.5395]]]), {'landcover': 2})
]);

// Bare Land
var bare = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Polygon([[[78.8400, 29.4800],[78.8405, 29.4800],[78.8405, 29.4795],[78.8400, 29.4795]]]), {'landcover': 3}),
    ee.Feature(ee.Geometry.Polygon([[[78.8500, 29.4700],[78.8505, 29.4700],[78.8505, 29.4695],[78.8500, 29.4695]]]), {'landcover': 3}),
    ee.Feature(ee.Geometry.Polygon([[[78.8300, 29.4900],[78.8305, 29.4900],[78.8305, 29.4895],[78.8300, 29.4895]]]), {'landcover': 3})
]);

var training = forest.merge(grassland).merge(water).merge(bare);

// Step 5: Sample Data 
var trainingData = image.sampleRegions({
  collection: training,
  properties: ['landcover'],
  scale: 10
});

//  Step 6: Train/Test Split 
var withRandom = trainingData.randomColumn('random');
var trainSet = withRandom.filter(ee.Filter.lt('random', 0.7));
var testSet = withRandom.filter(ee.Filter.gte('random', 0.7));

// Step 7: Random Forest 
var classifier = ee.Classifier.smileRandomForest(300).train({
  features: trainSet,
  classProperty: 'landcover',
  inputProperties: bands
});

// Step 8: Classify 
var classified = image.classify(classifier).clip(corbett);
var palette = ['006400', '7CFC00', '0000ff', 'deb887'];
Map.addLayer(classified, {min: 0, max: 3, palette: palette}, 'LULC Jim Corbett');

// Step 9: Accuracy 
var validated = testSet.classify(classifier);
var confMatrix = validated.errorMatrix('landcover', 'classification');
print('Confusion Matrix:', confMatrix);
print('Overall Accuracy:', confMatrix.accuracy());
print('Kappa Coefficient:', confMatrix.kappa());

//  Step 10: Area per Class 
var areaImage = ee.Image.pixelArea().divide(1e6).addBands(classified);
var areas = areaImage.reduceRegion({
  reducer: ee.Reducer.sum().group({
    groupField: 1,
    groupName: 'Class'
  }),
  geometry: corbett,
  scale: 10,
  maxPixels: 1e13
});
print('Area per class (sq km):', areas);

// Step 11: Legend 
var legend = ui.Panel({style: {position: 'bottom-left', padding: '8px 15px'}});
legend.add(ui.Label({value: 'LULC Legend', style: {fontWeight: 'bold', fontSize: '14px'}}));

var names = ['Forest', 'Grassland', 'Water', 'Bare Land'];
var makeRow = function(color, name) {
  var colorBox = ui.Label('', {backgroundColor: '#' + color, padding: '8px', margin: '0'});
  var label = ui.Label(name, {margin: '0 0 4px 6px'});
  return ui.Panel([colorBox, label], ui.Panel.Layout.Flow('horizontal'));
};
for (var i = 0; i < names.length; i++) {
  legend.add(makeRow(palette[i], names[i]));
}
Map.add(legend);

//  Step 12: Export 
Export.image.toDrive({
  image: classified,
  description: 'Jim_Corbett_LULC_Polygons',
  scale: 10,
  region: corbett,
  maxPixels: 1e13
});
