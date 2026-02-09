This paper presents a comprehensive comparison of image￾based geolocalization approaches. Our first approach was
hierarchical, coarse to fine grid based, trained with a
ResNet50 backbone on part of the YFCC100M database,
cleaned up and preprocessed to ensure a more informative
dataset. The second approach was starting from a different
CSV-based dataset from Mapillary, metadata processing and
progressing to a PlaNet-style convolutional neural network
for city-level classification. This simpler method treated ge￾olocalization as a multi-class classification problem instead
of a regression or area based approach. We later also im￾plemented a regression with the same base architecture to
compare it with the results previously obtained.
