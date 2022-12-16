# Content-based filtering recommendation system

## Problem definition
The following repository is an attempt to create a simple content based filtering recommendation system. It retrieves the closest items to a given item by calculating the closeness of the available information from the item to each other item.

## Package description
The package consists of the dataset creation script and the recommender class implementation.
The dataset creation script is used for processing the data and creating the necessary files for the recommendation class. The recommendation class is used 
for retrieving the closest items from the stored data. The retrieval was done only for existing data for performance purposes.
Example metadata with supported datatypes
```{python}
metadata = {
        'store':'categorical',
        'price':'numeric',
        'item description':'text',
        'image url':'image',
        'category':'categorical'
    }
```

## Running the example
To run the example for the package, run the dataset creation script to create the necessary files for the provided test dataset. Then either run the script containing the class implementation or create the recommender object elsewhere and run the predict method.
 This is a sample code for running the recommender class.
```{python}
from recommender import Recommender

rec = Recommender()
rec.predict(0,top_k = 5)
# sample output is a dataframe
```

## Notes
- The original dataset should also be saved in the data folder after running the dataset creation script
- Images should be provided as links