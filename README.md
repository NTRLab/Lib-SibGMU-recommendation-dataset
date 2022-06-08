# Lib-SibGMU-recommendation-dataset
Lib SibGMU recommendation dataset is a medical university library dataset. The dataset was obtained from the circulation history database of a medical school (Siberian State Medical University) library. The fields containing descriptions of documents, the time when the document was lent and returned, the encrypted (for anonymization purposes) user identifier, and the inventory number of the document issued to the user were unloaded. The history of circulation spans from September 2013 to December 2021. During this time, the users of the medical school repeatedly received books and other resources both for use in the educational process, and for conducting scientific research and self-development. The database of users includes students pursuing bachelors, masters, and graduate degrees, employees and professors of the medical school, as well as professionals from other medical institutions who had access to the resources of this medical library.

The textual field data are in Russian, encoded in UTF-8. The dataset contains 153.364 entries related to 7.149 users and 2.664 books. Share of students among the interaction data is about 97\%.

## Data structure
The main file is in XML format. Each interaction has a root tag \texttt{<record>} with the fields \texttt{<field tag=X>}, where X can be :
- "ID" - a unique identifier of user;
- "ID book" - a unique identifier of book;
- "name book" - string that contains book's author, name and brief description;
- "category reader" - category of a reader, e.g. student, professor, etc.;
- "birth YYYY" - user's birth year;
- "date circulation YYYYMMDD" - date in yyyy/mm/dd format when the event took place;

## Code structure
Please use `xml_parsing.py` to process `medlib.xml` into dataframe in case You need .csv format files. Notebooks with model training and metrics:
- `NeuFM.ipynb` - Uses NCF (NeuFM) model from cornac module.
- `kmean+als+lightfm.ipynb` - Uses cluster modeling, ALS and LightFM.
- `kmeans+fasttext.ipynb` - First creates txt file of users interactions history for fasttext model training (unsupervised). Then vector representations from fasttext model being used for clusterization.

  
## License and copyright
The Lib SibGMU recommendation dataset dataset is distributed under the Creative Commons Attribution 4.0 International License. 
