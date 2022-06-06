import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm

root = ET.parse('medlib.xml').getroot()

raw_samples = []
print("Processing xml file...")
for i in tqdm(range(len(root))):
    interaction_dict = {'id': str(root[i][0].text.replace('\n', '')), 'birth': str(root[i][1].text.replace('\n', '')),
                        'category_reader': str(root[i][2].text.replace('\n', '')),
                        'id_book': str(root[i][3].text.replace('\n', '')),
                        'name_book': str(root[i][4].text.replace('\n', '')),
                        'date_circulation': str(root[i][5].text.replace('\n', ''))}
    raw_samples.append(interaction_dict)

medlib_df = pd.DataFrame(raw_samples)

medlib_df.fillna('', inplace=True)

# Remove library visits
medlib_df.drop(medlib_df[medlib_df.id_book == ''].index, inplace=True)

# Remove spaces in book id
medlib_df['id_book'] = medlib_df['id_book'].apply(lambda x: x.replace(' ', '').replace('\n', '').replace('\t', ''))

# Remove spaces in category and book name and lowercase
medlib_df['category_reader'] = medlib_df['category_reader'].apply(lambda x: x.replace(' ', '').
                                                                  replace('\n', '').
                                                                  replace('\t', '').
                                                                  lower())
medlib_df['name_book'] = medlib_df['name_book'].apply(lambda x: x.replace(' ', '').
                                                      replace('\n', '').
                                                      replace('\t', '').
                                                      lower())

books_df = medlib_df.groupby(['id_book', 'name_book'])['id'].count().reset_index()

books_df['del'] = books_df['name_book'].apply(lambda x: 1 if str(x).find('посещение') != -1 else 0)
books_df = books_df[books_df['del'] != 1]
books_df.drop(['id', 'del'], inplace=True, axis=1)

books_df = books_df[books_df.name_book != ':']

print("Saving main dataframe as csv...")
medlib_df.to_csv('medlib.csv', index=False)

print("Saving books dataframe as csv...")
books_df.to_csv('books.csv', index=False)