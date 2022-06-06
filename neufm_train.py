import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm
from lightfm import LightFM
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


def add_categories(df):
    item_user = df[['id', 'id_book']]

    with pd.option_context('mode.chained_assignment', None):
        item_user.id = item_user.id.astype('category')
        item_user.id_book = item_user.id_book.astype('category')

        item_user['category_id'] = item_user.id.cat.codes
        item_user['category_book'] = item_user.id_book.cat.codes

    return item_user


def create_csr_matrix(matrix, users, items):
    n_users = users.max() + 1
    n_items = items.max() + 1
    matrix_shape = (n_users, n_items)

    data = np.ones(items.shape)
    user_item_matrix = csr_matrix((data, (users, items)), shape=matrix_shape)

    return user_item_matrix


def csr_row_set_nz_to_val(csr, row, value=0):
    if not isinstance(csr, csr_matrix):
        raise ValueError('Matrix given must be of CSR format.')
    csr.data[csr.indptr[row]:csr.indptr[row+1]] = value


def normalization_matrix(user_item_matrix):
    reader_normalized_matrix = normalize(user_item_matrix, axis=1, norm='l1')
    lazy_readers = np.array(user_item_matrix.sum(axis=1)).squeeze()
    lazy_readers = np.where(lazy_readers < 2)[0]
    for rId in lazy_readers:
        csr_row_set_nz_to_val(reader_normalized_matrix, rId)
    return reader_normalized_matrix


def normalize_matrix(train, val=None):
    users = train.category_id
    items = train.category_book

    csr_train = create_csr_matrix(train, users, items)
    csr_val = create_csr_matrix(val, users, items)

    normalization_train = normalization_matrix(csr_train)
    normalization_val = normalization_matrix(csr_val)

    return normalization_train, normalization_val


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(f'train shape:\t{train.shape}\ntest shape:\t{test.shape}')

categories = add_categories(train)

train = pd.merge(train, categories.drop_duplicates(),
                 left_on=['id', 'id_book'], right_on=['id', 'id_book'], how='inner').drop_duplicates()

test = test.merge(train[['id', 'category_id']].drop_duplicates(), on=['id'], how='left')
test = test.merge(train[['id_book', 'category_book']].drop_duplicates(), on=['id_book'], how='left')

# If test dataset contains books from train, add samples with them to train

test = test.dropna()
del_books = list(set(test.id_book.unique()) - set(train.id_book.unique()))

train = train.append(test[test.id_book.isin(del_books)], ignore_index=True)
train.fillna(100000000.0, inplace=True)

print('Processing test/train data...')

cnt = 1
max_items = train[train.category_book != 100000000.0].category_book.max()
for ind, val in tqdm(train[train.category_book == 100000000.0].iterrows()):
    train.loc[ind, 'category_book'] = max_items + cnt
    cnt += 1

normalization_train, normalization_test = normalize_matrix(train=train, val=test)
