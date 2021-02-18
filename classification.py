import pandas as pd


path = 'dataset\mpst_full_data.csv'

def import_dataset(path):
    #return pd.read_csv(path, names=['Title', 'Conditions', 'ONCOLOGY'], skiprows=1)
    return pd.read_csv(path)

if __name__ == '__main__':
    full_df = import_dataset(path)
    pd.set_option('max_columns', None)
    full_df.to_csv('dataset\\dataset.csv')

    train_df = pd.DataFrame(columns=['imdb_id', 'title','plot_synopsis', 'tags'])
    test_df = pd.DataFrame(columns=['imdb_id', 'title','plot_synopsis', 'tags'])
    val_df = pd.DataFrame(columns=['imdb_id', 'title','plot_synopsis', 'tags'])

    for index, row in full_df.iterrows():
        if row["split"] == "train":
            train_df = train_df.append({
                'imdb_id': full_df.loc[index, 'imdb_id'],
                 'title': full_df.loc[index, 'title'],
                 'plot_synopsis': full_df.loc[index, 'plot_synopsis'],
                 'tags': full_df.loc[index, 'tags']
                 }, ignore_index=True)
        elif row["split"] == "test":
            test_df = test_df.append({
                'imdb_id': full_df.loc[index, 'imdb_id'],
                 'title': full_df.loc[index, 'title'],
                 'plot_synopsis': full_df.loc[index, 'plot_synopsis'],
                 'tags': full_df.loc[index, 'tags']
                 }, ignore_index=True)
        elif row["split"] == "val":
            val_df = val_df.append({
                'imdb_id': full_df.loc[index, 'imdb_id'],
                'title': full_df.loc[index, 'title'],
                'plot_synopsis': full_df.loc[index, 'plot_synopsis'],
                'tags': full_df.loc[index, 'tags']
            }, ignore_index=True)

    train_df.to_csv('dataset\\train_dataset.csv')
    test_df.to_csv('dataset\\test_dataset.csv')
    val_df.to_csv('dataset\\validation_dataset.csv')