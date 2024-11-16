import datasets
import psutil

from masterthesis.data import get_dataset


if __name__ == "__main__":
    datasets.disable_caching()
    ds = get_dataset("cord-19", "sshleifer/distilbart-cnn-12-6", 1024)
    print(ds["train"].features)
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    ds.set_format("pandas")
    df_train = ds["train"][:]
    #print(df_train.loc[df_train["labels"] != None, ["labels"]])
    print(ds['train'][:][[ 'text_length', 'abstract_length']].describe())
    print(ds['val'][:][[ 'text_length', 'abstract_length']].describe())
    print(ds['test'][:][[ 'text_length', 'abstract_length']].describe())
    #print(f"Number of files in dataset : {ds.cache_files}")