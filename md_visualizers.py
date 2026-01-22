import random
import matplotlib.pyplot as plt


def visualize_results(df, resample_method='W', top_n_barcodes=5, kind='line'):
    random_barcode_sample = random.sample(list(df.barcode.unique()), top_n_barcodes)
    for barcode in random_barcode_sample:
        frag = df[df.barcode == barcode]
        fig = frag.resample(resample_method).agg({'actuals': sum, 'predictions': sum})
        fig.plot(kind=kind, figsize=(10, 6), title=f'BARCODE: {barcode}')
        plt.legend(loc='upper right')
        plt.show()
