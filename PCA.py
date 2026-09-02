import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def principal_component_analysis():
    data = pd.read_parquet('SIMA_Diario_Imputado.parquet')

    data = data.select_dtypes(include=['float64'])

    pca = PCA(n_components=3)
    pca.fit(data)

    X_pc = pca.transform(data)

    n_pcs= pca.components_.shape[0]
    print(n_pcs)

    # get the index of the most important feature on EACH component
    # LIST COMPREHENSION HERE
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

    initial_feature_names = [data.columns]
    # get the names
    most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs-1)]

    dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}

# build the dataframe
    df = pd.DataFrame(dic.items())

    print(f'Varianza explicada por cada componente {pca.explained_variance_ratio_}')
    print(f'Valores de cada uno: {pca.singular_values_}')
    print(f'')

def main():
    principal_component_analysis()

if __name__ == '__main__':
    main()