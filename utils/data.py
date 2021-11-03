import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['axes.formatter.offset_threshold'] = 2

class Data:
    
    def __init__(self, path, test=False):
        
        self.path = path
        self.df = pd.read_csv(path)
        self.column_names = list(self.df.columns)

        if not test:
            self.labels = self.df["Label"]
        else:
            self.labels = None
        
        self.vif = None

        self.num_bound_col_names = ["danceability", "energy", "speechiness","acousticness",
                              "instrumentalness", "liveness", "valence"]
        
        self.num_unbound_col_names = ["loudness", "tempo"]
        
        self.class_col_names = ["key"]
        self.binary_col_names = ["mode"]

        self.preprocessed = False

        self.bound_preprocessed = False
        self.unbound_preprocessed = False
        self.class_preprocessed  = False
        self.binary_preprocessed = False
        self.is_data_shuffled = False
        self.are_duplicates_removed = False

        pass
    

    def __str__(self):
        return str(self.path)
    
    
    def _shuffle(self):
        
        n_data = len(self.df)
        
        idx = np.random.permutation(n_data)
        self.df, self.labels = self.df.reindex(idx), self.labels.reindex(idx)
        
        self.is_data_shuffled = True


    def _split_df(self):

        assert self.preprocessed == False
        
        assert(len( self.num_bound_col_names) + len(self.num_unbound_col_names) + 
               len(self.class_col_names) + len(self.binary_col_names) + 1 == len(self.column_names))
               
        self.num_bound_col = self.df[self.num_bound_col_names]
        self.num_unbound_col = self.df[self.num_unbound_col_names]
        self.class_col = self.df[self.class_col_names]
        self.binary_col = self.df[self.binary_col_names]
        pass

        
    def _normalize_num_bound_col(self):
        
        self.mean_num_bound_col = self.num_bound_col.mean()
        self.std_num_bound_col = self.num_bound_col.std()
        
        self.num_bound_col = (self.num_bound_col - self.mean_num_bound_col) / self.std_num_bound_col

        self.bound_preprocessed = True
        pass
        
    def _normalize_num_unbound_col(self):

        self.mean_num_unbound_col = self.num_unbound_col.mean()
        self.std_num_unbound_col = self.num_unbound_col.std()

        self.num_unbound_col = (self.num_unbound_col - self.mean_num_unbound_col) / self.std_num_unbound_col

        self.unbound_preprocessed = True
        pass
        
    def _preprocess_class_col(self):
        
        self.class_col = pd.get_dummies(self.class_col["key"], prefix='key')

        self.class_preprocessed = True
        pass
    
    def _normalize_binary(self):
        
        self.mean_binary_col = self.binary_col.mean()
        self.std_binary_col = self.binary_col.std()
        
        self.binary_col = (self.binary_col - self.mean_binary_col) / self.std_binary_col

        self.binary_preprocessed = True
        pass
        
    pass


    def _append_cols(self):

        self.df = pd.concat([self.num_bound_col, self.num_unbound_col, self.class_col, self.binary_col], axis=1)
        
        
    def _get_vif(self):
        
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        vif = pd.DataFrame()
        vif["variables"] = self.df.columns
        vif["VIF"] = [variance_inflation_factor(self.df.values, i) for i in range(self.df.shape[1])]

        self.vif = vif
        

    def preprocess(self, bound_bool = True, unbound_bool = True, class_bool = True, binary_bool = False,
                    shuffle=True, remove_duplicates=True, vif=True):

        """
        Booleans stand for normalising the data types or not normalising the data types
        """

        self._split_df()

        if bound_bool : self._normalize_num_bound_col()
        if unbound_bool : self._normalize_num_unbound_col()
        if class_bool : self._preprocess_class_col()
        if binary_bool : self._normalize_binary()
        self._append_cols()     
        
        if remove_duplicates : self._remove_duplicates()
        if shuffle : self._shuffle()
        
        if vif : self._get_vif()
        
        self.preprocessed = True

        pass


    def _remove_duplicates(self):
        duplicated = self.df.duplicated()
        n_duplicated = np.sum(duplicated)

        idx = np.where(duplicated==True)[0]

        self.df = self.df.drop(idx).reset_index(drop=True)
        self.labels = self.labels.drop(idx).reset_index(drop=True)
        
        self.are_duplicates_removed = True

        print(f"There were {n_duplicated} duplicated elements in the dataset, and have been removed from the dataframe")
        

    
    def visualize(self, cmap=None, labels=None, diagonal="hist"):

        from matplotlib import colors

        try:
            if labels==None:
                labels = self.labels
        except: pass

        try:
            if cmap==None:
                cmap = colors.ListedColormap(['r', 'b'], 2)
        except: pass

        plotting_features = self.num_bound_col_names + self.num_unbound_col_names
        plotting_df = self.df[plotting_features]
        sc_matrix = pd.plotting.scatter_matrix(plotting_df, alpha=0.4, figsize=(18,14), c=labels, cmap=cmap, diagonal=diagonal, range_padding=0.1)

        plt.show()
        pass
    
    def preprocess_new_data(self, test_data):
        
        num_bound_col = test_data.df[self.num_bound_col_names]
        num_unbound_col = test_data.df[self.num_unbound_col_names]
        class_col = test_data.df[self.class_col_names]
        binary_col = test_data.df[self.binary_col_names]
        
        class_col = pd.get_dummies(class_col["key"], prefix='key')
        
        if self.bound_preprocessed:
            num_bound_col = (num_bound_col - self.mean_num_bound_col)/self.std_num_bound_col
        if self.unbound_preprocessed: 
            num_unbound_col = (num_unbound_col - self.mean_num_unbound_col)/self.std_num_unbound_col
            
        return pd.concat([num_bound_col, num_unbound_col, class_col, binary_col], axis=1)


