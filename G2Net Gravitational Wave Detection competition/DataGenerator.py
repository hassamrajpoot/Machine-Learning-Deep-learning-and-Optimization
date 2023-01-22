class DataGenerator(Sequence):
    def __init__(self, path, list_IDs, data, batch_size):
        self.path = path
        self.list_IDs = list_IDs
        self.data = data
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.list_IDs))

    def __len__(self):
        len_ = int(len(self.list_IDs) / self.batch_size)
        if len_ * self.batch_size < len(self.list_IDs):
            len_ += 1
        return len_

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def __data_generation(self, list_IDs_temp):
        X = np.zeros((self.batch_size, 3, 4096))
        y = np.zeros((self.batch_size, 1))
        for i, ID in enumerate(list_IDs_temp):
            id_ = self.data.loc[ID, 'id']
            file = id_ + '.npy'
            path_in = '/'.join([self.path, id_[0], id_[1], id_[2]]) + '/'
            data_array = np.load(path_in + file)
            data_array = (data_array - data_array.mean()) / data_array.std()
            X[i,] = data_array
            y[i,] = self.data.loc[ID, 'target']
        return X, y