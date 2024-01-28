import h5py

class Conductor:
    """Operate h5 files."""
    def __init__(self, file_path):
        self.file_path = file_path

    # Display the content of a specific dataset.
    def show_data(self, dataset_path):
        with h5py.File(self.file_path, 'r') as f:
            dataset = f[dataset_path][:]
        return dataset

    @property
    def get_structure_all(self):
        with h5py.File(self.file_path, 'r') as f:
            structure_all = []
            f.visit(lambda x: structure_all.append(x))
        return structure_all

    @property
    def get_structure_datasets(self):
        structure_all = self.get_structure_all
        with h5py.File(self.file_path, 'r') as f:
            structure_datasets = [path for path in structure_all if isinstance(f[path], h5py.Dataset)]
        return structure_datasets

    @property
    def get_structure_groups(self):
        structure_all = self.get_structure_all
        with h5py.File(self.file_path, 'r') as f:
            structure_groups = [path for path in structure_all if isinstance(f[path], h5py.Group)]
        return structure_groups


if __name__ == '__main__':
    pass
