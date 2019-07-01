from src.data_load import DatasetLoader


loader = DatasetLoader('../data/test_split/', '../data/test_split/', '../data/test_split/')

loader.generate_test_split()

