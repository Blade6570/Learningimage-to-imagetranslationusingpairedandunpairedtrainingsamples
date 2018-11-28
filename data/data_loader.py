
def CreateDataLoader(opt,k):
    if k==0:
       from data.custom_dataset_data_loader import CustomDatasetDataLoader
       data_loader = CustomDatasetDataLoader()
       print(data_loader.name())
       data_loader.initialize(opt)
       return data_loader
    else:
       from data.custom_dataset_data_loader_super import CustomDatasetDataLoader_super  
       data_loader = CustomDatasetDataLoader_super()
       print(data_loader.name())
       data_loader.initialize(opt)
       return data_loader
