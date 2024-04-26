from utils.datahandler import DataHandler


datahandler = DataHandler()

datahandler.load_data()
datahandler.standarization()

df = datahandler.get_data()
print(df)