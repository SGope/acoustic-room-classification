import pandas as pd
import os

class Prep():
    def convert(self, file):
        """"""
        print(file)
        path = "/home/issac/PycharmProjects/room_classification/dataset_for_students/Parameter_csv/Parameter_csv/"
        df = pd.read_csv(os.path.join(path, file))


    def get_files(self):
        path = "/home/issac/PycharmProjects/room_classification/dataset_for_students/Parameter_csv/Parameter_csv/"
        filenames = [file for file in os.listdir(path) if file.endswith(".csv")]

        for file in sorted(filenames):
            self.convert(file)
        print("Csv files have been saved")

def main():
    x = Prep()
    x.get_files()

if __name__ == "__main__":
    main()
