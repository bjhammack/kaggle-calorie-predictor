import pandas as pd


class DataObject(pd.DataFrame):
    @classmethod
    def from_csv(cls, file_path, **kwargs):
        df = pd.read_csv(file_path, **kwargs)
        return cls(df)

    def __init__(self, file_path, *args, **kwargs):
        df = pd.read_csv(file_path, **kwargs)
        super().__init__(df)
        self._enhance_data()

    def _enhance_data(self):
        if "Weight" in self.columns and "Height" in self.columns:
            self["BMI"] = self["Weight"] / ((self["Height"] / 100) ** 2)
        if "Heart_Rate" in self.columns and "Duration" in self.columns:
            self["Intensity"] = self["Heart_Rate"] / self["Duration"]
        # if "Body_Temp" in self.columns:
        #     self["Body_Temp_Delta"] = self["Body_Temp"] - self["Body_Temp"].iloc[0]
        if "Sex" in self.columns:
            self["Sex"] = self["Sex"].map({"male": 0, "female": 1})
