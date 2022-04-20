from deepts_forecasting.models.base_model import BaseModel


class mymodel(BaseModel):
    def __init__(self, a, b, c):
        super().__init__()
        self.save_hyperparameters()

    def output(self):
        # print(self.b)
        # print(self.c)
        print(dir(self))


model = mymodel(a=1, b=2, c=3)

model.output()
