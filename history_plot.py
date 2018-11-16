"""
    keras 模型history的绘图
"""
import matplotlib.pyplot as pyplot


class History_plot:
    def loss_plot(self, history, val=False):
        pyplot.plot(history.history['loss'], label='train_loss')
        if val:
            pyplot.plot(history.history['val_loss'], label='test_loss')
            pyplot.plot(history.history['val_acc'], label='test_acc')
        pyplot.plot(history.history['acc'], label='train_acc')
        pyplot.legend()
        pyplot.show()
