import os
import torch
import matplotlib.pyplot as plt
from I_Model_Zoo.Models.ObjectDetection.Util.Utils import mAP, get_bboxes

# 기능 : 실시간 plt 업데이트, 학습 정보 저장
# 필요 기능 : 그래프 생성(O), 그래프 업데이트(O), 모델 저장(O), 학습 정보 저장(O), 결과 정보 저장(O), 그래프 이미지 저장(O)

class Draw_Graph():
    def __init__(self, model, save_path, patience):
        self.model = model

        self.patience = patience
        self.save_path = save_path

        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

        self.top_accuracy_train = 0
        self.top_accuracy_validation = 0
        self.top_accuracy_train_epoch = 0
        self.top_accuracy_validation_epoch = 0

        self.bottom_loss_train = float('inf')
        self.bottom_loss_validation = float('inf')
        self.bottom_loss_train_epoch = 0
        self.bottom_loss_validation_epoch = 0

        os.makedirs(save_path, exist_ok=True)

        # 그래프 생성(O)
        plt.ion()
        self.fig, self.axs = plt.subplots(1, 2, figsize=(16, 4))

    # 모델 저장(O)
    def save_model(self, save_type='Best_Accuracy_Train'):
        model_path_make = self.save_path + '//' + save_type + '.pth'
        torch.save(self.model.state_dict(), model_path_make)

    # 그래프 이미지 저장(O)
    def save_plt(self):
        plt.savefig(self.save_path + '//Graphs.png')

    # 학습 정보 저장(O)
    def save_train_info(self, patience_count):
        self.save_plt()
        with open(self.save_path + '//Trian_Info.txt', "w") as file:
            file.write(
                f"Top Accuracy Train Epoch : {self.top_accuracy_train_epoch} Accuracy : {self.top_accuracy_train}\n"
                f"Top Accuracy Validation Epoch : {self.top_accuracy_validation_epoch} Accuracy : {self.top_accuracy_validation}\n"
                f"Bottom Loss Train Epoch : {self.bottom_loss_train_epoch} Loss : {self.bottom_loss_train}\n"
                f"Bottom Loss Validation Epoch : {self.bottom_loss_validation_epoch} Loss : {self.bottom_loss_validation}\n"
                f"Patience Count : {patience_count}/{self.patience}\n")

    # 결과 정보 저장(O)
    def save_test_info(self, total, correct, accuracy):
        self.save_plt()
        with open(self.save_path + '//Test_Result.txt', "w") as file:
            file.write(f"Total Num : {total}, Correct Num : {correct}\n"
                       f"Accuracy : {accuracy}")

    # 그래프 업데이트(O)
    def update_graph(self, train_acc, train_loss, val_acc, val_loss, epoch, patience_count):

        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)

        if self.top_accuracy_train < train_acc:
            self.top_accuracy_train = train_acc
            self.top_accuracy_train_epoch = epoch
            self.save_model('Best_Accuracy_Train')

        if self.top_accuracy_validation < val_acc:
            self.top_accuracy_validation = val_acc
            self.top_accuracy_validation_epoch = epoch
            self.save_model('Best_Accuracy_Validation')

        if self.bottom_loss_train > train_loss:
            self.bottom_loss_train = train_loss
            self.bottom_loss_train_epoch = epoch
            self.save_model('Bottom_Loss_Train')

        if self.bottom_loss_validation > val_acc:
            self.bottom_loss_validation = val_acc
            self.bottom_loss_validation_epoch = epoch
            self.save_model('Bottom_Loss_Validation')

        self.axs[0].clear()
        self.axs[1].clear()

        epochs = range(1, len(self.train_accuracies) + 1)

        self.axs[0].plot(epochs, self.train_accuracies, label='Train Accuracy', color='red', linewidth=0.5)
        self.axs[0].plot(epochs, self.val_accuracies, label='Validation Accuracy', color='blue', linewidth=0.5)

        self.axs[0].set_xlabel('Epochs')
        self.axs[0].set_ylabel('Accuracy')
        self.axs[0].set_title('Training and Validation Accuracy')
        self.axs[0].legend()

        self.axs[1].plot(epochs, self.train_losses, label='Train Loss', color='red', linewidth=0.5)
        self.axs[1].plot(epochs, self.val_losses, label='Validation Loss', color='blue', linewidth=0.5)

        self.axs[1].set_xlabel('Epochs')
        self.axs[1].set_ylabel('Loss')
        self.axs[1].set_title('Training and Validation Loss')
        self.axs[1].legend()

        for tick in self.axs[0].get_xticks():
            self.axs[0].axvline(x=tick, color='gray', linestyle='-', linewidth=0.1)

        for tick in self.axs[0].get_yticks():
            self.axs[0].axhline(y=tick, color='gray', linestyle='-', linewidth=0.1)

        for tick in self.axs[1].get_xticks():
            self.axs[1].axvline(x=tick, color='gray', linestyle='-', linewidth=0.1)

        for tick in self.axs[1].get_yticks():
            self.axs[1].axhline(y=tick, color='gray', linestyle='-', linewidth=0.1)

        plt.draw()
        plt.pause(0.1)

        self.save_train_info(patience_count)
        self.save_plt()
