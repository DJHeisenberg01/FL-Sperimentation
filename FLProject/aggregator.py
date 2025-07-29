import time
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

from conv_models.convolutional_net import ConvolutionalNet
import os


class Aggregator:

    def __init__(self, config, logger):
        self.model = None
        self.config = config
        self.logger = logger
        self.model_name = self.config['model_name']

        self.current_weights = self.get_init_parameters()

        # for stats
        self.train_losses = []
        self.avg_test_losses = []
        self.avg_test_f1 = []
        self.avg_test_acc = []
        self.avg_test_prec = []
        self.avg_test_recall = []
        self.epoch_duration = []

        self.map_metric = {}

        # for convergence check
        self.prev_test_loss = None
        self.best_loss = None
        self.best_weight = None

        # Best Parameters
        self.best_round = -1
        self.best_f1 = 0
        self.best_acc = 0
        self.best_prec = 0
        self.best_recall = 0

        self.best_round_test = -1
        self.best_f1_test = 0
        self.best_acc_test = 0
        self.best_prec_test = 0
        self.best_recall_test = 0

        self.training_start_time = int(round(time.time()))

    def get_init_parameters(self):
        print(self.config['device'])
        model = ConvolutionalNet("", self.model_name, self.config['device'])
        parameters = model.get_weight()
        self.logger.info("parameters loaded ... delete the model")
        del model
        return parameters

    def update_weights_weighted(self, client_weights, client_sizes):
        total_size = np.sum(client_sizes)
        if not isinstance(client_weights[0], str):
            new_weights = [np.zeros(param.shape) for param in client_weights[0]]
        else:
            new_weights = [np.zeros(param.shape) for param in client_weights[1]]
        try:
            for c in range(len(client_weights)):
                for i in range(len(new_weights)):
                    if isinstance(client_weights[c][i], str):
                        total_size -= c
                        break
                    new_weights[i] += (client_weights[c][i] * client_sizes[c])

            for i in range(len(new_weights)):
                new_weights[i] = new_weights[i] / total_size
        except:
            self.logger.warn(type(client_sizes[c]))
            self.logger.warn(type(total_size))
            self.logger.warn(type(client_weights[c][i]))
        self.current_weights = new_weights

    def update_weights(self, client_weights):
        total_size = len(client_weights)
        new_weights = [np.zeros(param.shape) for param in client_weights[0]]
        for c in range(len(client_weights)):
            for i in range(len(new_weights)):
                new_weights[i] += (client_weights[c][i] / total_size)
        self.current_weights = new_weights

    def aggregate_metrics(self, client_losses, client_f1, client_acc, client_prec, client_recall):
        total_size = len(client_losses)
        # weighted sum
        aggr_loss = sum(client_losses[i] / total_size
                        for i in range(len(client_losses)))
        aggr_f1 = sum(client_f1[i] / total_size
                      for i in range(len(client_losses)))
        aggr_acc = sum(client_acc[i] / total_size
                       for i in range(len(client_losses)))
        aggr_prec = sum(client_prec[i] / total_size
                        for i in range(len(client_losses)))
        aggr_recall = sum(client_recall[i] / total_size
                          for i in range(len(client_losses)))
        return aggr_loss, aggr_f1, aggr_acc, aggr_prec, aggr_recall

    def aggregate_train_loss(self, client_losses, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        total_size = len(client_losses)
        # weighted sum
        aggr_loss = sum(client_losses[i] / total_size
                        for i in range(len(client_losses)))
        self.train_losses += [[cur_round, cur_time, aggr_loss]]
        self.map_metric[cur_round] = {'train_loss': aggr_loss}
        return aggr_loss

    def aggregate_train_loss_weights(self, client_losses, client_sizes, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        total_size = sum(client_sizes)
        # weighted sum
        aggr_loss = sum(client_losses[i] / total_size * client_sizes[i]
                        for i in range(len(client_sizes)))
        self.train_losses += [[cur_round, cur_time, aggr_loss]]
        self.map_metric[cur_round] = {'train_loss': aggr_loss}
        return aggr_loss

    def aggregate_evaluation_metrics(self, client_losses, client_f1, client_acc, client_prec, client_recall, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_f1, aggr_acc, aggr_prec, aggr_recall = self.aggregate_metrics(client_losses, client_f1,
                                                                                      client_acc, client_prec,
                                                                                      client_recall)

        self.avg_test_losses += [[cur_round, cur_time, aggr_loss]]
        self.avg_test_f1 += [[cur_round, cur_time, aggr_f1]]
        self.avg_test_acc += [[cur_round, cur_time, aggr_acc]]
        self.avg_test_prec += [[cur_round, cur_time, aggr_prec]]
        self.avg_test_recall += [[cur_round, cur_time, aggr_recall]]

        if len(self.epoch_duration) == 0:
            self.epoch_duration.append(cur_time)
        else:
            self.epoch_duration.append(cur_time - self.epoch_duration[-1])

        epoch_map = self.map_metric[cur_round]
        epoch_map['test_loss'] = aggr_loss
        epoch_map['test_f1'] = aggr_f1
        epoch_map['test_acc'] = aggr_acc
        epoch_map['test_prec'] = aggr_prec
        epoch_map['test_recall'] = aggr_recall
        epoch_map['time_duration'] = self.epoch_duration[-1]

        return aggr_loss, aggr_f1, aggr_acc, aggr_prec, aggr_recall

    def aggregate_metrics_weighted(self, client_losses, client_f1, client_acc, client_prec, client_recall,
                                   client_sizes):
        total_size = sum(client_sizes)
        # weighted sum
        aggr_loss = sum(client_losses[i] / total_size * client_sizes[i]
                        for i in range(len(client_sizes)))
        aggr_f1 = sum(client_f1[i] / total_size * client_sizes[i]
                      for i in range(len(client_sizes)))
        aggr_acc = sum(client_acc[i] / total_size * client_sizes[i]
                       for i in range(len(client_sizes)))
        aggr_prec = sum(client_prec[i] / total_size * client_sizes[i]
                        for i in range(len(client_sizes)))
        aggr_recall = sum(client_recall[i] / total_size * client_sizes[i]
                          for i in range(len(client_sizes)))
        return aggr_loss, aggr_f1, aggr_acc, aggr_prec, aggr_recall

    def aggregate_evaluation_metrics_weighted(self, client_losses, client_f1, client_acc, client_prec,
                                              client_recall, client_size, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_f1, aggr_acc, aggr_prec, aggr_recall = self.aggregate_metrics_weighted(client_losses, client_f1,
                                                                                               client_acc, client_prec,
                                                                                               client_recall,
                                                                                               client_size)

        self.avg_test_losses += [[cur_round, cur_time, aggr_loss]]
        self.avg_test_f1 += [[cur_round, cur_time, aggr_f1]]
        self.avg_test_acc += [[cur_round, cur_time, aggr_acc]]
        self.avg_test_prec += [[cur_round, cur_time, aggr_prec]]
        self.avg_test_recall += [[cur_round, cur_time, aggr_recall]]

        if len(self.epoch_duration) == 0:
            self.epoch_duration.append(cur_time)
        else:
            self.epoch_duration.append(cur_time - self.epoch_duration[-1])

        epoch_map = self.map_metric[cur_round]
        epoch_map['test_loss'] = aggr_loss
        epoch_map['test_f1'] = aggr_f1
        epoch_map['test_acc'] = aggr_acc
        epoch_map['test_prec'] = aggr_prec
        epoch_map['test_recall'] = aggr_recall
        epoch_map['time_duration'] = self.epoch_duration[-1]

        return aggr_loss, aggr_f1, aggr_acc, aggr_prec, aggr_recall

    def get_stats(self):
        if self.map_metric:
            return pd.DataFrame(self.map_metric).transpose()
        else:
            return pd.DataFrame(
                columns=['train_loss', 'test_loss', 'test_f1', 'test_acc', 'test_precision', 'test_recall'])

    def save_model(self):
        datestr = time.strftime('%d%m')
        timestr = time.strftime('%m%d%H%M')
        actual_path = os.path.dirname(os.path.abspath(__file__))
        output_path = self.config['model_output_path']
        save_path = os.path.join(actual_path, output_path, datestr)
        os.makedirs(save_path, exist_ok=True)

        model = ConvolutionalNet("", self.model_name, 'cpu')
        model.set_weight(self.current_weights)
        path = model.save_model(save_path + "\\" + timestr + "_")
        del model
        return path

    def save_df(self, config):
        datestr = time.strftime('%d%m')
        timestr = time.strftime('%m%d%H%M')
        # Df of The Federated Training
        df = pd.DataFrame(self.map_metric)
        output_path = self.config['csv_output_path']
        actual_path = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(actual_path, output_path, datestr)
        os.makedirs(save_path, exist_ok=True)
        df = df.transpose()
        save_path = save_path + "\\" + str(timestr) + '_federated_metrics.csv'
        df.to_csv(save_path)

        # Save the Model Generated by the Federated Training
        model_path = ""
        # model_path = self.save_model() # TODO RIATTIAVRE SALVATAGGIO MODELLLO DOPO GRID SEARCH

        # Save the Training Parameter to a Pandas DataFrame
        parameter_map = {'model_name': self.config['model_name'], 'global_epoch': self.config['global_epoch'],
                         'models_percentage': self.config['models_percentage'],
                         'MIN_NUM_WORKERS': self.config['MIN_NUM_WORKERS'], 'local_epoch': self.config['local_epoch'],
                         'batch_size': self.config['batch_size'], 'learning_rate': self.config['learning_rate'],
                         'best_epoch': self.best_round, 'best_f1': self.best_f1, 'best_acc': self.best_acc,
                         'best_prec': self.best_prec, 'best_recall': self.best_recall, 'best_loss': self.best_loss,

                         'best_f1_test': self.best_f1_test, 'best_acc_test': self.best_acc_test,
                         'best_recall_test': self.best_recall_test, 'best_precision_test': self.best_prec_test,
                         'best_round_test': self.best_round_test,

                         'total_duration': sum(self.epoch_duration), 'dataframe_path': save_path,
                         'model_path': model_path
                         }

        parameter_path = os.path.join(actual_path, output_path, "training_parameters.csv")

        df_parameter = pd.DataFrame(parameter_map, index=[0])
        if os.path.exists(parameter_path):
            df = pd.read_csv(parameter_path)
            df_parameter = pd.concat([df, df_parameter], )

        df_parameter.to_csv(parameter_path, index=False)

    def get_parameter_plots(self):
        plots = []
        loss_parameters = [[self.train_losses, "Train Loss"], [self.avg_test_losses, "Test Loss"]]
        metric_parameters = [[self.avg_test_f1, "F1 Score"], [self.avg_test_acc, "Accuracy Score"],
                             [self.avg_test_prec, "Precision Score"], [self.avg_test_recall, "Recall Score"]]

        os.makedirs("static/Image", exist_ok=True)

        plt.title("Train and Test Loss Function")
        for parameter, name in loss_parameters:
            values = [v[2] for v in parameter]
            plt.plot(values, label=name)
        path = os.path.join("static/Image", "Loss.png")
        plt.legend(loc="upper right")
        plt.savefig(path)
        plt.clf()
        plots.append(path)

        plt.title("Metric Functions")
        for parameter, name in metric_parameters:
            values = [v[2] for v in parameter]
            plt.plot(values, label=name)
        path = os.path.join("static/Image", "Metric.png")
        plt.legend(loc="upper right")
        plt.savefig(path)
        plt.clf()
        plots.append(path)

        return plots

    def test(self):
        if self.model is None:
            self.model = ConvolutionalNet("", self.model_name, self.config['device'])
        self.model.set_weight(self.current_weights)
        valid_loss, metric_score, time_tot, data_len = self.model.validate_path("test_images")
        return metric_score
