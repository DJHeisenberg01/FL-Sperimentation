import logging
import time
import os
import socketio
import json
from conv_models.convolutional_net import ConvolutionalNet
from utilities import obj_to_pickle_string, pickle_string_to_obj

CONFIG_FILE = 'cfg/config.json'


def load_json(filename):
    with open(filename) as f:
        return json.load(f)


class FederatedClient(object):

    def __init__(self, config_file, dataset_path, client_id):
        '''
        Load with json parameters from a config file
        Params: ip_address, port, model_name, log_filename, global_epoch, models_percentage
                local_epochs, learning_rate, batch_size
        '''
        self.local_model = None
        self.config = load_json(config_file)
        self.dataset_path = dataset_path
        self.client_id = client_id

        # Set The Logger
        self.logger = logging.getLogger("Federated-Client")
        logging.getLogger("socketio").setLevel(logging.WARN)
        self.set_logger()

        self.sio = socketio.Client(logger=True, request_timeout=10, reconnection=True)  #engineio_logger=True
        self.sio.connect("http://" + self.config['ip_address'] + ":" + str(self.config['port']), )
        self.register_handles()
        self.logger.info("Sending Wake Up to the server")
        self.sio.emit('client_wake_up')
        self.sio.wait()

    def set_logger(self):
        datestr = time.strftime('%d%m')
        timestr = time.strftime('%m%d%H%M')
        self.logger.setLevel(logging.INFO)
        log_dir = os.path.join("logs", datestr, "FL-Client-LOG")
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, '{}.log'.format(timestr)))
        fh.setLevel(logging.INFO)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARN)
        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            f'%(asctime)s - %(name)s - %(levelname)s - ClientID-{self.client_id} - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def on_init(self):
        """
        The neural network is initialized and the server is notified that the client is ready for new training
        """
        self.logger.info('Received on init from server')
        self.local_model = ConvolutionalNet(self.dataset_path, self.config['model_name'], self.config['device'])
        self.logger.info("Local Model Initialized!")
        self.sio.emit('client_ready')

    def register_handles(self):
        ########## Socket IO messaging ##########

        def on_shutdown():
            self.logger.info('shutdown')
            self.sio.disconnect()

        def on_connect():
            self.logger.info('connect')

        def on_disconnect():
            self.logger.info('disconnect')

        def on_reconnect():
            self.logger.info('reconnect')

        def on_request_update(*args):
            """
            An update to the client is requested. The client loads the received model, trains it on the available data,
            and returns the obtained results such as training statistics and weights to the server.
            """
            req = args[0]
            self.logger.info("update requested")

            cur_round = req['round_number']
            self.logger.info("### Round {} ###".format(cur_round))

            self.logger.info("Received model from Server, Loading it..")
            weights = pickle_string_to_obj(req['current_weights'])
            self.local_model.set_weight(weights)

            epochs = req['epochs']
            lr = req['learning_rate']
            batch_size = req['batch_size']

            time_tot, train_map, train_loss, train_size = self.local_model.train(epochs=epochs, lr=lr,
                                                                                 batch_size=batch_size)
            my_weights = self.local_model.get_weight()
            pickle_string_weights = obj_to_pickle_string(my_weights)

            # Computation of the Train Metrics
            list_f1 = train_map['f1_score']
            list_acc = train_map['accuracy_score']
            avg_f1 = sum(list_f1) / len(list_f1)
            avg_acc = sum(list_acc) / len(list_acc)

            resp = {
                'round_number': cur_round,
                'weights': pickle_string_weights,
                'train_loss': train_loss,
                'avg_f1': avg_f1,
                'avg_acc': avg_acc,
                'train_size': train_size
            }

            self.logger.info("===================================")
            self.logger.info("client_train_loss {}".format(train_loss))
            self.logger.info("Average F1 Score {}".format(avg_f1))
            self.logger.info("Average Acc Score {}".format(avg_acc))
            self.logger.info("Time Tot of Training {}".format(time_tot))
            self.logger.info("===================================")

            self.sio.emit('client_update', resp)
            self.logger.info("Sent the Trained model and Metrics to the server")

        def on_stop_and_eval(*args):
            """
            Through this function, the client receives the model to be validated, validates it, and returns
            the validation result to the server
            """
            self.logger.info("Received aggregated model from server to evaluate")
            req = args[0]
            cur_time = time.time()
            weights = pickle_string_to_obj(req['current_weights'])
            self.local_model.set_weight(weights)

            self.logger.info("reciving weight time is {}".format(time.time() - cur_time))
            valid_loss, metric_score, time_tot, test_size = self.local_model.validate(req['batch_size'])
            resp = {
                'test_loss': valid_loss,
                'test_f1': metric_score['f1_score'],
                'test_acc': metric_score['accuracy'],
                'test_prec': metric_score['precision'],
                'test_recall': metric_score['recall'],
                'test_size': test_size,
                'time_tot': time_tot,
            }
            self.logger.info("Sending evaluation to the server..")
            self.sio.emit('client_eval', resp)

            if req['STOP']:
                self.logger.info("Federated training finished ...")
                exit(0)

        def on_check_client_resource(*args):
            """
            The client reports the current training round and the amount of load to the server.
            """
            req = args[0]
            self.logger.info("check client resource.")
            load_average = 0.15

            resp = {
                'round_number': req['round_number'],
                'load_rate': load_average
            }
            self.sio.emit('check_client_resource_done', resp)

        self.sio.on('shutdown', on_shutdown)
        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('reconnect', on_reconnect)
        self.sio.on('init', self.on_init)
        self.sio.on('request_update', on_request_update)
        self.sio.on('stop_and_eval', on_stop_and_eval)
        self.sio.on('check_client_resource', on_check_client_resource)

if __name__ == "__main__":
    client = FederatedClient(CONFIG_FILE,"dataset/Clients/client_0")
