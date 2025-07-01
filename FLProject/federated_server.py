import logging
import random
import time
from flask import Flask, request, render_template
from flask_socketio import *
from flask_socketio import SocketIO
import json
import os
import sys
from utilities import obj_to_pickle_string, pickle_string_to_obj
from aggregator import Aggregator

CONFIG_FILE = 'cfg/config.json'


def load_json(filename):
    with open(filename) as f:
        return json.load(f)


class FederatedServer(object):

    def __init__(self, config_file):
        '''
        Load with json parameters from a config file
        Params: ip_address, port, model_name, log_filename, global_epoch, models_percentage
                local_epochs, learning_rate, batch_size
        '''
        self.config = load_json(config_file)

        self.MIN_NUM_WORKERS = self.config['MIN_NUM_WORKERS']
        self.NUM_CLIENTS_CONTACTED_PER_ROUND = 0
        self.registered_clients = set()
        self.client_resource = {}
        self.current_round = -1  # -1 for not yet started
        self.current_round_client_updates = []
        self.eval_client_updates = []
        self.STOP = False
        self.wait_time = 0

        # Set Logger
        self.logger = logging.getLogger("Federated-Server")
        self.set_logger()
        self.logger.info("Config Params: " + str(self.config))
        #
        self.aggregator = Aggregator(self.config, self.logger)

        self.early_stop_tolerance = 0

        # Flask Parameter Configuration
        self.app = Flask(__name__)
        # self.app.debug = True
        self.socketio = SocketIO(self.app, ping_timeout=3600,
                                 ping_interval=3600,
                                 max_http_buffer_size=int(1e32))
        self.register_handles()

        @self.app.route('/')
        def status_page():
            df = self.aggregator.get_stats()
            images = self.aggregator.get_parameter_plots()
            texts = ["Number of Clients Connected: " + str(len(self.registered_clients)),
                     "====== Training Variable ======", "Model Name: " + str(self.config['model_name']),
                     "Global Epochs: " + str(self.config['global_epoch']),
                     "Learning Rate: " + str(self.config['learning_rate']),
                     "Batch Size: " + str(self.config['batch_size']),
                     "Local Epochs: " + str(self.config['local_epoch'])]
            return render_template('stats.html', tables=[df.to_html(classes='data')],
                                   texts=texts, images=images)

    def check_client_resource(self):
        """
        A random number of clients is selected between the number of total clients and registered clients
        from which to request availability.
        """
        self.client_resource = {}
        client_sids_selected = random.sample(list(self.registered_clients), int(self.NUM_CLIENTS_CONTACTED_PER_ROUND))
        self.logger.info('Sending weights to selected clients: ' + str(client_sids_selected))
        for rid in client_sids_selected:
            emit('check_client_resource', {
                'round_number': self.current_round,
            }, room=rid)

        # Note: we assume that during training the #workers will be >= MIN_NUM_WORKERS

    def train_next_round(self, client_sids_selected):
        """
        Selected clients are prompted for a model update. They are sent training parameters such as epochs,
        batch_size, learning_rate, round_number and current_weights.
        """
        self.current_round += 1
        # buffers all client updates
        self.current_round_client_updates = []
        self.logger.info("### Round {} ###".format(self.current_round))
        self.logger.info("Requesting updates from {}".format(client_sids_selected))
        current_weights = obj_to_pickle_string(self.aggregator.current_weights)
        for rid in client_sids_selected:
            emit('request_update', {
                'epochs': self.config['local_epoch'],
                'batch_size': self.config['batch_size'],
                'learning_rate': self.config['learning_rate'],
                'round_number': self.current_round,
                'current_weights': current_weights,
            }, room=rid)
            self.logger.info("Sent the model to {}".format(rid))

    def stop_and_eval(self):
        """
        The model is sent to clients for validation and they are notified whether or not it has reached a stopped state.
        """
        current_weights = obj_to_pickle_string(self.aggregator.current_weights)
        self.eval_client_updates = []
        for rid in self.registered_clients:
            emit('stop_and_eval', {
                'batch_size': self.config['batch_size'],
                'current_weights': current_weights,
                'STOP': self.STOP
            }, room=rid)
        self.logger.info("Sent the aggregated model to every client registered to validate it")

    def set_logger(self):
        datestr = time.strftime('%d%m')
        timestr = time.strftime('%m%d%H%M')
        self.logger.setLevel(logging.INFO)
        log_dir = os.path.join("logs", datestr, "FL-Server-LOG")
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, '{}.log'.format(timestr)))
        fh.setLevel(logging.INFO)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARN)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def run(self):
        self.socketio.run(self.app, host=self.config['ip_address'], port=self.config['port'], )
        self.logger.info("Federated Server Started")

    def register_handles(self):
        # single-threaded async, no need to lock

        @self.socketio.on('connect')
        def handle_connect():
            self.logger.info("Client Sid:" + str(request.sid) + "connected")

        @self.socketio.on('reconnect')
        def handle_reconnect():
            self.logger.info("Client Sid:" + str(request.sid) + "reconnected")

        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.logger.info("Client Sid:" + str(request.sid) + "disconnected")
            if request.sid in self.registered_clients:
                self.registered_clients.remove(request.sid)
                self.refresh_client_round()

        @self.socketio.on('client_wake_up')
        def handle_wake_up():
            """
            A wake-up signal is received from the various clients. The server initializes the
            clients for each signal received.
            """
            self.logger.info("Sending client wake_up to " + str(request.sid))
            emit('init')

        @self.socketio.on('client_ready')
        def handle_client_ready():
            """
            All clients ready for new training are reported. The ready clients are added to the clients
            registered by the server. When the threshold of ready clients is reached, an initial training
            of the model is started by exploiting some of the clients.
            """
            self.logger.info("The client " + str(request.sid) + " is ready for training")
            self.registered_clients.add(request.sid)
            self.refresh_client_round()

            # TODO CAMBIARE CONDIZIONE DI AVVIO PROCESSO DI TRAINING

            if len(self.registered_clients) >= self.MIN_NUM_WORKERS and self.current_round == -1:
                self.logger.info("The Federated Process is Starting")
                self.check_client_resource()
            elif len(self.registered_clients) < self.MIN_NUM_WORKERS:
                self.logger.info("Waiting for clients to start the process")
            else:
                self.logger.error("The current_round is not equal to -1, please restart server.")

        @self.socketio.on('check_client_resource_done')
        def handle_check_client_resource_done(data):
            """
            Each client is initialized in a dict with the load amount. If there are enough initialized clients,
            part of them are selected based on the total load amount available. If the percentage of available
            clients exceeds a certain threshold, training is started for the next round.
            """
            if data['round_number'] == self.current_round:
                self.client_resource[request.sid] = data['load_rate']
                if len(self.client_resource) == self.NUM_CLIENTS_CONTACTED_PER_ROUND:
                    satisfy = 0
                    client_sids_selected = []
                    for client_id, val in self.client_resource.items():
                        self.logger.info(str(client_id) + "cpu rate: " + str(val))
                        if float(val) < 0.4:
                            client_sids_selected.append(client_id)
                            self.logger.info(str(client_id) + "satisfy")
                            satisfy = satisfy + 1
                        else:
                            self.logger.warning(str(client_id) + "reject")

                    if satisfy / len(self.client_resource) > 0.5:
                        self.wait_time = min(self.wait_time, 3)
                        time.sleep(self.wait_time)
                        self.train_next_round(client_sids_selected)
                    else:
                        if self.wait_time < 10:
                            self.wait_time = self.wait_time + 1
                        time.sleep(self.wait_time)
                        self.check_client_resource()

        @self.socketio.on('client_update')
        def handle_client_update(data):
            """
            The server receives the updates processed by the clients. Model results are aggregated. The learning
            stop for a given threshold reached or number of epochs is evaluated, the model is evaluated against
            the others received, and is sent back to the clients for validation.
            """
            self.logger.info("received client update of bytes: {}".format(sys.getsizeof(data)))
            self.logger.info("handle client_update {}".format(request.sid))

            if data['round_number'] == self.current_round:
                self.current_round_client_updates += [data]
                self.current_round_client_updates[-1]['weights'] = pickle_string_to_obj(data['weights'])
                if len(self.current_round_client_updates) == self.NUM_CLIENTS_CONTACTED_PER_ROUND:

                    aggr_train_loss = 0
                    if self.config['weighted_aggregation']:
                        aggr_train_loss = self.aggregate_with_sizes()
                    else:
                        aggr_train_loss = self.aggregate_without_sizes()

                    self.logger.info("=== training ===")
                    self.logger.info("aggr_train_loss {}".format(aggr_train_loss))

                    # TODO CONDIZIONE DI STOP APPRENDIMENTO PER SOGLIA

                    if self.current_round >= self.config['global_epoch'] - 1:
                        self.logger.info("Reached the Maximum number of global epochs, the process is stopping..")
                        print("FINISHING FEDERATED PROCESS...")
                        self.STOP = True

                    self.test_on_selected()

                    self.stop_and_eval()

        @self.socketio.on('client_eval')
        def handle_client_eval(data):
            """
            The different validations are received from the server. The results from all clients are awaited, the
            weights are aggregated, the results modified if better than the previous ones, and it is considered
            whether to stop training or start a new iteration.
            """
            if self.eval_client_updates is None:
                return
            self.logger.info("handle client_eval {}".format(request.sid))
            self.eval_client_updates += [data]

            if len(self.eval_client_updates) == len(self.registered_clients):  # self.NUM_CLIENTS_CONTACTED_PER_ROUND:

                if self.config['weighted_aggregation']:
                    test_loss, test_f1, test_acc, test_prec, test_recall = (
                        self.aggregator.aggregate_evaluation_metrics_weighted(
                        [update['test_loss'] for update in self.eval_client_updates],
                        [update['test_f1'] for update in self.eval_client_updates],
                        [update['test_acc'] for update in self.eval_client_updates],
                        [update['test_prec'] for update in self.eval_client_updates],
                        [update['test_recall'] for update in self.eval_client_updates],
                        [update['test_size'] for update in self.eval_client_updates],
                        self.current_round,
                    ))
                else:
                    test_loss, test_f1, test_acc, test_prec, test_recall = self.aggregator.aggregate_evaluation_metrics(
                        [update['test_loss'] for update in self.eval_client_updates],
                        [update['test_f1'] for update in self.eval_client_updates],
                        [update['test_acc'] for update in self.eval_client_updates],
                        [update['test_prec'] for update in self.eval_client_updates],
                        [update['test_recall'] for update in self.eval_client_updates],
                        self.current_round
                    )

                self.logger.info("=== server test ===")
                self.logger.info("test_loss {}".format(test_loss))
                self.logger.info("test_f1 {}".format(test_f1))
                self.logger.info("test_acc {}".format(test_acc))
                self.logger.info("test_prec {}".format(test_prec))
                self.logger.info("test_recall {}".format(test_recall))

                if self.aggregator.prev_test_loss is not None and test_loss > self.aggregator.prev_test_loss:
                    self.early_stop_tolerance += 1

                if self.early_stop_tolerance == self.config['early_stop_patience']:
                    self.STOP = True
                    self.logger.info("Early stopping of the Process")
                    print("FINISHING FEDERATED PROCESS FOR EARLY STOPPING...")

                self.aggregator.prev_test_loss = test_loss

                if self.aggregator.best_f1 <= test_f1:
                    self.aggregator.best_f1 = test_f1
                    self.aggregator.best_loss = test_loss
                    self.aggregator.best_acc = test_acc
                    self.aggregator.best_prec = test_prec
                    self.aggregator.best_recall = test_recall
                    self.aggregator.best_weight = self.aggregator.current_weights
                    self.aggregator.best_round = self.current_round
                if self.STOP:
                    self.logger.info("== done ==")
                    self.eval_client_updates = None  # special value, forbid evaling again
                    self.logger.info("Federated training finished ... ")
                    self.logger.info("best model at round {}".format(self.aggregator.best_round))
                    self.logger.info("get best test loss {}".format(self.aggregator.best_loss))
                    self.logger.info("get best f1 {}".format(self.aggregator.best_f1))
                    self.logger.info("get best acc {}".format(self.aggregator.best_acc))
                    self.logger.info("get best precision {}".format(self.aggregator.best_prec))
                    self.logger.info("get best recall {}".format(self.aggregator.best_recall))
                    self.aggregator.save_df(self.config)
                    print("Federated Process ended, saved df of parameters")
                    for client in self.registered_clients:
                        emit("shutdown", room=client)
                    self.shutdown_server()
                else:
                    self.logger.info("start to next round...")
                    self.check_client_resource()

    def refresh_client_round(self):
        self.NUM_CLIENTS_CONTACTED_PER_ROUND = len(self.registered_clients) * self.config['models_percentage']

    def aggregate_with_sizes(self):
        self.aggregator.update_weights_weighted(
            [x['weights'] for x in self.current_round_client_updates],
            [x['train_size'] for x in self.current_round_client_updates]
        )
        aggr_train_loss = self.aggregator.aggregate_train_loss_weights(
            [x['train_loss'] for x in self.current_round_client_updates],
            [x['train_size'] for x in self.current_round_client_updates],
            self.current_round
        )
        return aggr_train_loss

    def aggregate_without_sizes(self):
        self.aggregator.update_weights(
            [x['weights'] for x in self.current_round_client_updates],
        )
        aggr_train_loss = self.aggregator.aggregate_train_loss(
            [x['train_loss'] for x in self.current_round_client_updates],
            self.current_round
        )
        return aggr_train_loss

    def shutdown_server(self):
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            raise RuntimeError('Not running with the Werkzeug Server')
        func()

    def test_on_selected(self):
        metric_score = self.aggregator.test()

        epoch_map = self.aggregator.map_metric[self.current_round]
        epoch_map['f1_score_test_yolo'] = metric_score['f1_score']
        epoch_map['accuracy_test_yolo'] = metric_score['accuracy']
        epoch_map['precision_test_yolo'] = metric_score['precision']
        epoch_map['recall_test_yolo'] = metric_score['recall']

        if self.aggregator.best_round_test == -1 or self.aggregator.best_f1_test < metric_score['f1_score']:
            self.aggregator.best_f1_test = metric_score['f1_score']
            self.aggregator.best_acc_test = metric_score['accuracy']
            self.aggregator.best_prec_test = metric_score['precision']
            self.aggregator.best_recall_test = metric_score['recall']
            self.aggregator.best_round_test = self.current_round


if __name__ == '__main__':
    server = FederatedServer(CONFIG_FILE)
    server.run()
