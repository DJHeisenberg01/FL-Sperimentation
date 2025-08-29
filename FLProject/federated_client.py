import logging
import time
import os
import socketio
import json
import numpy as np
from conv_models.convolutional_net import ConvolutionalNet
from utilities import obj_to_pickle_string, pickle_string_to_obj
from client_resources import ClientResources, ResourceManager

CONFIG_FILE = 'cfg/config.json'

def calculate_proximal_term(current_weights, global_weights, mu):
    """Calculate the proximal term for FedProx using numpy"""
    proximal_loss = 0
    for w, w_t in zip(current_weights, global_weights):
        # Calculate L2 norm squared
        proximal_loss += (mu / 2) * np.sum((w - w_t) ** 2)
    return proximal_loss

def load_json(filename):
    with open(filename) as f:
        return json.load(f)

class FederatedClient(object):

    def __init__(self, config_file, dataset_path, client_resources=None):
        '''
        Load with json parameters from a config file
        Params: ip_address, port, model_name, log_filename, global_epoch, models_percentage
                local_epochs, learning_rate, batch_size
        client_resources: Optional ClientResources object for simulation
        '''
        self.local_model = None
        self.config = load_json(config_file)
        self.dataset_path = dataset_path
        
        # Initialize client resources
        if client_resources is None:
            # Use default resources (backward compatibility)
            self.client_resources = ClientResources(
                compute_power=1.0,
                bandwidth=5.0,
                reliability=1.0
            )
        else:
            self.client_resources = client_resources

        # Set The Logger - Clear existing handlers to prevent duplication
        # Derive client name from dataset path if not in config
        if 'client_name' not in self.config:
            self.config['client_name'] = os.path.basename(dataset_path) or f"client_{id(self)}"
        
        logger_name = f"Federated-Client-{self.config['client_name']}"
        self.logger = logging.getLogger(logger_name)
        
        # Clear existing handlers to prevent log duplication
        if self.logger.handlers:
            self.logger.handlers.clear()
            
        logging.getLogger("socketio").setLevel(logging.WARN)
        self.set_logger()
        
        # Log client type and resources at startup
        client_category = self.client_resources.get_client_category()
        self.logger.info(f"[CLIENT] INITIALIZED: {self.client_resources.get_detailed_description()}")
        self.logger.info(f"[DATA] Dataset path: {dataset_path}")

        self.sio = socketio.Client(logger=True, request_timeout=60, reconnection=True)  #engineio_logger=True
        try:
            self.sio.connect("http://" + self.config['ip_address'] + ":" + str(self.config['port']), )
            self.register_handles()
            self.logger.info("Sending Wake Up to the server")
            self.sio.emit('client_wake_up')
        except Exception as e:
            self.logger.error(f"Failed to connect to server: {str(e)}")
            raise
        self.sio.wait()

    def set_logger(self):
        datestr = time.strftime('%d%m')
        timestr = time.strftime('%m%d%H')  # Removed minutes to keep same hour clients together
        self.logger.setLevel(logging.INFO)
        log_dir = os.path.join("logs", datestr, "FL-Client-LOG")
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

    def on_init(self):
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
            req = args[0]
            self.logger.info("update requested")

            cur_round = req['round_number']
            self.logger.info("### Round {} ###".format(cur_round))

            # Check client availability based on reliability
            if not self.client_resources.check_availability():
                self.logger.warning("Client not available for this round due to reliability")
                self.sio.emit('client_unavailable', {'round_number': cur_round})
                return

            self.logger.info("Received model from Server, Loading it..")
            try:
                weights = pickle_string_to_obj(req['current_weights'])
                if not isinstance(weights, list):
                    raise ValueError("Invalid weight format received from server")
                
                # Simulate transmission time (only if we have resource simulation)
                try:
                    model_size_mb = self._estimate_model_size(weights) / (1024 * 1024)  # Convert to MB
                    transmission_time = self.client_resources.simulate_transmission_time(model_size_mb)
                    self.logger.info(f"Simulating model download time: {transmission_time:.2f}s")
                    time.sleep(min(transmission_time, 3.0))  # Reduced cap to 3 seconds for faster testing
                except Exception as e:
                    self.logger.debug(f"Skipping transmission simulation: {e}")
                    
                self.local_model.set_weight(weights)

                epochs = req['epochs']
                lr = req['learning_rate']
                batch_size = req['batch_size']

                # Get FL algorithm parameters
                fl_params = req.get('fl_params', {'algorithm': 'fedavg'})
                self.logger.info(f"Using FL algorithm: {fl_params['algorithm']}")

                # Start timing for computation simulation
                training_start = time.time()
                
                time_tot, train_map, train_loss, train_size = self.local_model.train(
                    epochs=epochs, 
                    lr=lr,
                    batch_size=batch_size, 
                    fl_config=fl_params
                )
                
                # Simulate additional computation time based on client resources
                try:
                    actual_training_time = time.time() - training_start
                    simulated_time = self.client_resources.simulate_computation_time(actual_training_time)
                    additional_delay = max(0, simulated_time - actual_training_time)
                    if additional_delay > 0:
                        self.logger.info(f"Simulating additional computation time: {additional_delay:.2f}s")
                        time.sleep(min(additional_delay, 5.0))  # Reduced cap to 5 seconds for faster testing
                except Exception as e:
                    self.logger.debug(f"Skipping computation simulation: {e}")
                
                my_weights = self.local_model.get_weight()
                if not isinstance(my_weights, list):
                    raise ValueError("Invalid weight format after training")
                
                # Simulate upload time (only if we have resource simulation)
                try:
                    upload_time = self.client_resources.simulate_transmission_time(model_size_mb)
                    self.logger.info(f"Simulating model upload time: {upload_time:.2f}s")
                    time.sleep(min(upload_time, 3.0))  # Reduced cap to 3 seconds for faster testing
                except Exception as e:
                    self.logger.debug(f"Skipping upload simulation: {e}")
                    
                pickle_string_weights = obj_to_pickle_string(my_weights)
                
            except Exception as e:
                self.logger.error(f"Error processing weights or training: {str(e)}")
                # Send error status to server
                self.sio.emit('client_error', {
                    'error': str(e),
                    'round_number': req.get('round_number', -1)
                })
                return

            # Computation of the Train Metrics
            list_f1 = train_map['f1_score']
            list_acc = train_map['accuracy_score']
            avg_f1 = sum(list_f1) / len(list_f1)
            avg_acc = sum(list_acc) / len(list_acc)

            # METRICHE DA SALVARE
            metrics_to_save = {
                "round": cur_round,
                "train_loss": train_loss,
                "avg_f1": avg_f1,
                "avg_acc": avg_acc,
                "train_size": train_size,
                "compute_power": self.client_resources.compute_power,
                "bandwidth": self.client_resources.bandwidth,
                "reliability": self.client_resources.reliability,
                "training_time": time_tot,
                "client_name": self.config.get('client_name', f"client_{os.path.basename(self.dataset_path)}")
            }
            self.save_metrics(metrics_to_save, cur_round)

            # Include client resource information in response
            resp = {
                'round_number': cur_round,
                'client_id': req.get('client_id', 'unknown'),
                'weights': pickle_string_weights,
                'train_loss': train_loss,
                'avg_f1': avg_f1,
                'avg_acc': avg_acc,
                'train_size': train_size,
                'client_resources': {
                    'compute_power': self.client_resources.compute_power,
                    'bandwidth': self.client_resources.bandwidth,
                    'reliability': self.client_resources.reliability
                }
            }

            self.logger.info("===================================")
            self.logger.info("client_train_loss {}".format(train_loss))
            self.logger.info("Average F1 Score {}".format(avg_f1))
            self.logger.info("Average Acc Score {}".format(avg_acc))
            self.logger.info("Time Tot of Training {}".format(time_tot))
            self.logger.info("===================================")

            # Enhanced logging with client type
            client_category = self.client_resources.get_client_category()
            self.logger.info(f"[SEND] [{client_category}] Sending trained model and metrics to server")
            self.logger.info(f"[CLIENT] Client resources: {self.client_resources.get_detailed_description()}")

            self.sio.emit('client_update', resp)
            self.logger.info("Sent the Trained model and Metrics to the server")

        def on_stop_and_eval(*args):
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
            req = args[0]
            self.logger.info("check client resource.")
            load_average = 0.15

            resp = {
                'round_number': req['round_number'],
                'load_rate': load_average
            }
            self.sio.emit('check_client_resource_done', resp)

        def on_model_update(*args):
            """Handle model updates from server (for non-participating clients)"""
            req = args[0]
            self.logger.info(f"Received model update for round {req['round_number']} as {req.get('participation_status', 'observer')}")
            
            try:
                weights = pickle_string_to_obj(req['current_weights'])
                self.local_model.set_weight(weights)
                self.logger.info("Updated local model with latest server weights")
            except Exception as e:
                self.logger.error(f"Failed to update model: {e}")

        self.sio.on('shutdown', on_shutdown)
        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('reconnect', on_reconnect)
        self.sio.on('init', self.on_init)
        self.sio.on('request_update', on_request_update)
        self.sio.on('stop_and_eval', on_stop_and_eval)
        self.sio.on('check_client_resource', on_check_client_resource)
        self.sio.on('model_update', on_model_update)
    
    def _estimate_model_size(self, weights):
        """Estimate model size in bytes"""
        total_size = 0
        for weight in weights:
            if hasattr(weight, 'nbytes'):
                total_size += weight.nbytes
            else:
                # Fallback for other types
                total_size += len(str(weight).encode('utf-8'))
        return total_size

    def save_metrics(self, metrics_dict, round_number):
        metrics_dir = "metrics"
        os.makedirs(metrics_dir, exist_ok=True)
        # Usa il nome client per il file
        client_name = self.config.get('client_name', f"client_{os.path.basename(self.dataset_path)}")
        metrics_path = os.path.join(metrics_dir, f"metrics_{client_name}.json")
        # Se il file esiste, aggiorna la lista dei round
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r") as f:
                    old_data = json.load(f)
            except Exception:
                old_data = {}
        else:
            old_data = {}
        # Salva le metriche per ogni round
        old_data[str(round_number)] = metrics_dict
        with open(metrics_path, "w") as f:
            json.dump(old_data, f)

if __name__ == "__main__":
    client = FederatedClient(CONFIG_FILE,"dataset/Clients/client_0")
