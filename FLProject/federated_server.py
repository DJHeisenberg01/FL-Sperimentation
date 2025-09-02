import logging
import random
import sys
import time
from flask import Flask, request, render_template
from flask_socketio import *
from flask_socketio import SocketIO
import json
from utilities import obj_to_pickle_string, pickle_string_to_obj
from aggregator import Aggregator
from client_resources import ClientResources
from aggregation_policies import (
    UniformAggregation, PowerAwareAggregation, 
    ReliabilityAwareAggregation, BandwidthAwareAggregation, 
    HybridAggregation
)
import os

CONFIG_FILE = 'cfg/config.json'


def load_json(filename):
    with open(filename) as f:
        return json.load(f)
    
def save_server_metrics(round_num, test_loss, test_f1, test_acc, test_prec, test_recall):
        metrics_dir = "metrics"
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, "server_metrics.json")
        # Carica dati esistenti
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        else:
            data = {}
        # Aggiorna dati per il round corrente
        data[str(round_num)] = {
            "test_loss": test_loss,
            "test_f1": test_f1,
            "test_acc": test_acc,
            "test_prec": test_prec,
            "test_recall": test_recall
        }
        with open(metrics_path, "w") as f:
            json.dump(data, f)


class FederatedServer(object):

    def __init__(self, config_file, aggregation_policy=None):
        '''
        Load with json parameters from a config file
        Params: ip_address, port, model_name, log_filename, global_epoch, models_percentage
                local_epochs, learning_rate, batch_size
        aggregation_policy: Optional aggregation policy for resource-aware training
        '''
        self.config = load_json(config_file)

        # Federated Variables
        self.MIN_NUM_WORKERS = self.config['MIN_NUM_WORKERS']
        self.NUM_CLIENTS_CONTACTED_PER_ROUND = 0
        self.registered_clients = set()
        self.client_resource = {}
        self.client_resources_detailed = {}  # Store detailed ClientResources objects
        self.current_round = -1  # -1 for not yet started
        self.current_round_client_updates = []
        self.eval_client_updates = []
        self.STOP = False
        
        # Configure wait times based on config
        base_wait_time = self.config.get('round_wait_time', 10)
        self.wait_time = base_wait_time
        self.min_wait_time = max(5, base_wait_time // 2)  # At least 5 seconds, or half of configured time
        self.max_wait_time = base_wait_time * 3  # Up to 3x the configured time
        
        # Partial aggregation parameters
        self.min_clients_for_aggregation = self.config.get('min_clients_for_aggregation', max(1, self.MIN_NUM_WORKERS // 2))
        self.client_response_timeout = self.config.get('client_response_timeout', 60)  # seconds
        self.partial_aggregation_enabled = self.config.get('partial_aggregation_enabled', True)
        
        # Track client participation
        self.current_round_contacted_clients = []
        self.current_round_responding_clients = []
        self.eval_clients_contacted = []  # Track clients contacted for evaluation
        self.round_aggregation_done = False  # Prevent multiple aggregations per round
        self.partial_aggregation_timer = None  # Timer for partial aggregation
        self.partial_aggregation_wait_time = self.config.get('partial_aggregation_wait_time', 30)  # seconds to wait before partial aggregation
        
        # Initialize aggregation policy
        if aggregation_policy is None:
            self.aggregation_policy = UniformAggregation()
        else:
            self.aggregation_policy = aggregation_policy

        # Set Logger
        self.logger = logging.getLogger("Federated-Server")
        self.set_logger()
        self.logger.info("Config Params: " + str(self.config))
        
        # Determine expected number of clients
        self.expected_clients = self._get_expected_client_count()
        
        # Log client configuration if available
        client_config = self.config.get('client_configuration', {})
        if client_config:
            num_clients = client_config.get('num_clients', 'Not specified')
            client_quality = client_config.get('client_quality', 'Not specified')
            self.logger.info(f"[CLIENT] PROFILE CONFIGURATION:")
            self.logger.info(f"   Number of clients: {num_clients}")
            self.logger.info(f"   Client quality profile: {client_quality}")
            
            # Log quality ranges if profile is valid
            if client_quality in ClientResources.QUALITY_PROFILES:
                ranges = ClientResources.get_quality_ranges(client_quality)
                self.logger.info(f"   Profile ranges - Compute: {ranges['compute_power']}, "
                               f"Bandwidth: {ranges['bandwidth']} Mbps, "
                               f"Reliability: {ranges['reliability']}")
            elif client_quality == 'misti':
                mix_dist = client_config.get('mix_distribution', {})
                self.logger.info(f"   Mixed distribution: {mix_dist}")
        else:
            self.logger.info("[CLIENT] PROFILE: Using legacy configuration (no client_configuration section)")
        
        self.aggregator = Aggregator(self.config, self.logger)

        self.early_stop_tolerance = 0

        # Flask Parameter Configuration
        self.app = Flask(__name__)
        # self.app.debug = True
        self.socketio = SocketIO(self.app, ping_timeout=3600,
                                 ping_interval=3600,
                                 max_http_buffer_size=int(1e32))
        
        # Setup Flask routes
        self.setup_flask_routes()
        
        # Determine expected number of clients after socketio is initialized
        self.expected_clients = self._get_expected_client_count()
        self.logger.info(f"[SERVER] Expecting {self.expected_clients} clients to connect before starting")
        
        self.register_handles()
        
    

    def _get_expected_client_count(self):
        """Determine the expected number of clients based on configuration"""
        # First check if client_configuration specifies number of clients
        client_config = self.config.get('client_configuration', {})
        if client_config and 'num_clients' in client_config:
            return client_config['num_clients']
        
        # Fallback to num_clients or MIN_NUM_WORKERS
        if 'num_clients' in self.config:
            return self.config['num_clients']
        
        return self.MIN_NUM_WORKERS

    def setup_flask_routes(self):
        """Setup Flask routes for the web interface"""
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
            return render_template('stats.html', tables=[df.to_html(classes='data')], texts=texts, images=images)

    def check_client_resource(self):
        self.client_resource = {}
        client_sids_selected = random.sample(list(self.registered_clients), int(self.NUM_CLIENTS_CONTACTED_PER_ROUND))
        self.logger.info('Sending weights to selected clients: ' + str(client_sids_selected))
        for rid in client_sids_selected:
            emit('check_client_resource', {
                'round_number': self.current_round,
            }, room=rid)

        # Note: we assume that during training the #workers will be >= MIN_NUM_WORKERS

    def train_next_round(self, client_sids_selected):
        self.current_round += 1
        # buffers all client updates
        self.current_round_client_updates = []
        self.current_round_contacted_clients = client_sids_selected.copy()
        self.current_round_responding_clients = []
        self.round_aggregation_done = False  # Reset aggregation flag for new round
        # Reset partial aggregation timer
        if self.partial_aggregation_timer:
            self.partial_aggregation_timer.cancel()
            self.partial_aggregation_timer = None
        self.logger.info("### Round {} ###".format(self.current_round))
        self.logger.info("Requesting updates from {}".format(client_sids_selected))
        current_weights = obj_to_pickle_string(self.aggregator.current_weights)
        # Prepare FL algorithm parameters
        fl_params = {
            'algorithm': self.config.get('fl_algorithm', 'fedavg')
        }
        
        # Add algorithm-specific parameters
        if fl_params['algorithm'] == 'fedprox':
            fl_params['proximal_term'] = self.config.get('proximal_term', 0.01)
        elif fl_params['algorithm'] in ['fedyogi', 'fedadam']:
            fl_params.update({
                'beta1': self.config.get('beta1', 0.9),
                'beta2': self.config.get('beta2', 0.99),
                'eta': self.config.get('eta', 0.01),
                'tau': self.config.get('tau', 1e-3)
            })

        for rid in client_sids_selected:
            emit('request_update', {
                'epochs': self.config['local_epoch'],
                'batch_size': self.config['batch_size'],
                'learning_rate': self.config['learning_rate'],
                'round_number': self.current_round,
                'current_weights': current_weights,
                'fl_params': fl_params,
                'client_id': rid
            }, room=rid)
            self.logger.info("Sent the model to {} with FL algorithm: {}".format(rid, fl_params['algorithm']))

    def stop_and_eval(self):
        current_weights = obj_to_pickle_string(self.aggregator.current_weights)
        self.eval_client_updates = []
        self.eval_clients_contacted = []  # Track which clients we're expecting evaluations from
        for rid in self.registered_clients:
            self.socketio.emit('stop_and_eval', {
                'batch_size': self.config['batch_size'],
                'current_weights': current_weights,
                'STOP': self.STOP
            }, room=rid)
            self.eval_clients_contacted.append(rid)
        self.logger.info("Sent the aggregated model to every client registered to validate it")

    def set_logger(self):
        datestr = time.strftime('%d%m')
        timestr = time.strftime('%m%d%H%M')
        self.logger.setLevel(logging.INFO)
        log_dir = os.path.join("logs", datestr, "FL-Server-LOG")
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, '{}.log'.format(timestr)))
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARN)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def run(self):
        self.socketio.run(self.app, host=self.config['ip_address'], port=self.config['port'], )
        self.logger.info("Federated Server Started")

    def register_handles(self):
        # single-threaded async, no need to lock

        @self.socketio.on('connect')
        def handle_connect():
            self.logger.info("[CONNECT] Client Sid:" + str(request.sid) + " connected")

        @self.socketio.on('reconnect')
        def handle_reconnect():
            self.logger.info("[RECONNECT] Client Sid:" + str(request.sid) + " reconnected")

        @self.socketio.on('disconnect')
        def handle_disconnect():
            # Log with client type if available
            if request.sid in self.client_resources_detailed:
                client_resources = self.client_resources_detailed[request.sid]
                category = client_resources.get_client_category()
                self.logger.info(f"[DISCONNECT] Client Sid:{request.sid} [{category}] disconnected")
            else:
                self.logger.info("[DISCONNECT] Client Sid:" + str(request.sid) + " disconnected")
                
            if request.sid in self.registered_clients:
                self.registered_clients.remove(request.sid)
                self.refresh_client_round()

        @self.socketio.on('client_wake_up')
        def handle_wake_up():
            self.logger.info("Sending client wake_up to " + str(request.sid))
            emit('init')

        @self.socketio.on('client_ready')
        def handle_client_ready():
            self.logger.info("The client " + str(request.sid) + " is ready for training")
            self.registered_clients.add(request.sid)
            self.refresh_client_round()

            # Start federated process only when we have the expected number of clients
            if len(self.registered_clients) >= self.expected_clients and self.current_round == -1:
                self.logger.info("The Federated Process is Starting")
                self.check_client_resource()
            elif len(self.registered_clients) < self.expected_clients:
                remaining = self.expected_clients - len(self.registered_clients)
                self.logger.info(f"Waiting for {remaining} more clients to start the process ({len(self.registered_clients)}/{self.expected_clients} connected)")
            else:
                self.logger.error("The current_round is not equal to -1, please restart server.")

        @self.socketio.on('check_client_resource_done')
        def handle_check_client_resource_done(data):
            if data['round_number'] == self.current_round:
                self.client_resource[request.sid] = data['load_rate']
                
                # Initialize default resources if not available
                if request.sid not in self.client_resources_detailed:
                    self.client_resources_detailed[request.sid] = ClientResources(
                        compute_power=1.0,
                        bandwidth=5.0,
                        reliability=1.0
                    )
                
                if len(self.client_resource) == self.NUM_CLIENTS_CONTACTED_PER_ROUND:
                    # Use aggregation policy for client selection
                    available_clients = list(self.client_resource.keys())
                    
                    # Apply traditional CPU rate filtering first
                    max_cpu_rate = self.config.get('max_cpu_rate', 0.8)
                    cpu_filtered_clients = []
                    
                    for client_id, val in self.client_resource.items():
                        self.logger.info(f"{client_id} CPU rate: {val}")
                        if float(val) <= max_cpu_rate:
                            cpu_filtered_clients.append(client_id)
                            self.logger.info(f"{client_id} satisfied CPU requirement (threshold: {max_cpu_rate})")
                        else:
                            self.logger.warning(f"{client_id} rejected - CPU rate too high ({val} > {max_cpu_rate})")
                    
                    # Apply aggregation policy selection
                    try:
                        client_sids_selected = self.aggregation_policy.select_clients(
                            cpu_filtered_clients, 
                            self.client_resources_detailed
                        )
                        self.logger.info(f"Aggregation policy selected clients: {client_sids_selected}")
                    except Exception as e:
                        self.logger.warning(f"Aggregation policy failed, falling back to all CPU-filtered clients: {e}")
                        client_sids_selected = cpu_filtered_clients

                    if len(client_sids_selected) > 0:
                        self.wait_time = max(self.min_wait_time, self.wait_time - 1)  # Gradually reduce wait time but keep minimum
                        self.logger.info(f"Starting training round with {len(client_sids_selected)} clients. Wait time: {self.wait_time}s")
                        time.sleep(self.wait_time)
                        self.train_next_round(client_sids_selected)
                    else:
                        if self.wait_time < self.max_wait_time:
                            self.wait_time = min(self.wait_time + 2, self.max_wait_time)  # Increase wait time more gradually
                        self.logger.info(f"No clients selected, waiting {self.wait_time}s before retrying...")
                        time.sleep(self.wait_time)
                        self.check_client_resource()

        @self.socketio.on('client_update')
        def handle_client_update(data):
            self.logger.info("received client update of bytes: {}".format(sys.getsizeof(data)))
            self.logger.info("handle client_update {}".format(request.sid))

            if data['round_number'] == self.current_round:
                self.current_round_client_updates += [data]
                self.current_round_client_updates[-1]['weights'] = pickle_string_to_obj(data['weights'])
                self.current_round_responding_clients.append(request.sid)
                
                # Store client resource information if available
                if 'client_resources' in data:
                    resources_data = data['client_resources']
                    client_resources = ClientResources(
                        compute_power=resources_data['compute_power'],
                        bandwidth=resources_data['bandwidth'],
                        reliability=resources_data['reliability']
                    )
                    self.client_resources_detailed[request.sid] = client_resources
                    
                    # Enhanced logging with client category
                    category = client_resources.get_client_category()
                    self.logger.info(f"[CLIENT] Client {request.sid} resources updated: {client_resources.get_detailed_description()}")
                else:
                    self.logger.info(f"[CLIENT] Client {request.sid} update received (no resource info)")
                
                # Check if we can proceed with aggregation
                self.check_aggregation_conditions()

        @self.socketio.on('client_eval')
        def handle_client_eval(data):
            if self.eval_client_updates is None:
                return
            self.logger.info("handle client_eval {}".format(request.sid))
            self.eval_client_updates += [data]

            if len(self.eval_client_updates) == len(self.eval_clients_contacted):  # Wait for evaluations from contacted clients only

                if self.config['weighted_aggregation']:
                    test_loss, test_f1, test_acc, test_prec, test_recall = self.aggregator.aggregate_evaluation_metrics_weighted(
                        [update['test_loss'] for update in self.eval_client_updates],
                        [update['test_f1'] for update in self.eval_client_updates],
                        [update['test_acc'] for update in self.eval_client_updates],
                        [update['test_prec'] for update in self.eval_client_updates],
                        [update['test_recall'] for update in self.eval_client_updates],
                        [update['test_size'] for update in self.eval_client_updates],
                        self.current_round,
                    )
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
                    
                # Check stopping conditions after evaluation is complete
                if self.current_round >= self.config['global_epoch'] - 1:
                    self.logger.info("Reached the Maximum number of global epochs, the process is stopping..")
                    print("FINISHING FEDERATED PROCESS...")
                    obj_to_pickle_string(self.aggregator.current_weights, save=True,
                                                       file_path="weights.pkl")
                    self.STOP = True
                    
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
                        self.socketio.emit("shutdown", room=client)
                    self.shutdown_server()
                else:
                    self.logger.info("start to next round...")
                    # Add a pause between rounds to reduce frequency
                    round_pause = self.config.get('round_wait_time', 10)
                    self.logger.info(f"Waiting {round_pause}s before starting next round...")
                    time.sleep(round_pause)
                    self.check_client_resource()
                # Dopo aver calcolato test_loss, test_f1, test_acc, test_prec, test_recall
                save_server_metrics(self.current_round, test_loss, test_f1, test_acc, test_prec, test_recall)

    def refresh_client_round(self):
        self.NUM_CLIENTS_CONTACTED_PER_ROUND = len(self.registered_clients) * self.config['models_percentage']

    def check_aggregation_conditions(self):
        """Check if we can proceed with aggregation (either all clients responded or partial aggregation conditions met)"""
        # Prevent multiple aggregations for the same round
        if self.round_aggregation_done:
            self.logger.info(f"Aggregation already completed for Round {self.current_round}. Ignoring additional client updates.")
            return
            
        num_responded = len(self.current_round_client_updates)
        num_contacted = len(self.current_round_contacted_clients)
        
        # Case 1: All contacted clients have responded
        if num_responded == num_contacted:
            self.logger.info(f"All {num_contacted} contacted clients responded. Proceeding with aggregation.")
            # Cancel partial aggregation timer if running
            if self.partial_aggregation_timer:
                self.partial_aggregation_timer.cancel()
                self.partial_aggregation_timer = None
            self.proceed_with_aggregation()
            return
            
        # Case 2: Partial aggregation conditions
        if self.partial_aggregation_enabled and num_responded >= self.min_clients_for_aggregation:
            # Start timer for partial aggregation if not already started
            if self.partial_aggregation_timer is None:
                self.logger.info(f"Minimum clients reached ({num_responded}/{num_contacted}). "
                               f"Starting {self.partial_aggregation_wait_time}s timer for partial aggregation...")
                import threading
                self.partial_aggregation_timer = threading.Timer(
                    self.partial_aggregation_wait_time, 
                    self._partial_aggregation_timeout
                )
                self.partial_aggregation_timer.start()
            else:
                self.logger.info(f"Waiting for more clients: {num_responded}/{num_contacted} responded, "
                               f"minimum needed: {self.min_clients_for_aggregation} (timer running)")
            return
        
        # Case 3: Not enough clients yet, continue waiting
        self.logger.info(f"Waiting for more clients: {num_responded}/{num_contacted} responded, "
                        f"minimum needed: {self.min_clients_for_aggregation}")

    def _partial_aggregation_timeout(self):
        """Called when partial aggregation timeout expires"""
        if not self.round_aggregation_done:
            num_responded = len(self.current_round_client_updates)
            num_contacted = len(self.current_round_contacted_clients)
            self.logger.info(f"Partial aggregation timeout reached. Proceeding with {num_responded}/{num_contacted} clients.")
            self.partial_aggregation_timer = None
            self.proceed_with_aggregation()

    def proceed_with_aggregation(self):
        """Perform aggregation and continue to next round or finish training"""
        # Mark aggregation as done to prevent multiple calls
        self.round_aggregation_done = True
        
        num_participated = len(self.current_round_client_updates)
        num_contacted = len(self.current_round_contacted_clients)
        
        self.logger.info(f"=== AGGREGATION (Round {self.current_round}) ===")
        self.logger.info(f"Participating clients: {num_participated}/{num_contacted}")
        
        # Perform aggregation
        aggr_train_loss = 0
        if self.config['weighted_aggregation']:
            aggr_train_loss = self.aggregate_with_sizes()
        else:
            aggr_train_loss = self.aggregate_without_sizes()

        self.logger.info("=== training ===")
        self.logger.info("aggr_train_loss {}".format(aggr_train_loss))

        self.test_on_selected()
        
        # Send updated model to ALL registered clients (including non-participants)
        self.broadcast_updated_model()
        
        self.stop_and_eval()

    def broadcast_updated_model(self):
        """Send the updated model to all registered clients, including those who didn't participate"""
        current_weights = obj_to_pickle_string(self.aggregator.current_weights)
        
        # Send to all registered clients, not just participants
        non_participants = set(self.registered_clients) - set(self.current_round_responding_clients)
        
        if non_participants:
            self.logger.info(f"Broadcasting updated model to {len(non_participants)} non-participating clients: {list(non_participants)}")
            for client_id in non_participants:
                self.socketio.emit('model_update', {
                    'round_number': self.current_round,
                    'current_weights': current_weights,
                    'participation_status': 'observer'
                }, room=client_id)
        
        self.logger.info(f"Model broadcast completed to all {len(self.registered_clients)} registered clients")

    def aggregate_with_sizes(self):
        # Get client IDs from updates
        client_ids = [update.get('client_id', f"unknown_{i}") for i, update in enumerate(self.current_round_client_updates)]
        
        # Try to use aggregation policy weights if available
        try:
            # Only use policy weights if we have resource information for all clients
            if all(client_id in self.client_resources_detailed for client_id in client_ids):
                # Create client_data_sizes dict for proper FedAvg weighting
                client_data_sizes = {
                    client_id: update['train_size'] 
                    for client_id, update in zip(client_ids, self.current_round_client_updates)
                }
                
                aggregation_weights = self.aggregation_policy.compute_weights(
                    client_ids, 
                    self.client_resources_detailed,
                    client_data_sizes  # Now passing the data sizes for correct weighting
                )
                # Convert to list format for aggregator
                policy_weights = [aggregation_weights.get(client_id, 1.0) for client_id in client_ids]
                
                self.logger.info(f"Using aggregation policy weights (with data sizes): {dict(zip(client_ids, policy_weights))}")
                
                # Use policy weights instead of train sizes
                self.aggregator.update_weights_weighted(
                    [x['weights'] for x in self.current_round_client_updates],
                    policy_weights
                )
                aggr_train_loss = self.aggregator.aggregate_train_loss_weights(
                    [x['train_loss'] for x in self.current_round_client_updates],
                    policy_weights,
                    self.current_round
                )
            else:
                raise ValueError("Missing resource information for some clients")
                
        except Exception as e:
            self.logger.warning(f"Failed to use aggregation policy weights, falling back to train sizes: {e}")
            # Fallback to original implementation
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
    # Backward compatibility: if no aggregation policy specified, use default
    server = FederatedServer(CONFIG_FILE)
    server.run()
