import numpy as np
import logging
from .fl_algorithm_base import FLAlgorithm

class FedAvg(FLAlgorithm):
    def aggregate_weights(self, client_weights, client_sizes=None):
        if client_sizes is not None:
            # Weighted average based on client dataset sizes
            total_size = np.sum(client_sizes)
            new_weights = [np.zeros(param.shape) for param in client_weights[0]]
            
            for c in range(len(client_weights)):
                for i in range(len(new_weights)):
                    if isinstance(client_weights[c][i], str):
                        total_size -= client_sizes[c]
                        break
                    new_weights[i] += (client_weights[c][i] * client_sizes[c])

            for i in range(len(new_weights)):
                new_weights[i] = new_weights[i] / total_size
                
        else:
            # Simple average
            total_size = len(client_weights)
            new_weights = [np.zeros(param.shape) for param in client_weights[0]]
            for c in range(len(client_weights)):
                for i in range(len(new_weights)):
                    new_weights[i] += (client_weights[c][i] / total_size)
                    
        return new_weights

class FedProx(FLAlgorithm):
    def __init__(self, config):
        super().__init__(config)
        self.mu = config.get('proximal_term', 0.01)  # Proximal term coefficient
        
    def aggregate_weights(self, client_weights, client_sizes=None):
        """
        FedProx implementation with proximal term
        The proximal term is added during client training, not during aggregation
        For aggregation, we use the same logic as FedAvg
        """
        if client_sizes is not None:
            total_size = np.sum(client_sizes)
            new_weights = [np.zeros(param.shape) for param in client_weights[0]]
            
            for c in range(len(client_weights)):
                for i in range(len(new_weights)):
                    if isinstance(client_weights[c][i], str):
                        total_size -= client_sizes[c]
                        break
                    new_weights[i] += (client_weights[c][i] * client_sizes[c])

            for i in range(len(new_weights)):
                new_weights[i] = new_weights[i] / total_size
        else:
            total_size = len(client_weights)
            new_weights = [np.zeros(param.shape) for param in client_weights[0]]
            for c in range(len(client_weights)):
                for i in range(len(new_weights)):
                    new_weights[i] += (client_weights[c][i] / total_size)
                    
        return new_weights

class FedYogi(FLAlgorithm):
    def __init__(self, config, initial_weights=None):
        super().__init__(config, initial_weights)
        self.beta1 = config.get('beta1', 0.9)
        self.beta2 = config.get('beta2', 0.99)
        self.eta = config.get('eta', 0.01)
        self.tau = config.get('tau', 1e-3)
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0    # Time step
        
    def aggregate_weights(self, client_weights, client_sizes=None):
        try:
            # Validate input weights
            if not client_weights or len(client_weights) == 0:
                raise ValueError("No client weights provided")
            
            # First compute the average of client weights like FedAvg
            if client_sizes is not None:
                total_size = np.sum(client_sizes)
                if total_size == 0:
                    raise ValueError("Total client size is 0")
                    
                avg_weights = [np.zeros(param.shape) for param in client_weights[0]]
                
                for c in range(len(client_weights)):
                    for i in range(len(avg_weights)):
                        if isinstance(client_weights[c][i], str):
                            total_size -= client_sizes[c]
                            break
                        try:
                            avg_weights[i] += (client_weights[c][i] * client_sizes[c])
                        except Exception as e:
                            raise ValueError(f"Error processing weights for client {c}: {str(e)}")

                if total_size == 0:
                    raise ValueError("All clients were rejected due to invalid weights")
                    
                for i in range(len(avg_weights)):
                    avg_weights[i] = avg_weights[i] / total_size
            else:
                total_size = len(client_weights)
                if total_size == 0:
                    raise ValueError("No valid client weights provided")
                    
                avg_weights = [np.zeros(param.shape) for param in client_weights[0]]
                for c in range(len(client_weights)):
                    for i in range(len(avg_weights)):
                        try:
                            avg_weights[i] += (client_weights[c][i] / total_size)
                        except Exception as e:
                            raise ValueError(f"Error processing weights for client {c}: {str(e)}")

            # Initialize momentum if not exists
            if self.m is None:
                try:
                    self.m = [np.zeros_like(w) for w in avg_weights]
                    self.v = [np.zeros_like(w) for w in avg_weights]
                except Exception as e:
                    raise ValueError(f"Error initializing momentum: {str(e)}")
                    
            return avg_weights
                    
        except Exception as e:
            logging.error(f"Error in FedYogi aggregation: {str(e)}")
            raise

        # Update with Yogi optimizer logic
        new_weights = []
        for i, (w_t, m_t, v_t) in enumerate(zip(avg_weights, self.m, self.v)):
            # Se non ci sono pesi precedenti, usa i pesi medi come pesi correnti
            if self.current_weights is None:
                self.current_weights = avg_weights
                grad = np.zeros_like(w_t)
            else:
                grad = w_t - self.current_weights[i]  # Gradient is difference from current
            
            # Update momentum
            m_t = self.beta1 * m_t + (1 - self.beta1) * grad
            v_t = v_t - (1 - self.beta2) * np.sign(v_t - grad ** 2) * grad ** 2
            
            # Update weights
            update = self.eta * m_t / (np.sqrt(np.abs(v_t)) + self.tau)
            new_weights.append(w_t + update)
            
            # Store updated moments
            self.m[i] = m_t
            self.v[i] = v_t
            
        return new_weights

class FedAdam(FLAlgorithm):
    def __init__(self, config):
        super().__init__(config)
        self.beta1 = config.get('beta1', 0.9)
        self.beta2 = config.get('beta2', 0.99)
        self.eta = config.get('eta', 0.01)
        self.tau = config.get('tau', 1e-3)
        self.m = None
        self.v = None
        
    def aggregate_weights(self, client_weights, client_sizes=None):
        # First compute the average of client weights like FedAvg
        if client_sizes is not None:
            total_size = np.sum(client_sizes)
            avg_weights = [np.zeros(param.shape) for param in client_weights[0]]
            
            for c in range(len(client_weights)):
                for i in range(len(avg_weights)):
                    if isinstance(client_weights[c][i], str):
                        total_size -= client_sizes[c]
                        break
                    avg_weights[i] += (client_weights[c][i] * client_sizes[c])

            for i in range(len(avg_weights)):
                avg_weights[i] = avg_weights[i] / total_size
        else:
            total_size = len(client_weights)
            avg_weights = [np.zeros(param.shape) for param in client_weights[0]]
            for c in range(len(client_weights)):
                for i in range(len(avg_weights)):
                    avg_weights[i] += (client_weights[c][i] / total_size)

        # Initialize momentum if not exists
        if self.m is None:
            self.m = [np.zeros_like(w) for w in avg_weights]
            self.v = [np.zeros_like(w) for w in avg_weights]

        # Update with Adam optimizer logic
        new_weights = []
        for i, (w_t, m_t, v_t) in enumerate(zip(avg_weights, self.m, self.v)):
            grad = w_t - self.current_weights[i]
            
            # Update biased first moment estimate
            m_t = self.beta1 * m_t + (1 - self.beta1) * grad
            # Update biased second raw moment estimate
            v_t = self.beta2 * v_t + (1 - self.beta2) * np.square(grad)
            
            # Update weights
            update = self.eta * m_t / (np.sqrt(v_t) + self.tau)
            new_weights.append(w_t + update)
            
            # Store updated moments
            self.m[i] = m_t
            self.v[i] = v_t
            
        return new_weights

def get_fl_algorithm(algorithm_name: str, config: dict, initial_weights=None) -> FLAlgorithm:
    """
    Factory method to get the appropriate FL algorithm
    Args:
        algorithm_name: Name of the FL algorithm to use
        config: Configuration dictionary
        initial_weights: Initial weights for the algorithm (needed for FedYogi and FedAdam)
    Returns:
        FLAlgorithm instance
    """
    algorithms = {
        'fedavg': FedAvg,
        'fedprox': FedProx,
        'fedyogi': FedYogi,
        'fedadam': FedAdam,
    }
    
    algorithm_class = algorithms.get(algorithm_name.lower())
    if algorithm_class is None:
        raise ValueError(f"Unknown FL algorithm: {algorithm_name}")
        
    algorithm = algorithm_class(config)
    if initial_weights is not None:
        algorithm.update_current_weights(initial_weights)
    
    return algorithm
