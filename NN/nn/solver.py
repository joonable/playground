import numpy as np
import nn.optim as optim

class Solver:
    """A encapsulated module for training neural network model.
    This class is in charge of 1) handle training settings,
    2) training the model and 3) validate the model and shows the results

    Args:
        - model: Network instance. It must have modules attribute.
        - data: Dataset to train and validate.
    Optional Args:
        - optim (str): Update rule. Only support sgd method.
        - optim_config (dict): Additional config of optim rule. e.g. lr.
        - lr_decay (float): Decaying rate of LR. LR is decayed at every epoch.
        - batch_size (int): Batch size when training.
        - num_epochs (int): Maximum epochs of training.
        - verbose (bool): Decide wheter print the results or not.
        - flatten_input (bool): If True, flatten the input to make 
          suitable for the FC (linear) layer.
    """
    def __init__(self, model, data, **kwargs):
        self.model = model
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]
        
        # Unpack keyword arguments
        self.optim = kwargs.pop("optim", "sgd")
        self.optim_config = kwargs.pop("optim_config", {})
        self.batch_size = kwargs.pop("batch_size", 64)
        self.num_epochs = kwargs.pop("num_epochs", 10)
        self.verbose = kwargs.pop("verbose", True)
        self.flatten_input = kwargs.pop("flatten_input", True)
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        
        if self.optim == "sgd":
            self.optim = optim.SGD
        else:
            raise ValueError("Invalid optim")
        self.optim = self.optim(model.modules, **self.optim_config)

        # Set up some variables for book-keeping
        self.epoch = 0
        self.loss_history = list()
        self.train_acc_history = list()
        self.val_acc_history = list()

    def _step(self):
        # Make a minibatch of training data
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        if self.flatten_input:
            X_batch = X_batch.reshape(X_batch.shape[0], -1)

        # Compute loss and gradient
        loss = self.model.loss(X_batch, y_batch)
        self.optim.step()

        return loss

    def train(self):
        """Train the model
        """
        num_train = self.X_train.shape[0]
        steps_per_epoch = max(num_train//self.batch_size, 1)
        num_steps = self.num_epochs * steps_per_epoch

        for t in range(num_steps):
            loss = self._step()
            self.loss_history.append(loss)

            epoch_end = (t+1) % steps_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                # decay learning rate
                self.optim.lr *= self.lr_decay
            
            first_step, last_step = t==0, t==num_steps+1
            if first_step or last_step or epoch_end:
                train_acc = self.validate(
                    self.X_train, self.y_train, num_samples=1000
                )
                val_acc = self.validate(
                    self.X_val, self.y_val, num_samples=1000
                )
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)

                if self.verbose:
                    print("Epoch {}/{}".format(self.epoch, self.num_epochs))
                    print("  Loss: {:.3f}".format(loss))
                    print("  Train accuracy: {:.3f}".format(train_acc))
                    print("  Val accuracy: {:.3f}".format(val_acc))

    def validate(self, X, y, num_samples=None, batch_size=100):
        """Validate the model

        Args:
        - X: Data X (feature).
        - y: Data y (label).
        - num_samples (None or int): If None, use all data for test 
          and if specifed only cover num_samples data.
        - batch_size (int): Batch size when validate.
        """
        # Maybe subsample the data
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        
        if self.flatten_input:
            X = X.reshape(X.shape[0], -1)

        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc 
