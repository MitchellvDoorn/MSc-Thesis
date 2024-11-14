import numpy as np
import deepxde as dde
from deepxde import config, optimizers, display
import tensorflow as tf
import objective_functions

class LossHistory(dde.model.LossHistory):
    # overwrote and added y_pred_test to the append method
    def __init__(self):
        super().__init__()
        self.y_pred_test = []

    def append(self, step, loss_train, loss_test, metrics_test, y_pred_test):
        self.steps.append(step)
        self.loss_train.append(loss_train)
        if loss_test is None:
            loss_test = self.loss_test[-1]
        if metrics_test is None:
            metrics_test = self.metrics_test[-1]
        self.loss_test.append(loss_test)
        self.metrics_test.append(metrics_test)
        self.y_pred_test.append(y_pred_test)

class ModelPCNN(dde.Model):
    '''
    Model inherits the DeepXDE Model Class and adds y_pred_test to the losshistory

    Commented out is an overwritten _compile_tensorflow method that adds an objective as a regularization term to the loss function
    '''

    def __init__(self, data, net, loss_weights):
        super().__init__(data, net)
        self.losshistory = LossHistory() # important to import after the super() line because otherwise other LossHistory object from the LossHistory class in model.py will be inherited
        self.loss_weigts_new = loss_weights
    def _compile_tensorflow(self, lr, loss_fn, decay):
        """tensorflow"""

        @tf.function(jit_compile=config.xla_jit)
        def outputs(training, inputs):
            return self.net(inputs, training=training)

        def outputs_losses(training, inputs, targets, auxiliary_vars, losses_fn):
            self.net.auxiliary_vars = auxiliary_vars
            # Don't call outputs() decorated by @tf.function above, otherwise the
            # gradient of outputs wrt inputs will be lost here.
            outputs_ = self.net(inputs, training=training)
            # Data losses
            losses = losses_fn(targets, outputs_, loss_fn, inputs, self)
            if not isinstance(losses, list):
                losses = [losses]
            # Regularization loss
            if self.net.regularizer is None:
                losses += [objective_functions.OptimalFuel.call(self, inputs, outputs_, losses)]
                # losses += [self.net.regularizer(inputs, outputs_, losses)]
            losses = tf.convert_to_tensor(losses)
            # Weighted losses
            if self.loss_weigts_new is not None:
                losses *= self.loss_weigts_new
            return outputs_, losses

        @tf.function(jit_compile=config.xla_jit)
        def outputs_losses_train(inputs, targets, auxiliary_vars):
            return outputs_losses(
                True, inputs, targets, auxiliary_vars, self.data.losses_train
            )

        @tf.function(jit_compile=config.xla_jit)
        def outputs_losses_test(inputs, targets, auxiliary_vars):
            return outputs_losses(
                False, inputs, targets, auxiliary_vars, self.data.losses_test
            )

        opt = optimizers.get(self.opt_name, learning_rate=lr, decay=decay)

        @tf.function(jit_compile=config.xla_jit)
        def train_step(inputs, targets, auxiliary_vars):
            # inputs and targets are np.ndarray and automatically converted to Tensor.
            with tf.GradientTape() as tape:
                losses = outputs_losses_train(inputs, targets, auxiliary_vars)[1]
                total_loss = tf.math.reduce_sum(losses)
            trainable_variables = (
                self.net.trainable_variables + self.external_trainable_variables
            )
            grads = tape.gradient(total_loss, trainable_variables)
            opt.apply_gradients(zip(grads, trainable_variables))

        def train_step_tfp(
            inputs, targets, auxiliary_vars, previous_optimizer_results=None
        ):
            def build_loss():
                losses = outputs_losses_train(inputs, targets, auxiliary_vars)[1]
                return tf.math.reduce_sum(losses)

            trainable_variables = (
                self.net.trainable_variables + self.external_trainable_variables
            )
            return opt(trainable_variables, build_loss, previous_optimizer_results)

        # Callables
        self.outputs = outputs
        self.outputs_losses_train = outputs_losses_train
        self.outputs_losses_test = outputs_losses_test
        self.train_step = (
            train_step
            if not optimizers.is_external_optimizer(self.opt_name)
            else train_step_tfp
        )

    def _test(self):
        # Overwrote and added train_state.y_pred_test to the losshistory
        # TODO Now only print the training loss in rank 0. The correct way is to print the average training loss of all ranks.
        (
            self.train_state.y_pred_train,
            self.train_state.loss_train,
        ) = self._outputs_losses(
            True,
            self.train_state.X_train,
            self.train_state.y_train,
            self.train_state.train_aux_vars,
        )
        self.train_state.y_pred_test, self.train_state.loss_test = self._outputs_losses(
            False,
            self.train_state.X_test,
            self.train_state.y_test,
            self.train_state.test_aux_vars,
        )

        if isinstance(self.train_state.y_test, (list, tuple)):
            self.train_state.metrics_test = [
                m(self.train_state.y_test[i], self.train_state.y_pred_test[i])
                for m in self.metrics
                for i in range(len(self.train_state.y_test))
            ]
        else:
            self.train_state.metrics_test = [
                m(self.train_state)
                for m in self.metrics
            ]

        self.train_state.update_best()
        self.losshistory.append(
            self.train_state.step,
            self.train_state.loss_train,
            self.train_state.loss_test,
            self.train_state.metrics_test,
            self.train_state.y_pred_test,
        )

        if (
            np.isnan(self.train_state.loss_train).any()
            or np.isnan(self.train_state.loss_test).any()
        ):
            self.stop_training = True

        # if self.display_progress:
        if config.rank == 0:
            display.training_display(self.train_state)


class TimeDomain_with_std(dde.geometry.TimeDomain):
    def __init__(self, t0, t1, sampler_std = None):
        super().__init__(t0, t1)
        self.t0 = t0
        self.t1 = t1
        self.sampler_std = sampler_std