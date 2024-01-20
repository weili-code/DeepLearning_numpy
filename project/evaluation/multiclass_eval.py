import os
import numpy as np
import time


def trainer_multiclass(
    model,
    optimizers,
    criterion,
    train_data,
    valid_data,
    num_epochs,
    batch_size,
    print_all=False,
):
    """
    A trainder for multiclass predictor.

    Key workflow:
        for each b in batches:
            optimizer.zero_grad()
            yhat=model.forward
            loss=criterion.forward(yhat, ylabels)
            delta = criterion.backward()
            model.backward(delta)
            optimizer.step()

    Args:
        model (obj-MLP <nn.modules.mlp>, or obj-CNN <nn.modules.cnn>), or obj-RNNClassifier<nn.modules.rnn_classifier>):
        optimizers ( <nn.optim.sgd> object):
        criterion (obj-CrossEntropyLoss <nn.optim.loss>):
                here we use CrossEntropyLoss classification
        train_data (list): [training data input, targets]
                input is (num_examples, num_features)
                      or (num_examples, in_channles, num_features) for cnn
                      or (num_examples, in_channles, in_height, in_width) for cnn
                      or (num_examples, seq_length, input_size) for rnn model
                targets is (num_examples, num_classes) one-hot encoded
        valid_data (list):
        num_epochs (int):
        batch_size (int):
        print_all (bool, optional): Defaults to False.
            Whether to print info for each batch iteration.

    Returns:
        training_losses: running average of losses of minibatches for each epoch
        training_errors: error rates in the training data
        validation_losses
        validation_errors
    """

    trainx, trainy = train_data
    valx, valy = valid_data

    idxs = np.arange(len(trainx))

    training_losses = np.zeros(num_epochs)
    training_errors = np.zeros(num_epochs)
    validation_losses = np.zeros(num_epochs)
    validation_errors = np.zeros(num_epochs)

    num_batch_train = np.ceil(trainx.shape[0] / batch_size)
    num_batch_valid = np.ceil(valx.shape[0] / batch_size)

    for e in range(num_epochs):
        # Per epoch setup ...
        start_time = time.time()
        # in the begining of each epoch, shuffle training data
        np.random.shuffle(idxs)
        x_train = trainx[idxs]
        y_train = trainy[idxs]

        model.train()
        for b in range(0, trainx.shape[0], batch_size):
            # 1. Zero out grad after each batch
            for k in optimizers:
                k.zero_grad()
            # 2. Forward
            y_scores = model.forward(x_train[b : b + batch_size])
            # y_scores are logits
            y_true = y_train[b : b + batch_size]
            # 3. Backward
            # recall criterionr function CrossEntropyLoss takes two arguments
            # A (np.ndarray): (batch size, num_classes) -- logits
            # Y (np.ndarray): (batch size, num_classes)
            loss = criterion.forward(y_scores, y_true)
            delta = criterion.backward()
            model.backward(delta)
            # 4. Update with gradients
            for k in optimizers:
                k.step()
            # 5. collect the training loss
            training_losses[
                e
            ] += loss  # obtaining sum loss for all minibatches for e-th epoch
            # 6. Calculate training error count
            for i in range(y_scores.shape[0]):
                if np.argmax(y_scores[i]) != np.argmax(y_true[i]):
                    training_errors[
                        e
                    ] += 1  # obtaining total errors for whole training data for e-th epoch
            # 7 logging
            if print_all is True and not b % batch_size:
                print(
                    "Epoch: %03d/%03d | Batch %03d~%03d/%03d | Train loss: %.4f"
                    % (e + 1, num_epochs, b, b + batch_size - 1, trainx.shape[0], loss)
                )

        model.eval()
        for b in range(0, valx.shape[0], batch_size):
            # 1. Zero out grad after each batch
            for k in optimizers:
                k.zero_grad()
            # 2. Forward
            y_scores = model.forward(valx[b : b + batch_size])
            y_true = valy[b : b + batch_size]
            # 3. Calculate validation loss
            loss = criterion.forward(y_scores, y_true)
            validation_losses[
                e
            ] += loss  # total loss for whole validation for e-th epoch
            # 4. Calculate validation error count
            for i in range(y_scores.shape[0]):
                if np.argmax(y_scores[i]) != np.argmax(y_true[i]):
                    validation_errors[e] += 1
            # 5 logging
            if print_all is True and not b % batch_size:
                print(
                    "Epoch: %03d/%03d | Batch %03d~%03d/%03d | Validation loss: %.4f"
                    % (e + 1, num_epochs, b, b + batch_size - 1, valx.shape[0], loss)
                )

        end_time = time.time()

        # Accumulate data (running average of losses of minibatches over one epoch)
        training_losses[e] = (
            training_losses[e] / num_batch_train
        )  # divided by number of batches
        validation_losses[e] = validation_losses[e] / num_batch_valid
        # error rate for each epoch
        training_errors[e] = training_errors[e] / trainx.shape[0]
        validation_errors[e] = validation_errors[e] / valx.shape[0]

        # logging for epochs
        print(
            "Epoch: %03d/%03d | Train loss: %.4f | Validation loss: %.4f "
            % (e + 1, num_epochs, training_losses[e], validation_losses[e])
        )
        print(
            "Epoch: %03d/%03d | Train error: %.4f | Validation error: %.4f "
            % (e + 1, num_epochs, training_errors[e], validation_errors[e])
        )
        print("Time elapsed: %.2f min" % ((end_time - start_time) / 60))

    return (training_losses, training_errors, validation_losses, validation_errors)


def evaluator_multiclass(model, criterion, test_data, batch_size):
    """
    Evaluate the model's performance on test data

    Args:
        model (MLP <nn.modules.mlp> object): trained model.
        criterion (CrossEntropyLoss <nn.optim.loss> object):
                here we use CrossEntropyLoss classification
        test_data (list): [training data input, targets]
                input is (num_examples, num_features)
                targets is (num_examples, num_classes) one-hot encoded
        batch_size (int): batch size for loading the data.
                This should not affect the error rate or accuracy.

    Returns:
        running_loss: running average of losses of minibatches
        test_error: error rate in the whole data
    """

    test_x, test_y = test_data
    idxs = np.arange(len(test_x))
    num_batch_test = np.ceil(test_x.shape[0] / batch_size)

    running_loss = 0.0
    error_count = 0

    model.eval()
    for b in range(0, test_x.shape[0], batch_size):
        # 1. Forward
        y_scores = model.forward(test_x[b : b + batch_size])
        y_true = test_y[b : b + batch_size]
        # 3. Calculate loss
        loss = criterion.forward(y_scores, y_true)
        running_loss += loss  # total loss for the test data
        # 4. Calculate test error count
        for i in range(y_scores.shape[0]):
            if np.argmax(y_scores[i]) != np.argmax(y_true[i]):
                error_count += 1

    # total loss
    running_loss = running_loss / num_batch_test
    # error rate
    test_error = error_count / test_x.shape[0]

    # logging
    print("Testing Loss: ", running_loss)
    print("Testing Error: ", test_error)

    return (running_loss, test_error)
