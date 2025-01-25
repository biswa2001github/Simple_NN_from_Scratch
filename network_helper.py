def train(network, loss, loss_prime, x_train, y_train, epochs = 100, epoch_step=10, learning_rate = 0.01, verbose = True):
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)
        if verbose and (e + 1) % epoch_step == 0:
            print(f"{e + 1}/{epochs}, error={error}")

            

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output
