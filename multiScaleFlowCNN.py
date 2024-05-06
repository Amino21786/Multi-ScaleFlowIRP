import jax
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import matplotlib.pyplot as plt

# Creation of the seven layer CNN in JAX
class CNN(eqx.Module):
    layers: list

    def __init__(self, key):
        subkeys = jr.split(key, 4)

        self.layers = [
            eqx.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=4, key=subkeys[0]),
            eqx.nn.MaxPool2d(kernel_size=2, stride=2),
            eqx.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=4, key=subkeys[1]),
            eqx.nn.MaxPool2d(kernel_size=2, stride=2),
            eqx.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=4, key=subkeys[2]),
            eqx.nn.MaxPool2d(kernel_size=2, stride=2),
            jnp.ravel,
            eqx.nn.Linear(in_features=50, out_features=4, key=subkeys[3]),
        ]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
# MSE loss function
def loss(model, xs, ys):
    ys_pred = jax.vmap(model)(xs)
    return jnp.mean((ys_pred - ys)**2)

# Training the model with the vmaps, K tensors (train and test), with the model, optimiser, steps and epochs
def train(vmap_train, 
          Ks_train, 
          vmap_test, 
          Ks_test, 
          model, 
          optim, 
          steps,
          epochs,
          key
):
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(model, opt_state, xs, ys):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, xs, ys)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value
    
    for epoch in range(epochs):
        for step in range(steps):
            key, subkey = jr.split(key, 2)
            x = jr.choice(subkey, vmap_train, (1, ))
            y = jr.choice(subkey, Ks_train, (1, ))

            model, opt_state, train_loss = make_step(model, opt_state, x, y)
        
        train_loss = loss(model, vmap_train, Ks_train)
        test_loss = loss(model, vmap_test, Ks_test)
            
        print(f"{epoch =}")
        print(f"Train loss: {train_loss}")
        print(f"Test loss: {test_loss}")
    
    return model
    
# Main running of the code 
if __name__ == "__main__":

    # Loads the numpy arrays of the velocity maps (FNO output, (x,y)) and the normalised K tensors from data.py file
    vmap = np.load('dataCNN/vmap.npy', allow_pickle=True)
    Ks = np.load('dataCNN/Ksnorms.npy', allow_pickle=True)

    # Conversion to JNP arrays for use
    vmap = jnp.array(vmap)
    Ks = jnp.array(Ks, dtype='float32')

    key = jr.PRNGKey(0)
    key, subkey = jr.split(key)

    vmap = jr.permutation(subkey, vmap)
    Ks = jr.permutation(subkey, Ks)

    # Initiate the train and test split
    split = 700
    vmap_train = vmap[:split]
    vmap_test = vmap[split:]
    Ks_train = Ks[:split]
    Ks_test = Ks[split:]
    
    # Model setup
    key, subkey = jr.split(key)
    model = CNN(key)
    optim = optax.adam(0.003)

    # Train the model with the training data
    model = train(vmap_train, Ks_train, vmap_test, Ks_test, model, optim, 800, 200, subkey)

    # Testing the model and saving the np arrays
    Kspred = jax.vmap(model)(vmap_test)
    print(Kspred)
    print(Ks_test)

    np.save('dataCNN/Kspred.npy', Kspred)
    np.save('dataCNN/Kstrue.npy', Ks_test)