import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    torch.manual_seed(42) 
    np.random.seed(42)

    # Check which GPU is available
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(f'Using device: {device}')
    return (
        DataLoader,
        device,
        nn,
        np,
        optim,
        plt,
        torch,
        torchvision,
        transforms,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Module 3: Practical 1 - Fully Connected Neural Networks
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Loss Functions

    We start with a brief review of Fully Connected Neural Networks. Recall that in Supervised Learning we tune the parameters of the neural network to minimize the "Loss Function" that represents how close the neural network outputs are to the labels defined in the dataset. There are many choices for loss functions, but the most common are Mean Square Loss for regression problems and Categorial Cross Entropy Loss for classification problems.

    #### **Mean Square Loss**
    $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$

    #### **Categorical Cross Entropy Loss**
    $-\frac{1}{n}\sum_{i=1}^{n}\sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$

    For the classification problems the network typically outputs $\hat{y}_{i, c}$ the probability of the example $i$ belonging to the class $c$. In the formula above, $y_{i,c}$ is equal to $1$ for the correct class, and $0$ otherwise, so we are really just averaging the logarithms of the probability the network outputs for the correct label in the training data.

    - Loss: $-\log(p_{\text{correct class}})$
    - Directly uses the index of the correct class
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### **Where does Cross Entropy come from? Maximum Likelihood**

    The Categorical Cross Entropy Loss is not an arbitrary choice — it falls out of the **maximum likelihood** principle, and it is exactly the **cross-entropy** we introduced in **Module 1, Practical 1 (Probability)**.

    For a classification dataset of \( N \) i.i.d. examples \( (x_i, y_i) \), the network outputs \( Q(y \mid x_i; \theta) = \hat{y}_{i,c} \), the predicted probability of class \( c \) (here \( \theta \) collects all the network weights). The **likelihood** of the observed labels is the product of the probabilities the model assigns to each correct label, and maximum likelihood chooses the \( \theta \) that makes them most probable:

    $$
    \theta^\star = \arg\max_{\theta} \prod_{i=1}^{N} Q(y_i \mid x_i; \theta).
    $$

    Taking a logarithm (monotonic, so the maximizer is unchanged) turns the product into a sum; negating and scaling by \( \tfrac{1}{N} \) (a positive constant) turns the maximization into a minimization. Using the one-hot encoding \( y_{i,c} \) (equal to \( 1 \) for the correct class and \( 0 \) otherwise), the log-probability of the correct label becomes a sum over all classes:

    $$
    \theta^\star = \arg\min_{\theta} \underbrace{-\frac{1}{N} \sum_{i=1}^{N} \log Q(y_i \mid x_i; \theta)}_{\text{negative log-likelihood}}
    = \arg\min_{\theta} \underbrace{-\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log \hat{y}_{i,c}}_{\text{Categorical Cross Entropy Loss}}.
    $$

    So minimizing the Categorical Cross Entropy Loss **is** maximum likelihood estimation of the class probabilities.

    **Connecting back to the Module 1 definition.** Each per-example inner sum is *literally* the Module 1 cross-entropy \( H[P, Q] = -\sum_x P(x) \log Q(x) \), read with the **class index \( c \) playing the role of the outcome \( x \)**:

    $$
    -\sum_{c=1}^{C} y_{i,c} \log \hat{y}_{i,c} \;=\; H[P_i, Q_i],
    $$

    where the one-hot label \( y_{i,\cdot} \) is the **true distribution** \( P_i \) over classes (all its mass on the correct class) and the softmax outputs \( \hat{y}_{i,\cdot} \) are the **model distribution** \( Q_i \). The loss is just the average of these per-example cross-entropies, \( \frac{1}{N} \sum_i H[P_i, Q_i] \). Since \( P_i \) is one-hot, \( H[P_i] = 0 \), so cross-entropy and KL divergence coincide here (\( H[P_i, Q_i] = D_{\text{KL}}(P_i \,\|\, Q_i) \)): minimizing the loss drives each \( Q_i \) toward the point-mass target \( P_i \).
    """)
    # > **What are we really approximating? (population view)**
    # >
    # > This identity is *exact*, but it uses the **empirical** label distribution — the one-hot vectors are point masses on the labels we happened to observe. The average \( \frac{1}{N}\sum_i \) is a Monte Carlo estimate of an expectation over the true \( p_{\text{data}} \), and by the definition \( H[P,Q] = \mathbb{E}_{z\sim P}[-\log Q(z)] \) that expectation is itself a cross-entropy:
    # >
    # > $$
    # > \frac{1}{N} \sum_{i=1}^{N} \big[-\log Q(y_i \mid x_i; \theta)\big]
    # > \;\;\xrightarrow{\;N \to \infty\;}\;\;
    # > \mathbb{E}_{x \sim p_{\text{data}}}\Big[\, H\big[p_{\text{data}}(y \mid x),\, Q(y \mid x)\big] \Big].
    # > $$
    # >
    # > This is the **conditional** cross-entropy (over the label, averaged over inputs) — *not* a joint cross-entropy over \( (x,y) \). So minimizing the loss approximately pushes \( Q(y \mid x) \) toward the true conditional \( p_{\text{data}}(y \mid x) \), which need **not** be one-hot: for a given input there may be genuine label ambiguity. The one-hot target is simply the empirical stand-in from a single observed label per example.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Recall that each artificial neuron in a neural network takes a linear combination of its inputs followed by the application of an activation function. For classification problems, the `softmax` activation function turns the outputs of the final layer into probabilities summing up to 1.
    """)
    return


@app.cell
def _(np):
    # Activation functions
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def softmax(x):
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / exp_x.sum()

    return sigmoid, softmax


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here is an example of what a simple Fully Connected Neural Network might looks like. Feel free to experiment with the inputs and the weights to see how they influence the output probabilities. Of course, the goal of training a neural network is to find the weights (coefficients) in each artificial neuron to minimize the loss function on the training examples.
    """)
    return


@app.cell(hide_code=True)
def _():
    # Create sample data for the network

    # Define input variables
    weather_examples = [
        [28.5, 45.0, 1015.2],
        [18.7, 78.3, 1008.5],
        [22.1, 85.6, 998.7]
    ]
    weather_categories = ["Sunny", "Rainy", "Stormy"]

    # Define input variables
    var_names = ["bias", "temperature (°C)", "humidity (%)", "pressure (hPa)"]
    return var_names, weather_categories, weather_examples


@app.cell(hide_code=True)
def _(mo):
    h1_weight_array = mo.ui.array([mo.ui.slider(start=-2, stop=2, step=0.2, label=f'''w[{i}]''', value=0) for i in range(4)])
    h2_weight_array = mo.ui.array([mo.ui.slider(start=-2, stop=2, step=0.2, label=f'''w[{i}]''', value=0) for i in range(4)])
    y1_weight_array = mo.ui.array([mo.ui.slider(start=-2, stop=2, step=0.2, label=f'''w[{i}]''', value=0) for i in range(2)])
    y2_weight_array = mo.ui.array([mo.ui.slider(start=-2, stop=2, step=0.2, label=f'''w[{i}]''', value=0) for i in range(2)])
    y3_weight_array = mo.ui.array([mo.ui.slider(start=-2, stop=2, step=0.2, label=f'''w[{i}]''', value=0) for i in range(2)])
    return (
        h1_weight_array,
        h2_weight_array,
        y1_weight_array,
        y2_weight_array,
        y3_weight_array,
    )


@app.cell(hide_code=True)
def _(mo):
    dropdown = mo.ui.dropdown(options={"Example 1":0, "Example 2":1, "Example 3":2}, value="Example 1")
    return (dropdown,)


@app.cell(hide_code=True)
def _(dropdown, mo, var_names, weather_categories, weather_examples):
    input_array = mo.ui.array([mo.ui.number(label=f'''{var_names[i]}''', value=weather_examples[dropdown.value][i-1]) for i in range(1,4)])
    label_dropdown = mo.ui.dropdown(options={weather_categories[0]:0, weather_categories[1]:1, weather_categories[2]:2}, value=weather_categories[dropdown.value])
    return input_array, label_dropdown


@app.cell(hide_code=True)
def _(
    dropdown,
    h_inputs,
    h_values,
    np,
    plt,
    sigmoid,
    v,
    var_values,
    w,
    weather_categories,
    y_inputs,
    y_values,
):
    import networkx as nx
    import math
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, height_ratios=[1, 3])

    # Main network diagram in top area (spanning 3 columns)
    ax_main = fig.add_subplot(gs[1, :])

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes with positions
    pos = {
        "x1": (0, 0), "x2": (0, 0.5), "x3": (0, 1), "bias": (0, -0.5), # Input layer
        "h1": (1.5, 0.25), "h2": (1.5, 0.75),            # Hidden layer
        "y1": (3, 0), "y2": (3, 0.5), "y3": (3, 1)               # Output layer
    }

    # Add nodes
    G.add_nodes_from(["x1", "x2", "x3", "bias", "h1", "h2", "y1", "y2", "y3"])

    # Add edges
    edges = [
        ("x1", "h1"), ("x1", "h2"), 
        ("x2", "h1"), ("bias", "h1"), ("x2", "h2"),
        ("x3", "h1"), ("x3", "h2"),
        ("h1", "y1"), ("h2", "y1"),
        ("h1", "y2"), ("h2", "y2"),
        ("h1", "y3"), ("h2", "y3")
    ]
    G.add_edges_from(edges)

    # Draw the graph
    nx.draw(G, pos, with_labels=False, node_size=3000, 
            node_color=["lightblue", "lightblue", "lightblue", "lightblue", "lightgreen", "lightgreen", "salmon", "salmon", "salmon"],
            arrowsize=20, arrowstyle='->', width=1.5, ax=ax_main)

    # Add custom node labels with interpolated values
    node_labels = {
        "bias": "1",
        "x1": f"x₁\n{var_values[1]}", 
        "x2": f"x₂\n{var_values[2]}", 
        "x3": f"x₃\n{var_values[3]}", 
        "h1": f"\n\nh₁\n\n" + fr"$\sigma({h_inputs[0]}) = {h_values[0]}$", 
        "h2": f"\n\nh₂\n\n" + fr"$\sigma({h_inputs[1]}) = {h_values[1]}$", 
        "y1": f"\n\ny₁\n\n ${y_values[0]}$",
        "y2": f"\n\ny₂\n\n ${y_values[1]}$",
        "y3": f"\n\ny₃\n\n ${y_values[2]}$"
    }
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=18, font_weight="bold", ax=ax_main)

    # Add edge labels with actual weight values
    edge_labels = {
        ("bias", "h1"): fr"$\times({w[0,0]})$", 
        ("x1", "h1"): fr"$\times({w[0,1]})$", 
        # ("x1", "h2"): fr"$\times({w[0,1]})$", 
        ("x2", "h1"): fr"$\times({w[0,2]})$", 
        # ("x2", "h2"): f"w3={w[1,1]}",
        ("x3", "h1"): fr"$\times({w[0,3]})$", 
        # ("x3", "h2"): f"w5={w[2,1]}",
        ("h1", "y1"): fr"$\times({v[0, 0]})$", 
        ("h2", "y1"): fr"$\times({v[0, 1]})$"
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=20, ax=ax_main)

    # Add layer titles
    ax_main.text(0, 1.5, "INPUT LAYER", fontsize=16, ha="center", fontweight="bold")
    ax_main.text(1.5, 1.5, "HIDDEN LAYER", fontsize=16, ha="center", fontweight="bold")
    ax_main.text(3, 1.5, "OUTPUT LAYER", fontsize=16, ha="center", fontweight="bold")

    ax_main.text(-0.5, 1, "Pressure", fontsize=16, ha="center", style='italic')
    ax_main.text(-0.5, 0.5, "Humidity", fontsize=16, ha="center", style='italic')
    ax_main.text(-0.5, 0, "Temperature", fontsize=16, ha="center", style='italic')


    # Add activation function labels
    ax_main.text(1.5, -0.3, "Activation: Sigmoid", fontsize=12, ha="center", style='italic')
    ax_main.text(3, -0.5, "Activation: Softmax", fontsize=12, ha="center", style='italic')

    example = dropdown.value
    ax_main.text(3.5, 1, f"{weather_categories[2]}", fontsize=16, ha="center", color = 'blue' if example == 2 else 'black')
    ax_main.text(3.5, 0.5, f"{weather_categories[1]}", fontsize=16, ha="center", color = 'blue' if example == 1 else 'black')
    ax_main.text(3.5, 0, f"{weather_categories[0]}", fontsize=16, ha="center", color = 'blue' if example == 0 else 'black')

    ax_main.text(2, -0.7, f"Loss: -log({y_values[example]}) = {-math.log(y_values[example])}", fontsize=20, ha="center", style='italic')


    # Color-code the layers with background shapes
    input_layer = plt.Rectangle((-0.5, -0.3), 0.8, 1.6, fill=True, alpha=0.1, color='blue')
    hidden_layer = plt.Rectangle((1, -0.05), 1, 1.1, fill=True, alpha=0.1, color='green')
    output_layer = plt.Rectangle((2.6, -0.3), 2, 1.5, fill=True, alpha=0.1, color='red')

    ax_main.add_patch(input_layer)
    ax_main.add_patch(hidden_layer)
    ax_main.add_patch(output_layer)

    # Add a sigmoid activation function plot in the bottom area
    ax_sigmoid = fig.add_subplot(gs[0, 1:2])
    x = np.linspace(-6, 6, 100)
    y = sigmoid(x)

    # Plot the sigmoid function
    ax_sigmoid.plot(x, y, 'b-', linewidth=2)
    ax_sigmoid.set_title('Sigmoid Activation Function: σ(z) = 1/(1+e^(-z))', fontsize=14)
    ax_sigmoid.set_xlabel('Input (z)', fontsize=12)
    ax_sigmoid.set_ylabel('Output: σ(z)', fontsize=12)
    ax_sigmoid.grid(True, alpha=0.3)
    ax_sigmoid.set_xlim(-6, 6)
    ax_sigmoid.set_ylim(-0.1, 1.1)


    ax_main.text(3, 1.4, f"softmax({y_inputs[2], y_inputs[1], y_inputs[1]})", fontsize=18, ha="center")

    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(
    dropdown,
    h1_weight_array,
    h2_weight_array,
    input_array,
    label_dropdown,
    mo,
    np,
    sigmoid,
    softmax,
    y1_weight_array,
    y2_weight_array,
    y3_weight_array,
):
    var_values = [1, input_array[0].value, input_array[1].value, input_array[2].value]
    #[1, example["temp"], example["humidity"], example["pressure"]]

    # Define weights (randomly initialized)
    w = np.round(np.random.uniform(-1, 1, size=(2, 4)), 2)  # 3 inputs × 2 hidden neurons
    # w = np.array([
    #     [ 0.3, -0.2,  0.1,  0.4],  # temperature weights
    #     [-0.1,  0.5,  0.2, -0.3],  # humidity weights
    #     [ 0.2,  0.1, -0.4,  0.1]   # pressure weights
    # ])

    v = np.round(np.random.uniform(-1, 1, size=(3, 4)), 2)       # 2 hidden neurons × 1 output
    # Weights for output layer (4 hidden neurons × 3 output classes)
    # v = np.array([
    #     [ 0.5,  0.1, -0.3],  # h1 to outputs
    #     [-0.2,  0.4,  0.1],  # h2 to outputs
    #     [ 0.1, -0.1,  0.5],  # h3 to outputs
    #     [-0.1,  0.3,  0.2]   # h4 to outputs
    # ])


    w[0,0] = h2_weight_array[3].value
    w[0,1] = h2_weight_array[2].value
    w[0,2] = h2_weight_array[1].value
    w[0,3] = h2_weight_array[0].value

    w[1,0] = h1_weight_array[3].value
    w[1,1] = h1_weight_array[2].value
    w[1,2] = h1_weight_array[1].value
    w[1,3] = h1_weight_array[0].value

    v[0,0] = y1_weight_array[0].value
    v[0,1] = y1_weight_array[1].value
    v[1,0] = y2_weight_array[0].value
    v[1,1] = y2_weight_array[1].value
    v[2,0] = y3_weight_array[0].value
    v[2,1] = y3_weight_array[1].value


    # Calculate hidden layer values
    h_inputs = np.zeros(2)
    for i in range(2):
        h_inputs[i] = np.round(np.sum([w[i, j] * var_values[j] for j in range(4)]), 2)
    h_values = np.round(sigmoid(h_inputs), 2)

    # Calculate output values
    y_inputs = np.zeros(3)
    for i in range(3):
        y_inputs[i] = np.round(np.sum([v[i, j] * h_values[j] for j in range(2)]), 2)
    y_values = np.round(softmax(y_inputs), 2) # Using softmax for multi-class classification

    mo.vstack([dropdown, mo.hstack([input_array[::-1], label_dropdown], justify="start"), mo.hstack([h1_weight_array, h2_weight_array, mo.vstack([y3_weight_array, y2_weight_array, y1_weight_array])])], justify="start")
    return h_inputs, h_values, v, var_values, w, y_inputs, y_values


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Clearly, we need a systematic way to calculate these weights. Below we go over a typical neural network setup and training in PyTorch for the problem of classification of images from the CIFAR10 dataset.
    """)
    return


@app.cell
def _(np, torchvision, transforms):
    BATCH_SIZE = 32
    CLASSES = np.array(
        [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
    )

    # Prepare the Data
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return BATCH_SIZE, CLASSES, test_dataset, train_dataset


@app.cell
def _(train_dataset):
    # check the shape of the first image
    train_dataset[0][0].shape
    return


@app.cell
def _(CLASSES, plt, train_dataset):
    _img, _label = train_dataset[0]
    print("Label: ", _label)
    print("Class: ", CLASSES[_label])
    plt.imshow(_img.permute(1,2,0).squeeze())
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that each training example is a tuple containing the three dimensional image tensor (C by H by W) and the label.
    """)
    return


@app.cell
def _(train_dataset):
    print("Some individual pixel: ", train_dataset[54][0][1, 12, 13])
    print("Corresponding Label: ", train_dataset[54][1])
    return


@app.cell
def _(CLASSES, np, plt, train_dataset):
    _random_index = np.random.randint(len(train_dataset))
    _img, _label = train_dataset[_random_index]
    print("Label: ", _label)
    print("Class: ", CLASSES[_label])
    plt.imshow(_img.permute(1,2,0).squeeze())
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    PyTorch has a special *DataLoader* class that takes care of some of the tedious details of constructing batches from the dataset.
    """)
    return


@app.cell
def _(BATCH_SIZE, DataLoader, test_dataset, train_dataset):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader, train_loader


@app.cell
def _(train_dataset, train_loader):
    print("Length of the train dataset: ", len(train_dataset))
    print("Length of list(train_loader): ", len(list(train_loader)))
    print("Type of the first element of list(train_loader): ", type(list(train_loader)[0]))
    print("Type of the first element of list(train_loader)[0]: ", type(list(train_loader)[0][0]))
    print("Shape of the first element of list(train_loader)[0]: ", list(train_loader)[0][0].shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Dataloaders** provide multiple ways to access the data, either by converting it into a **Python list** or by using an **iterable**.

    Using `list(train_loader)`, as we have, loads the **entire dataset into memory**, which can be **slow** and even **fail** when dealing with large datasets.

    Since **neural network training algorithms process data in batches**, it is more efficient to use an **iterator**. Instead of retrieving the first batch like this:
    ```python
    list(train_loader)[0]
    ```
    which loads everything into memory, we use:
    ```python
    next(iter(train_loader))
    ```
    This approach retrieves only the first batch without loading the entire dataset, making it memory-efficient and faster.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Let's load the first batch of our data (image and label) and display it using the `matplotlib` library.

    Recall that the shape returned by
    ```python
    next(iter(train_loader))
    ```
    is 32 by 3 by 32 by 32. This shape represents the batch size, number of channels, height, and width of the image, respectively.
    """)
    return


@app.cell
def _(CLASSES, plt, train_loader):
    next_batch_images, next_batch_labels = next(iter(train_loader))
    _first_img = next_batch_images[0] # retrieve the first image from the batch of 32
    _first_label = next_batch_labels[0] # retrieve the first label from the batch of 32
    plt.imshow(_first_img.permute(1, 2, 0)) # imshow requires the image to be in height x width x channels format
    print("Label: ", CLASSES[_first_label])
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Why is the first image different from when we used the dataset directly?
    """)
    return


@app.cell
def _():
    # Parameters
    NUM_CLASSES = 10
    EPOCHS = 10
    return EPOCHS, NUM_CLASSES


@app.cell
def _(NUM_CLASSES, device, nn, torch):
    # Build the model
    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(32 * 32 * 3, 200)
            self.fc2 = nn.Linear(200, 150)
            self.fc3 = nn.Linear(150, NUM_CLASSES)

        def forward(self, x):
            x = self.flatten(x)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.softmax(self.fc3(x), dim=1)
            return x

    model = MLP().to(device)
    print(model)

    # Compare to TensorFlow
    # input_layer = layers.Input(shape=(32, 32, 3))
    # x = layers.Flatten()(input_layer)
    # x = layers.Dense(units=200, activation = 'relu')(x)
    # x = layers.Dense(units=150, activation = 'relu')(x)
    # output_layer = layers.Dense(units=10, activation = 'softmax')(x)
    # model = models.Model(input_layer, output_layer)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Model Checkpoints

    **Checkpoints** are snapshots of the model's state (e.g. model weights) during training that allow you to:

    - Resume training from any point if interrupted
    - Compare performance across different epochs
    - Save the best performing model automatically
    - Experiment with different training strategies

    #### What Gets Saved in a Checkpoint?

    In particular, in this example, the checkpoint will contain:

    - **Model weights** (`model.state_dict()`) - The learned parameters
    - **Optimizer state** (`optimizer.state_dict()`) - Learning rates, momentum, etc.
    - **Epoch number** - Which training epoch this represents
    - **Loss and accuracy** - Performance metrics at this point

    #### Why Save Optimizer State?

    The optimizer state is crucial because modern optimizers like **Adam** maintain:

    - **Adaptive learning rates** per parameter
    - **Momentum** from previous gradient updates
    - **Internal counters** for learning rate scheduling

    Without saving optimizer state, resuming training would:

    - Reset learning rates to initial values
    - Lose accumulated momentum
    - Potentially cause training instability or slower convergence
    """)
    return


@app.cell
def _(torch):
    import os

    def save_checkpoint(model, optimizer, epoch, loss, accuracy, checkpoint_dir='checkpoints_fcnn'):
        """Save model checkpoint"""
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy
        }

        # Save latest checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'fcnn_epoch_{epoch:03d}.pth')
        torch.save(checkpoint, checkpoint_path)

        return checkpoint_path

    def load_checkpoint(model, optimizer, checkpoint_path, device):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        accuracy = checkpoint['accuracy']

        print(f"Loaded checkpoint from epoch {epoch}")
        print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")

        return epoch, loss, accuracy

    return (save_checkpoint,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Training the model
    """)
    return


@app.cell
def _(
    EPOCHS,
    device,
    losschart,
    model,
    nn,
    optim,
    save_checkpoint,
    torch,
    train_loader,
    widget,
):
    from tqdm import tqdm

    datalogs = []
    best_accuracy = 0.0

    # Train the model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # To load from a checkpoint, uncomment the line below:
    # start_epoch, _, _ = load_checkpoint(model, optimizer, 'checkpoints_fcnn/fcnn_epoch_005.pth', device)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        running_correct, running_total = 0, 0

        model.train()
        train_loader_with_progress = tqdm(iterable=train_loader, ncols=120, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for batch_number, (inputs, labels) in enumerate(train_loader_with_progress):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            # predicted = torch.argmax(outputs.data)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # log data for tracking
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)
            running_loss += loss.item()  

            if (batch_number % 100 == 99):
                train_loader_with_progress.set_postfix({'avg accuracy': f'{running_correct/running_total:.3f}', 'avg loss': f'{running_loss/(batch_number+1):.4f}'})

                datalogs.append({
                    "epoch": epoch + batch_number / len(train_loader), 
                    "train_loss": running_loss / (batch_number + 1),
                    "train_accuracy": running_correct/running_total,
                })

        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * running_correct / running_total

        datalogs.append({
            "epoch": epoch + 1, 
            "train_loss": epoch_loss,
            "train_accuracy": running_correct/running_total,
        })

        # Save checkpoint every epoch
        checkpoint_path = save_checkpoint(
            model, optimizer, epoch + 1, epoch_loss, epoch_accuracy
        )

        # Save best model
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_path = save_checkpoint(
                model, optimizer, epoch + 1, epoch_loss, epoch_accuracy, 
                checkpoint_dir='checkpoints_fcnn/best'
            )
            print(f"New best model saved! Accuracy: {epoch_accuracy:.2f}%")

        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.2f}%")
        print(f"Checkpoint saved: {checkpoint_path}")

        widget.src = losschart(datalogs)

    print("Finished Training")
    return


@app.cell
def _(plt):
    import altair as alt
    from mofresh import refresh_matplotlib, ImageRefreshWidget
    import polars as pl

    widget = ImageRefreshWidget(src="")

    @refresh_matplotlib
    def losschart(data):
        df = pl.DataFrame(data)
        plt.plot(df["epoch"], df["train_loss"])
        plt.ylabel("Loss")
        plt.xlabel("Epoch")

    widget
    return losschart, widget


@app.cell
def _(device, model, test_loader, torch):
    # Evaluation
    test_correct = 0
    test_total = 0
    model.eval()
    with torch.no_grad():
        for test_images, test_labels in test_loader:
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)
            test_outputs = model(test_images)
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_total += test_labels.size(0)
            test_correct += (test_predicted == test_labels).sum().item()

    test_accuracy = 100 * test_correct / test_total
    print(f"Accuracy of the network on the 10000 test images: {test_accuracy:.2f}%")
    return (test_accuracy,)


@app.cell(hide_code=True)
def _(mo, test_accuracy):
    mo.md(fr"""
    The model has an **accuracy of {test_accuracy:.2f}%** on the test set, which is **better than random guessing** (10 classes).  

    However, this accuracy is **low** compared to **state-of-the-art models**.  

    The **simple model** we built has **limited capacity** to learn the **complex patterns** in the CIFAR-10 dataset.  

    Next, we will build a **more advanced model** using convolutional neural network (CNN) to **improve accuracy** and **learn more complex patterns** in the data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We end with seeing how the trained model classified 10 random images from a test batch.
    """)
    return


@app.cell
def _(CLASSES, device, model, np, plt, test_loader, torch):
    _images, _labels = next(iter(test_loader))
    _images = _images.to(device)
    _labels = _labels.to(device)
    _outputs = model(_images).to(device)
    _, preds = torch.max(_outputs, 1)
    preds_single = CLASSES[preds.cpu().numpy()]
    actual_single = CLASSES[_labels.cpu().numpy()]
    n_to_show = 10
    indices = np.random.choice(range(len(_images)), n_to_show)

    _fig = plt.figure(figsize=(15, 3))
    _fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for _i, idx in enumerate(indices):
        img = _images[idx].cpu().numpy().transpose((1, 2, 0))
        ax = _fig.add_subplot(1, n_to_show, _i + 1)
        ax.axis("off")
        ax.text(
            0.5,
            -0.35,
            "pred = " + str(preds_single[idx]),
            fontsize=10,
            ha="center",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            -0.7,
            "act = " + str(actual_single[idx]),
            fontsize=10,
            ha="center",
            transform=ax.transAxes,
        )
        ax.imshow(img)
    plt.gcf()
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Loading Checkpoints

    The `load_checkpoint()` function restores both the model and optimizer to their exact state from a saved checkpoint.

    #### Function Usage:
    ```python
    epoch, loss, accuracy = load_checkpoint(model, optimizer, checkpoint_path, device)
    ```

    #### Parameters:
    - **`model`**: The neural network to load weights into
    - **`optimizer`**: The optimizer to restore state for
    - **`checkpoint_path`**: Path to the saved checkpoint file
    - **`device`**: Device to load the checkpoint on ('cpu', 'cuda', 'mps')

    #### Returns:
    - **`epoch`**: The epoch number when this checkpoint was saved
    - **`loss`**: The training loss at that epoch
    - **`accuracy`**: The training accuracy at that epoch

    #### Common Use Cases:

    **1. Resume Training After Interruption:**
    ```python
    # Load latest checkpoint and continue from where you left off
    epoch, _, _ = load_checkpoint(model, optimizer, 'checkpoints_fcnn/fcnn_epoch_025.pth', device)

    # Continue training from epoch + 1
    for new_epoch in range(epoch, EPOCHS):
        # training code...
    ```

    **2. Load Best Model for Inference:**
    ```python
    # Load the best performing model (no need for optimizer state in inference)
    load_checkpoint(model, optimizer, 'checkpoints_fcnn/best/fcnn_epoch_015.pth', device)
    model.eval()  # Set to evaluation mode
    # Use model for predictions...
    ```

    **3. Compare Different Epochs:**
    ```python
    # Test epoch 5 performance
    load_checkpoint(model, optimizer, 'checkpoints_fcnn/fcnn_epoch_005.pth', device)
    test_model(model)

    # Test epoch 10 performance
    load_checkpoint(model, optimizer, 'checkpoints_fcnn/fcnn_epoch_010.pth', device)
    test_model(model)
    ```

    #### Important Notes:
    - The model architecture must match the saved checkpoint
    - Device mapping handles loading checkpoints across different devices
    - For inference only, you can pass a dummy optimizer (but it's still required)
    """)
    return


@app.cell
def _():
    # Example: Load a specific checkpoint and test it
    # Uncomment the lines below to load and test a checkpoint

    # # Create a fresh optimizer for loading
    # _test_optimizer = optim.Adam(model.parameters(), lr=0.0005)
    # _epoch, _loss, _accuracy = load_checkpoint(model, _test_optimizer, 'checkpoints_fcnn/fcnn_epoch_005.pth', device)

    # # Test the loaded model
    # model.eval()
    # _test_images, _test_labels = next(iter(test_loader))
    # _test_images = _test_images.to(device)
    # _test_labels = _test_labels.to(device)
    # _test_outputs = model(_test_images)
    # _, _test_preds = torch.max(_test_outputs, 1)

    # print(f"Loaded model from epoch {_epoch}")
    # print(f"Sample predictions: {CLASSES[_test_preds[:5].cpu().numpy()]}")
    # print(f"Actual labels: {CLASSES[_test_labels[:5].cpu().numpy()]}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
