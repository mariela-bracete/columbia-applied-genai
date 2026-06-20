import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Module 3: Practical 3 - Convolutional Neural Networks""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""In this practical, we "upgrade" our Fully Connected Neural Network to a Convolutional Neural Network (CNN). Except for the model definition, most of the code here is identical to Practical 1.""")
    return


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
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    print(f"Using device: {device}")
    return (
        DataLoader,
        F,
        device,
        nn,
        np,
        optim,
        plt,
        torch,
        torchvision,
        transforms,
    )


@app.cell
def _(torchvision, transforms):
    BATCH_SIZE = 32

    # Prepare the Data
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    return BATCH_SIZE, test_dataset, train_dataset


@app.cell
def _(train_dataset):
    train_dataset[0][0].shape
    return


@app.cell
def _(plt, train_dataset):
    first_img, first_label = train_dataset[0]
    print("Label: ", first_label)
    plt.imshow(first_img.permute(1, 2, 0).squeeze())
    plt.show()
    return (first_label,)


@app.cell
def _(first_label, np):
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
    CLASSES[first_label]
    return (CLASSES,)


@app.cell
def _(train_dataset):
    print("Some individual pixel: ", train_dataset[54][0][1, 12, 13])
    print("Corresponding Label: ", train_dataset[54][1])
    return


@app.cell
def _(CLASSES, np, plt, train_dataset):
    _random_index = np.random.randint(len(train_dataset))
    _img, _label = train_dataset[_random_index]
    print("Label: ", _label, CLASSES[_label])
    plt.imshow(_img.permute(1, 2, 0).squeeze())
    plt.show()
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
    print(
        "Shape of the first element of list(train_loader)[0]: ",
        list(train_loader)[0][0].shape,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Dataloaders** provide multiple ways to access the data, either by converting it into a **Python list** or by using an **iterable**.  

    Using `list(train_loader)`, as we have shown in Practical 1, loads the **entire dataset into memory**, which can be **slow** and even **fail** when dealing with large datasets.  

    Since **neural network training algorithms process data in batches**, it is more efficient to use an **iterator**. Instead of retrieving the first batch like this:  
    ```python
    list(train_loader)[0]
    ```
    which loads everything into memory, we use:
    ```python
    next(iter(train_loader))
    ```
    This approach retrieves only the first batch without loading the entire dataset, making it memory-efficient and faster.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    Let's load the first batch of our data (image and label) and display it using the `matplotlib` library.

    Recall that the shape returned by 
    ```python
    next(iter(train_loader))
    ```
    is 32 by 3 by 32 by 32. This shape represents the batch size, number of channels, height, and width of the image, respectively.
    """
    )
    return


@app.cell
def _(CLASSES, plt, train_loader):
    next_batch_images, next_batch_labels = next(iter(train_loader))
    _first_img = next_batch_images[0]  # retrieve the first image from the batch of 32
    _first_label = next_batch_labels[0]  # retrieve the first label from the batch of 32
    plt.imshow(
        _first_img.permute(1, 2, 0)
    )  # imshow requires the image to be in height x width x channels format
    plt.show()
    print("Label: ", CLASSES[_first_label])
    return


@app.cell
def _():
    # Parameters
    NUM_CLASSES = 10
    EPOCHS = 10
    return (EPOCHS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We now define our first CNN architecture. We will use this building block throughout the semester, so make sure you understand how it works. In particular, if we use the same padding and kernel (filter) size values for for both the width and height dimensions, the formula for the image size after the convolution or a max pooling layer is given by: 

    $$
    \frac{width + 2 * padding - (kernelsize -1) - 1}{S} + 1
    $$

    Verify this with `torch.nn.Conv2d` and `torch.nn.MaxPool2d` function documentation.
    """
    )
    return


@app.cell
def _(F, nn):
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(
                3, 16, kernel_size=3, padding=1
            )  # Input channels = 3, Output channels = 16
            self.pool = nn.MaxPool2d(
                kernel_size=2, stride=2
            )  # Pooling layer, will half the dimensions
            self.conv2 = nn.Conv2d(
                16, 32, kernel_size=3, padding=1
            )  # Input channels = 16, Output channels = 32
            self.fc1 = nn.Linear(32 * 8 * 8, 128)  # Fully connected layer
            self.fc2 = nn.Linear(128, 10)  # Output layer for 10 classes

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 32 * 8 * 8)  # Flatten
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    return (SimpleCNN,)


@app.cell
def _(SimpleCNN):
    net = SimpleCNN()
    return (net,)


@app.cell(hide_code=True)
def _(mo):
    input_shape_ui = mo.ui.array(
        [mo.ui.number(value=3 if i == 0 else 32) for i in range(3)]
    )

    layer_0_shape_ui = mo.ui.array([mo.ui.number(value=None) for i in range(3)])
    layer_1_shape_ui = mo.ui.array([mo.ui.number(value=None) for i in range(3)])
    layer_2_shape_ui = mo.ui.array([mo.ui.number(value=None) for i in range(3)])
    layer_3_shape_ui = mo.ui.array([mo.ui.number(value=None) for i in range(3)])
    layer_4_shape_ui = mo.ui.array([mo.ui.number(value=None) for i in range(1)])
    layer_5_shape_ui = mo.ui.array([mo.ui.number(value=None) for i in range(1)])

    mo.md(
        rf"""
        ## Fill in the table based on the input shape

        Input shape:

        $B \times$ {input_shape_ui[0]} $\times$ {input_shape_ui[1]} $\times$ {input_shape_ui[2]}

        After the first convolution ($B \times C \times H \times W$): 

        $B \times$ {layer_0_shape_ui[0]} $\times$ {layer_0_shape_ui[1]} $\times$ {layer_0_shape_ui[2]}


        After the first pooling ($B \times C \times H \times W$): 

        $B \times$ {layer_1_shape_ui[0]} $\times$ {layer_1_shape_ui[1]} $\times$ {layer_1_shape_ui[2]}

        After the second convolution ($B \times C \times H \times W$): 

        $B \times$ {layer_2_shape_ui[0]} $\times$ {layer_2_shape_ui[1]} $\times$ {layer_2_shape_ui[2]}

        After the second pooling ($B \times C \times H \times W$): 

        $B \times$ {layer_3_shape_ui[0]} $\times$ {layer_3_shape_ui[1]} $\times$ {layer_3_shape_ui[2]}

        After the first fully connected layer ($B \times C$): 

        $B \times$ {layer_4_shape_ui[0]}

        After the second fully connected layer ($B \times C$): 

        $B \times$ {layer_5_shape_ui[0]}

        """
    )
    return (
        input_shape_ui,
        layer_0_shape_ui,
        layer_1_shape_ui,
        layer_2_shape_ui,
        layer_3_shape_ui,
        layer_4_shape_ui,
        layer_5_shape_ui,
    )


@app.cell(hide_code=True)
def _(
    input_shape_ui,
    layer_0_shape_ui,
    layer_1_shape_ui,
    layer_2_shape_ui,
    layer_3_shape_ui,
    layer_4_shape_ui,
    layer_5_shape_ui,
    mo,
    net,
):
    _C = input_shape_ui[0].value
    _H = input_shape_ui[1].value
    _W = input_shape_ui[2].value

    input_shape = [_C, _H, _W]
    layer_shape = []

    _Ht = (
        _H + 2 * net.conv1.padding[0] - (net.conv1.kernel_size[0] - 1) - 1
    ) / net.conv1.stride[0] + 1
    _W = (
        _W + 2 * net.conv1.padding[1] - (net.conv1.kernel_size[1] - 1) - 1
    ) / net.conv1.stride[1] + 1
    _H = _Ht
    _C = net.conv1.out_channels
    layer_shape.append([_C, _H, _W])

    _Ht = (
        _H + 2 * net.pool.padding - (net.pool.kernel_size - 1) - 1
    ) / net.pool.stride + 1
    _W = (
        _W + 2 * net.pool.padding - (net.pool.kernel_size - 1) - 1
    ) / net.pool.stride + 1
    _H = _W
    layer_shape.append([_C, _H, _W])

    _Ht = (
        _H + 2 * net.conv2.padding[0] - (net.conv2.kernel_size[0] - 1) - 1
    ) / net.conv2.stride[0] + 1
    _W = (
        _W + 2 * net.conv2.padding[1] - (net.conv2.kernel_size[1] - 1) - 1
    ) / net.conv2.stride[1] + 1
    _H = _Ht
    _C = net.conv2.out_channels
    layer_shape.append([_C, _H, _W])

    _Ht = (
        _H + 2 * net.pool.padding - (net.pool.kernel_size - 1) - 1
    ) / net.pool.stride + 1
    _W = (
        _W + 2 * net.pool.padding - (net.pool.kernel_size - 1) - 1
    ) / net.pool.stride + 1
    _H = _W
    layer_shape.append([_C, _H, _W])

    layer_shape.append([net.fc1.out_features])
    layer_shape.append([net.fc2.out_features])

    mo.md(
        rf"""
        After the first convolution layer $B \times$ {layer_0_shape_ui[0].value} $\times$ {layer_0_shape_ui[1].value} $\times$ {layer_0_shape_ui[2].value}
        {'✅' if (not any([layer_shape[0][i] != layer_0_shape_ui[i].value for i in range(3)])) else '❌'}

        After the first pooling layer $B \times$ {layer_1_shape_ui[0].value} $\times$ {layer_1_shape_ui[1].value} $\times$ {layer_1_shape_ui[2].value}
        {'✅' if (not any([layer_shape[1][i] != layer_1_shape_ui[i].value for i in range(3)])) else '❌'}   

        After the second convolution layer $B \times$ {layer_2_shape_ui[0].value} $\times$ {layer_2_shape_ui[1].value} $\times$ {layer_2_shape_ui[2].value}
        {'✅' if (not any([layer_shape[2][i] != layer_2_shape_ui[i].value for i in range(3)])) else '❌'}

        After the second pooling layer $B \times$ {layer_3_shape_ui[0].value} $\times$ {layer_3_shape_ui[1].value} $\times$ {layer_3_shape_ui[2].value}
        {'✅' if (not any([layer_shape[3][i] != layer_3_shape_ui[i].value for i in range(3)])) else '❌'}   

        After the first fully connected layer $B \times$ {layer_4_shape_ui[0].value}
        {'✅' if (not any([layer_shape[4][i] != layer_4_shape_ui[i].value for i in range(1)])) else '❌'}    

        After the second fully connected layer $B \times$ {layer_5_shape_ui[0].value}
        {'✅' if (not any([layer_shape[5][i] != layer_5_shape_ui[i].value for i in range(1)])) else '❌'}     
        """
    )

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can improve the architecture by adding two more common operations: Batch Normalization and Dropout.

    - Batch Normalization (BatchNorm) is a technique that normalizes the inputs of each layer across a mini-batch to have zero mean and unit variance, then scales and shifts them with learnable parameters. It stabilizes and speeds up training by reducing internal covariate shift.

    - Dropout is a regularization method where, during training, a random fraction of neurons are “dropped” (set to zero) in each forward pass, preventing co-adaptation of features and reducing overfitting.
    """
    )
    return


@app.cell
def _(F, nn):
    class EnhancedCNN(nn.Module):
        def __init__(self):
            super(EnhancedCNN, self).__init__()
            # Convolutional Layer 1 with BatchNorm
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            # Convolutional Layer 2 with BatchNorm
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)  # Batch Normalization after Conv2

            # Third convolutional layer
            self.conv3 = nn.Conv2d(
                32, 64, kernel_size=3, padding=1
            )  # Output channels = 64
            self.bn3 = nn.BatchNorm2d(64)  # Batch Normalization after Conv3

            # Fourth convolutional layer
            self.conv4 = nn.Conv2d(
                64, 128, kernel_size=3, padding=1
            )  # Output channels = 128
            self.bn4 = nn.BatchNorm2d(128)  # Batch Normalization after Conv4

            # Fully connected layers with Dropout
            self.fc1 = nn.Linear(128 * 2 * 2, 128)
            self.dropout = nn.Dropout(0.5)  # Dropout with 50% probability
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            # Conf and pooling layers
            x = self.pool(
                F.relu(self.bn1(self.conv1(x)))
            )  # Conv -> BatchNorm -> ReLU -> Pool
            x = self.pool(
                F.relu(self.bn2(self.conv2(x)))
            )  # Conv -> BatchNorm -> ReLU -> Pool
            x = self.pool(
                F.relu(self.bn3(self.conv3(x)))
            )  # Conv -> BatchNorm -> ReLU -> Pool
            x = self.pool(
                F.relu(self.bn4(self.conv4(x)))
            )  # Conv -> BatchNorm -> ReLU -> Pool

            # Flatten the feature map
            x = x.view(-1, 128 * 2 * 2)

            # Fully connected layer 1 with Dropout
            x = F.relu(self.fc1(x))
            x = self.dropout(x)

            # Fully connected layer 2 (output)
            x = self.fc2(x)
            return x

    return (EnhancedCNN,)


@app.cell
def _(EnhancedCNN, device):
    model = EnhancedCNN().to(device)
    print(model)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### Training the model

    The training is exactly the same as for our Fully Connected Neural Network.
    """
    )
    return


@app.cell
def _(
    EPOCHS,
    device,
    losschart,
    model,
    nn,
    optim,
    torch,
    train_loader,
    widget,
):
    from tqdm import tqdm

    datalogs = []

    # Train the model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        running_correct, running_total = 0, 0

        model.train()
        train_loader_with_progress = tqdm(
            iterable=train_loader, ncols=120, desc=f"Epoch {epoch+1}/{EPOCHS}"
        )
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

            if batch_number % 100 == 99:
                train_loader_with_progress.set_postfix(
                    {
                        "avg accuracy": f"{running_correct/running_total:.3f}",
                        "avg loss": f"{running_loss/(batch_number+1):.4f}",
                    }
                )

                datalogs.append(
                    {
                        "epoch": epoch + batch_number / len(train_loader),
                        "train_loss": running_loss / (batch_number + 1),
                        "train_accuracy": running_correct / running_total,
                    }
                )

        datalogs.append(
            {
                "epoch": epoch + 1,
                "train_loss": running_loss / len(train_loader),
                "train_accuracy": running_correct / running_total,
            }
        )

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
    mo.md(rf"""The model has an **{test_accuracy:.2f}%** on the test set, much better than the accuracy we got from a fully connected network and due to parameter sharing of convolutional layers the CNN we constructed has less parameters!""")
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

    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for _i, idx in enumerate(indices):
        img = _images[idx].cpu().numpy().transpose((1, 2, 0))
        ax = fig.add_subplot(1, n_to_show, _i + 1)
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
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
