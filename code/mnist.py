from torchvision import datasets
from torchvision.transforms import ToTensor

def main():
    train_data = datasets.MNIST(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True,
    )

    test_data = datasets.MNIST(
        root='data',
        train=False,
        transform=ToTensor()
    )

    print(train_data)
    print(train_data.data.size())
    print(train_data.targets.size())
    print(test_data)

if __name__ == "__main__":
    main()
