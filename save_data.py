from pathlib import Path
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as T


def dump_dataset(dataset, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (img, label) in enumerate(dataset):
        # img is PIL Image
        img = img.convert("RGB")  # diffusion expects 3 channels
        img.save(out_dir / f"{i:06d}.png")


def main():
    root = Path("data")
    root.mkdir(exist_ok=True)

    transform = T.ToTensor()  # no-op here, torchvision still gives PIL

    # -------------------------
    # MNIST
    # -------------------------
    mnist_train = datasets.MNIST(
        root="torch_datasets",
        train=True,
        download=True,
        transform=None,
    )
    mnist_test = datasets.MNIST(
        root="torch_datasets",
        train=False,
        download=True,
        transform=None,
    )

    dump_dataset(mnist_train, root / "mnist" / "train")
    dump_dataset(mnist_test, root / "mnist" / "test")

    # -------------------------
    # Fashion-MNIST
    # -------------------------
    fashion_train = datasets.FashionMNIST(
        root="torch_datasets",
        train=True,
        download=True,
        transform=None,
    )
    fashion_test = datasets.FashionMNIST(
        root="torch_datasets",
        train=False,
        download=True,
        transform=None,
    )

    dump_dataset(fashion_train, root / "fashion" / "train")
    dump_dataset(fashion_test, root / "fashion" / "test")

    print("Done! MNIST and Fashion-MNIST dumped to ./data/")


if __name__ == "__main__":
    main()
