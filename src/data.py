"""Dataset loaders, dataset metadata, and torchvision transforms."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as T
from medmnist import (
    OCTMNIST,
    BloodMNIST,
    BreastMNIST,
    DermaMNIST,
    PathMNIST,
    PneumoniaMNIST,
    RetinaMNIST,
    TissueMNIST,
)
from PIL import Image
from probly.datasets.torch import CIFAR10C, CIFAR10H, MedMNISTC, TinyImageNet
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from paths import DATA_PATH

if TYPE_CHECKING:
    from collections.abc import Sequence

# Number of classes per dataset.
DATASET_NUM_CLASSES = {
    "bloodmnist": 8,
    "cifar10": 10,
    "pathmnist": 9,
}

TRANSFORM_TRAIN = {
    "cifar10": T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
    # Leads with T.RGB() for the same reason pathmnist does: get_ood_data applies this transform
    # to the OOD sets too, and bloodmnist's grid includes single-channel ones. bloodmnist is itself
    # RGB, so this is a verified bitwise no-op on its own images and the artifacts already trained
    # against the previous composition stay valid.
    "bloodmnist": T.Compose(
        [
            T.RGB(),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
    # pathmnist is RGB, so T.RGB() is a no-op on its own images. It leads the composition anyway
    # because get_ood_data applies this transform to the OOD sets too, and pathmnist's near/far
    # grid includes single-channel ones (tissuemnist, octmnist, pneumoniamnist, breastmnist).
    "pathmnist": T.Compose(
        [
            T.RGB(),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
}

TRANSFORM_TEST = {
    "cifar10": T.Compose(
        [
            T.Resize((32, 32), antialias=True),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
    "bloodmnist": TRANSFORM_TRAIN["bloodmnist"],
    "pathmnist": TRANSFORM_TRAIN["pathmnist"],
}

TRANSFORM_SHIFT = {
    "cifar10": TRANSFORM_TEST["cifar10"],
    "bloodmnist": T.Compose([T.Resize((28, 28), antialias=True), TRANSFORM_TEST["bloodmnist"]]),
    "pathmnist": T.Compose([T.Resize((28, 28), antialias=True), TRANSFORM_TEST["pathmnist"]]),
}

SHIFT_CORRUPTIONS = {
    "bloodmnist": MedMNISTC.corruptions["bloodmnist"],
    "cifar10": tuple(CIFAR10C.corruptions[:15]),
    "pathmnist": MedMNISTC.corruptions["pathmnist"],
}


def resolve_shift_corruptions(dataset: str, corruptions: str | Sequence[str] | None) -> list[str]:
    """Resolve a shift config's corruptions field to a concrete corruption list for a dataset.

    Args:
        dataset: Dataset name; must have a corruption benchmark wired in get_shift_data.
        corruptions: The config value: None, "all", or an empty list sweeps the dataset's full
            benchmark set (SHIFT_CORRUPTIONS); a list of names runs that subset.

    Returns:
        The corruption names to sweep.

    Raises:
        ValueError: dataset has no corruption benchmark, or a named corruption is unknown for it.
    """
    if dataset not in SHIFT_CORRUPTIONS:
        raise ValueError(
            f"No corruption benchmark wired for dataset={dataset!r}. Available: {sorted(SHIFT_CORRUPTIONS)}."
        )
    available = SHIFT_CORRUPTIONS[dataset]
    if corruptions is None or corruptions == "all" or not list(corruptions):
        return list(available)
    named = [str(c) for c in corruptions]
    unknown = [c for c in named if c not in available]
    if unknown:
        raise ValueError(
            f"Unknown corruptions {unknown} for dataset={dataset!r}. Choose from {list(available)} or 'all'."
        )
    return named


# MedMNIST returns labels as shape (1,) ndarrays; squeeze so DataLoader batching yields
# (B,) integer tensors that F.cross_entropy expects.
MEDMNIST_TARGET_TRANSFORM = np.squeeze

# Native resolution the MedMNIST-C cells are downsampled to before the clean test transform. The
# benchmark ships 224x224, so every access otherwise pays an 8x antialiased downsample; see
# _medmnistc_cell. Keyed into the cache filenames, so changing it rebuilds rather than reuses.
MEDMNISTC_SIZE = 28
# Where the downsampled cells are cached. Small: a whole shift benchmark at 28x28 is on the
# order of 120 MB, versus 300 MB for a single corruption/severity cell at 224x224.
SHIFT_CACHE_PATH = DATA_PATH / "shift_cache"


class _CachedShiftDataset(Dataset):
    """A MedMNIST-C cell already downsampled to the model's native resolution.

    Holds the small uint8 images in memory and rebuilds the PIL image per access, so everything
    downstream of the resize is byte-for-byte the path MedMNISTC would have taken; only the
    repeated 224 to 28 antialiased downsample is skipped. scratch/test_shift_cache_equivalence.py
    checks the equivalence.
    """

    def __init__(self, images: np.ndarray, targets: np.ndarray, transform: Any) -> None:  # noqa: ANN401
        """Store the cell.

        Args:
            images: Downsampled images, shape (N, H, W) or (N, H, W, C), uint8.
            targets: Integer labels, shape (N,).
            transform: Transform applied to each PIL image; the clean test transform, since the
                resize is already baked into images.
        """
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self) -> int:
        """Number of images in the cell."""
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        """Return the (transformed image, label) pair at index."""
        img = Image.fromarray(self.images[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.targets[index])


def _medmnistc_cell(name: str, corruption: str, severity: int) -> tuple[np.ndarray, np.ndarray]:
    """Downsampled images and labels for one MedMNIST-C cell, built once and cached on disk.

    MedMNIST-C ships 224x224 images that every model here immediately downsamples to 28x28, so
    the antialiased resize is repeated identical work in every run that touches the cell. Doing
    it once and caching the result removes it from every later job. Loading the source npz also
    decompresses all five severities, so building one corruption's five cells costs five reads,
    but only once ever.

    Args:
        name: MedMNIST dataset flag.
        corruption: Corruption name, from SHIFT_CORRUPTIONS[name].
        severity: Corruption severity in 1..5.

    Returns:
        (images, targets): uint8 images at MEDMNISTC_SIZE and their integer labels.
    """
    path = SHIFT_CACHE_PATH / name / f"{corruption}_s{severity}_{MEDMNISTC_SIZE}.npz"
    if path.exists():
        with np.load(path) as cached:
            return cached["images"], cached["targets"]
    raw = MedMNISTC(root=DATA_PATH, dataset=name, corruption=corruption, severity=severity, download=True)
    resize = T.Resize((MEDMNISTC_SIZE, MEDMNISTC_SIZE), antialias=True)
    # Resize the same PIL image MedMNISTC.__getitem__ would have built, so the cached pixels are
    # exactly what the old path produced at this point in the pipeline.
    images = np.stack([np.asarray(resize(Image.fromarray(img))) for img in raw.data])
    targets = np.asarray(raw.targets, dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Uncompressed: these are small, and compression would cost more time than the read saves.
    np.savez(path, images=images, targets=targets)
    return images, targets


def get_train_data(
    name: str,
    val_split: float = 0.0,
    num_train: int | None = None,
    seed: int | None = None,
    train_transforms: bool = True,
    **loader_kwargs: Any,
) -> tuple[DataLoader[Any], None | DataLoader[Any], DataLoader[Any]]:
    """Return train, val, test loaders for a named dataset.

    Each dataset case owns its own val-split logic (random_split with transform swap,
    or use the dataset's built-in val, etc.). val_loader is None when val_split == 0.

    Args:
        name: Dataset name.
        val_split: Fraction of training data held out for validation. 0 disables it.
            For datasets with a built-in val split (MEDMNIST), val_split > 0 just toggles
            it on; the size is whatever the dataset provides.
        num_train: Random subsample size for the training split (after val-splitting).
            None uses the full training set.
        seed: RNG seed for shuffling, splitting, and subsampling.
        train_transforms: Apply the augmenting train transforms to the training split. False
            applies the evaluation transforms instead, for methods that fit feature references
            on the training data (e.g. credal_rl_multinomial) and must not see augmentation.
            The split indices are unaffected.
        **loader_kwargs: Forwarded to every returned DataLoader (batch_size, num_workers, ...).

    Returns:
        (train_loader, val_loader_or_none, test_loader).

    Raises:
        ValueError: num_train exceeds the training-split size.
    """
    rng = torch.Generator().manual_seed(seed) if seed is not None else None
    train_transform = (TRANSFORM_TRAIN if train_transforms else TRANSFORM_TEST).get(name)
    match name:
        case "cifar10":
            train = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=train_transform)
            test = torchvision.datasets.CIFAR10(
                root=DATA_PATH, train=False, download=True, transform=TRANSFORM_TEST[name]
            )
            val = None
            if val_split > 0:
                n = len(train)
                n_val = int(round(val_split * n))
                train, val = random_split(train, [n - n_val, n_val], generator=rng)
                # Use the eval transform on val. Subset shares its underlying dataset with
                # train, so shallow-copy first to avoid mutating train's transform.
                val.dataset = copy.copy(val.dataset)
                val.dataset.transform = TRANSFORM_TEST[name]  # ty: ignore[unresolved-attribute]
        case "bloodmnist":
            train = BloodMNIST(
                root=str(DATA_PATH),
                split="train",
                download=True,
                transform=train_transform,
                target_transform=MEDMNIST_TARGET_TRANSFORM,
            )
            test = BloodMNIST(
                root=str(DATA_PATH),
                split="test",
                download=True,
                transform=TRANSFORM_TEST[name],
                target_transform=MEDMNIST_TARGET_TRANSFORM,
            )
            val = None
            if val_split > 0:
                val = BloodMNIST(
                    root=str(DATA_PATH),
                    split="val",
                    download=True,
                    transform=TRANSFORM_TEST[name],
                    target_transform=MEDMNIST_TARGET_TRANSFORM,
                )
        case "pathmnist":
            train = PathMNIST(
                root=str(DATA_PATH),
                split="train",
                download=True,
                transform=train_transform,
                target_transform=MEDMNIST_TARGET_TRANSFORM,
            )
            test = PathMNIST(
                root=str(DATA_PATH),
                split="test",
                download=True,
                transform=TRANSFORM_TEST[name],
                target_transform=MEDMNIST_TARGET_TRANSFORM,
            )
            val = None
            if val_split > 0:
                val = PathMNIST(
                    root=str(DATA_PATH),
                    split="val",
                    download=True,
                    transform=TRANSFORM_TEST[name],
                    target_transform=MEDMNIST_TARGET_TRANSFORM,
                )
        case _:
            raise ValueError(f"Unknown dataset: {name}")

    if num_train is not None:
        if num_train > len(train):
            raise ValueError(f"num_train={num_train} exceeds the training-split size ({len(train)}).")
        train, _ = random_split(train, [num_train, len(train) - num_train], generator=rng)

    train_loader = DataLoader(train, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val, **loader_kwargs) if val is not None else None
    test_loader = DataLoader(test, **loader_kwargs)
    return train_loader, val_loader, test_loader


def get_test_data(name: str, **loader_kwargs: Any) -> DataLoader[Any]:
    """Return only the test loader for a named dataset."""
    match name:
        case "cifar10":
            test = torchvision.datasets.CIFAR10(
                root=DATA_PATH, train=False, download=True, transform=TRANSFORM_TEST[name]
            )
        case "bloodmnist":
            test = BloodMNIST(
                root=str(DATA_PATH),
                split="test",
                download=True,
                transform=TRANSFORM_TEST[name],
                target_transform=MEDMNIST_TARGET_TRANSFORM,
            )
        case "pathmnist":
            test = PathMNIST(
                root=str(DATA_PATH),
                split="test",
                download=True,
                transform=TRANSFORM_TEST[name],
                target_transform=MEDMNIST_TARGET_TRANSFORM,
            )
        case _:
            raise ValueError(f"Unknown dataset: {name}")
    return DataLoader(test, **loader_kwargs)


def get_first_order_data(name: str, **loader_kwargs: Any) -> DataLoader[Any]:
    """Return a test loader whose targets are first-order label distributions instead of class indices.

    Maps a base dataset name to its re-annotated counterpart over the same test instances:
    cifar10 -> CIFAR-10H (Peterson et al., 2019), human annotation counts over the 10000 CIFAR-10
    test images, normalized per instance to a distribution of shape (10,). Models trained on the
    base dataset can therefore be evaluated directly against these targets, e.g. distribution
    (box-containment) coverage of credal sets in experiments/coverage_efficiency.py.

    Args:
        name: Base dataset name (the one the artifact was trained on). Currently only cifar10 has
            a first-order counterpart. Missing files (CIFAR-10 images, CIFAR-10H counts) are
            downloaded on first use.
        **loader_kwargs: Forwarded to the DataLoader.

    Returns:
        Test DataLoader yielding (input, target) batches with targets of shape (batch, num_classes).

    Raises:
        ValueError: name has no first-order counterpart.
    """
    match name:
        case "cifar10":
            test = CIFAR10H(root=DATA_PATH, transform=TRANSFORM_TEST["cifar10"], download=True)
        case _:
            raise ValueError(f"No first-order counterpart for dataset: {name}")
    return DataLoader(test, **loader_kwargs)


def get_ood_data(
    name_id: str,
    name_ood: str,
    seed: int,
    **loader_kwargs: Any,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Return matched (id_loader, ood_loader) for OOD detection evaluation.

    Both loaders apply TRANSFORM_TEST[name_id] so OOD inputs are preprocessed
    the same way ID inputs are. Both are deterministically truncated to
    n = min(N_id_test, N_ood_test) via torch.randperm seeded by seed. Neither
    loader is shuffled.

    Args:
        name_id: In-distribution test dataset name. Must be a key in TRANSFORM_TEST.
            One of 'cifar10', 'pathmnist' or 'bloodmnist'.
        name_ood: Out-of-distribution test dataset name. For name_id='cifar10', one of
            'cifar100', 'tin', 'mnist', 'svhn', 'textures', 'places365', which are the six OOD
            sets of OpenOOD's CIFAR-10 benchmark: cifar100 and tin are its near-OOD pair, the
            rest its far-OOD sets. tin is
            TinyImageNet's 10000-image validation split, uncurated: OpenOOD additionally removes
            the TinyImageNet images that duplicate CIFAR content, this does not. DTD uses
            split='test' (1880 images), not the full ~5640 imageset. Places365
            uses the small val split (~36500 images) without OpenOOD's
            overlap-removal curation.
            For name_id='pathmnist', one of 'bloodmnist' or 'tissuemnist' (the near-OOD pair:
            also cells under a microscope, but a different specimen and label set) or
            'pneumoniamnist', 'octmnist', 'breastmnist', 'dermamnist' (the far-OOD sets, one per
            imaging modality). Near and far are defined by imaging modality relative to the ID
            set. Each OOD set uses its own test split, so the smaller ones cap n. Against the 7180
            ID test images the caps are bloodmnist 3421, dermamnist 2005, octmnist 1000,
            pneumoniamnist 624 and breastmnist 156; only tissuemnist does not bind.
            For name_id='bloodmnist', one of 'pathmnist' or 'tissuemnist' (the near-OOD pair:
            also cells under a microscope, but a different specimen and label set) or
            'dermamnist', 'retinamnist', 'octmnist', 'breastmnist' (the far-OOD sets, one per
            imaging modality). pathmnist, dermamnist and retinamnist are RGB like bloodmnist
            itself, so those three columns carry no channel-statistics shortcut; the
            single-channel sets are promoted by T.RGB() and a detector can separate them on
            colour variance alone. Against the 3421 ID test images the caps are breastmnist 156,
            retinamnist 400, octmnist 1000 and dermamnist 2005; pathmnist and tissuemnist do
            not bind.
        seed: RNG seed for the deterministic subset selection.
        **loader_kwargs: Forwarded to both DataLoaders (batch_size, num_workers, pin_memory, ...).

    Returns:
        (id_loader, ood_loader), neither shuffled, each yielding n examples.

    Raises:
        ValueError: name_id or name_ood not supported.
    """
    if name_id not in TRANSFORM_TEST:
        raise ValueError(f"Unsupported ID dataset for OOD evaluation: {name_id!r}.")
    test_transform = TRANSFORM_TEST[name_id]

    match name_id:
        case "cifar10":
            id_test = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=test_transform)
        case "pathmnist":
            id_test = PathMNIST(
                root=str(DATA_PATH),
                split="test",
                download=True,
                transform=test_transform,
                target_transform=MEDMNIST_TARGET_TRANSFORM,
            )
        case "bloodmnist":
            id_test = BloodMNIST(
                root=str(DATA_PATH),
                split="test",
                download=True,
                transform=test_transform,
                target_transform=MEDMNIST_TARGET_TRANSFORM,
            )
        case _:
            raise ValueError(f"Unsupported ID dataset for OOD evaluation: {name_id!r}.")

    match name_ood:
        case "cifar100":
            ood_test = torchvision.datasets.CIFAR100(
                root=DATA_PATH, train=False, download=True, transform=test_transform
            )
        case "tin":
            ood_test = TinyImageNet(root=DATA_PATH, split="val", transform=test_transform, download=True)
        case "mnist":
            ood_test = torchvision.datasets.MNIST(
                root=DATA_PATH,
                train=False,
                download=True,
                transform=T.Compose([T.RGB(), *test_transform.transforms]),
            )
        case "svhn":
            ood_test = torchvision.datasets.SVHN(root=DATA_PATH, split="test", download=True, transform=test_transform)
        case "textures":
            ood_test = torchvision.datasets.DTD(root=DATA_PATH, split="test", download=True, transform=test_transform)
        case "places365":
            ood_test = torchvision.datasets.Places365(
                root=DATA_PATH, split="val", small=True, download=True, transform=test_transform
            )
        # The bloodmnist and pathmnist test transforms lead with T.RGB(), so the single-channel
        # MedMNIST sets below are promoted to three channels exactly as the ID data is, and it is
        # a no-op on the RGB ones (bloodmnist, dermamnist).
        case "bloodmnist":
            ood_test = BloodMNIST(
                root=str(DATA_PATH),
                split="test",
                download=True,
                transform=test_transform,
                target_transform=MEDMNIST_TARGET_TRANSFORM,
            )
        case "pneumoniamnist":
            ood_test = PneumoniaMNIST(
                root=str(DATA_PATH),
                split="test",
                download=True,
                transform=test_transform,
                target_transform=MEDMNIST_TARGET_TRANSFORM,
            )
        case "tissuemnist":
            ood_test = TissueMNIST(
                root=str(DATA_PATH),
                split="test",
                download=True,
                transform=test_transform,
                target_transform=MEDMNIST_TARGET_TRANSFORM,
            )
        case "octmnist":
            ood_test = OCTMNIST(
                root=str(DATA_PATH),
                split="test",
                download=True,
                transform=test_transform,
                target_transform=MEDMNIST_TARGET_TRANSFORM,
            )
        case "breastmnist":
            ood_test = BreastMNIST(
                root=str(DATA_PATH),
                split="test",
                download=True,
                transform=test_transform,
                target_transform=MEDMNIST_TARGET_TRANSFORM,
            )
        case "dermamnist":
            ood_test = DermaMNIST(
                root=str(DATA_PATH),
                split="test",
                download=True,
                transform=test_transform,
                target_transform=MEDMNIST_TARGET_TRANSFORM,
            )
        case "pathmnist":
            ood_test = PathMNIST(
                root=str(DATA_PATH),
                split="test",
                download=True,
                transform=test_transform,
                target_transform=MEDMNIST_TARGET_TRANSFORM,
            )
        case "retinamnist":
            ood_test = RetinaMNIST(
                root=str(DATA_PATH),
                split="test",
                download=True,
                transform=test_transform,
                target_transform=MEDMNIST_TARGET_TRANSFORM,
            )
        case _:
            raise ValueError(f"Unsupported OOD dataset: {name_ood!r}.")

    n = min(len(id_test), len(ood_test))
    rng = torch.Generator().manual_seed(seed)
    id_idx = torch.randperm(len(id_test), generator=rng)[:n].tolist()
    ood_idx = torch.randperm(len(ood_test), generator=rng)[:n].tolist()
    id_subset = Subset(id_test, id_idx)
    ood_subset = Subset(ood_test, ood_idx)

    id_loader = DataLoader(id_subset, shuffle=False, **loader_kwargs)
    ood_loader = DataLoader(ood_subset, shuffle=False, **loader_kwargs)
    return id_loader, ood_loader


def get_shift_data(
    name: str,
    corruption: str,
    severity: int,
    **loader_kwargs: Any,
) -> DataLoader[Any]:
    """Return a corrupted test loader for distribution-shift evaluation.

    Loads a corruption-benchmark version of the named dataset's test set (CIFAR-10-C for
    'cifar10', MedMNIST-C for the MedMNIST datasets): the clean test set with a single corruption
    applied at a single severity, for evaluating a model trained on the clean data under
    shift. Inputs are preprocessed with TRANSFORM_SHIFT[name]: like the clean test images,
    with an extra resize to the clean resolution where the benchmark ships larger images
    (MedMNIST-C is 224x224). The loader is not shuffled and yields all corrupted test
    images for the requested corruption and severity.

    Args:
        name: Base dataset name, selecting the corruption dataset and its transform.
        corruption: Corruption type. One of the names in SHIFT_CORRUPTIONS[name].
        severity: Corruption severity in 1..5.
        **loader_kwargs: Forwarded to the DataLoader (batch_size, num_workers, pin_memory, ...).

    Returns:
        The unshuffled corrupted test loader.

    Raises:
        ValueError: name has no shift dataset, or corruption or severity is invalid
            (the latter two raised by the benchmark dataset class).
    """
    match name:
        case "cifar10":
            test = CIFAR10C(
                root=DATA_PATH,
                corruption=corruption,
                severity=severity,
                transform=TRANSFORM_SHIFT[name],
                download=True,
            )
        case "bloodmnist" | "pathmnist":
            # MedMNIST-C ships 224x224 corrupted images while the models train on native 28x28
            # MedMNIST. The downsample is identical in every run, so it is done once per cell and
            # cached (_medmnistc_cell); what remains here is the clean test preprocessing, which
            # is what TRANSFORM_SHIFT applies after its resize. Labels are plain ints, so no
            # target_transform. Building the cache downloads the dataset's archive (up to several
            # GB) from Zenodo into DATA_PATH/medmnist_c on first use.
            images, targets = _medmnistc_cell(name, corruption, severity)
            test = _CachedShiftDataset(images, targets, TRANSFORM_TEST[name])
        case _:
            raise ValueError(f"No shift dataset available for: {name!r}")
    return DataLoader(test, shuffle=False, **loader_kwargs)
