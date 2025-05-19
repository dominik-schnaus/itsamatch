"""Define datasets for image-text matching tasks."""

import shutil
from abc import ABC
from os import listdir, makedirs
from pathlib import Path
from typing import Callable

import requests
from torchvision.datasets import (
    CIFAR10 as CIFAR10Base,
)
from torchvision.datasets import (
    CIFAR100 as CIFAR100Base,
)
from torchvision.datasets import (
    CocoCaptions as CocoCaptionsBase,
)
from torchvision.datasets import (
    ImageFolder,
    ImageNet,
    VisionDataset,
)
from torchvision.datasets.imagenet import META_FILE
from torchvision.datasets.utils import download_and_extract_archive


class Dataset(VisionDataset, ABC):
    """
    Abstract base class for all datasets.

    Attributes:
        is_paired (bool): Whether the dataset contains paired image-text data.
        prompts (list[str] | None): A list of prompt templates for classification datasets.
        classes (list[str] | None): A list of class names for classification datasets.
    """

    is_paired: bool
    prompts: list[str] | None = None
    classes: list[str] | None = None

    def __repr__(self):
        """Return a string representation of the dataset."""
        return f"{self.__class__.__name__}({', '.join([f'{k}={v}' for k, v in self.__dict__.items()])})"

    def __str__(self):
        """Return the name of the dataset."""
        return self.__class__.__name__


class CocoCaptions(Dataset, CocoCaptionsBase):
    """
    COCO Captions dataset. This is a paired image-text dataset.
    """

    is_paired = True
    prompts = None
    classes = None


class CIFAR10(Dataset, CIFAR10Base):
    """
    CIFAR-10 dataset. This is a classification dataset.
    """

    is_paired = False
    prompts = [
        "a photo of a {}.",
        "a blurry photo of a {}.",
        "a black and white photo of a {}.",
        "a low contrast photo of a {}.",
        "a high contrast photo of a {}.",
        "a bad photo of a {}.",
        "a good photo of a {}.",
        "a photo of a small {}.",
        "a photo of a big {}.",
        "a photo of the {}.",
        "a blurry photo of the {}.",
        "a black and white photo of the {}.",
        "a low contrast photo of the {}.",
        "a high contrast photo of the {}.",
        "a bad photo of the {}.",
        "a good photo of the {}.",
        "a photo of the small {}.",
        "a photo of the big {}.",
    ]
    classes = [
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

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        """
        Initialize CIFAR10 dataset.

        Args:
            root (str or ``pathlib.Path``): Root directory of dataset where directory
                ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
            train (bool, optional): If True, creates dataset from training set, otherwise
                creates from test set.
            transform (callable, optional): A function/transform that takes in a PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
        """

        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    def __str__(self):
        """Return the name of the dataset."""
        return "CIFAR-10"


class CINIC10(Dataset, ImageFolder):
    """
    CINIC-10 dataset. This is a classification dataset, an extension of CIFAR-10.
    """

    is_paired = False
    prompts = [
        "itap of a {}.",
        "a bad photo of the {}.",
        "a origami {}.",
        "a photo of the large {}.",
        "a {} in a video game.",
        "art of the {}.",
        "a photo of the small {}.",
    ]

    archive = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"
    filename = "CINIC-10.tar.gz"
    base_folder = "cinic-10"

    def __init__(
        self,
        root: str,
        split: str = "train",
        download: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize CINIC10 dataset.

        Args:
            root (str): Root directory of dataset.
            split (str, optional): The dataset split, supports "train", "val" (maps to "valid"), or "test".
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            **kwargs: Additional arguments for ImageFolder.
        """
        self.root = Path(root)

        # Map "val" split to "valid" as used in the CINIC-10 directory structure
        if split == "val":
            split = "valid"

        self.split = split

        target_folder = self.root / self.base_folder / self.split
        # Download the dataset if it's not found and download is requested
        if download and not target_folder.exists():
            print("CINIC-10 is not found, downloading...")
            self.download()
            print("CINIC-10 is downloaded")

        super().__init__(target_folder, **kwargs)

    def download(self):
        """Download the CINIC-10 data if it doesn't exist in processed_folder already."""
        # Use torchvision utility to download and extract the archive
        download_and_extract_archive(
            url=self.archive,
            download_root=self.root,
            extract_root=self.root / self.base_folder,
            filename=self.filename,
            md5=None,  # CINIC-10 does not provide an MD5 checksum
        )

    def __str__(self):
        """Return the name of the dataset."""
        return "CINIC-10"


class CIFAR100(Dataset, CIFAR100Base):
    """
    CIFAR-100 dataset. This is a classification dataset.
    """

    is_paired = False
    prompts = [
        "a photo of a {}.",
        "a blurry photo of a {}.",
        "a black and white photo of a {}.",
        "a low contrast photo of a {}.",
        "a high contrast photo of a {}.",
        "a bad photo of a {}.",
        "a good photo of a {}.",
        "a photo of a small {}.",
        "a photo of a big {}.",
        "a photo of the {}.",
        "a blurry photo of the {}.",
        "a black and white photo of the {}.",
        "a low contrast photo of the {}.",
        "a high contrast photo of the {}.",
        "a bad photo of the {}.",
        "a good photo of the {}.",
        "a photo of the small {}.",
        "a photo of the big {}.",
    ]
    classes = [
        "apple",
        "aquarium fish",
        "baby",
        "bear",
        "beaver",
        "bed",
        "bee",
        "beetle",
        "bicycle",
        "bottle",
        "bowl",
        "boy",
        "bridge",
        "bus",
        "butterfly",
        "camel",
        "can",
        "castle",
        "caterpillar",
        "cattle",
        "chair",
        "chimpanzee",
        "clock",
        "cloud",
        "cockroach",
        "couch",
        "crab",
        "crocodile",
        "cup",
        "dinosaur",
        "dolphin",
        "elephant",
        "flatfish",
        "forest",
        "fox",
        "girl",
        "hamster",
        "house",
        "kangaroo",
        "keyboard",
        "lamp",
        "lawn mower",
        "leopard",
        "lion",
        "lizard",
        "lobster",
        "man",
        "maple tree",
        "motorcycle",
        "mountain",
        "mouse",
        "mushroom",
        "oak tree",
        "orange",
        "orchid",
        "otter",
        "palm tree",
        "pear",
        "pickup truck",
        "pine tree",
        "plain",
        "plate",
        "poppy",
        "porcupine",
        "possum",
        "rabbit",
        "raccoon",
        "ray",
        "road",
        "rocket",
        "rose",
        "sea",
        "seal",
        "shark",
        "shrew",
        "skunk",
        "skyscraper",
        "snail",
        "snake",
        "spider",
        "squirrel",
        "streetcar",
        "sunflower",
        "sweet pepper",
        "table",
        "tank",
        "telephone",
        "television",
        "tiger",
        "tractor",
        "train",
        "trout",
        "tulip",
        "turtle",
        "wardrobe",
        "whale",
        "willow tree",
        "wolf",
        "woman",
        "worm",
    ]

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        """
        Initialize CIFAR100 dataset.

        Args:
            root (str or ``pathlib.Path``): Root directory of dataset where directory
                ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
            train (bool, optional): If True, creates dataset from training set, otherwise
                creates from test set.
            transform (callable, optional): A function/transform that takes in a PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
        """
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    def __str__(self):
        """Return the name of the dataset."""
        return "CIFAR-100"


class ImageNet100(Dataset, ImageNet):
    """
    ImageNet-100 dataset. A subset of ImageNet with 100 classes.
    This dataset can be generated from a full ImageNet dataset.
    It supports using original class names or class names with definitions.
    """

    is_paired = False
    prompts_original = [
        "itap of a {}.",
        "a bad photo of the {}.",
        "a origami {}.",
        "a photo of the large {}.",
        "a {} in a video game.",
        "art of the {}.",
        "a photo of the small {}.",
    ]
    classes_original = [
        "American robin",
        "Gila monster",
        "eastern hog-nosed snake",
        "garter snake",
        "green mamba",
        "European garden spider",
        "lorikeet",
        "goose",
        "rock crab",
        "fiddler crab",
        "American lobster",
        "little blue heron",
        "American coot",
        "Chihuahua",
        "Shih Tzu",
        "Papillon",
        "toy terrier",
        "Treeing Walker Coonhound",
        "English foxhound",
        "borzoi",
        "Saluki",
        "American Staffordshire Terrier",
        "Chesapeake Bay Retriever",
        "Vizsla",
        "Kuvasz",
        "Komondor",
        "Rottweiler",
        "Dobermann",
        "Boxer",
        "Great Dane",
        "Standard Poodle",
        "Mexican hairless dog (xoloitzcuintli)",
        "coyote",
        "African wild dog",
        "red fox",
        "tabby cat",
        "meerkat",
        "dung beetle",
        "stick insect",
        "leafhopper",
        "hare",
        "wild boar",
        "gibbon",
        "langur",
        "ambulance",
        "baluster / handrail",
        "bassinet",
        "boathouse",
        "poke bonnet",
        "bottle cap",
        "car wheel",
        "bell or wind chime",
        "movie theater",
        "cocktail shaker",
        "computer keyboard",
        "Dutch oven",
        "football helmet",
        "gas mask or respirator",
        "hard disk drive",
        "harmonica",
        "honeycomb",
        "clothes iron",
        "jeans",
        "lampshade",
        "laptop computer",
        "milk can",
        "mixing bowl",
        "modem",
        "moped",
        "graduation cap",
        "mousetrap",
        "obelisk",
        "park bench",
        "pedestal",
        "pickup truck",
        "pirate ship",
        "purse",
        "fishing casting reel",
        "rocking chair",
        "rotisserie",
        "safety pin",
        "sarong",
        "balaclava ski mask",
        "slide rule",
        "stretcher",
        "front curtain",
        "throne",
        "tile roof",
        "tripod",
        "hot tub",
        "vacuum cleaner",
        "window screen",
        "airplane wing",
        "cabbage",
        "cauliflower",
        "pineapple",
        "carbonara",
        "chocolate syrup",
        "gyromitra",
        "stinkhorn mushroom",
    ]

    prompts_definition = None
    classes_definition = [
        "American robin bird - large American thrush having a rust-red breast and abdomen",
        "Gila monster lizard - large orange and black lizard of southwestern United States; not dangerous unless molested",
        "Hognose snake - harmless North American snake with upturned nose; may spread its head and neck or play dead when disturbed",
        "Garter snake - any of numerous nonvenomous longitudinally-striped viviparous North American and Central American snakes",
        "Green mamba - green phase of the black mamba",
        "Spider - a spider common in European gardens",
        "Lorikeet parrot bird - any of various small brightly colored Australasian parrots having a brush-tipped tongue for feeding on nectar and soft fruits",
        "Goose - web-footed long-necked typically gregarious migratory aquatic birds usually larger and less aquatic than ducks",
        "Rock crab - crab of eastern coast of North America",
        "Fiddler crab - burrowing crab of American coastal regions having one claw much enlarged in the male",
        "Lobster - lobster of Atlantic coast of America",
        "Little blue heron bird - small bluish-grey heron of the western hemisphere",
        "American coot bird - a coot found in North America",
        "Chihuahua dog - an old breed of tiny short-haired dog with protruding eyes from Mexico held to antedate Aztec civilization",
        "Shih-Tzu dog - a Chinese breed of small dog similar to a Pekingese",
        "Papillon dog - small slender toy spaniel with erect ears and a black-spotted brown to white coat",
        "Toy terrier dog - a small active dog",
        "Walker foxhound dog - an American breed of foxhound",
        "English foxhound dog - an English breed slightly larger than the American foxhounds originally used to hunt in packs",
        "Russian wolfhound dog - tall fast-moving dog breed",
        "Saluki gazelle hound dog - old breed of tall swift keen-eyed hunting dogs resembling greyhounds; from Egypt and southwestern Asia",
        "American Staffordshire pit bull terrier dog - American breed of muscular terriers with a short close-lying stiff coat",
        "Chesapeake Bay retriever dog - American breed having a short thick oily coat ranging from brown to light tan",
        "Hungarian pointer dog - Hungarian hunting dog resembling the Weimaraner but having a rich deep red coat",
        "Kuvasz dog - long-established Hungarian breed of tall light-footed but sturdy white dog; used also as a hunting dog",
        "Komondor dog - Hungarian breed of large powerful shaggy-coated white dog; used also as guard dog",
        "Rottweiler dog - German breed of large vigorous short-haired cattle dogs",
        "Doberman pinscher dog - medium large breed of dog of German origin with a glossy black and tan coat; used as a watchdog",
        "Boxer dog - a breed of stocky medium-sized short-haired dog with a brindled coat and square-jawed muzzle developed in Germany",
        "Great Dane dog - very large powerful smooth-coated breed of dog",
        "Standard poodle dog - a breed or medium-sized poodles",
        "Mexican hairless dog - any of an old breed of small nearly hairless dogs of Mexico",
        "Coyote - small wolf native to western North America",
        "Wild African hunting dog - a powerful doglike mammal of southern and eastern Africa that hunts in large packs; now rare in settled area",
        "Red fox - the common Old World fox; having reddish-brown fur; commonly considered a single circumpolar species",
        "Tabby cat - a cat with a grey or tawny coat mottled with black",
        "Meerkat - a mongoose-like viverrine of South Africa having a face like a lemur and only four toes",
        "Beetle - any of numerous beetles that roll balls of dung on which they feed and in which they lay eggs",
        "Walking stick insect - any of various mostly tropical insects having long twiglike bodies",
        "Leafhopper - small leaping insect that sucks the juices of plants",
        "Hare - swift timid long-eared mammal larger than a rabbit having a divided upper lip and long hind legs; young born furred and with open eyes",
        "Wild boar - Old World wild swine having a narrow body and prominent tusks from which most domestic swine come; introduced in United States",
        "Gibbon - smallest and most perfectly anthropoid arboreal ape having long arms and no tail; of southern Asia and East Indies",
        "Langur - slender long-tailed monkey of Asia",
        "Ambulance - a vehicle that takes people to and from hospitals",
        "Handrail - a railing at the side of a staircase or balcony to prevent people from falling",
        "Bassinet - a basket (usually hooded) used as a baby's bed",
        "Boathouse - a shed at the edge of a river or lake; used to store boats",
        "Poke bonnet - a hat tied under the chin",
        "Bottlecap - a cap that seals a bottle",
        "Car wheel - a wheel that has a tire and rim and hubcap; used to propel the car",
        "Gong - a percussion instrument consisting of a set of tuned bells that are struck with a hammer; used as an orchestral instrument",
        "Movie theater - a theater where films are shown",
        "Cocktail shaker - a shaker for mixing cocktails",
        "Computer keyboard - a keyboard that is a data input device for computers; arrangement of keys is modelled after the typewriter keyboard",
        "Dutch oven - an oven consisting of a metal box for cooking in front of a fire",
        "Football helmet - a padded helmet with a face mask to protect the head of football players",
        "Gasmask - a protective mask with a filter; protects the face and lungs against poisonous gases",
        "Hard disk - a rigid magnetic disk mounted permanently in a drive unit",
        "Harmonica - a small rectangular free-reed instrument having a row of free reeds set back in air holes and played by blowing into the desired hole",
        "Honeycomb - a framework of hexagonal cells resembling the honeycomb built by bees",
        "Smoothing iron - home appliance consisting of a flat metal base that is heated and used to smooth cloth",
        "Denim jeans - (usually plural) close-fitting trousers of heavy denim for manual work or casual wear",
        "Lamp shade - a protective ornamental shade used to screen a light bulb from direct view",
        "Laptop computer - a portable computer small enough to use in your lap",
        "Milk can - large can for transporting milk",
        "Mixing bowl - bowl used with an electric mixer",
        "Modem - (from a combination of MOdulate and DEModulate) electronic equipment consisting of a device used to connect computers by a telephone line",
        "Moped - a motorbike that can be pedaled or driven by a low-powered gasoline engine",
        "Mortarboard - an academic cap with a flat square with a tassel on top",
        "Mousetrap - a trap for catching mice",
        "Obelisk - a stone pillar having a rectangular cross section tapering towards a pyramidal top",
        "Park bench - a bench in a public park",
        "Pedestal - an architectural support or base (as for a column or statue)",
        "Pickup truck - a light truck with an open body and low sides and a tailboard",
        "Pirate ship - a ship that is manned by pirates",
        "Purse - a small bag for carrying money",
        "Fishing casting reel - winder consisting of a revolving spool with a handle; attached to a fishing rod",
        "Rocking chair - a chair mounted on rockers",
        "rotisserie - an oven or broiler equipped with a rotating spit on which meat cooks as it turns",
        "Safety pin - a pin in the form of a clasp; has a guard so the point of the pin will not stick the user",
        "Sarong - a loose skirt consisting of brightly colored fabric wrapped around the body; worn by both women and men in the South Pacific",
        "Ski mask - a woolen face mask to protect the face from cold while skiing on snow",
        "Slide rule - analog computer consisting of a handheld instrument used for rapid calculations; have been replaced by pocket calculators",
        "Stretcher - a litter for transporting people who are ill or wounded or dead; usually consists of a sheet of canvas stretched between two poles",
        "Theater curtain - a hanging cloth that conceals the stage from the view of the audience; rises or parts at the beginning and descends or closes between acts and at the end of a performance",
        "Throne - the chair of state for a monarch, bishop, etc.",
        "Tile roof - a roof made of fired clay tiles",
        "Tripod - a three-legged rack used for support",
        "Hot tub - a large open vessel for holding or storing liquids",
        "Vacuum cleaner - an electrical home appliance that cleans by suction",
        "Window screen - screen to keep insects from entering a building through the open window",
        "Airplane wing - one of the horizontal airfoils on either side of the fuselage of an airplane",
        "Cabbage - any of several varieties of cabbage having a large compact globular head; may be steamed or boiled or stir-fried or used raw in coleslaw",
        "Cauliflower - compact head of white undeveloped flowers",
        "Pineapple - large sweet fleshy tropical fruit with a terminal tuft of stiff leaves; widely cultivated",
        "Carbonara - sauce for pasta; contains eggs and bacon or ham and grated cheese",
        "Chocolate syrup - sauce made with unsweetened chocolate or cocoa and sugar and water",
        "Gyromitra mushroom - any fungus of the genus Gyromitra",
        "Stinkhorn mushroom - any of various ill-smelling brown-capped fungi of the order Phallales",
    ]

    target_class_url = "https://raw.githubusercontent.com/danielchyeh/ImageNet-100-Pytorch/refs/heads/main/IN100.txt"

    def __init__(
        self,
        root: str,
        imagenet_root: str,
        split: str = "train",
        definitions: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize ImageNet100 dataset.

        Args:
            root (str): Root directory where ImageNet-100 subset will be stored/generated.
            imagenet_root (str): Root directory of the full ImageNet dataset.
            split (str, optional): The dataset split, typically "train" or "val".
            definitions (bool, optional): If True, use class names with definitions.
                                          Otherwise, use original class names.
            **kwargs: Additional arguments for ImageNet (VisionDataset).
        """
        root = Path(root)
        imagenet_root = Path(imagenet_root)
        imagenet100_root = root / "imagenet100"
        target_folder = imagenet100_root / split
        # Generate ImageNet-100 subset if it doesn't exist
        if not target_folder.exists():
            print("ImageNet-100 is not found, generating...")
            makedirs(target_folder, exist_ok=True)
            target_class_file = imagenet100_root / "IN100.txt"
            source_folder = imagenet_root / split
            self.generate_data(source_folder, target_folder, target_class_file)
            print("ImageNet-100 is generated")

        # Initialize the parent ImageNet class with the path to the generated ImageNet-100 subset
        super().__init__(imagenet100_root, split, **kwargs)

        self.definitions = definitions
        # Set prompts and classes based on whether definitions are used
        if definitions:
            self.prompts = self.prompts_definition
            self.classes = self.classes_definition
        else:
            self.prompts = self.prompts_original
            self.classes = self.classes_original

    def generate_data(
        self, source_folder: Path, target_folder: Path, target_class_file: Path
    ):
        """
        Generates the ImageNet-100 subset by copying relevant class folders
        from the full ImageNet dataset.

        Args:
            source_folder (Path): Path to the split (train/val) in the full ImageNet dataset.
            target_folder (Path): Path where the ImageNet-100 subset for the split will be created.
            target_class_file (Path): Path to the file containing the list of 100 target class IDs.
        """
        # From https://github.com/danielchyeh/ImageNet-100-Pytorch/blob/main/generate_IN100.py
        # Download the target class list if it doesn't exist
        if not target_class_file.exists():
            response = requests.get(self.target_class_url)
            with open(target_class_file, "wb") as f:
                f.write(response.content)

        # Read the target class IDs
        target_class_ids = []
        with open(target_class_file, "r") as txt_data:
            for line in txt_data:
                s = str(line.split("\n")[0])
                target_class_ids.append(s)

        # Copy class folders from source to target if they are in the target_class_ids list
        for class_dir_name in listdir(source_folder):
            if class_dir_name in target_class_ids:
                shutil.copytree(
                    source_folder / class_dir_name,
                    target_folder / class_dir_name,
                )

        # Copy the META_FILE (contains class to index mapping) to the parent of the target_folder
        # This is required by the torchvision.datasets.ImageNet class
        target_parent = target_folder.parent
        meta_file_path_target = target_parent / META_FILE
        if not meta_file_path_target.exists():
            # Source META_FILE is in the parent of the source_folder (e.g., imagenet_root/META_FILE)
            meta_file_path_source = source_folder.parent / META_FILE
            if meta_file_path_source.exists():
                shutil.copyfile(meta_file_path_source, meta_file_path_target)
            else:
                print(
                    f"Warning: {META_FILE} not found at {meta_file_path_source}. ImageNet dataset might not load correctly."
                )

    def __str__(self):
        """Return the name of the dataset."""
        return "ImageNet-100"
