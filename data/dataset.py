import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np

# Dataset for 2025 AI Challenge
class DNA2025Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, label_transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.label_transform = label_transform

        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Data root directory not found: {self.root_dir}")

        self.image_dir = os.path.join(root_dir, 'image', split)
        self.label_dir = os.path.join(root_dir, 'labelmap', split)

        self.data_pairs = []
        self._load_data()

        print(f"\n✅ Total loaded Image: {len(self.data_pairs)}\n")

    def _get_label_filename(self, image_filename):
        if image_filename.endswith('.jpg'):
            return image_filename.replace('.jpg', '_CategoryId.png')
        elif 'leftImg8bit.png' in image_filename:
            return image_filename.replace('_leftImg8bit.png', '_gtFine_CategoryId.png')
        elif image_filename.endswith('.png'):
            return image_filename.replace('.png', '_CategoryId.png')
        else:
            base_name = os.path.splitext(image_filename)[0]
            return f"{base_name}_CategoryId.png"

    def _load_data(self):
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found {self.image_dir}")
        
        target_folders = sorted(os.listdir(self.image_dir))
        print(f"Folder to use\n->{target_folders}\n")
        
        total_matched_count = 0
        current_folder_matched_count = 0
        missing_labels = []

        for sub_folder in target_folders:
            print(f"sub_folder: {sub_folder} -> ", end='')
            sub_image_path = os.path.join(self.image_dir, sub_folder)
            sub_label_path = os.path.join(self.label_dir, sub_folder)

            if not os.path.isdir(sub_image_path):
                continue

            if not os.path.exists(sub_image_path):
                print(f"Warning: Label folder not found for {sub_folder}")
                continue

            for img_name in sorted(os.listdir(sub_image_path)):
                if not img_name.lower().endswith(('.png', '.jpg', 'jpeg')):
                    continue

                img_path = os.path.join(sub_image_path, img_name)

                label_name = self._get_label_filename(img_name)
                label_path = os.path.join(sub_label_path, label_name)

                if os.path.exists(label_path):
                    self.data_pairs.append({
                        'image': img_path,
                        'label': label_path,
                        'folder': sub_folder,
                        'image_name': img_name,
                        'label_name': label_name
                    })
                    total_matched_count += 1
                    current_folder_matched_count += 1
                else:
                    missing_labels.append({
                        'folder': sub_folder,
                        'image': img_name,
                        'expected_label': label_name
                    })
            
            print(f"Successed matching count: {current_folder_matched_count}")
            current_folder_matched_count = 0
            if missing_labels:
                print(f"❌ Matching failed {len(missing_labels)}")

                for item in missing_labels:
                    print(f"  -  {item['folder']}/{item['image']} -> {item['folder']}/{item['expected_label']} (Not exist)")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        data_info = self.data_pairs[idx]

        image = Image.open(data_info['image']).convert('RGB')
        label = Image.open(data_info['label']).convert('')

        if self.transform:
            image = self.transform(image)

        if self.label_transform:
            label = self.label_transform(label)
        else:
            label = torch.from_numpy(np.array(label)).long()

        return image, label, data_info['folder']

if __name__ == "__main__":
    dataset = DNA2025Dataset('./data/SemanticDataset_final')