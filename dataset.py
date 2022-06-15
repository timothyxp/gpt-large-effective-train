from torch.utils.data import Dataset, DataLoader
import random


class BatchDataset(Dataset):
    def __init__(self, data, batch_size, tokenizer):
        self.data = data
        self.batch_size = batch_size
        self.tokenizer = tokenizer

    def __len__(self):
        return (len(self.data) - self.batch_size) // self.batch_size

    def shuffle(self):
        random.shuffle(self.data)

    def __getitem__(self, index):
        index = random.randint(0, len(self.data) - self.batch_size)

        result = self.tokenizer(
            self.data[index: index + self.batch_size],
            max_length=1024, padding=True,
            pad_to_multiple_of=256, return_tensors='pt'
        )

        result['labels'] = result['input_ids']

        return result