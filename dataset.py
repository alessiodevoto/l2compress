from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch

import logging

logger = logging.getLogger(__name__)


class ChunkDataset(IterableDataset):
    """
    Iterable dataset
    Yields sequences with a fixed length

    Args:
        dataset: huggingface dataset
        chunk_size: the length of each sequence
        tokenizer: tokenizer
    """

    def __init__(self, dataset, tokenizer, chunk_size, merge_chunks=True, text_col="text"):
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.tokenizer = tokenizer
        self.merge_chunks = merge_chunks  
        self.text_col = text_col #   content

    def get_token_ids(self, text):
        '''
        Args:
            text: string text
        Returns:
            the list of tokenized text; ends with SEP
        '''
        token_ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        token_ids.append(self.tokenizer.eos_token_id)
        return token_ids

    def random_chunk_iterator(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id
        num_workers = worker_info.num_workers

        cur_chunk = [self.tokenizer.bos_token_id] 
        cur_chunk_remain = self.chunk_size - 1  # we have one token "BOS" in the chunk

        for idx, item in enumerate(self.dataset):

            if idx % num_workers != worker_id:
                continue

            token_ids = self.get_token_ids(item[self.text_col])

            num_tokens = len(token_ids)
            item_offset = 0

            while num_tokens:
                num_to_take = min(num_tokens, cur_chunk_remain)
                cur_chunk.extend(token_ids[item_offset:item_offset + num_to_take])
                item_offset += num_to_take
                cur_chunk_remain -= num_to_take
                num_tokens -= num_to_take

                if cur_chunk_remain == 0:
                    yield {"input_ids": cur_chunk}
                    cur_chunk = [self.tokenizer.bos_token_id]
                    cur_chunk_remain = self.chunk_size - 1
    

    def basic_chunk_iterator(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id
        num_workers = worker_info.num_workers

        cur_chunk = [self.tokenizer.bos_token_id]

        for idx, item in enumerate(self.dataset):

            if idx % num_workers != worker_id:
                continue

            token_ids = self.get_token_ids(item[self.text_col])

            yield {"input_ids": cur_chunk + token_ids[:self.chunk_size-1]}
            cur_chunk = [self.tokenizer.bos_token_id]
            

    def __iter__(self):
        iterator = self.random_chunk_iterator() if self.merge_chunks else self.basic_chunk_iterator()
        try:
            while True:
                yield next(iterator)
        except Exception as e:
            logger.info(f"stop interation")

    def get_dataloader(self, batch_size=1, num_workers=1):

        def data_collator(examples: list):
            input_ids = torch.tensor([item["input_ids"] for item in examples])
            attention_mask = torch.ones_like(input_ids)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        return DataLoader(self, batch_size=batch_size, num_workers=num_workers, collate_fn=data_collator)


class PromptDataset(Dataset):
    def __init__(self, prompt_list, tokenizer):
        self.data = prompt_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def get_dataloader(self, batch_size, max_length):
        def collate_fn(items):
            batch = [item["prompt"] for item in items]
            targets = [item["target"] for item in items]
            # todo: check if it starts with BOS token
            inputs = self.tokenizer(batch, padding=True, return_tensors="pt", truncation=True, max_length=max_length)
            return {"inputs": inputs, "targets": targets}

        return DataLoader(self, batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8)
